import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
# Store embeddings in Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def store_in_qdrant(collection_name: str, vectors: np.ndarray, port: int, batch_size=1000):
    """
    Recreates a Qdrant collection on the given port, then upserts vectors in smaller batches.
    Uses wait=False so it doesn't block until indexing is done.
    """
    client = QdrantClient(host="localhost", port=port, timeout=600.0)

    dim = vectors.shape[1]
    print(f"Recreating '{collection_name}' on port {port} (dim={dim})...")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    n = len(vectors)
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        chunk = [
            PointStruct(id=idx, vector=vectors[idx].tolist())
            for idx in range(start, end)
        ]
        client.upsert(collection_name=collection_name, points=chunk, wait=False)
        start = end
        print(f"Upserted {start}/{n} -> {collection_name}")

    print(f"Finished storing {n} vectors in '{collection_name}' on port {port}")

# Load precomputed Wikipedia embeddings
def load_embeddings():
    orig_embeddings = np.load("data/wikipedia_original_embeddings.npy")
    return orig_embeddings

# ShuffleDistModelMulti class remains unchanged
class ShuffleDistModelMulti(keras.Model):
    def __init__(self, encoder, X_orig, X_shufA, X_shufB, batch_size):
        super().__init__()
        self.encoder = encoder
        self.X_orig = tf.constant(X_orig, dtype=tf.float32)
        self.X_shufA = tf.Variable(X_shufA, trainable=False, dtype=tf.float32)
        self.X_shufB = tf.Variable(X_shufB, trainable=False, dtype=tf.float32)
        self.log_alpha = tf.Variable(0.0, trainable=True)
        self.batch_size = batch_size
        self.N = X_orig.shape[0]
        self.steps_per_epoch = self.N // self.batch_size
        self.batch_idx = 0

    def train_step(self, data):
        x, _ = data
        current_bs = tf.shape(x)[0]
        start_i = self.batch_idx * self.batch_size
        end_i = start_i + current_bs

        Xa_sub = self.X_shufA[start_i:end_i]
        Xb_sub = self.X_shufB[start_i:end_i]

        with tf.GradientTape() as tape:
            alpha = tf.exp(self.log_alpha)
            Z = self.encoder(x, training=True)
            Za = self.encoder(Xa_sub, training=True)
            diff_xa = x - Xa_sub
            orig_distA = tf.reduce_sum(diff_xa * diff_xa, axis=1)
            diff_za = Z - Za
            embed_distA = tf.reduce_sum(diff_za * diff_za, axis=1)

            Zb = self.encoder(Xb_sub, training=True)
            diff_xb = x - Xb_sub
            orig_distB = tf.reduce_sum(diff_xb * diff_xb, axis=1)
            diff_zb = Z - Zb
            embed_distB = tf.reduce_sum(diff_zb * diff_zb, axis=1)

            eps = 1e-9
            log_origA = tf.math.log(orig_distA + eps)
            log_embedA = tf.math.log(alpha * embed_distA + eps)
            ratio_diffA = log_origA - log_embedA

            log_origB = tf.math.log(orig_distB + eps)
            log_embedB = tf.math.log(alpha * embed_distB + eps)
            ratio_diffB = log_origB - log_embedB

            lossA = ratio_diffA * ratio_diffA
            lossB = ratio_diffB * ratio_diffB
            loss_vec = 0.5 * (lossA + lossB)

            loss = tf.reduce_mean(loss_vec)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.batch_idx += 1
        if self.batch_idx >= self.steps_per_epoch:
            self.batch_idx = 0

        return {"loss": loss}

class ShuffleTwoEpochCallback(keras.callbacks.Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.N = model.N

    def on_epoch_begin(self, epoch, logs=None):
        permA = np.random.permutation(self.N)
        permB = np.random.permutation(self.N)
        X_orig_np = self.model.X_orig.numpy()
        self.model.X_shufA.assign(X_orig_np[permA])
        self.model.X_shufB.assign(X_orig_np[permB])
        self.model.batch_idx = 0

# Main function
def main():
    import sys
    from tensorflow.keras.layers import Dense, LeakyReLU
    from tensorflow.keras import Sequential

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--batch_size", default=2500, type=int)
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()

    # Load Wikipedia embeddings
    X = load_embeddings()
    N, D = X.shape
    print(f"[INFO] Loaded Wikipedia embeddings shape=({N},{D}).")

    # Two initial permutations for shuffle
    permA_init = np.random.permutation(N)
    permB_init = np.random.permutation(N)
    X_shufA_init = X[permA_init]
    X_shufB_init = X[permB_init]

    # Build the encoder (D->512->256->128)
    encoder = Sequential([
        keras.Input((D,)),
        Dense(2048),
        LeakyReLU(alpha=0.01),
        #Dense(1024),
        #LeakyReLU(alpha=0.01),
        Dense(512),
        LeakyReLU(alpha=0.01),
        Dense(128),
        LeakyReLU(alpha=0.01),
        Dense(64)  # Output layer (same as PCA/UMAP)
    ], name="encoder")

    # Create ShuffleDistModelMulti
    model = ShuffleDistModelMulti(
        encoder=encoder,
        X_orig=X,
        X_shufA=X_shufA_init,
        X_shufB=X_shufB_init,
        batch_size=args.batch_size
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=5.0)
    model.compile(optimizer=optimizer, run_eagerly=True)

    # Keras dataset setup
    dummy_labels = np.zeros((N, 1), dtype=np.float32)

    # Callback for shuffling
    shuffle_cb = ShuffleTwoEpochCallback(model)

    print(f"[INFO] Training for {args.epochs} epochs, batch_size={args.batch_size} ...")

    model.fit(
        x=X,
        y=dummy_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=False,
        callbacks=[shuffle_cb]
    )

    # Get final embeddings
    embeddings = model.encoder(X).numpy()
    print(f"[INFO] Final embeddings shape={embeddings.shape}")

    # Save outputs
    outdir = args.outdir
    #if os.path.exists(outdir):
    #    import shutil
    #    shutil.rmtree(outdir)
    #os.makedirs(outdir)

    #model.encoder.save(os.path.join(outdir, "encoder.keras"))
    model.encoder.save(
        os.path.join(outdir, "encoder_savedmodel"),
        save_format="tf"
    )
    emb_file = os.path.join(outdir, "wikipedia_ae_embeddings.npy")
    np.save(emb_file, embeddings)
    print(f"[INFO] Saved embeddings to {emb_file}")

    # Store embeddings in qdrant
    # Store embeddings in Qdrant on port 6336
    store_in_qdrant(collection_name="wikipedia_ae", vectors=embeddings, port=6336, batch_size=1000)

    print(f"[INFO] Done.")

if __name__ == "__main__":
    main()

