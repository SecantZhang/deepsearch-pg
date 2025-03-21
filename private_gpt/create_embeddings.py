import os
import numpy as np
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import umap
import joblib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ========== CONFIG ==========
SUBSET_PERCENTAGE = 1
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PCA_COMPONENTS = 64
UMAP_COMPONENTS = 64

# Qdrant Collections & Ports
QDRANT_COLLECTIONS = {
    "original": "wikipedia_original",
    "pca":      "wikipedia_pca",
    "umap":     "wikipedia_umap"
}
QDRANT_PORTS = {
    "original": 6333,
    "pca":      6334,
    "umap":     6335
}

os.makedirs("data", exist_ok=True)

#######################################
# 1) LOAD OR CACHE THE DATASET
#######################################
subset_path = os.path.join("data", "wikipedia_subset_dataset")
if os.path.exists(subset_path):
    print(f"Loading dataset subset from '{subset_path}'...")
    dataset = Dataset.load_from_disk(subset_path)
else:
    print(f"Loading {SUBSET_PERCENTAGE}% of the Wikipedia dataset from Hugging Face...")
    dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{SUBSET_PERCENTAGE}%]")
    dataset.save_to_disk(subset_path)
    print(f"Saved dataset subset to '{subset_path}' for future runs.")

#######################################
# 2) LOAD EMBEDDING MODEL
#######################################
print(f"Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

#######################################
# 3) GENERATE EMBEDDINGS
#######################################
print("Generating embeddings...")
emb_list = []
for i, article in enumerate(dataset):
    # limit text for demonstration
    text = article["text"][:100]
    emb = model.encode(text)
    emb_list.append(emb)
    if (i + 1) % 100 == 0:
        print(f"Processed {i+1} articles...")

emb_array = np.array(emb_list, dtype=np.float32)
orig_file = os.path.join("data", "wikipedia_original_embeddings.npy")
np.save(orig_file, emb_array)
print(f"Saved original embeddings to {orig_file}")
print(f"Original dimension = {emb_array.shape[1]}")

#######################################
# 4) PCA
#######################################
print(f"Applying PCA -> {PCA_COMPONENTS} dims")
pca = PCA(n_components=PCA_COMPONENTS)
pca_array = pca.fit_transform(emb_array).astype(np.float32)
pca_file = os.path.join("data", "wikipedia_pca_embeddings.npy")
np.save(pca_file, pca_array)
print(f"Saved PCA embeddings to {pca_file}")

#######################################
# 5) UMAP
#######################################
print(f"Applying UMAP -> {UMAP_COMPONENTS} dims")
umap_reducer = umap.UMAP(n_components=UMAP_COMPONENTS, metric="cosine")
umap_array = umap_reducer.fit_transform(emb_array).astype(np.float32)
umap_file = os.path.join("data", "wikipedia_umap_embeddings.npy")
np.save(umap_file, umap_array)
print(f"Saved UMAP embeddings to {umap_file}")

umap_model = os.path.join("data", "umap_reducer.joblib")
joblib.dump(umap_reducer, umap_model)
print(f"Saved UMAP model to {umap_model}")

#######################################
# 6) STORE IN QDRANT
#######################################
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
        chunk = []
        for idx in range(start, end):
            chunk.append(PointStruct(id=idx, vector=vectors[idx].tolist()))
        client.upsert(collection_name=collection_name, points=chunk, wait=False)
        start = end
        print(f"Upserted {start}/{n} -> {collection_name}")

    print(f"Finished storing {n} vectors in '{collection_name}' on port {port}")

print("\n===== Storing embeddings in each Qdrant server =====")
# Original → port 6333
store_in_qdrant(QDRANT_COLLECTIONS["original"], emb_array,  QDRANT_PORTS["original"], batch_size=1000)
# PCA → port 6334
store_in_qdrant(QDRANT_COLLECTIONS["pca"], pca_array,      QDRANT_PORTS["pca"],      batch_size=1000)
# UMAP → port 6335
store_in_qdrant(QDRANT_COLLECTIONS["umap"], umap_array,    QDRANT_PORTS["umap"],     batch_size=1000)

print("All done!")