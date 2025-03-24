import os
import time
import numpy as np
import joblib
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
import traceback
from sklearn.decomposition import PCA
import umap

import tensorflow as tf
from tensorflow import keras

# ========== CONFIG ==========

DATA_DIR = "private_gpt/data"
DATASET_DIR = os.path.join(DATA_DIR, "wikipedia_subset_dataset")

ORIGINAL_EMB_FILE   = os.path.join(DATA_DIR, "wikipedia_original_embeddings.npy")
PCA_DIM             = 64
UMAP_MODEL_FILE     = os.path.join(DATA_DIR, "umap_reducer.joblib")
AE_MODEL_DIR        = os.path.join(DATA_DIR, "encoder_savedmodel")  # from your create-ae-embedding script

TOP_K = 10

# Qdrant ports + collections for each method
QDRANT_PORTS = {
    "original": 6333,
    "pca":      6334,
    "umap":     6335,
    "ae":       6336
}
QDRANT_COLLECTIONS = {
    "original": "wikipedia_original",
    "pca":      "wikipedia_pca",
    "umap":     "wikipedia_umap",
    "ae":       "wikipedia_ae"
}

# ========== LOAD DATA / MODELS ==========

print(f"[INFO] Checking dataset folder at: {DATASET_DIR}")
if not os.path.exists(DATASET_DIR):
    raise RuntimeError(f"Dataset dir '{DATASET_DIR}' not found. Did you run ingestion?")

print("[INFO] Loading partial Wikipedia subset from disk...")
dataset: Dataset = load_from_disk(DATASET_DIR)
print(f"[INFO] Dataset length = {len(dataset)} articles")

print("[INFO] Initializing the base embedding model: all-MiniLM-L6-v2")
base_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load original embeddings so we can re-fit PCA
if not os.path.exists(ORIGINAL_EMB_FILE):
    raise RuntimeError(f"Cannot find {ORIGINAL_EMB_FILE}, aborting.")
original_embeddings = np.load(ORIGINAL_EMB_FILE)
print(f"[INFO] Original embeddings shape = {original_embeddings.shape}")

# Fit PCA(64) from the original array
print("[INFO] Fitting PCA(64) from original array...")
pca_model = PCA(n_components=PCA_DIM, random_state=42)
pca_model.fit(original_embeddings)

# Load UMAP if present
umap_reducer = None
if os.path.exists(UMAP_MODEL_FILE):
    print(f"[INFO] Loading UMAP reducer from {UMAP_MODEL_FILE} ...")
    umap_reducer = joblib.load(UMAP_MODEL_FILE)
else:
    print("[WARN] No umap_reducer.joblib found. 'umap' method won't be available.")

# Load AE model if present
ae_encoder = None
if os.path.isdir(AE_MODEL_DIR):
    print(f"[INFO] Loading AE encoder from {AE_MODEL_DIR} ...")
    try:
        ae_encoder = keras.models.load_model(AE_MODEL_DIR)
    except Exception as e:
        # print(f"[WARN] Could not load AE model: {e}")
        print(f"[ERROR] Failed to load AE model:")
        traceback.print_exc()
else:
    print("[WARN] No AE model directory found at {AE_MODEL_DIR}. 'ae' method won't be available.")

print("[INFO] Setup complete. We can now do multi-method searches (original/pca/umap/ae).")


# ========== HELPER FUNCTIONS ==========

def _transform_query(query_384: np.ndarray, method: str) -> np.ndarray:
    """
    Helper to apply dimension reduction based on 'method'.
    - 'original' -> no reduction
    - 'pca'      -> PCA(64)
    - 'umap'     -> UMAP(64) if available
    - 'ae'       -> AE(64) if available
    """
    if method == "pca":
        return pca_model.transform(query_384.reshape(1, -1)).astype(np.float32)[0]

    elif method == "umap":
        if umap_reducer is None:
            raise RuntimeError("UMAP reducer not loaded, cannot do 'umap' method.")
        return umap_reducer.transform(query_384.reshape(1, -1)).astype(np.float32)[0]

    elif method == "ae":
        if ae_encoder is None:
            raise RuntimeError("AE model not loaded, cannot do 'ae' method.")
        # AE encoder expects shape [batch_size, 384], returns shape [batch_size, 64].
        # Convert to float32 if needed.
        query_tensor = tf.convert_to_tensor(query_384.reshape(1, -1), dtype=tf.float32)
        ae_out = ae_encoder(query_tensor, training=False)
        return ae_out.numpy().astype(np.float32)[0]

    else:
        # 'original' => no reduction
        return query_384


# ========== MAIN SEARCH FUNCTION ==========

def multi_search_pipeline(query_text: str, top_k: int = TOP_K):
    """
    Perform retrieval for *all* methods (original, pca, umap, ae) in one go:
      1) Embed the query to 384-d once via all-MiniLM-L6-v2.
      2) For each method, reduce dimension if needed, then search its Qdrant instance.
      3) Record retrieval time, top-K doc IDs, doc texts, etc.
      4) Compute overlap with 'original' doc IDs (ground truth).
      5) Return a dict of all results.

    Return format:
    {
      "original": {
         "doc_ids":        list[int],
         "doc_texts":      list[str],
         "retrieval_time": float,
         "overlap_count":  int,
         "overlap_list":   list[bool]
      },
      "pca": { ... },
      "umap": { ... },
      "ae":   { ... }
    }
    """
    # Define which methods to attempt
    all_methods = ["original", "pca", "umap", "ae"]

    # 1) Embed the user query in 384-d
    query_384 = base_embed_model.encode(query_text).astype(np.float32)

    results_dict = {}

    # 2) Retrieve for each method
    for method in all_methods:
        # Dimension reduction
        query_vec = _transform_query(query_384, method)

        # Qdrant port & collection
        if method not in QDRANT_PORTS:
            raise ValueError(f"Unknown method: {method}")
        port = QDRANT_PORTS[method]
        collection_name = QDRANT_COLLECTIONS[method]

        client = QdrantClient(host="localhost", port=port)

        # Time the search
        start_time = time.time()
        qdrant_results: list[ScoredPoint] = client.search(
            collection_name=collection_name,
            query_vector=query_vec,
            limit=top_k
        )
        end_time = time.time()

        # Gather IDs & texts
        doc_ids = [int(r.id) for r in qdrant_results]
        doc_texts = [dataset[idx]["text"] for idx in doc_ids]

        results_dict[method] = {
            "doc_ids": doc_ids,
            "doc_texts": doc_texts,
            "retrieval_time": end_time - start_time
        }

    # 3) Compute overlap with 'original' ground truth
    if "original" in results_dict:
        original_ids = set(results_dict["original"]["doc_ids"])
        # For 'original' itself
        results_dict["original"]["overlap_count"] = len(results_dict["original"]["doc_ids"])
        results_dict["original"]["overlap_list"] = [True] * len(results_dict["original"]["doc_ids"])

        # For each other method
        for method, data in results_dict.items():
            if method == "original":
                continue
            competitor_ids = data["doc_ids"]
            overlap = [doc_id in original_ids for doc_id in competitor_ids]
            overlap_count = sum(overlap)
            data["overlap_count"] = overlap_count
            data["overlap_list"] = overlap
    else:
        # If original isn't in the dict, no ground truth to compare
        for method, data in results_dict.items():
            data["overlap_count"] = 0
            data["overlap_list"] = [False] * len(data["doc_ids"])

    return results_dict