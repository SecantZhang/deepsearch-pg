"""
search_pipeline.py

Generates a query vector in 384-d, optionally reduces it to 64-d via PCA or UMAP,
searches the correct Qdrant instance for top-K results, and returns the original
Wikipedia articles from a local dataset subset.

No file output, no main() function. Just a library function `search_pipeline()`.
"""

import os
import numpy as np
import joblib
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from sklearn.decomposition import PCA  # We'll re-fit PCA from the original array
import umap

# import os
# print("Current Working Directory:", os.getcwd())

# ========== CONFIG ==========
DATA_DIR = "private_gpt/data"
DATASET_DIR = os.path.join(DATA_DIR, "wikipedia_subset_dataset")

ORIGINAL_EMB_FILE = os.path.join(DATA_DIR, "wikipedia_original_embeddings.npy")
UMAP_MODEL_FILE   = os.path.join(DATA_DIR, "umap_reducer.joblib")

# Dimensions
ORIGINAL_DIM = 384  # "all-MiniLM-L6-v2" outputs
PCA_DIM = 64
UMAP_DIM = 64

# Qdrant ports + collections for each method
QDRANT_PORTS = {
    "original": 6333,
    "pca":      6334,
    "umap":     6335
}
QDRANT_COLLECTIONS = {
    "original": "wikipedia_original",
    "pca":      "wikipedia_pca",
    "umap":     "wikipedia_umap"
}

TOP_K = 10

# ========== LOAD OR CACHE ==========

print(f"[INFO] Checking dataset folder at: {DATASET_DIR}")
if not os.path.exists(DATASET_DIR):
    raise RuntimeError(f"Dataset dir '{DATASET_DIR}' not found. Did you run ingestion?")

print(f"[INFO] Loading partial Wikipedia subset from disk...")
dataset: Dataset = load_from_disk(DATASET_DIR)
print(f"[INFO] Dataset length = {len(dataset)} articles")

print("[INFO] Initializing the base embedding model: all-MiniLM-L6-v2")
base_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 1) Load original embeddings so we can re-fit PCA
if not os.path.exists(ORIGINAL_EMB_FILE):
    raise RuntimeError(f"Cannot find {ORIGINAL_EMB_FILE}, aborting.")
original_embeddings = np.load(ORIGINAL_EMB_FILE)
print(f"[INFO] Original embeddings shape = {original_embeddings.shape}")

# 2) Fit or re-fit PCA(64) from the original array
print("[INFO] Re-fitting PCA(64) from original array, so we can transform queries.")
pca_model = PCA(n_components=PCA_DIM, random_state=42)
pca_model.fit(original_embeddings)

# 3) Load UMAP if present
umap_reducer = None
if os.path.exists(UMAP_MODEL_FILE):
    print(f"[INFO] Loading UMAP reducer from {UMAP_MODEL_FILE} ...")
    umap_reducer = joblib.load(UMAP_MODEL_FILE)
else:
    print("[WARN] No umap_reducer.joblib found. 'umap' method won't be available.")

print("[INFO] Setup complete. We can now do search queries with original/pca/umap.")


def search_pipeline(query_text: str, method: str = "original") -> list[str]:
    """
    1) Embed query (384-d) via all-MiniLM-L6-v2.
    2) If method=='pca', reduce to 64-d via pca_model.
       If method=='umap', reduce to 64-d via umap_reducer.
       If 'original', keep 384-d.
    3) Search correct Qdrant instance (port 6333, 6334, 6335).
    4) Retrieve top-K indices, then fetch 'dataset[idx]["text"]'.
    5) Return list of article texts.
    """
    print("Selected method: ", method)
    # 1) embed
    query_384 = base_embed_model.encode(query_text).astype(np.float32)
    # 2) dimension reduction
    if method == "pca":
        query_vec = pca_model.transform(query_384.reshape(1, -1)).astype(np.float32)[0]
    elif method == "umap":
        if umap_reducer is None:
            raise RuntimeError("UMAP reducer not loaded, cannot do 'umap' method.")
        query_vec = umap_reducer.transform(query_384.reshape(1, -1)).astype(np.float32)[0]
    else:
        # 'original'
        query_vec = query_384

    # 3) pick Qdrant server
    if method not in QDRANT_PORTS:
        raise ValueError(f"Unknown method: {method}")

    port = QDRANT_PORTS[method]
    collection_name = QDRANT_COLLECTIONS[method]
    client = QdrantClient(host="localhost", port=port)

    # 4) search top-K
    results: list[ScoredPoint] = client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=TOP_K
    )
    idx_list = [r.id for r in results]

    # 5) retrieve from dataset
    retrieved_texts = []
    for idx in idx_list:
        int_idx = int(idx)
        text_item = dataset[int_idx]["text"]
        retrieved_texts.append(text_item)

    return retrieved_texts
