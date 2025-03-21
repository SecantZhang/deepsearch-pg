from typing import Literal, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

vector_store_router = APIRouter()

class VectorStoreRequest(BaseModel):
    method: Literal["original", "pca", "umap"]

class VectorStoreResponse(BaseModel):
    status: Literal["success"]
    message: str

AVAILABLE_METHODS: Dict[str, Dict[str, str]] = {
    "original": {
        "type": "qdrant",
        "url": "http://localhost:6333",
    },
    "pca": {
        "type": "qdrant",
        "url": "http://localhost:6334",
    },
    "umap": {
        "type": "qdrant",
        "url": "http://localhost:6335",
    }
}

current_vector_store_config: Dict[str, str] = {}

@vector_store_router.post("/switch_vector_store", tags=["Vector Store"], response_model=VectorStoreResponse)
async def switch_vector_store(request: VectorStoreRequest) -> VectorStoreResponse:
    method = request.method
    if method not in AVAILABLE_METHODS:
        raise HTTPException(status_code=400, detail="Invalid method selected")

    selected_config = AVAILABLE_METHODS[method]
    current_vector_store_config.update(selected_config)
    print(f"[BACKEND] Switched to vector store: {method}, new config = {current_vector_store_config}")
    return VectorStoreResponse(status="success", message=f"Switched to vector store: {method}")

@vector_store_router.get("/current_vector_store", tags=["Vector Store"])
async def get_current_vector_store() -> Dict[str, str]:
    return current_vector_store_config