"""
Workflow Memory — Qdrant vector store for semantic workflow recall.
Stores workflows as embeddings so the agent can find similar past workflows.
"""
import json
import os
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "workflows")
VECTOR_SIZE = 384   # all-MiniLM-L6-v2 embedding size


_client: Optional[QdrantClient] = None
_embedder = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
        _ensure_collection()
    return _client


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _ensure_collection():
    client = QdrantClient(url=QDRANT_URL)
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection: {COLLECTION}")


def _embed(text: str) -> list[float]:
    return _get_embedder().encode(text).tolist()


def save_workflow_step(task: str, plan: list, history: list):
    """
    Save a completed workflow to Qdrant for future reuse.
    The task description becomes the searchable embedding.
    """
    client = _get_client()
    vector = _embed(task)

    payload = {
        "task": task,
        "plan": json.dumps(plan),
        "history_summary": f"{len(history)} steps completed",
        "success_count": sum(1 for h in history if h.get("success")),
        "total_steps": len(history),
    }

    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload,
        )],
    )


def find_similar_workflow(task: str, threshold: float = 0.80) -> Optional[dict]:
    """
    Search Qdrant for a workflow similar to the given task.

    Returns the best match if similarity >= threshold, else None.
    """
    client = _get_client()
    vector = _embed(task)

    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=1,
        score_threshold=threshold,
    ).points

    if not results:
        return None

    best = results[0]
    return {
        "task": best.payload["task"],
        "plan": json.loads(best.payload["plan"]),
        "similarity": best.score,
        "success_rate": (
            best.payload["success_count"] / best.payload["total_steps"]
            if best.payload["total_steps"] > 0 else 0
        ),
    }


def list_workflows(limit: int = 20) -> list[dict]:
    """List all stored workflows."""
    client = _get_client()
    results, _ = client.scroll(
        collection_name=COLLECTION,
        limit=limit,
        with_payload=True,
    )
    return [
        {
            "task": r.payload["task"],
            "steps": r.payload["total_steps"],
            "success_rate": (
                r.payload["success_count"] / r.payload["total_steps"]
                if r.payload["total_steps"] > 0 else 0
            ),
        }
        for r in results
    ]
