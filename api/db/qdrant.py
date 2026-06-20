"""
Qdrant connection manager — per-user document collections for semantic search.

Fail-soft: if Qdrant is unreachable at startup, the app still boots; vector ops
return empty/no-op until it's available.
"""
import logging

from api.config import settings

logger = logging.getLogger(__name__)

VECTOR_SIZE = 384  # all-MiniLM-L6-v2
_client = None
_available = False


def connect_qdrant() -> bool:
    global _client, _available
    try:
        from qdrant_client import QdrantClient

        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=10)
        _client.get_collections()  # connection check
        _available = True
        logger.info("Qdrant connected: %s", settings.qdrant_url)
    except Exception as exc:
        logger.warning("Qdrant unavailable — vector search disabled: %s", exc)
        _client = None
        _available = False
    return _available


def get_qdrant_client():
    return _client


def is_available() -> bool:
    return _available


def collection_name(user_id: str) -> str:
    return f"user_{user_id}_documents"


def ensure_collection(user_id: str) -> bool:
    """Create the user's collection if missing (idempotent). Returns True on success."""
    if not _available or _client is None:
        return False
    try:
        from qdrant_client.models import Distance, VectorParams

        name = collection_name(user_id)
        existing = {c.name for c in _client.get_collections().collections}
        if name not in existing:
            _client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", name)
        return True
    except Exception as exc:
        logger.warning("ensure_collection failed for %s: %s", user_id, exc)
        return False
