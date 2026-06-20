"""
Vector store — embeds transcripts with sentence-transformers (all-MiniLM-L6-v2,
384-dim) and stores/searches them in per-user Qdrant collections.
"""
import logging
import uuid
from datetime import datetime, timezone

from api.config import settings
from api.db import qdrant

logger = logging.getLogger(__name__)

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", settings.embedding_model)
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def embed_text(text: str):
    """Return a 384-dim list embedding for `text`."""
    vec = _get_embedder().encode(text, normalize_embeddings=False)
    return vec.tolist()


def store_document(
    user_id: str,
    text: str,
    session_id: str,
    timestamp: datetime | None = None,
    emotion_label: str | None = None,
) -> bool:
    if not text or not text.strip():
        return False
    if not qdrant.is_available():
        logger.warning("Qdrant unavailable — document not indexed (session %s)", session_id)
        return False
    if not qdrant.ensure_collection(user_id):
        return False

    try:
        from qdrant_client.models import PointStruct

        vector = embed_text(text)
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": text,
                "session_id": session_id,
                "user_id": user_id,
                "timestamp": ts,
                "emotion_label": emotion_label,
            },
        )
        qdrant.get_qdrant_client().upsert(
            collection_name=qdrant.collection_name(user_id), points=[point]
        )
        return True
    except Exception as exc:
        logger.warning("store_document failed (session %s): %s", session_id, exc)
        return False


def search_documents(user_id: str, query: str, top_k: int = 5, query_embedding=None) -> list[dict]:
    if not qdrant.is_available():
        return []
    try:
        vector = query_embedding if query_embedding is not None else embed_text(query)
        # qdrant-client >=1.12 replaced .search() with .query_points()
        response = qdrant.get_qdrant_client().query_points(
            collection_name=qdrant.collection_name(user_id),
            query=vector,
            limit=top_k,
        )
        hits = response.points
        return [
            {
                "text": h.payload.get("text", ""),
                "session_id": h.payload.get("session_id"),
                "timestamp": h.payload.get("timestamp"),
                "emotion_label": h.payload.get("emotion_label"),
                "score": h.score,
            }
            for h in hits
        ]
    except Exception as exc:
        logger.warning("search_documents failed for %s: %s", user_id, exc)
        return []
