"""
Semantic query cache (Redis, optional).

A hit needs the SAME top_k AND cosine(query_embedding, cached) >= threshold (0.95).
Per-user FIFO eviction at max_per_user; TTL on each entry. All ops are fail-soft.
"""
import hashlib
import json
import logging
import time

import numpy as np

from api.config import settings
from api.db import redis as redis_db

logger = logging.getLogger(__name__)


def _key(user_id: str, query: str, top_k: int) -> str:
    h = hashlib.sha256(f"{query.strip().lower()}:{top_k}".encode()).hexdigest()[:16]
    return f"query_cache:{user_id}:{h}"


def _index_key(user_id: str) -> str:
    return f"query_cache_keys:{user_id}"


def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def check_cache(user_id: str, query: str, top_k: int, query_embedding) -> dict | None:
    if not redis_db.is_available() or query_embedding is None:
        return None
    try:
        client = redis_db.get_redis_client()
        keys = client.zrange(_index_key(user_id), 0, -1)
        best, best_sim = None, -1.0
        for k in keys:
            raw = client.get(k)
            if not raw:
                continue
            entry = json.loads(raw)
            if entry.get("top_k") != top_k:
                continue
            sim = _cosine(query_embedding, entry.get("query_embedding", []))
            if sim > best_sim:
                best, best_sim = entry, sim
        if best and best_sim >= settings.query_cache_similarity_threshold:
            return {"answer": best["answer"], "sources": best["sources"],
                    "query": query, "_cache_hit": True, "_similarity": round(best_sim, 4)}
    except Exception as exc:
        logger.warning("cache check failed: %s", exc)
    return None


def store_in_cache(user_id: str, query: str, top_k: int, query_embedding,
                   answer: str, sources: list) -> None:
    if not redis_db.is_available() or query_embedding is None:
        return
    try:
        client = redis_db.get_redis_client()
        key = _key(user_id, query, top_k)
        payload = json.dumps({
            "original_query": query, "query_embedding": list(map(float, query_embedding)),
            "top_k": top_k, "answer": answer, "sources": sources, "timestamp": time.time(),
        })
        client.setex(key, settings.query_cache_ttl_seconds, payload)
        idx = _index_key(user_id)
        client.zadd(idx, {key: time.time()})
        # FIFO eviction beyond max_per_user
        size = client.zcard(idx)
        if size > settings.query_cache_max_per_user:
            oldest = client.zrange(idx, 0, size - settings.query_cache_max_per_user - 1)
            if oldest:
                client.delete(*oldest)
                client.zrem(idx, *oldest)
    except Exception as exc:
        logger.warning("cache store failed: %s", exc)


def invalidate_user_cache(user_id: str) -> None:
    if not redis_db.is_available():
        return
    try:
        client = redis_db.get_redis_client()
        idx = _index_key(user_id)
        keys = client.zrange(idx, 0, -1)
        if keys:
            client.delete(*keys)
        client.delete(idx)
    except Exception as exc:
        logger.warning("cache invalidate failed: %s", exc)
