"""Redis connection manager (optional — caching degrades gracefully if down)."""
import logging

from api.config import settings

logger = logging.getLogger(__name__)

_client = None
_available = False


def connect_redis() -> bool:
    global _client, _available
    try:
        import redis as redis_lib

        _client = redis_lib.from_url(settings.redis_url, decode_responses=True, socket_timeout=3)
        _client.ping()
        _available = True
        logger.info("Redis connected: %s", settings.redis_url)
    except Exception as exc:
        logger.warning("Redis unavailable — query cache disabled: %s", exc)
        _client = None
        _available = False
    return _available


def get_redis_client():
    return _client


def is_available() -> bool:
    return _available
