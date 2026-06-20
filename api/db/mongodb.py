import logging

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database

from api.config import settings

logger = logging.getLogger(__name__)

_client: MongoClient | None = None
_db: Database | None = None


def connect_mongodb() -> None:
    global _client, _db
    try:
        _client = MongoClient(settings.mongodb_url, serverSelectionTimeoutMS=5000)
        _client.server_info()  # force connection check
        _db = _client[settings.mongodb_db_name]
        _create_indexes(_db)
        logger.info("MongoDB connected: %s / %s", settings.mongodb_url, settings.mongodb_db_name)
    except Exception as exc:
        logger.warning("MongoDB unavailable — continuing without DB: %s", exc)
        _client = None
        _db = None


def disconnect_mongodb() -> None:
    global _client
    if _client:
        _client.close()
        _client = None
        logger.info("MongoDB disconnected")


def get_database() -> Database | None:
    return _db


def _create_indexes(db: Database) -> None:
    _idx = [
        ("user_feedback", [("user_id", ASCENDING)], {}),
        ("user_feedback", [("timestamp", ASCENDING)], {}),
        ("user_feedback", [("user_id", ASCENDING), ("timestamp", ASCENDING)], {}),
        ("training_jobs", [("user_id", ASCENDING)], {}),
        ("training_jobs", [("job_id", ASCENDING)], {"unique": True}),
        ("training_jobs", [("user_id", ASCENDING), ("created_at", DESCENDING)], {}),
        ("user_models", [("user_id", ASCENDING)], {"unique": True}),
        ("user_models", [("updated_at", ASCENDING)], {}),
        ("speaker_profiles", [("user_id", ASCENDING)], {"unique": True}),
        ("speaker_profiles", [("updated_at", ASCENDING)], {}),
    ]
    for collection, keys, opts in _idx:
        try:
            db[collection].create_index(keys, **opts)
        except Exception as exc:
            logger.warning("Index creation skipped (%s %s): %s", collection, keys, exc)
