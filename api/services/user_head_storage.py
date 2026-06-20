"""
Per-user head storage — file backend (default): models/user_heads/<user_id>.pt.

`model_version()` returns the file mtime so the in-memory LRU cache busts after a
retrain. (The reference also defines a gzip-in-Mongo backend gated by
USE_MONGODB_STORAGE; file backend is implemented here.)
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

USER_HEADS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "user_heads"


def _path(user_id: str) -> Path:
    return USER_HEADS_DIR / f"{user_id}.pt"


def save_model(user_id: str, state_dict) -> dict:
    import torch

    USER_HEADS_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(user_id)
    torch.save(state_dict, path)
    size_kb = round(path.stat().st_size / 1024, 1)
    return {"storage_mode": "file", "path": str(path), "size_kb": size_kb}


def load_model(user_id: str):
    path = _path(user_id)
    if not path.exists():
        return None
    import torch

    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        logger.warning("user head load failed for %s: %s", user_id, exc)
        return None


def exists(user_id: str) -> bool:
    return _path(user_id).exists()


def model_version(user_id: str) -> str:
    path = _path(user_id)
    return str(path.stat().st_mtime_ns) if path.exists() else "none"
