"""
Per-user emotion head — identical architecture to the global head: Linear(768, 8).
Lazy-loaded per user with an LRU cache so personalization stays cheap.
"""
import logging
from functools import lru_cache

import numpy as np

from api.feature_config import EMBEDDING_DIM, NUM_CLASSES, USER_HEAD_CACHE_SIZE

logger = logging.getLogger(__name__)


def build_user_head():
    import torch.nn as nn

    return nn.Linear(EMBEDDING_DIM, NUM_CLASSES)


@lru_cache(maxsize=USER_HEAD_CACHE_SIZE)
def _load_user_head_cached(user_id: str, version: str):
    """Cached by (user_id, version). `version` busts the cache after retraining."""
    import torch

    from api.services.user_head_storage import load_model

    state = load_model(user_id)
    if state is None:
        return None
    head = build_user_head()
    head.load_state_dict(state)
    head.eval()
    return head


def get_user_head(user_id: str):
    from api.services.user_head_storage import model_version

    return _load_user_head_cached(user_id, model_version(user_id))


def predict_user_proba(embedding: np.ndarray, user_id: str) -> np.ndarray | None:
    head = get_user_head(user_id)
    if head is None:
        return None
    import torch

    x = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        probs = torch.softmax(head(torch.from_numpy(x)), dim=1).squeeze(0).cpu().numpy()
    return probs.astype(np.float32)
