"""
Dual-head classifier — blends the global emotion head with a per-user head via
the Alpha Engine.

    P_final = alpha * P_global + (1 - alpha) * P_user
    (renormalized; argmax → emotion, max → confidence)

Stage 4: only the global head exists, so this always returns the global-only
result (blend_weight=1.0, user_*=None). Stage 7 adds the user-head path.
"""
import logging

import numpy as np

from api.feature_config import EMOTION_LABELS
from api.services import global_emotion_head
from api.services.alpha_engine import compute_blend_weight

logger = logging.getLogger(__name__)


def classify_with_dual_heads(embedding: np.ndarray, user_id: str, feedback_count: int) -> dict:
    p_global = global_emotion_head.predict_proba(embedding)

    # No global head → fixed neutral fallback (graceful degradation)
    if p_global is None:
        return {
            "emotion": "neutral",
            "confidence": 0.5,
            "global_emotion": "neutral",
            "global_confidence": 0.5,
            "user_emotion": None,
            "user_confidence": None,
            "blend_weight": 1.0,
            "alpha_data": None,
            "alpha_conf": None,
            "alpha_formula": "sigmoid",
            "probabilities": None,
        }

    g_idx = int(np.argmax(p_global))
    g_emotion = EMOTION_LABELS[g_idx]
    g_conf = float(p_global[g_idx])

    # Stage 7 will attempt to load a user head here. For now, global-only.
    p_user = _try_user_head(embedding, user_id)

    if p_user is None:
        return {
            "emotion": g_emotion,
            "confidence": g_conf,
            "global_emotion": g_emotion,
            "global_confidence": g_conf,
            "user_emotion": None,
            "user_confidence": None,
            "blend_weight": 1.0,
            "alpha_data": None,
            "alpha_conf": None,
            "alpha_formula": "linear",
            "probabilities": p_global.tolist(),
        }

    # Blend (Stage 7 path)
    alpha_info = compute_blend_weight(g_conf, feedback_count, user_id)
    alpha = alpha_info["alpha"]
    p_final = alpha * p_global + (1.0 - alpha) * p_user
    s = p_final.sum()
    if s > 0:
        p_final = p_final / s

    f_idx = int(np.argmax(p_final))
    u_idx = int(np.argmax(p_user))
    return {
        "emotion": EMOTION_LABELS[f_idx],
        "confidence": float(p_final[f_idx]),
        "global_emotion": g_emotion,
        "global_confidence": g_conf,
        "user_emotion": EMOTION_LABELS[u_idx],
        "user_confidence": float(p_user[u_idx]),
        "blend_weight": alpha,
        "alpha_data": alpha_info["alpha_data"],
        "alpha_conf": alpha_info["alpha_conf"],
        "alpha_formula": alpha_info["formula"],
        "probabilities": p_final.tolist(),
    }


def _try_user_head(embedding: np.ndarray, user_id: str):
    """Stage 4 stub — no user heads yet. Stage 7 loads from user_head_storage (LRU)."""
    return None
