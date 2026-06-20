"""
Alpha Engine — computes the blend weight between the global and per-user
emotion heads.

    Final = alpha * Global + (1 - alpha) * User

`alpha` is the weight on the GLOBAL head. New users / low feedback → alpha≈1
(trust the crowd). As feedback grows and when the global model is unconfident →
alpha shrinks (trust the user's personal model).
"""
import math

from api.feature_config import (
    ALPHA_CONFIDENCE_THRESHOLD_TAU,
    ALPHA_FEEDBACK_SCALE_K,
    ALPHA_SIGMOID_SHARPNESS_BETA,
    USE_SIGMOID_ALPHA,
)


def compute_blend_weight(global_confidence: float, feedback_count: int, user_id: str | None = None) -> dict:
    """
    Returns {alpha, alpha_data, alpha_conf, formula}.

    Sigmoid (active):
        alpha_data = 1 / (1 + N/K)                      # data availability   (K=50)
        alpha_conf = 1 / (1 + exp(-beta*(C_g - tau)))   # confidence S-curve  (tau=0.6, beta=10)
        alpha      = alpha_data * alpha_conf             # multiplicative, naturally in (0,1)
    """
    n = max(0, int(feedback_count))
    c_g = float(global_confidence)

    if USE_SIGMOID_ALPHA:
        alpha_data = 1.0 / (1.0 + n / ALPHA_FEEDBACK_SCALE_K)
        alpha_conf = 1.0 / (1.0 + math.exp(-ALPHA_SIGMOID_SHARPNESS_BETA * (c_g - ALPHA_CONFIDENCE_THRESHOLD_TAU)))
        alpha = alpha_data * alpha_conf
        return {
            "alpha": alpha,
            "alpha_data": alpha_data,
            "alpha_conf": alpha_conf,
            "formula": "sigmoid",
        }

    # Linear (legacy fallback)
    if n < 20:
        alpha = 1.0
    else:
        alpha = 0.5 + 0.3 * c_g - 0.2 * min(n / 100.0, 1.0)
        alpha = max(0.3, min(1.0, alpha))
    return {"alpha": alpha, "alpha_data": None, "alpha_conf": None, "formula": "linear"}
