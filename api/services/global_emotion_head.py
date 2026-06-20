"""
Global emotion head — a single Linear(768, 8) trained on public datasets
(RAVDESS + CREMA-D). Loaded eagerly at startup; predicts an emotion + confidence
from a Wav2Vec2 embedding.

Artifacts (produced by training/train_wav2vec2.py):
  - models/global_emotion_head.pt     (state_dict of the Linear layer)
  - models/embedding_scaler.joblib    (StandardScaler fit on training embeddings)

Graceful degradation: if the artifact is missing, predict() returns
("neutral", 0.5) so the API still serves.
"""
import logging
from pathlib import Path

import numpy as np

from api.feature_config import EMBEDDING_DIM, EMOTION_LABELS, NUM_CLASSES

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
HEAD_PATH = MODELS_DIR / "global_emotion_head.pt"
SCALER_PATH = MODELS_DIR / "embedding_scaler.joblib"
FINETUNED_HEAD_PATH = MODELS_DIR / "global_emotion_head_finetuned.pt"
FINETUNED_BACKBONE = MODELS_DIR / "wav2vec2_finetuned"

_head = None
_scaler = None
_loaded = False
_available = False
_active = "none"


def _build_head():
    import torch.nn as nn
    return nn.Linear(EMBEDDING_DIM, NUM_CLASSES)


def _use_finetuned() -> bool:
    from api.config import settings

    return (
        settings.use_finetuned_emotion_model
        and FINETUNED_HEAD_PATH.exists()
        and FINETUNED_BACKBONE.exists()
    )


def load_global_head() -> bool:
    """
    Load the active head. Prefers the fine-tuned head (no scaler — it was trained on
    raw pooled features of the fine-tuned backbone) when present; else the baseline
    head + StandardScaler. Returns True if available. Idempotent.
    """
    global _head, _scaler, _loaded, _available, _active
    if _loaded:
        return _available

    _loaded = True
    import torch

    if _use_finetuned():
        _head = _build_head()
        _head.load_state_dict(torch.load(FINETUNED_HEAD_PATH, map_location="cpu"))
        _head.eval()
        _scaler = None  # fine-tuned head consumes raw fine-tuned-backbone embeddings
        _available = True
        _active = "finetuned"
        logger.info("Emotion head loaded: FINE-TUNED (%s)", FINETUNED_HEAD_PATH.name)
        return True

    if not HEAD_PATH.exists():
        logger.warning("Global head missing at %s — using neutral fallback", HEAD_PATH)
        _available = False
        return False

    _head = _build_head()
    _head.load_state_dict(torch.load(HEAD_PATH, map_location="cpu"))
    _head.eval()

    if SCALER_PATH.exists():
        import joblib
        _scaler = joblib.load(SCALER_PATH)
    else:
        logger.warning("Scaler missing at %s — predicting on raw embeddings", SCALER_PATH)
        _scaler = None

    _available = True
    _active = "baseline"
    logger.info("Emotion head loaded: BASELINE (%s)", HEAD_PATH.name)
    return True


def active_model() -> str:
    if not _loaded:
        load_global_head()
    return _active


def is_available() -> bool:
    if not _loaded:
        load_global_head()
    return _available


def predict_proba(embedding: np.ndarray) -> np.ndarray | None:
    """Return the (8,) softmax probability vector, or None if head unavailable."""
    if not is_available():
        return None

    import torch

    x = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    if _scaler is not None:
        x = _scaler.transform(x)
    with torch.no_grad():
        logits = _head(torch.from_numpy(x.astype(np.float32)))
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs.astype(np.float32)


def predict(embedding: np.ndarray) -> tuple[str, float]:
    """Return (emotion_label, confidence). Falls back to ('neutral', 0.5)."""
    probs = predict_proba(embedding)
    if probs is None:
        return ("neutral", 0.5)
    idx = int(np.argmax(probs))
    return (EMOTION_LABELS[idx], float(probs[idx]))
