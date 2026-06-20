"""
Wav2Vec2 audio encoder — turns a waveform into a 768-dim utterance embedding.

Lazy module-singleton of facebook/wav2vec2-base in eval() mode. The embedding is
the mean-pool of last_hidden_state over time. Robust to NaNs and very short clips.
This same function is used at inference (audio upload) and offline (training), so
features are identical on both sides.
"""
import logging

import librosa
import numpy as np

from pathlib import Path

from api.feature_config import EMBEDDING_DIM, SAMPLE_RATE, WAV2VEC2_MODEL_NAME

logger = logging.getLogger(__name__)

FINETUNED_BACKBONE = Path(__file__).resolve().parent.parent.parent / "models" / "wav2vec2_finetuned"

_model = None
_processor = None
_MIN_SAMPLES = 4800  # 0.3 s at 16 kHz


def _backbone_source() -> str:
    """Fine-tuned backbone if enabled + present, else the base pretrained model."""
    from api.config import settings

    if settings.use_finetuned_emotion_model and FINETUNED_BACKBONE.exists():
        return str(FINETUNED_BACKBONE)
    return WAV2VEC2_MODEL_NAME


def _load():
    global _model, _processor
    if _model is None or _processor is None:
        import torch  # noqa: F401  (ensure torch present)
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        source = _backbone_source()
        logger.info("Loading Wav2Vec2 backbone: %s", source)
        # Processor (feature normalization) is identical for base and fine-tuned.
        _processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_NAME)
        _model = Wav2Vec2Model.from_pretrained(source)
        _model.eval()
    return _model, _processor


def extract_wav2vec2_embedding(path: str) -> np.ndarray:
    """Load 16 kHz mono audio at `path` → (768,) float32 embedding."""
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        logger.warning("Audio load failed (%s): %s — returning zero embedding", path, exc)
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    return embed_waveform(y)


def embed_waveform(y: np.ndarray) -> np.ndarray:
    """Embed an in-memory 16 kHz mono waveform → (768,) float32."""
    import torch

    if y is None or len(y) == 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    y = np.nan_to_num(np.asarray(y, dtype=np.float32))
    if len(y) < _MIN_SAMPLES:
        y = np.pad(y, (0, _MIN_SAMPLES - len(y)))

    model, processor = _load()
    inputs = processor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs.input_values)
    emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return np.nan_to_num(emb).astype(np.float32)
