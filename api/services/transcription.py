"""
Speech-to-text via OpenAI Whisper.

Lazy module-singleton (loaded on first use). Whisper can use the GPU if available,
else CPU. Returns "" on any failure so the upload pipeline never breaks on transcription.
"""
import logging

from api.config import settings

logger = logging.getLogger(__name__)

_model = None


def _load(model_size: str | None = None):
    global _model
    if _model is None:
        import whisper

        size = model_size or settings.whisper_model_size
        logger.info("Loading Whisper model: %s", size)
        _model = whisper.load_model(size)
    return _model


def transcribe_audio(path: str, model_size: str | None = None, language: str | None = None) -> str:
    try:
        model = _load(model_size)
        result = model.transcribe(
            path,
            language=language or settings.whisper_language,
            beam_size=5,
            best_of=5,
            temperature=0,
            condition_on_previous_text=True,
            fp16=False,
        )
        return (result.get("text") or "").strip()
    except Exception as exc:
        logger.warning("Transcription failed for %s: %s", path, exc)
        return ""
