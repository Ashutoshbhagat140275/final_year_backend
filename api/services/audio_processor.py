"""
Audio processing pipeline — Stage 3.

Steps implemented here:
  1. Validate extension + size
  2. Save to uploads/<user_id>/<timestamp>.<ext>
  3. Preprocess: resample 16 kHz, VAD trim, peak-normalize, overwrite
  4. Stub emotion: ("neutral", 0.5)  ← replaced in Stage 4
  5. Stub transcription: ""           ← replaced in Stage 5
  6. Persist AudioSession to MongoDB
  7. Return result dict

Stage 4 adds: Wav2Vec2 embedding + real classifier
Stage 5 adds: Whisper transcription + Qdrant indexing
Stage 6 adds: speaker diarization + owner detection
"""
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile, status

from api.config import settings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "flac", "ogg"}


# ── Public entry point ─────────────────────────────────────────────────────────

def process_audio(user_id: str, file: UploadFile) -> dict:
    _validate_file(file)
    audio_path = _save_file(user_id, file)
    audio_path = _preprocess(audio_path)

    # Stage 3 stubs — replaced in later stages
    emotion, confidence = _stub_classify()
    transcription = ""
    session_id = _persist_session(user_id, audio_path, emotion, confidence, transcription)

    return {
        "session_id": session_id,
        "emotion": emotion,
        "confidence": confidence,
        "global_emotion": emotion,
        "global_confidence": confidence,
        "user_emotion": None,
        "user_confidence": None,
        "blend_weight": 1.0,
        "alpha_data": None,
        "alpha_conf": None,
        "alpha_formula": "sigmoid",
        "transcription": transcription,
        "timestamp": datetime.now(timezone.utc),
        "owner_speech_ratio": None,
        "owner_segments_count": None,
        "other_segments_count": None,
        "owner_detection_status": None,
        "speaker_timeline": [],
    }


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate_file(file: UploadFile) -> None:
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '.{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )
    # Read content length header if available; full size check happens after save
    content_length = file.size  # FastAPI UploadFile exposes .size (may be None for streams)
    if content_length and content_length > settings.max_audio_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds {settings.max_audio_size_mb} MB limit",
        )


# ── Save ───────────────────────────────────────────────────────────────────────

def _save_file(user_id: str, file: UploadFile) -> Path:
    ext = (file.filename or "audio.wav").rsplit(".", 1)[-1].lower()
    upload_dir = Path(settings.audio_upload_dir) / user_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = upload_dir / f"{ts}_{uuid.uuid4().hex[:8]}.{ext}"

    content = file.file.read()
    # Post-read size check
    if len(content) > settings.max_audio_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds {settings.max_audio_size_mb} MB limit",
        )
    dest.write_bytes(content)
    return dest


# ── Preprocess ─────────────────────────────────────────────────────────────────

def _preprocess(path: Path) -> Path:
    try:
        y, sr = librosa.load(str(path), sr=16000, mono=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode audio: {exc}",
        )

    # VAD trim — remove leading/trailing silence
    intervals = librosa.effects.split(y, top_db=30)
    if len(intervals) > 0:
        voiced = np.concatenate([y[s:e] for s, e in intervals])
        if len(voiced) > 0:
            y = voiced

    # Peak-normalize
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # Overwrite with cleaned 16 kHz WAV
    out_path = path.with_suffix(".wav")
    sf.write(str(out_path), y, 16000)
    if out_path != path:
        try:
            os.remove(str(path))
        except OSError:
            pass

    return out_path


# ── Stub emotion (Stage 3) ─────────────────────────────────────────────────────

def _stub_classify() -> tuple[str, float]:
    return ("neutral", 0.5)


# ── Persist session ────────────────────────────────────────────────────────────

def _persist_session(
    user_id: str,
    audio_path: Path,
    emotion: str,
    confidence: float,
    transcription: str,
) -> str:
    from api.db.mongodb import get_database
    from api.models.audio_session import AudioSession

    session_id = str(uuid.uuid4())
    session = AudioSession(
        session_id=session_id,
        user_id=user_id,
        audio_file_path=str(audio_path),
        emotion_data={"label": emotion, "emotion": emotion, "confidence": confidence},
        transcription_text=transcription,
        personalization_trainable=True,
    )

    db = get_database()
    if db is not None:
        AudioSession.get_collection(db).insert_one(session.to_dict())
    else:
        logger.warning("MongoDB unavailable — session %s not persisted", session_id)

    return session_id
