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

    # Stage 4: real Wav2Vec2 embedding + dual-head classification (global-only here)
    from api.services.dual_head_classifier import classify_with_dual_heads
    from api.services.wav2vec2_encoder import extract_wav2vec2_embedding

    embedding = extract_wav2vec2_embedding(str(audio_path))
    feedback_count = _get_feedback_count(user_id)
    result = classify_with_dual_heads(embedding, user_id, feedback_count)

    # Stage 5: transcribe (Whisper) + index transcript embedding (Qdrant)
    from api.services.transcription import transcribe_audio

    transcription = transcribe_audio(str(audio_path))
    session_id = _persist_session(
        user_id, audio_path, result["emotion"], result["confidence"], transcription, embedding
    )
    _index_transcript(user_id, transcription, session_id, result["emotion"])

    return {
        "session_id": session_id,
        "emotion": result["emotion"],
        "confidence": result["confidence"],
        "global_emotion": result["global_emotion"],
        "global_confidence": result["global_confidence"],
        "user_emotion": result["user_emotion"],
        "user_confidence": result["user_confidence"],
        "blend_weight": result["blend_weight"],
        "alpha_data": result["alpha_data"],
        "alpha_conf": result["alpha_conf"],
        "alpha_formula": result["alpha_formula"],
        "transcription": transcription,
        "timestamp": datetime.now(timezone.utc),
        "owner_speech_ratio": None,
        "owner_segments_count": None,
        "other_segments_count": None,
        "owner_detection_status": None,
        "speaker_timeline": [],
    }


def _get_feedback_count(user_id: str) -> int:
    from api.db.mongodb import get_database
    from api.models.user import User

    db = get_database()
    if db is None:
        return 0
    doc = User.get_collection(db).find_one({"user_id": user_id}, {"feedback_count": 1})
    return int(doc.get("feedback_count", 0)) if doc else 0


def _index_transcript(user_id: str, transcription: str, session_id: str, emotion: str) -> None:
    """Embed + upsert the transcript into the user's Qdrant collection (fail-soft)."""
    if not transcription:
        return
    try:
        from api.services.vector_store import store_document

        store_document(user_id, transcription, session_id, emotion_label=emotion)
    except Exception as exc:
        logger.warning("Transcript indexing failed (session %s): %s", session_id, exc)


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


# ── Persist session ────────────────────────────────────────────────────────────

def _persist_session(
    user_id: str,
    audio_path: Path,
    emotion: str,
    confidence: float,
    transcription: str,
    embedding=None,
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
        # Store the 768-dim embedding for later user-head training (Stage 5+ model)
        if embedding is not None:
            _persist_emotion_analysis(db, user_id, session_id, emotion, confidence, embedding)
    else:
        logger.warning("MongoDB unavailable — session %s not persisted", session_id)

    return session_id


def _persist_emotion_analysis(db, user_id, session_id, emotion, confidence, embedding) -> None:
    """Store the Wav2Vec2 embedding (historical field name: mfcc_features)."""
    try:
        from api.models.emotion_analysis import EmotionAnalysis

        ea = EmotionAnalysis(
            user_id=user_id,
            session_id=session_id,
            emotion_label=emotion,
            confidence=confidence,
            mfcc_features=[float(x) for x in embedding],
        )
        EmotionAnalysis.get_collection(db).insert_one(ea.to_dict())
    except Exception as exc:
        logger.warning("EmotionAnalysis not persisted: %s", exc)
