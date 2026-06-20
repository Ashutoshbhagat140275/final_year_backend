"""
Speaker service — classical (CPU, no neural speaker model) diarization + owner
matching, plus voice enrollment.

Speaker embedding: concat of mean+std of 16-MFCC, mean+std of 12-chroma, and
[mean,std] of spectral-centroid / ZCR / RMS -> 62-dim, L2-normalized.

Owner matching: VAD-split -> embed each segment -> AgglomerativeClustering
(cosine, average, threshold 0.35) -> compare each cluster centroid to the enrolled
owner embedding by cosine. Best score >=0.72 -> verified, >=0.55 -> low_confidence,
else not_found.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
_SCALER_PATH = _MODELS_DIR / "speaker_feature_scaler.joblib"
_scaler = None
_scaler_loaded = False
_voice_encoder = None
_voice_encoder_tried = False
_RESEMBLYZER_DIM = 256

# Constants
MIN_ENROLLMENT_SAMPLES = 3
MAX_ENROLLMENT_SAMPLES = 5
# Thresholds for Resemblyzer d-vector cosine (same-speaker ~0.75-0.95, cross ~0.4-0.70).
OWNER_THRESHOLD = 0.75
LOW_CONF_THRESHOLD = 0.65
MIN_OWNER_RATIO_FOR_TRAINING = 0.25
MIN_SEGMENT_DURATION = 0.35
MAX_SEGMENTS_TRANSCRIBE = 12
SAMPLE_RATE = 16000
EMB_DIM = 62


# ── Speaker embedding ──────────────────────────────────────────────────────────

def _load_scaler():
    """Per-dimension StandardScaler so features are comparable before L2 (makes cosine
    speaker-discriminative). Without it, large MFCC-energy dims dominate → cosine≈1 for all."""
    global _scaler, _scaler_loaded
    if not _scaler_loaded:
        _scaler_loaded = True
        if _SCALER_PATH.exists():
            try:
                import joblib
                _scaler = joblib.load(_SCALER_PATH)
                logger.info("Speaker feature scaler loaded")
            except Exception as exc:
                logger.warning("Speaker scaler load failed: %s", exc)
                _scaler = None
    return _scaler


def raw_speaker_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray | None:
    """The unscaled 62-dim feature vector (used by the scaler fitter and the embedder)."""
    if y is None or len(y) < int(0.05 * sr):
        return None
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        feat = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            chroma.mean(axis=1), chroma.std(axis=1),
            [centroid.mean(), centroid.std()],
            [zcr.mean(), zcr.std()],
            [rms.mean(), rms.std()],
        ]).astype(np.float32)
        return np.nan_to_num(feat)
    except Exception as exc:
        logger.warning("speaker feature extraction failed: %s", exc)
        return None


def _extract_speaker_embedding_from_waveform(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    feat = raw_speaker_features(y, sr)
    if feat is None:
        return np.zeros(EMB_DIM, dtype=np.float32)

    scaler = _load_scaler()
    if scaler is not None:
        feat = scaler.transform(feat.reshape(1, -1))[0].astype(np.float32)

    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat.astype(np.float32)


def _get_voice_encoder():
    """Lazy Resemblyzer VoiceEncoder — neural speaker d-vectors (the real owner-ID engine).
    Falls back to classical librosa features if unavailable."""
    global _voice_encoder, _voice_encoder_tried
    if not _voice_encoder_tried:
        _voice_encoder_tried = True
        try:
            from resemblyzer import VoiceEncoder

            _voice_encoder = VoiceEncoder(verbose=False)
            logger.info("Resemblyzer VoiceEncoder loaded (neural speaker embeddings)")
        except Exception as exc:
            logger.warning("Resemblyzer unavailable — using classical features: %s", exc)
            _voice_encoder = None
    return _voice_encoder


def _speaker_embedding(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Primary speaker embedding: Resemblyzer d-vector, else classical fallback.
    A run uses one space consistently (enrollment + analysis call this)."""
    enc = _get_voice_encoder()
    if enc is not None:
        try:
            if y is None or len(y) < int(0.1 * sr):
                return np.zeros(_RESEMBLYZER_DIM, dtype=np.float32)
            from resemblyzer import preprocess_wav

            # Resemblyzer requires ITS loudness normalization (target dBFS) for consistent
            # d-vectors — peak-normalized audio alone yields unstable embeddings.
            wav = preprocess_wav(np.asarray(y, dtype=np.float32), source_sr=sr)
            if wav is None or len(wav) < int(0.1 * 16000):
                return np.zeros(_RESEMBLYZER_DIM, dtype=np.float32)
            return enc.embed_utterance(wav).astype(np.float32)
        except Exception as exc:
            logger.warning("resemblyzer embed failed: %s", exc)
            return np.zeros(_RESEMBLYZER_DIM, dtype=np.float32)
    return _extract_speaker_embedding_from_waveform(y, sr)


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Profile helpers ────────────────────────────────────────────────────────────

def _get_profile(db, user_id):
    from api.models.speaker_profile import SpeakerProfile

    doc = SpeakerProfile.get_collection(db).find_one({"user_id": user_id})
    return SpeakerProfile.from_dict(doc) if doc else None


def _save_profile(db, profile):
    from api.models.speaker_profile import SpeakerProfile

    profile.updated_at = datetime.now(timezone.utc)
    SpeakerProfile.get_collection(db).update_one(
        {"user_id": profile.user_id}, {"$set": profile.to_dict()}, upsert=True
    )


def _embed_file(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return _speaker_embedding(y)


# ── Enrollment ─────────────────────────────────────────────────────────────────

def start_enrollment(db, user_id: str) -> dict:
    from api.models.speaker_profile import SpeakerProfile

    profile = _get_profile(db, user_id) or SpeakerProfile(user_id=user_id)
    profile.enrollment_state = "collecting"
    profile.pending_embeddings = []
    profile.sample_count = 0
    profile.enrolled = False
    profile.owner_embedding = None
    _save_profile(db, profile)
    return {"message": "Enrollment started. Upload 3-5 clean owner-voice clips."}


def add_enrollment_sample(db, user_id: str, path: str) -> dict:
    profile = _get_profile(db, user_id)
    if not profile or profile.enrollment_state != "collecting":
        start_enrollment(db, user_id)
        profile = _get_profile(db, user_id)

    if profile.sample_count >= MAX_ENROLLMENT_SAMPLES:
        return {"message": f"Already have the maximum {MAX_ENROLLMENT_SAMPLES} samples.",
                "samples_collected": profile.sample_count}

    emb = _embed_file(path)
    profile.pending_embeddings.append([float(x) for x in emb])
    profile.sample_count = len(profile.pending_embeddings)
    _save_profile(db, profile)
    return {"message": f"Sample {profile.sample_count} collected.",
            "samples_collected": profile.sample_count}


def complete_enrollment(db, user_id: str) -> dict:
    profile = _get_profile(db, user_id)
    if not profile or profile.sample_count < MIN_ENROLLMENT_SAMPLES:
        have = profile.sample_count if profile else 0
        raise ValueError(f"Need at least {MIN_ENROLLMENT_SAMPLES} samples (have {have}).")

    mean_emb = _l2norm(np.mean(np.array(profile.pending_embeddings, dtype=np.float32), axis=0))
    profile.owner_embedding = [float(x) for x in mean_emb]
    profile.enrolled = True
    profile.enrollment_state = "completed"
    profile.pending_embeddings = []
    _save_profile(db, profile)
    return {"message": "Enrollment complete. Owner voice profile saved.",
            "enrolled": True, "sample_count": profile.sample_count}


def get_enrollment_status(db, user_id: str) -> dict:
    profile = _get_profile(db, user_id)
    if not profile:
        return {
            "enrolled": False, "enrollment_state": "not_started",
            "samples_collected": 0, "required_samples": MIN_ENROLLMENT_SAMPLES,
            "max_samples": MAX_ENROLLMENT_SAMPLES, "updated_at": None,
        }
    return {
        "enrolled": profile.enrolled,
        "enrollment_state": profile.enrollment_state,
        "samples_collected": profile.sample_count,
        "required_samples": MIN_ENROLLMENT_SAMPLES,
        "max_samples": MAX_ENROLLMENT_SAMPLES,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }


# ── Diarization + owner matching ───────────────────────────────────────────────

def analyze_speakers_and_extract_owner_audio(path: str, user_id: str, db) -> dict | None:
    """
    Returns speaker metadata dict, or None if no usable segments (caller falls back
    to full audio). When the user isn't enrolled, owner detection is 'not_found'.
    """
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        logger.warning("speaker analysis load failed: %s", exc)
        return None

    intervals = librosa.effects.split(y, top_db=30)
    segments = [(int(s), int(e)) for s, e in intervals if (e - s) / sr >= MIN_SEGMENT_DURATION]
    if not segments:
        return None

    embeddings = np.array([_speaker_embedding(y[s:e], sr) for s, e in segments])

    # Cluster segments
    if len(segments) == 1:
        labels = np.array([0])
    else:
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(
            metric="cosine", linkage="average", distance_threshold=0.35, n_clusters=None
        ).fit_predict(embeddings)

    profile = _get_profile(db, user_id)
    owner_emb = np.array(profile.owner_embedding, dtype=np.float32) if (profile and profile.owner_embedding) else None

    # Per-cluster centroid + cosine vs owner
    cluster_ids = sorted(set(labels.tolist()))
    cluster_centroids = {c: _l2norm(embeddings[labels == c].mean(axis=0)) for c in cluster_ids}
    cluster_owner_score = {
        c: (_cosine(cluster_centroids[c], owner_emb) if owner_emb is not None else -1.0)
        for c in cluster_ids
    }

    best_cluster = max(cluster_ids, key=lambda c: cluster_owner_score[c]) if owner_emb is not None else None
    best_score = cluster_owner_score[best_cluster] if best_cluster is not None else -1.0
    if owner_emb is None:
        status = "not_found"
    elif best_score >= OWNER_THRESHOLD:
        status = "verified"
    elif best_score >= LOW_CONF_THRESHOLD:
        status = "low_confidence"
    else:
        status = "not_found"

    owner_cluster = best_cluster if status in ("verified", "low_confidence") else None

    # Build timeline + collect owner audio
    other_label_map: dict[int, int] = {}
    timeline = []
    owner_chunks = []
    owner_time = 0.0
    total_time = 0.0
    owner_segments = 0
    other_segments = 0
    for (s, e), c in zip(segments, labels):
        dur = (e - s) / sr
        total_time += dur
        is_owner = (c == owner_cluster)
        if is_owner:
            label = "OWNER"
            owner_chunks.append(y[s:e])
            owner_time += dur
            owner_segments += 1
        else:
            if c not in other_label_map:
                other_label_map[c] = len(other_label_map) + 1
            label = f"OTHER_{other_label_map[c]}"
            other_segments += 1
        timeline.append({
            "speaker_label": label,
            "start": round(s / sr, 3),
            "end": round(e / sr, 3),
            "owner_confidence": round(float(cluster_owner_score[c]), 4) if owner_emb is not None else 0.0,
        })

    owner_speech_ratio = (owner_time / total_time) if total_time > 0 else 0.0

    # Write owner-only audio
    owner_audio_path = None
    if owner_chunks:
        owner_y = np.concatenate(owner_chunks)
        stem = Path(path)
        owner_audio_path = str(stem.with_name(f"{stem.stem}_owner.wav"))
        sf.write(owner_audio_path, owner_y, SAMPLE_RATE)

    trainable = (owner_segments > 0 and owner_speech_ratio >= MIN_OWNER_RATIO_FOR_TRAINING
                 and status == "verified")

    return {
        "owner_audio_path": owner_audio_path,
        "speaker_timeline": timeline,
        "owner_speech_ratio": round(owner_speech_ratio, 4),
        "owner_segments_count": owner_segments,
        "other_segments_count": other_segments,
        "owner_detection_status": status,
        "personalization_trainable": bool(trainable),
        "num_segments": len(segments),
    }
