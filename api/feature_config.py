import math

# ── Emotion labels ─────────────────────────────────────────────────────────────
EMOTION_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
NUM_CLASSES = 8
EMOTION_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}

# ── Wav2Vec2 ──────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 768
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
SAMPLE_RATE = 16000

# ── Alpha Engine ──────────────────────────────────────────────────────────────
USE_SIGMOID_ALPHA = True
ALPHA_FEEDBACK_SCALE_K = 50          # K — feedback volume scale
ALPHA_CONFIDENCE_THRESHOLD_TAU = 0.6  # τ — sigmoid midpoint on global confidence
ALPHA_SIGMOID_SHARPNESS_BETA = 10    # β — sigmoid sharpness

# ── Training triggers / hyperparameters (user head) ───────────────────────────
MIN_FEEDBACK_FOR_TRAINING = 20
INCREMENTAL_TRAINING_INTERVAL = 10
TRAINING_EPOCHS = 20
TRAINING_BATCH_SIZE = 16
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
USER_HEAD_CACHE_SIZE = 100

# ── Dataset label maps (offline training) ─────────────────────────────────────
RAVDESS_CODE_TO_IDX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
CREMAD_CODE_TO_IDX = {"NEU": 0, "HAP": 2, "SAD": 3, "ANG": 4, "FEA": 5, "DIS": 6}


def _validate_alpha_config() -> None:
    assert ALPHA_FEEDBACK_SCALE_K > 0, "K must be > 0"
    assert 0 < ALPHA_CONFIDENCE_THRESHOLD_TAU < 1, "τ must be in (0, 1)"
    assert ALPHA_SIGMOID_SHARPNESS_BETA > 0, "β must be > 0"


_validate_alpha_config()
