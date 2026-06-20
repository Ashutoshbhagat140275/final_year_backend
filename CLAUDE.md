# CLAUDE.md — RAG Audio Emotion Backend

## Project
Voice "second brain": upload audio → transcribe (Whisper) + classify emotion (Wav2Vec2 dual-head) + speaker diarization → store in MongoDB + Qdrant → query via RAG (Ollama/mistral). Personalizes per user via a feedback-trained linear head.

## Layout
```
api/
├── main.py            # App factory, startup, router registration
├── config.py          # pydantic-settings (.env overridable)
├── feature_config.py  # ML constants shared by training + inference
├── routers/           # Thin HTTP controllers
├── middleware/        # JWT auth dependency
├── services/          # All ML + business logic
├── models/            # MongoDB document models
├── schemas/           # Pydantic request/response schemas
└── db/                # MongoDB, Qdrant, Redis connection managers
training/              # Offline training scripts (global head, user head)
models/                # Saved artifacts: global_emotion_head.pt, scaler, user_heads/
uploads/               # Per-user audio (gitignored)
```

## Build stages
- Stage 1: Scaffold + Auth (register/login/JWT) — `/health`, `/api/auth/*`
- Stage 2: MongoDB models + real DB persistence
- Stage 3: Audio upload + preprocessing (16 kHz/VAD/normalize)
- Stage 4: Wav2Vec2 encoder + global emotion classifier
- Stage 5: Whisper transcription + Qdrant + RAG query
- Stage 6: Speaker enrollment + diarization + owner detection
- Stage 7: Feedback + user-head training + Alpha Engine
- Stage 8: Redis cache + dashboard + admin

## Key constants (feature_config.py)
- Emotion labels: neutral, calm, happy, sad, angry, fearful, disgusted, surprised (8 classes)
- Wav2Vec2: `facebook/wav2vec2-base`, 768-dim embeddings
- Alpha Engine: sigmoid blend `α = alpha_data × alpha_conf`, K=50, τ=0.6, β=10
- Training trigger: N ≥ 20 feedback AND N % 10 == 0

## Running
```bash
cd api
uvicorn main:app --reload --port 8000
```

## Services topology (all local)
- FastAPI :8000 ↔ MongoDB :27017, Qdrant :6333, Ollama :11434, Redis :6379 (optional)

## Important notes
- PyMongo is **sync** under async FastAPI — fine at dev scale
- BackgroundTasks are in-process; swap Celery for production
- Owner-safety gate: feedback only accepted for `personalization_trainable=True` sessions
- Change `JWT_SECRET_KEY` before any real deployment
- CORS is currently `*` — scope it for production
