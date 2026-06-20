import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import audio, auth, rag, speaker

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Audio Emotion Backend",
    description="Voice second brain: transcription, emotion analysis, personalization, RAG",
    version="0.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(audio.router)
app.include_router(rag.router)
app.include_router(speaker.router)


# ── Lifecycle ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    from api.db.mongodb import connect_mongodb
    connect_mongodb()

    # Connect Qdrant (fail-soft — vector search disabled if down)
    from api.db.qdrant import connect_qdrant
    connect_qdrant()

    # Detect + eager-load the global emotion head (graceful if missing)
    from api.services.global_emotion_head import load_global_head
    if load_global_head():
        logger.info("Active emotion model: global-head")
    else:
        logger.warning("No global head found — emotion falls back to neutral/0.5")


@app.on_event("shutdown")
def shutdown():
    from api.db.mongodb import disconnect_mongodb
    disconnect_mongodb()


# ── Root + health ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "RAG Audio Emotion Backend", "version": "0.6.0", "docs": "/docs"}


@app.get("/health")
def health():
    from api.db.mongodb import get_database
    db_status = "connected" if get_database() is not None else "unavailable"
    return {"status": "healthy", "mongodb": db_status}
