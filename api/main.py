import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import auth

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Audio Emotion Backend",
    description="Voice second brain: transcription, emotion analysis, personalization, RAG",
    version="0.2.0",
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


# ── Lifecycle ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    from api.db.mongodb import connect_mongodb
    connect_mongodb()


@app.on_event("shutdown")
def shutdown():
    from api.db.mongodb import disconnect_mongodb
    disconnect_mongodb()


# ── Root + health ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "RAG Audio Emotion Backend", "version": "0.2.0", "docs": "/docs"}


@app.get("/health")
def health():
    from api.db.mongodb import get_database
    db_status = "connected" if get_database() is not None else "unavailable"
    return {"status": "healthy", "mongodb": db_status}
