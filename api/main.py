from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import auth

app = FastAPI(
    title="RAG Audio Emotion Backend",
    description="Voice second brain: transcription, emotion analysis, personalization, RAG",
    version="0.1.0",
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


# ── Root + health ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "RAG Audio Emotion Backend", "version": "0.1.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}
