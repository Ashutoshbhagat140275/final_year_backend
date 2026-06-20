"""
RAG query service — answer natural-language questions grounded in a user's
own transcribed recordings.

Flow: embed query once → (Stage 7: semantic cache check) → Qdrant top-k search →
build context → local LLM (Ollama) → {answer, sources, query}.
"""
import logging

import requests

from api.config import settings
from api.services.vector_store import embed_text, search_documents

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Based on the following context from audio transcriptions, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""


def query_rag(user_id: str, query: str, top_k: int = 5) -> dict:
    # 1) embed query once (reused for cache + search)
    try:
        query_embedding = embed_text(query)
    except Exception as exc:
        logger.warning("Query embedding failed: %s", exc)
        query_embedding = None

    # 2) Stage 7: semantic cache check goes here

    # 3) retrieve
    sources = search_documents(user_id, query, top_k=top_k, query_embedding=query_embedding)
    if not sources:
        return {
            "answer": "I couldn't find any recordings to answer that. Try uploading some audio first.",
            "sources": [],
            "query": query,
        }

    # 4) build context
    context = "\n\n".join(
        f"Document {i+1} (from session {s.get('session_id')}):\n{s.get('text','')}"
        for i, s in enumerate(sources)
    )
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    # 5) local LLM
    answer = _ollama_generate(prompt)

    # 6) format sources (+ Stage 7: store in cache)
    formatted = [
        {
            "text": s.get("text", ""),
            "session_id": s.get("session_id"),
            "timestamp": s.get("timestamp"),
            "score": s.get("score"),
        }
        for s in sources
    ]
    return {"answer": answer, "sources": formatted, "query": query}


def _ollama_generate(prompt: str) -> str:
    try:
        resp = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()
    except Exception as exc:
        logger.warning("Ollama generate failed: %s", exc)
        return "The language model is currently unavailable, so I can't compose an answer right now."
