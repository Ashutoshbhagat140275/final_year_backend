# What This Backend Does — Purpose & End Goal

> A plain-language explanation of the `api/` backend: what it is, what problem it
> solves, how it works, and where it's ultimately headed. For the "is it working"
> health/test report, see [BACKEND_TEST_REPORT.md](BACKEND_TEST_REPORT.md).

---

## 1. In One Sentence

It is the **brain of a "Second Brain" for your voice** — a multi-tenant backend that
takes audio you record, figures out **what you said** (transcription), **how you felt**
(emotion), and **who was speaking** (speaker identity), stores it privately per user,
and lets you **ask questions about your own recordings in natural language**.

---

## 2. The End Goal

> **Build a personal, voice-first memory system that understands not just your words,
> but your emotions — and gets more accurate about *you* the more you use it.**

Most note-taking and search tools only capture *text* and treat *everyone the same*.
This project aims higher on two axes:

1. **Emotion-aware memory.** Every recording is tagged with the emotion in your voice,
   so the system can answer not only *"what did I say about the project?"* but support
   questions like *"when was I stressed last week?"* — turning raw audio into an
   emotional + factual timeline of your life.

2. **Personalization that learns you.** Emotion in speech is deeply individual — your
   "angry" sounds different from someone else's. The end goal is a model that **adapts
   to each user's voice over time** through their own feedback, so predictions become
   personally accurate rather than generically average.

The north star: **a private, self-improving audio knowledge base** where you speak,
it remembers and understands, and you can converse with your own history.

---

## 3. The Problem It Solves

| Pain point | What this backend provides |
|------------|----------------------------|
| Voice notes pile up and become unsearchable | Auto-transcription + semantic (meaning-based) search over everything you've recorded |
| Generic emotion AI is inaccurate for *you* | A per-user model trained on *your* corrections |
| "What did I talk about / how did I feel?" is hard to answer | A RAG chatbot that answers in natural language, grounded in your actual recordings |
| Privacy concerns with cloud AI | Runs entirely on **local/self-hosted** infrastructure (local LLM, self-hosted DBs) — your audio never leaves your machine |
| Multi-user data leaking together | Strict **multi-tenancy** — each user's data, vectors, and personal model are isolated |

---

## 4. What It Actually Does (Core Capabilities)

1. **User accounts & security** — register / login with JWT-protected, per-user access.
2. **Audio ingestion** — accept WAV/MP3/M4A/FLAC/OGG, validate, clean (resample to
   16 kHz, trim silence, normalize).
3. **Emotion analysis** — classify the speaker's emotion into 8 classes
   (`neutral, calm, happy, sad, angry, fearful, disgusted, surprised`) using
   neural audio embeddings (Wav2Vec2) + a trained classifier.
4. **Speech-to-text** — transcribe with Whisper, tagged by speaker.
5. **Speaker identification** — distinguish the **owner's** voice from other people in
   the recording (via a voice-enrollment profile), so emotion/personalization focus on
   *you*, not bystanders.
6. **Private memory store** — embed each transcript and store it in a per-user vector
   collection for semantic retrieval.
7. **Conversational querying (RAG)** — answer natural-language questions by retrieving
   the most relevant snippets and having a local LLM compose a grounded answer with
   sources.
8. **Personalization loop** — let users correct wrong emotion predictions; once enough
   corrections accumulate, train a personalized model just for them.
9. **Dashboards** — emotion history and usage statistics over time.
10. **Admin tooling** — manage and clean up per-user model artifacts.

---

## 5. How It Works — The Journey of One Recording

```
 You speak ──► Upload audio
                  │
                  ▼
       ┌──────────────────────────────┐
       │ 1. Clean: 16 kHz, trim, norm │
       └──────────────────────────────┘
                  │
                  ▼
       ┌──────────────────────────────┐
       │ 2. Who is talking? (speaker  │  ← isolates YOUR voice from others
       │    diarization vs. your      │
       │    enrolled voice profile)   │
       └──────────────────────────────┘
                  │
        ┌─────────┴───────────┐
        ▼                     ▼
 ┌───────────────┐   ┌──────────────────┐
 │ 3. Emotion    │   │ 4. Transcribe    │
 │ (Wav2Vec2 +   │   │ (Whisper, with   │
 │  dual-head    │   │  speaker tags)   │
 │  classifier)  │   └──────────────────┘
 └───────────────┘            │
        │                     │
        └─────────┬───────────┘
                  ▼
       ┌──────────────────────────────┐
       │ 5. Save:                     │
       │   • MongoDB (session,emotion)│
       │   • Qdrant  (text embedding) │  ← your private, searchable memory
       └──────────────────────────────┘

 Later… you ask: "What did I talk about when I was happy?"
                  │
                  ▼
   Retrieve relevant snippets (Qdrant) ──► Local LLM (Ollama) ──► grounded answer + sources
```

And the loop that makes it smarter:

```
 Wrong emotion? ──► You correct it ──► stored as feedback
                                          │
                          (at 20 corrections, then every 10)
                                          ▼
                            Train a personal model just for YOU
                                          │
                                          ▼
                  Future predictions blend "general" + "your" model
```

---

## 6. The Standout Idea — Dual-Head Personalization & the "Alpha Engine"

This is the project's signature innovation and the heart of its end goal.

- A **Global Head**: one emotion model trained on public datasets (everyone's patterns).
- A **User Head**: a small model trained on *your* feedback corrections (your patterns).
- The **Alpha Engine** decides, for every prediction, **how much to trust each**:

  ```
  Final = α · Global  +  (1 − α) · User
  ```

  `α` is computed from two things: **how much feedback you've given** (more feedback →
  trust your personal model more) and **how confident the global model is**. The current
  configuration uses a smooth *sigmoid* formula so the system transitions gracefully from
  "trust the crowd" (new user) to "trust you" (experienced user).

**Why it matters:** it directly serves the end goal — the system literally **becomes
more *you* over time**, without ever discarding the safety net of general knowledge.

---

## 7. Design Principles Baked In

- **Privacy-first / local:** local LLM (Ollama) + self-hosted vector DB (Qdrant) +
  local MongoDB. Audio and insights stay on-prem.
- **Multi-tenant isolation:** separate vector collections and personal models per user.
- **Graceful degradation:** if optional services (e.g. the Redis cache) are down, the
  app keeps working with reduced features rather than crashing.
- **Owner-safety:** personalization training only uses audio confirmed to be the
  owner's own voice, so other people's voices can't corrupt your personal model.
- **Separation of concerns:** thin HTTP controllers → services (business + ML) →
  models/DB, making the ML pipeline swappable and testable.

---

## 8. Technology Stack

| Concern | Technology |
|---------|------------|
| API framework | FastAPI (Python) |
| Audio embeddings | Wav2Vec2 (HuggingFace Transformers) |
| Emotion classifier | PyTorch (global + per-user heads) |
| Speech-to-text | OpenAI Whisper |
| Speaker analysis | librosa / parselmouth-based diarization + voice profiles |
| Vector search | Qdrant (per-user collections) |
| Text embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM / answers | Ollama (local, e.g. `mistral`) |
| User & metadata DB | MongoDB |
| Query cache (optional) | Redis |
| Auth | JWT (HS256) |

---

## 9. Who It's For

- **Individuals** wanting a private, searchable, emotion-aware journal of their spoken
  thoughts — a true "second brain" for voice.
- **Researchers/students** exploring personalized affective computing (this project is
  built around the paper *"Improving Audio Emotion Classification with a Deep Neural
  Network"*).
- **Developers** who need a template for privacy-preserving, multi-tenant RAG systems
  that combine speech, emotion, and personalization.

---

## 10. Summary

This backend turns spoken audio into a **private, understanding, conversational
memory**. It hears your words, reads your emotions, knows your voice from others, and —
crucially — **learns to understand *you* specifically** through your feedback. The end
goal is a self-improving, emotion-aware personal knowledge base you can simply talk to.
