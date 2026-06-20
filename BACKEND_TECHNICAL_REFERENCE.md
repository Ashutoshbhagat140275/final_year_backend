# Backend Technical Reference — RAG Audio Emotion Analysis

> **Purpose of this document:** A complete, implementation-level blueprint of the
> `api/` backend, written so it can be used as **context to design and build a similar
> system from scratch**. It covers the architecture, every component's algorithm and
> data shapes, the database schemas, the request flows, the full API surface, and all
> the magic constants/hyperparameters.
>
> Companions: [BACKEND_OVERVIEW.md](BACKEND_OVERVIEW.md) (the "what & why" in plain
> language) and [BACKEND_TEST_REPORT.md](BACKEND_TEST_REPORT.md) (the "is it working").

---

## 0. What you're building

A **multi-tenant, privacy-first, voice "second brain"**: users upload audio; the system
transcribes it (Whisper), classifies the speaker's **emotion** (Wav2Vec2 + a dual-head
neural classifier that *personalizes per user*), identifies **who is speaking** (owner
vs. others via speaker diarization), stores everything in **per-user** stores, and lets
users **query their recordings in natural language** via RAG over a local LLM (Ollama).
A **feedback loop** lets users correct emotion predictions; once enough corrections
accumulate, a **personal model** is trained for that user.

Three ideas define the system and are worth replicating deliberately:
1. **Dual-head emotion model + Alpha Engine** — blend a global model with a per-user
   model, weighted dynamically by feedback volume and global confidence.
2. **Owner-aware processing** — isolate the account owner's voice from other speakers so
   emotion and personalization are about *the user*, not bystanders.
3. **Graceful multi-service degradation** — the app boots and serves even when optional
   backends (Redis cache, model artifacts) are missing.

---

## 1. Technology Stack

| Layer | Choice | Version (pinned) | Notes |
|-------|--------|------------------|-------|
| Language/runtime | Python | 3.10 | venv at `api/venv` |
| Web framework | FastAPI | 0.104.1 | ASGI, served by Uvicorn 0.24.0 |
| Audio embeddings | Wav2Vec2 (`facebook/wav2vec2-base`) | transformers 4.57.6 | 768-dim |
| Emotion classifier | PyTorch | 2.10.0 | Linear heads (global + per-user) |
| Speech-to-text | OpenAI Whisper | 20231117 | model size `small`, lang `en` |
| Speaker embeddings | librosa hand-crafted features | librosa 0.10.1 | MFCC/chroma/spectral (NOT a neural speaker model) |
| Text embeddings | sentence-transformers `all-MiniLM-L6-v2` | 5.2.2 | 384-dim |
| Vector DB | Qdrant | qdrant-client 1.7.0 | per-user collections, cosine |
| LLM | Ollama (local) | REST API | default model `mistral` |
| Primary DB | MongoDB | pymongo 4.6.0 (**sync**) | users, sessions, feedback, jobs, profiles, models |
| Cache (optional) | Redis | redis 5.0.1 | semantic query cache |
| Auth | JWT (HS256) + bcrypt | python-jose 3.3.0, passlib/bcrypt | 24h tokens |
| Config | pydantic-settings | 2.1.0 | `.env` + defaults |

**Runtime topology:** FastAPI process (port 8000) ↔ MongoDB (27017), Qdrant (6333),
Ollama (11434), Redis (6379, optional). All local/self-hosted.

---

## 2. Architecture & Project Layout

### 2.1 Layered design (clean separation)
```
app/
├── main.py            # App factory, CORS, router registration, startup/shutdown, model-format detection
├── config.py          # pydantic-settings: all env-configurable settings + defaults
├── routers/           # HTTP controllers (thin) — auth, audio, rag, dashboard, feedback, speaker, admin
├── middleware/auth.py # JWT bearer dependency: get_current_user / _user_id / _admin
├── services/          # ALL business logic + ML (the heart of the system)
├── models/            # MongoDB document models (plain classes w/ to_dict/from_dict/get_collection)
├── schemas/           # Pydantic request/response contracts
└── db/                # Connection managers: mongodb.py, qdrant.py, redis.py
training/              # Offline scripts: train_wav2vec2 (global head), train_user_head, prepare_dataset
models/                # Saved artifacts: global_emotion_head.pt, embedding_scaler.joblib, user_heads/*.pt
uploads/               # Per-user uploaded audio: uploads/<user_id>/<timestamp>.<ext>
```

**Dependency direction:** `routers → services → (models, db, external clients)`.
Routers never touch the DB driver directly except via models' `get_collection(db)`.
Services hold all ML and orchestration. This keeps the ML pipeline swappable.

### 2.2 Request lifecycle
1. Uvicorn receives HTTP request → FastAPI routes to a router function.
2. Protected routes resolve the `get_current_user*` dependency → decode JWT → verify
   user exists in Mongo → inject `user_id`.
3. Router validates input (Pydantic schema) and delegates to a service.
4. Service performs work (ML, DB, vector ops, LLM) and returns a plain dict.
5. Router wraps the dict in a Pydantic response model → JSON.

### 2.3 Startup sequence (`main.py` `@app.on_event("startup")`)
1. Connect MongoDB (creates indexes). On failure: **log warning, continue**.
2. Connect Qdrant (verifies via `get_collections()`). On failure: warn, continue.
3. Connect Redis (`ping`). On failure: warn, continue (caching disabled).
4. `detect_active_model_format()` → one of `dual-head | global-only | legacy | unavailable`
   based on which artifacts exist (`global_emotion_head.pt`, `user_heads/*.pt`,
   `embedding_classifier.pt`).
5. Log Alpha Engine config (sigmoid/linear + K/τ/β).
6. Eager-load the global emotion head; user heads are lazy-loaded per request (LRU).
> **Design principle: graceful degradation.** Every external dependency is optional at
> boot. Missing models fall back to a fixed `("neutral", 0.5)` prediction; missing Redis
> disables caching; missing Qdrant collection returns "no documents yet".

---

## 3. Configuration (`config.py`)

All settings are env-overridable (`.env`, case-insensitive). Defaults:

| Setting | Default | Used by |
|---------|---------|---------|
| `mongodb_url` / `mongodb_db_name` | `mongodb://localhost:27017` / `rag_audio_db` | Mongo |
| `qdrant_url` / `qdrant_api_key` | `http://localhost:6333` / `None` | Qdrant |
| `jwt_secret_key` | `your-secret-key-change-in-production` ⚠️ | Auth (CHANGE THIS) |
| `jwt_algorithm` / `jwt_expiration_hours` | `HS256` / `24` | Auth |
| `ollama_base_url` / `ollama_model` | `http://localhost:11434` / `mistral` | RAG |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Vector store (384-dim) |
| `redis_url` | `redis://localhost:6379/0` | Cache |
| `query_cache_ttl_seconds` | `3600` | Cache |
| `query_cache_max_per_user` | `20` | Cache eviction |
| `query_cache_similarity_threshold` | `0.95` | Semantic cache hit |
| `whisper_model_size` / `whisper_language` | `small` / `en` | Transcription |
| `audio_upload_dir` / `max_audio_size_mb` | `./uploads` / `50` | Audio I/O |
| `enable_speaker_aware_processing` | `True` | Speaker pipeline toggle |
| `USE_MONGODB_STORAGE` / `DUAL_SAVE_MODE` / `MONGODB_STORAGE_COMPRESSION` | `False` / `False` / `gzip` | User-head storage backend |

A separate **`feature_config.py`** holds ML constants (see §5, §8, §11) — it is the single
source of truth shared by training scripts and inference to guarantee feature parity.

---

## 4. Data Model

### 4.1 MongoDB (database `rag_audio_db`, **sync PyMongo**)
`get_database()` returns a global `Database`. Each model exposes `get_collection(db)`,
`to_dict()`, `from_dict()`.

| Collection | Model | Key fields |
|------------|-------|-----------|
| `users` | User | `email`, `password_hash` (bcrypt), `feedback_count:int=0`, `is_admin:bool=False`, `created_at` |
| `audio_sessions` | AudioSession | `user_id:str`, `audio_file_path`, `emotion_data:{label,emotion,confidence}`, `transcription_text`, `qdrant_collection_id`, `speaker_timeline`, `owner_detection_status`, `owner_speech_ratio`, `owner_segments_count`, `other_segments_count`, `personalization_trainable:bool=True`, `timestamp` |
| `emotion_analyses` | EmotionAnalysis | `user_id`, `session_id`, `emotion_label`, `confidence`, `mfcc_features:List[float]` (**actually the 768-dim Wav2Vec2 embedding**), `timestamp` |
| `user_feedback` | UserFeedback | `user_id`, `session_id`, `embedding:List[float]` (768-dim), `predicted_emotion`, `corrected_emotion`, `timestamp` |
| `speaker_profiles` | SpeakerProfile | `user_id` (unique), `owner_embedding:List[float]|None`, `enrolled:bool`, `enrollment_state`, `pending_embeddings:List[List[float]]`, `sample_count`, `created_at`, `updated_at` |
| `training_jobs` | TrainingJob | `user_id`, `job_id` (unique UUID), `status∈{queued,running,completed,failed}`, `created_at`, `updated_at`, `started_at`, `completed_at`, `error_message`, `metrics:dict|None` |
| `user_models` | UserModelStorage | `user_id` (unique), `model_blob:Binary` (gzip-compressed `torch.save` state_dict), `version:int=1`, `metadata:dict`, `created_at`, `updated_at` |

**Indexes created at startup:**
- `user_feedback`: `user_id`, `timestamp`, compound `(user_id, timestamp)`
- `training_jobs`: `user_id`, **unique** `job_id`, compound `(user_id, created_at desc)`
- `user_models`: **unique** `user_id`, `updated_at`
- `speaker_profiles`: **unique** `user_id`, `updated_at`

> Note: `EmotionAnalysis.mfcc_features` is a historical name — it stores the **768-dim
> Wav2Vec2 embedding**, not MFCCs. This embedding is later reused as training input for
> the user head (so feedback training needs no re-extraction).

### 4.2 Qdrant (vector store)
- **Collection per user:** `user_{user_id}_documents` (created on demand, idempotent).
- **Vector config:** `size=384`, `distance=Cosine`.
- **Point:** `id = uuid4()`, `vector = MiniLM embedding of transcript`, payload:
  ```json
  {"text": "...", "session_id": "...", "user_id": "...", "timestamp": "ISO", "emotion_label": "happy"}
  ```
- **Search:** top-k (default 5), **no score threshold** (returns best k regardless).

### 4.3 Redis (optional semantic query cache)
- Entry key: `query_cache:{user_id}:{sha256(normalized_query + ":" + top_k)[:16]}`
- Per-user index: sorted set `query_cache_keys:{user_id}`, score = insert timestamp (FIFO eviction).
- Value (JSON): `{original_query, query_embedding:[...384], top_k, answer, sources, timestamp}`.
- TTL 3600s; max 20 entries/user; semantic hit needs **same top_k AND cosine ≥ 0.95**.

---

## 5. Domain Concepts & Constants (`feature_config.py`)

```python
EMOTION_LABELS = ["neutral","calm","happy","sad","angry","fearful","disgusted","surprised"]  # idx 0..7
NUM_CLASSES = 8
EMBEDDING_DIM = 768
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
SAMPLE_RATE = 16000

# Alpha Engine
USE_SIGMOID_ALPHA = True            # sigmoid formula active (linear is legacy fallback)
ALPHA_FEEDBACK_SCALE_K = 50         # K
ALPHA_CONFIDENCE_THRESHOLD_TAU = 0.6  # τ
ALPHA_SIGMOID_SHARPNESS_BETA = 10   # β

# Training triggers / hyperparameters (user head)
MIN_FEEDBACK_FOR_TRAINING = 20
INCREMENTAL_TRAINING_INTERVAL = 10
TRAINING_EPOCHS = 20
TRAINING_BATCH_SIZE = 16
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
USER_HEAD_CACHE_SIZE = 100          # LRU of loaded user heads

# Dataset label maps (training)
RAVDESS_CODE_TO_IDX = {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7}
CREMAD_CODE_TO_IDX  = {"NEU":0,"HAP":2,"SAD":3,"ANG":4,"FEA":5,"DIS":6}  # no calm/surprised
```
A `_validate_alpha_config()` runs at import to enforce `K>0`, `0<τ<1`, `β>0`.

---

## 6. Component Deep-Dive

### 6.1 Authentication (`services/auth.py`, `middleware/auth.py`)
- **Hashing:** bcrypt (`bcrypt.hashpw`/`checkpw`).
- **`create_user(email, password)`** → enforces unique email, stores `password_hash`,
  `feedback_count=0`, `is_admin=False`.
- **`authenticate_user(email, password)`** → returns User or None.
- **`create_access_token(data)`** → JWT with custom claims `{user_id, email}` + `exp`
  (now + 24h), signed HS256.
- **`get_current_user`** (dependency) → decode token, require `user_id` & `email`,
  **re-check the user exists in Mongo**, return `{user_id, email}`. Invalid → 401.
- **`get_current_admin`** → also require `user.is_admin == True`, else 403.

### 6.2 Audio processing pipeline (`services/audio_processor.py`)
`process_audio(user_id, UploadFile) -> dict` — the orchestrator. Steps:
1. **Validate** — ext ∈ `{wav,mp3,m4a,flac,ogg}`, size ≤ 50 MB.
2. **Save** — `uploads/<user_id>/<YYYYMMDD_HHMMSS>.<ext>`.
3. **Preprocess** — `librosa.load(sr=16000)`, VAD trim `librosa.effects.split(top_db=30)`
   (concatenate voiced spans), **peak-normalize** (`y/max(|y|)`), overwrite file.
4. **Speaker-aware processing** (if enabled) — `analyze_speakers_and_extract_owner_audio()`
   returns `speaker_meta` (owner audio path, timeline, ratios, status, trainable flag).
   On any error → log + fall back to full audio.
5. **Audio validation** — load emotion-source audio; if `duration < 0.5s` or
   `mean_RMS < 0.001` → fall back to full audio (avoids garbage diarized clips).
6. **Embedding** — `extract_wav2vec2_embedding(emotion_source)` → 768-dim vector.
7. **Feedback count** — read `users.feedback_count` (drives blend weight).
8. **Classify** — `classify_with_dual_heads(embedding, user_id, feedback_count)`.
9. **Transcribe** — `speaker_meta.full_transcription` (speaker-tagged) or
   `transcribe_audio(full)`.
10. **Persist** — insert `AudioSession` + `EmotionAnalysis`(with embedding) to Mongo.
11. **Index** — `store_document()` → Qdrant; `invalidate_user_cache()`.
Returns the full prediction + transcription + speaker metadata dict (see §9 upload response).

### 6.3 Wav2Vec2 encoder (`services/wav2vec2_encoder.py`)
- Lazy module-singleton of `Wav2Vec2Model`/`Wav2Vec2Processor` from `facebook/wav2vec2-base`,
  in `eval()` mode.
- `extract_wav2vec2_embedding(path) -> np.ndarray(768,)`:
  load 16 kHz → pad to ≥0.3 s (4800 samples) → `nan_to_num` → processor → model →
  **mean-pool `last_hidden_state` over time** → `(768,)` float32. Robust to NaNs.

### 6.4 Speaker service (`services/speaker_service.py`) — diarization + owner matching
**This is fully classical/CPU (no neural speaker model).** Constants:
`MIN_ENROLLMENT_SAMPLES=3`, `MAX=5`, `OWNER_THRESHOLD=0.72`, `LOW_CONF=0.55`,
`MIN_OWNER_RATIO_FOR_TRAINING=0.25`, `MIN_SEGMENT_DURATION=0.35s`, `MAX_SEGMENTS_TRANSCRIBE=12`.

**Speaker embedding** (`_extract_speaker_embedding_from_waveform`): concatenate
`mean+std` of 16-MFCC, `mean+std` of 12-chroma, and `[mean,std]` of spectral-centroid,
ZCR, RMS → ~**62-dim** vector, **L2-normalized** (empty audio → 46-dim zeros fallback).

**Enrollment:**
- `start_enrollment` → profile `enrollment_state="collecting"`, `pending_embeddings=[]`.
- `add_enrollment_sample` → extract embedding, append (cap at 5).
- `complete_enrollment` → require ≥3, `owner_embedding = L2norm(mean(pending))`,
  `enrolled=True`, state `"completed"`, clear pending.
- `get_enrollment_status` → `{enrolled, enrollment_state, samples_collected, required=3, max=5, updated_at}`.

**`analyze_speakers_and_extract_owner_audio(path, user_id, db)`:**
1. VAD-split into segments ≥0.35 s.
2. Embed each segment.
3. **Cluster** segments: `AgglomerativeClustering(metric="cosine", linkage="average",
   distance_threshold=0.35, n_clusters=None)`.
4. For each cluster, centroid = L2norm(mean of member embeddings); **cosine vs owner_embedding**.
5. Best cluster: score ≥0.72 → `"verified"`; ≥0.55 → `"low_confidence"`; else `"not_found"`.
6. Build `speaker_timeline` (`OWNER` / `OTHER_n` per segment with start/end/owner_confidence).
7. `owner_speech_ratio = owner_speech_time / total_speech_time`; concat owner segments → `<stem>_owner.wav`.
8. **Segment-wise transcription** (≤12 segments) → `"[start-end] LABEL: text"` lines.
9. `personalization_trainable = (owner_segments>0 AND ratio≥0.25 AND status=="verified")`.

### 6.5 Transcription (`services/transcription.py`)
Lazy-singleton Whisper (`settings.whisper_model_size`, default `small`).
`transcribe_audio(path, model_size=None, language=None) -> str` with opts
`{language=en, beam_size=5, best_of=5, temperature=0, condition_on_previous_text=True, fp16=False}`.

### 6.6 Dual-head classifier (`services/dual_head_classifier.py`) + Alpha Engine
**Heads** (`global_emotion_head.py`, `user_emotion_head.py`): identical architecture —
a single `nn.Linear(768, 8)`. Inference: `softmax(linear(x))`; confidence = `max(prob)`.
User heads cached with `@lru_cache(maxsize=100)`.

**`classify_with_dual_heads(embedding, user_id, feedback_count) -> dict`:**
- Always run global head → `(P_g, C_g)`.
- Try user head. **If none:** return global-only, `blend_weight=1.0`,
  `alpha_formula="linear"`, `user_*=None`.
- **If present:** `α = compute_blend_weight(C_g, feedback_count)`, then blend:
  ```python
  P_f = α * P_g + (1 - α) * P_u
  P_f = P_f / P_f.sum()           # renormalize
  emotion = argmax(P_f); confidence = max(P_f)
  ```
- Return: `{emotion, confidence, global_emotion, global_confidence, user_emotion,
  user_confidence, blend_weight, alpha_data, alpha_conf, alpha_formula, probabilities}`.

**Alpha Engine (`services/alpha_engine.py`) — `compute_blend_weight(C_g, N, user_id=None)`**
returns `{alpha, alpha_data, alpha_conf, formula}`. Switched by `USE_SIGMOID_ALPHA`:

*Sigmoid (active):*
```
alpha_data = 1 / (1 + N/K)                       # data availability  (K=50)
alpha_conf = 1 / (1 + exp(-β (C_g - τ)))         # confidence S-curve (τ=0.6, β=10)
alpha      = alpha_data * alpha_conf             # multiplicative; natural (0,1) bounds
```
- N=0 → alpha_data=1.0; N=K → 0.5; N→∞ → 0. C_g<τ favors user; C_g>τ favors global.

*Linear (legacy fallback):*
```
if N < 20:  alpha = 1.0
else:       alpha = clamp(0.5 + 0.3*C_g - 0.2*min(N/100, 1.0), 0.3, 1.0)
```

**Interpretation:** `alpha` = weight on the **global** head. New users (no head / low N)
→ trust global; as feedback grows and when global is unconfident → shift to the user head.

### 6.7 Vector store + RAG (`services/vector_store.py`, `rag_service.py`)
- **Embedding:** lazy-singleton `SentenceTransformer(all-MiniLM-L6-v2)` (384-dim).
- **`store_document(user_id, text, session_id, timestamp, emotion_label=None)`** →
  ensure collection → upsert point (uuid id, payload as §4.2).
- **`search_documents(user_id, query, top_k=5, query_embedding=None)`** → embed query
  (reuse if provided) → Qdrant `search(limit=top_k)` → `[{text, session_id, timestamp, score}]`.
- **`query_rag(user_id, query, top_k=5)`:**
  1. Embed query once (reused for cache + search).
  2. `check_cache` → return on semantic hit (skip LLM).
  3. `search_documents`; if empty → friendly "upload audio first" (not cached).
  4. Build context: `Document i (from session ...):\n<text>` joined by blank lines.
  5. Prompt template → Ollama `POST {base}/api/generate {model, prompt, stream:false}`, 120 s timeout.
  6. Format `sources` (text/session_id/timestamp/score); `store_in_cache`; return `{answer, sources, query}`.

  Prompt template:
  ```
  Based on the following context from audio transcriptions, answer the user's question.

  Context:
  {context}

  Question: {query}

  Answer:
  ```

### 6.8 Semantic query cache (`services/query_cache.py`)
- `check_cache` → load all `query_cache_keys:{user_id}` entries; among those with the same
  `top_k`, compute cosine of stored vs current query embedding; if best ≥ `0.95` → hit
  (returns cached answer/sources, with `_cache_hit/_similarity`). Redis down → `None` (miss).
- `store_in_cache` → `setex` (TTL 3600) + add to user zset; if size > 20, evict oldest by score.
- `invalidate_user_cache` → delete all of a user's entries (called after each upload).

### 6.9 Feedback service (`services/feedback_service.py`)
`submit_feedback(db, user_id, session_id, corrected_emotion, task_queue=None) -> dict`:
1. Validate `corrected_emotion ∈ EMOTION_LABELS` (else ValueError → 400).
2. Parse ObjectIds; fetch `AudioSession`; **ownership check** (`session.user_id == user_id`, else PermissionError → 403).
3. **Owner-safety gate:** if `session.personalization_trainable == False` → ValueError
   ("owner voice could not be confidently verified") → 400.
4. Fetch the session's `EmotionAnalysis.mfcc_features` (768-dim embedding); store a
   `UserFeedback` doc `{embedding, predicted_emotion, corrected_emotion}`.
5. Atomically `increment_feedback_count` on the user.
6. **Trigger rule** `should_trigger_training(N)`: `N ≥ 20 AND N % 10 == 0` → fires at 20,30,40,…
   If triggered and a `task_queue` is provided → enqueue async training + create `TrainingJob`.
7. Return `{status, feedback_count, training_triggered, message, training_job_id?}`.

### 6.10 Training — user head (`training/train_user_head.py`)
- **`train_user_head(user_id, force_retrain=False) -> metrics`:** load all of the user's
  `UserFeedback` (embedding + corrected-label-index), require ≥20; create fresh head
  (`force_retrain`) or load existing; `TensorDataset`, `batch=min(16,n)`,
  `Adam(lr=1e-3, weight_decay=1e-4)`, `CrossEntropyLoss`, **20 epochs**; save state_dict
  via the storage service. **Metrics:** `{final_loss, final_accuracy, num_samples,
  num_epochs, storage_mode, save_latency_ms}` — *note: no `model_path` key.*
- **Incremental vs initial:** initial = fresh weights; incremental = load existing weights
  and re-train on the **full** feedback history (mitigates catastrophic forgetting). Same
  hyperparameters.
- **`train_user_head_async(job_id, user_id, db)`:** background entry; transitions the
  `TrainingJob` `queued → running → completed/failed`, stores metrics or error.

### 6.11 Training — global head (`training/train_wav2vec2.py`, `prepare_dataset.py`)
- `prepare_dataset.py` scans **RAVDESS** (`modality 03` audio-only; emotion code → idx) and
  **CREMA-D** (`NEU/HAP/SAD/ANG/FEA/DIS`) into a unified manifest.
- `train(... save_as_global_head=True)`: extract+cache Wav2Vec2 embeddings (`.npz`),
  `StandardScaler`, stratified 70/15/15 split, `Adam(1e-3, wd 1e-4)`,
  `CrossEntropyLoss(weight=inverse-freq)`, ≤50 epochs, early-stop patience 10,
  `ReduceLROnPlateau`. Saves `global_emotion_head.pt` + `embedding_scaler.joblib` + test set.
- (There is no separate `train_global_head.py`; the global head is produced by this script.)

### 6.12 User-head storage (`services/user_head_storage.py`)
Pluggable backend via `USE_MONGODB_STORAGE` / `DUAL_SAVE_MODE`:
- **file** (default): `models/user_heads/<user_id>.pt`.
- **mongodb**: `user_models` doc with **gzip-compressed** `torch.save` blob (`Binary`),
  SHA256 checksum + sizes in `metadata`, compressed size capped (<~1 MB).
- **dual**: write both; read MongoDB-first with file fallback. `load_model` returns a
  `state_dict | None`; `exists` checks either backend.

### 6.13 Background jobs (`services/task_queue.py`, `training_job_tracker.py`)
- **`TaskQueue`** abstract; **`FastAPITaskQueue`** wraps FastAPI `BackgroundTasks`:
  `enqueue(func, *a, **kw)` → uuid `job_id`, schedules `func(job_id, *a, **kw)`, returns id
  immediately. (A `CeleryTaskQueue` placeholder exists for a distributed upgrade.)
- **Job tracker:** `create_training_job`, `update_job_status` (auto-stamps `started_at`/
  `completed_at`, validates status), `get_latest_job`, `get_job_by_id`.
  > Limitation: FastAPI BackgroundTasks are in-process, non-persistent — a restart loses
  > queued work. Swap in Celery/RQ for production durability.

### 6.14 Admin (`routers/admin.py`, `services/admin_service.py`)
Admin-only (`get_current_admin`): `GET /admin/user-models` (list user models with
`feedback_count`, `model_size_kb`, `last_trained`) and `DELETE /admin/user-models/cleanup`
(prune by `min_days_inactive` / `max_feedback_count` / explicit `user_ids`).

---

## 7. End-to-End Flows

### 7.1 Audio upload (synchronous)
```
Client → POST /api/audio/upload (JWT, multipart file)
  validate → save → preprocess(16k, VAD, normalize)
  → speaker diarization (cluster → match owner → owner audio + timeline + transcript)
  → Wav2Vec2 embedding(768)
  → dual-head classify (global + user, blended by α)
  → Whisper transcript (speaker-tagged)
  → persist AudioSession + EmotionAnalysis(embedding) to Mongo
  → upsert transcript embedding to Qdrant; invalidate cache
  → 200 {emotion, confidence, global/user preds, blend, transcription, speaker meta}
```

### 7.2 RAG query
```
Client → POST /api/rag/query (JWT, {query, top_k})
  embed query (MiniLM) → semantic cache check (≥0.95? return)
  → Qdrant top-k search → build context → Ollama /api/generate
  → 200 {answer, sources[], query}; store in cache
```

### 7.3 Feedback → personalization
```
(prerequisite) enroll speaker (3–5 clips) so uploads are owner-verified & trainable
Client → POST /api/feedback (JWT, {session_id, corrected_emotion})
  validate emotion → ownership + owner-safety gate
  → store UserFeedback(embedding) → increment feedback_count
  → if N≥20 and N%10==0: enqueue train_user_head_async + create TrainingJob
  → 200 {feedback_count, training_triggered, training_job_id?}
Background: train head (20 epochs) → save → job status=completed (metrics)
Client → GET /api/training-status/{user_id} to poll job
Next upload: user head now exists → predictions blend global+user via α
```

---

## 8. API Reference (18 routes)

| Method | Path | Auth | Body / Params | Returns |
|--------|------|------|---------------|---------|
| GET | `/` | – | – | banner |
| GET | `/health` | – | – | `{status:"healthy"}` |
| POST | `/api/auth/register` | – | `{email,password}` | `{message,user_id}` |
| POST | `/api/auth/login` | – | `{email,password}` | `{access_token,token_type,user_id}` |
| POST | `/api/audio/upload` | JWT | multipart `file` | upload response (below) |
| POST | `/api/audio/feedback` | JWT | `{session_id,corrected_emotion}` | feedback response *(legacy duplicate of `/api/feedback`)* |
| POST | `/api/rag/query` | JWT | `{query, top_k=5}` | `{answer, sources[], query}` |
| GET | `/api/dashboard/emotions/{user_id}` | JWT (self) | `?start_date&end_date&limit` | `{emotions[], total}` |
| GET | `/api/dashboard/stats/{user_id}` | JWT (self) | – | `{total_sessions, emotion_distribution, avg_confidence}` |
| POST | `/api/feedback` | JWT | `{session_id, corrected_emotion}` | `{status, feedback_count, training_triggered, training_job_id?, message}` |
| GET | `/api/training-status/{user_id}` | JWT (self/admin) | – | job status + metrics |
| POST | `/api/trigger-training/{user_id}` | JWT (admin) | – | `{status, training_job_id, message}` |
| POST | `/api/speaker/enroll/start` | JWT | – | enrollment start |
| POST | `/api/speaker/enroll/upload` | JWT | multipart `file` | samples collected |
| POST | `/api/speaker/enroll/complete` | JWT | – | `{enrolled, sample_count, ...}` |
| GET | `/api/speaker/enroll/status` | JWT | – | `{enrolled, enrollment_state, samples_collected, required, max}` |
| GET | `/admin/user-models` | JWT (admin) | – | `{total_count, models[]}` |
| DELETE | `/admin/user-models/cleanup` | JWT (admin) | `{min_days_inactive?, max_feedback_count?, user_ids?}` | `{deleted_count, deleted_user_ids[], total_space_freed_kb}` |

**Upload response shape:**
```json
{
  "session_id": "str", "emotion": "str", "confidence": 0.0,
  "global_emotion": "str", "global_confidence": 0.0,
  "user_emotion": "str|null", "user_confidence": "float|null",
  "blend_weight": 1.0, "alpha_data": "float|null", "alpha_conf": "float|null",
  "alpha_formula": "sigmoid|linear", "transcription": "str", "timestamp": "datetime",
  "owner_speech_ratio": "float|null", "owner_segments_count": "int|null",
  "other_segments_count": "int|null", "owner_detection_status": "verified|low_confidence|not_found|null",
  "speaker_timeline": [{"speaker_label":"OWNER|OTHER_1","start":0.0,"end":0.0,"owner_confidence":0.0}]
}
```

**Auth model for protected routes:** `Authorization: Bearer <jwt>`; dashboard routes also
enforce **path `user_id` == token user_id** (403 otherwise); admin routes require `is_admin`.

---

## 9. Key Constants & Hyperparameters (single table)

| Domain | Constant | Value |
|--------|----------|-------|
| Audio | sample rate / VAD top_db / max size | 16 kHz / 30 dB / 50 MB |
| Audio fallback | min duration / min RMS | 0.5 s / 0.001 |
| Wav2Vec2 | model / dim / min len | `wav2vec2-base` / 768 / 0.3 s (4800) |
| Speaker | enroll min/max | 3 / 5 |
| Speaker | owner thr / low-conf thr | 0.72 / 0.55 |
| Speaker | cluster metric/linkage/threshold | cosine / average / 0.35 |
| Speaker | min seg dur / max seg transcribe / min owner ratio | 0.35 s / 12 / 0.25 |
| Whisper | size / lang / beam / best_of / temp | small / en / 5 / 5 / 0 |
| Text embed | model / dim | MiniLM-L6-v2 / 384 |
| Qdrant | distance / default top_k | cosine / 5 |
| RAG | Ollama model / timeout | mistral / 120 s |
| Cache | TTL / max-per-user / sim threshold | 3600 s / 20 / 0.95 |
| Alpha | K / τ / β / formula | 50 / 0.6 / 10 / sigmoid |
| Train (user) | min feedback / interval / epochs / batch / lr / wd | 20 / 10 / 20 / min(16,n) / 1e-3 / 1e-4 |
| Train (global) | epochs / batch / split / patience | ≤50 / 64 / 70-15-15 / 10 |
| User head cache | LRU size | 100 |
| JWT | algo / expiry | HS256 / 24 h |

---

## 10. Reproduction Checklist (build a similar system)

1. **Scaffold** FastAPI with the `routers/services/models/schemas/db/middleware` layering;
   put all ML in services and one `feature_config.py` shared with training.
2. **Stand up infra:** MongoDB (metadata), Qdrant (per-user vectors, cosine, dim = your
   text-embedder), Ollama (a local model), Redis (optional cache). Make every connection
   fail-soft at startup.
3. **Auth:** bcrypt + JWT(HS256) dependency that re-verifies the user in DB; add an
   `is_admin` gate.
4. **Audio pipeline:** validate → save per-user → 16 kHz/VAD/normalize → speaker
   diarization (Agglomerative on cheap librosa embeddings) → owner matching by cosine vs an
   enrolled mean embedding → Wav2Vec2 768-dim → dual-head classify → Whisper transcript →
   persist + index. Keep an owner-audio fallback for short/quiet clips.
5. **Dual-head + Alpha Engine:** two `Linear(768,8)` heads; blend by
   `α = data(N) · conf(C_g)` (sigmoid) with global-only when no user head; expose alpha
   components in the response for observability.
6. **RAG:** embed query (MiniLM) → semantic cache → Qdrant top-k → context-stuffed prompt
   → local LLM; return answer + sources; invalidate cache on new uploads.
7. **Feedback loop:** owner-safety gate → store `(embedding, corrected_label)` → count →
   trigger training at `N≥20 && N%10==0` via a background task; track jobs in DB; lazy-load
   per-user heads with an LRU and reuse the stored embedding as training input.
8. **Offline training:** prepare RAVDESS+CREMA-D manifest → cache Wav2Vec2 embeddings →
   train a linear global head (weighted CE, early stopping) → ship `global_emotion_head.pt`.
9. **Storage strategy:** start with file-based per-user heads; add a gzip-in-Mongo backend
   with a dual-write migration mode when you need centralization.
10. **Admin/ops:** endpoints to list and prune stale user models.

---

## 11. Design Decisions, Trade-offs & Gotchas (worth knowing before copying)

- **Per-user model + LRU** keeps personalization cheap: heads are tiny `Linear(768,8)`
  (~tens of KB), loaded on demand, capped at 100 in memory.
- **Reusing the stored 768-dim embedding** as the training feature means feedback training
  never re-decodes audio — fast and consistent with inference.
- **Speaker diarization is classical, not neural** (librosa features + Agglomerative
  clustering). Cheap and CPU-only, but less robust than pyannote/embedding models — an
  obvious upgrade point. (Minor inconsistency: empty-audio fallback is 46-dim while real
  features are ~62-dim; align these if you reimplement.)
- **Whisper hallucinates on non-speech** (e.g. "Thanks for watching!" on tones) — expect
  this with synthetic/empty audio in tests.
- **No score threshold on vector search** — low-relevance chunks can enter the LLM context;
  add a min-score filter if precision matters.
- **Sync PyMongo under async FastAPI** — DB calls block the event loop; fine at small scale,
  but use Motor or a threadpool for throughput.
- **BackgroundTasks are in-process & non-persistent** — training is lost on restart; use a
  real queue (Celery/RQ) for durability and concurrency limits.
- **`USE_SIGMOID_ALPHA` flips behavior globally** and is the source of most "stale test"
  failures — keep tests and the active formula in lockstep (see test report).
- **Security defaults to fix before any real deployment:** change `JWT_SECRET_KEY`, scope
  CORS (currently `*` with credentials), and complete `.env`.
- **Two feedback endpoints** (`/api/audio/feedback` and `/api/feedback`) overlap — pick one.
- **Owner-safety prerequisite:** feedback/personalization only works after speaker
  enrollment makes a session `personalization_trainable=True`.

---

*Generated from a full read of the backend source (routers, services, models, schemas, db,
middleware, training scripts) plus a live end-to-end test run. For exact line references and
the verified runtime behavior, see [BACKEND_TEST_REPORT.md](BACKEND_TEST_REPORT.md).*
