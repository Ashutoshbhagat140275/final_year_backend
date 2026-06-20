"""
Train a per-user emotion head from the user's feedback corrections.

Reuses the stored 768-dim Wav2Vec2 embeddings (no audio re-decoding). Initial run
= fresh weights; incremental = load existing weights and retrain on the FULL feedback
history (mitigates catastrophic forgetting).
"""
import logging

import numpy as np

from api.feature_config import (
    EMOTION_LABELS,
    MIN_FEEDBACK_FOR_TRAINING,
    TRAINING_BATCH_SIZE,
    TRAINING_EPOCHS,
    TRAINING_LEARNING_RATE,
    TRAINING_WEIGHT_DECAY,
)

logger = logging.getLogger(__name__)
_EMOTION_TO_IDX = {l: i for i, l in enumerate(EMOTION_LABELS)}


def train_user_head(user_id: str, force_retrain: bool = False) -> dict:
    import time

    import torch
    import torch.nn as nn

    from api.db.mongodb import get_database
    from api.models.user_feedback import UserFeedback
    from api.services.user_emotion_head import build_user_head
    from api.services.user_head_storage import load_model, save_model

    db = get_database()
    if db is None:
        raise RuntimeError("Database unavailable")

    docs = list(UserFeedback.get_collection(db).find({"user_id": user_id}))
    if len(docs) < MIN_FEEDBACK_FOR_TRAINING:
        raise ValueError(f"Need >= {MIN_FEEDBACK_FOR_TRAINING} feedback (have {len(docs)})")

    X = np.array([d["embedding"] for d in docs], dtype=np.float32)
    y = np.array([_EMOTION_TO_IDX[d["corrected_emotion"]] for d in docs], dtype=np.int64)

    head = build_user_head()
    if not force_retrain:
        existing = load_model(user_id)
        if existing is not None:
            head.load_state_dict(existing)  # incremental: warm-start

    optimizer = torch.optim.Adam(head.parameters(), lr=TRAINING_LEARNING_RATE,
                                 weight_decay=TRAINING_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = torch.utils.data.DataLoader(ds, batch_size=min(TRAINING_BATCH_SIZE, len(docs)), shuffle=True)

    head.train()
    final_loss = 0.0
    for _ in range(TRAINING_EPOCHS):
        epoch_loss = 0.0
        for xb, yb in dl:
            optimizer.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
        final_loss = epoch_loss / len(docs)

    head.eval()
    with torch.no_grad():
        preds = head(torch.from_numpy(X)).argmax(1).numpy()
    accuracy = float((preds == y).mean())

    t0 = time.time()
    store = save_model(user_id, head.state_dict())
    save_latency_ms = round((time.time() - t0) * 1000, 1)

    return {
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(accuracy, 4),
        "num_samples": len(docs),
        "num_epochs": TRAINING_EPOCHS,
        "storage_mode": store.get("storage_mode"),
        "save_latency_ms": save_latency_ms,
    }


def train_user_head_async(job_id: str, user_id: str, db=None) -> None:
    """Background entry: queued → running → completed/failed, recording metrics/error."""
    from api.db.mongodb import get_database
    from api.services.training_job_tracker import update_job_status

    db = db or get_database()
    try:
        update_job_status(db, job_id, "running")
        metrics = train_user_head(user_id)
        update_job_status(db, job_id, "completed", metrics=metrics)
        logger.info("User head trained for %s: %s", user_id, metrics)
    except Exception as exc:
        logger.warning("User head training failed for %s: %s", user_id, exc)
        update_job_status(db, job_id, "failed", error_message=str(exc))
