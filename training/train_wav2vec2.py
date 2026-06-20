"""
Train the GLOBAL emotion head (Linear 768->8) on the prepared dataset.

Pipeline:
  1. Read datasets/prepared/manifest.json (stratified train/val/test).
  2. Extract Wav2Vec2 embeddings for every clip (cached to .npz so reruns are fast).
  3. Fit StandardScaler on TRAIN embeddings only.
  4. Train Linear(768,8) with class-weighted CrossEntropy (inverse frequency),
     Adam(lr=1e-3, wd=1e-4), <=50 epochs, early-stop patience 10, ReduceLROnPlateau.
  5. Evaluate on the held-out test split.
  6. Save models/global_emotion_head.pt + models/embedding_scaler.joblib + metrics.

Run:
    python training/train_wav2vec2.py
    python training/train_wav2vec2.py --manifest datasets/prepared/manifest.json --epochs 50
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.feature_config import (  # noqa: E402
    EMBEDDING_DIM,
    EMOTION_LABELS,
    NUM_CLASSES,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "datasets" / "embeddings_cache"


# ── Embedding extraction (cached) ──────────────────────────────────────────────

SAVE_EVERY = 500  # checkpoint embeddings every N files (resumable across runs)


def extract_split(name: str, entries: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract Wav2Vec2 embeddings for a split, incrementally and resumably.

    The single Bash invocation is capped at 10 min, and CPU extraction of ~9k
    clips takes longer — so progress is checkpointed to <name>.npz every
    SAVE_EVERY files with a `done` counter. Re-running resumes from `done`.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{name}.npz"

    n = len(entries)
    X = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    y = np.array([e["label_idx"] for e in entries], dtype=np.int64)
    done = 0

    if cache.exists():
        data = np.load(cache)
        cached_done = int(data["done"]) if "done" in data else len(data["X"])
        if len(data["X"]) == n and cached_done >= n:
            print(f"  [{name}] complete — {n} cached embeddings")
            return data["X"], data["y"]
        if len(data["X"]) == n:
            X = data["X"]
            y = data["y"]
            done = cached_done
            print(f"  [{name}] resuming from {done}/{n}")

    from api.services.wav2vec2_encoder import extract_wav2vec2_embedding

    t0 = time.time()
    for i in range(done, n):
        X[i] = extract_wav2vec2_embedding(entries[i]["path"])
        if (i + 1) % SAVE_EVERY == 0 or i + 1 == n:
            np.savez_compressed(cache, X=X, y=y, done=np.int64(i + 1))
            rate = (i + 1 - done) / max(1e-6, time.time() - t0)
            print(f"  [{name}] {i+1}/{n}  ({rate:.1f}/s, checkpointed)", flush=True)

    return X, y


# ── Training ───────────────────────────────────────────────────────────────────

def train(manifest_path: Path, epochs: int = 50, patience: int = 10) -> dict:
    import joblib
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    manifest = json.loads(manifest_path.read_text())
    splits = manifest["splits"]

    print("Extracting embeddings ...")
    Xtr, ytr = extract_split("train", splits["train"])
    Xva, yva = extract_split("val", splits["val"])
    Xte, yte = extract_split("test", splits["test"])

    # Scale on train only
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xva_s = scaler.transform(Xva).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    # Class weights (inverse frequency)
    counts = np.bincount(ytr, minlength=NUM_CLASSES).astype(np.float32)
    weights = np.where(counts > 0, counts.sum() / (NUM_CLASSES * counts), 0.0)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    Xtr_t = torch.from_numpy(Xtr_s)
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva_s)
    yva_t = torch.from_numpy(yva)

    model = nn.Linear(EMBEDDING_DIM, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    print("Training global head ...")
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva_t)
            val_loss = criterion(val_logits, yva_t).item()
            val_acc = (val_logits.argmax(1) == yva_t).float().mean().item()
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(f"  epoch {epoch:>2}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"lr={optimizer.param_groups[0]['lr']:.1e}  {'*' if bad_epochs==0 else ''}")
        if bad_epochs >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test eval
    model.eval()
    with torch.no_grad():
        te_logits = model(torch.from_numpy(Xte_s))
        te_pred = te_logits.argmax(1).numpy()
    test_acc = float((te_pred == yte).mean())
    per_class = {}
    for idx, lbl in enumerate(EMOTION_LABELS):
        mask = yte == idx
        per_class[lbl] = round(float((te_pred[mask] == idx).mean()), 3) if mask.sum() else None

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "global_emotion_head.pt")
    joblib.dump(scaler, MODELS_DIR / "embedding_scaler.joblib")

    metrics = {
        "train_samples": int(len(ytr)),
        "val_samples": int(len(yva)),
        "test_samples": int(len(yte)),
        "best_val_loss": round(best_val, 4),
        "test_accuracy": round(test_acc, 4),
        "per_class_test_accuracy": per_class,
        "class_weights": {EMOTION_LABELS[i]: round(float(w), 3) for i, w in enumerate(weights)},
    }
    (MODELS_DIR / "global_head_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="datasets/prepared/manifest.json")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    metrics = train(Path(args.manifest), epochs=args.epochs, patience=args.patience)
    print("\n" + "=" * 56)
    print("  GLOBAL HEAD TRAINING COMPLETE")
    print("=" * 56)
    print(f"  Test accuracy : {metrics['test_accuracy']}")
    print("  Per-class test accuracy:")
    for lbl, acc in metrics["per_class_test_accuracy"].items():
        print(f"    {lbl:<11} {acc}")
    print(f"\n  Saved: models/global_emotion_head.pt, models/embedding_scaler.joblib")
    print("=" * 56)


if __name__ == "__main__":
    main()
