"""
Apples-to-apples model comparison on the SAME speaker-independent test set.

Why this exists: the Stage 4 baseline's 62.4% was measured on a *random* split,
while the fine-tuned model is measured on a *speaker-independent* split. To compare
fairly, both must be scored on the identical test set. This tool does that, and prints
a confusion matrix so we can see WHICH emotions get confused (e.g. happy vs surprised).

Evaluates:
  - baseline:   frozen facebook/wav2vec2-base + scaler + models/global_emotion_head.pt
  - finetuned:  models/wav2vec2_finetuned/ + models/global_emotion_head_finetuned.pt (if present)

Run:
    python training/eval_compare.py
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.feature_config import EMBEDDING_DIM, EMOTION_LABELS, NUM_CLASSES  # noqa: E402
from training.finetune_wav2vec2 import build_speaker_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def confusion(preds, labels) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


def print_result(name, preds, labels):
    preds, labels = np.array(preds), np.array(labels)
    acc = float((preds == labels).mean())
    print("\n" + "=" * 64)
    print(f"  {name}  —  accuracy {acc:.4f}  (n={len(labels)})")
    print("=" * 64)
    print("  Per-class accuracy:")
    for idx, lbl in enumerate(EMOTION_LABELS):
        m = labels == idx
        a = round(float((preds[m] == idx).mean()), 3) if m.sum() else None
        print(f"    {lbl:<11} {a}")
    cm = confusion(preds, labels)
    short = [l[:4] for l in EMOTION_LABELS]
    print("\n  Confusion matrix (rows=true, cols=pred):")
    print("           " + " ".join(f"{s:>5}" for s in short))
    for i, lbl in enumerate(EMOTION_LABELS):
        print(f"    {lbl[:8]:<8} " + " ".join(f"{cm[i,j]:>5}" for j in range(NUM_CLASSES)))
    return acc


def eval_baseline(test_entries):
    import joblib
    import torch

    from api.services.wav2vec2_encoder import extract_wav2vec2_embedding

    head_path = MODELS_DIR / "global_emotion_head.pt"
    if not head_path.exists():
        print("baseline head missing — skipping")
        return None
    head = torch.nn.Linear(EMBEDDING_DIM, NUM_CLASSES)
    head.load_state_dict(torch.load(head_path, map_location="cpu"))
    head.eval()
    scaler = joblib.load(MODELS_DIR / "embedding_scaler.joblib") if (MODELS_DIR / "embedding_scaler.joblib").exists() else None

    preds, labels = [], []
    for i, e in enumerate(test_entries):
        emb = extract_wav2vec2_embedding(e["path"]).reshape(1, -1)
        if scaler is not None:
            emb = scaler.transform(emb)
        with torch.no_grad():
            logits = head(torch.from_numpy(emb.astype(np.float32)))
        preds.append(int(logits.argmax(1)))
        labels.append(e["label_idx"])
        if (i + 1) % 300 == 0:
            print(f"  baseline {i+1}/{len(test_entries)}", flush=True)
    return preds, labels


def eval_finetuned(test_entries):
    import torch
    from transformers import Wav2Vec2Model

    ft_dir = MODELS_DIR / "wav2vec2_finetuned"
    head_path = MODELS_DIR / "global_emotion_head_finetuned.pt"
    if not ft_dir.exists() or not head_path.exists():
        print("fine-tuned model not found — run finetune_wav2vec2.py first")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = Wav2Vec2Model.from_pretrained(ft_dir).to(device).eval()
    head = torch.nn.Linear(EMBEDDING_DIM, NUM_CLASSES).to(device)
    head.load_state_dict(torch.load(head_path, map_location=device))
    head.eval()

    preds, labels = [], []
    for i, e in enumerate(test_entries):
        y, _ = sf.read(e["path"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        x = torch.from_numpy(np.nan_to_num(y).astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pooled = backbone(x).last_hidden_state.mean(dim=1)
            logits = head(pooled)
        preds.append(int(logits.argmax(1)))
        labels.append(e["label_idx"])
        if (i + 1) % 300 == 0:
            print(f"  finetuned {i+1}/{len(test_entries)}", flush=True)
    return preds, labels


def main():
    splits = build_speaker_split()
    test = splits["test"]
    print(f"Speaker-independent test set: {len(test)} clips")

    base = eval_baseline(test)
    base_acc = print_result("BASELINE (frozen probe)", *base) if base else None

    ft = eval_finetuned(test)
    if ft:
        ft_acc = print_result("FINE-TUNED (end-to-end)", *ft)
        if base_acc is not None:
            delta = ft_acc - base_acc
            print(f"\n  >>> Fine-tuned vs baseline on SAME test set: "
                  f"{base_acc:.4f} -> {ft_acc:.4f}  ({'+' if delta>=0 else ''}{delta:.4f})")


if __name__ == "__main__":
    main()
