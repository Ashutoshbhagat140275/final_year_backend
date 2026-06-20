"""
GPU fine-tuning of the wav2vec2 backbone for speech emotion (the real accuracy lever).

Unlike train_wav2vec2.py (frozen backbone + linear probe on cached embeddings), this
trains the wav2vec2 transformer END-TO-END together with a Linear(768,8) head, so the
"ears" themselves adapt to emotion. Expected: well above the 62.4% frozen-probe baseline.

Design choices for a 6GB laptop GPU (RTX 3050):
  - fp16 autocast + GradScaler
  - freeze the conv feature-extractor + the bottom FREEZE_BOTTOM_LAYERS transformer layers
  - small batch + gradient accumulation
  - SPEAKER-INDEPENDENT split (actors disjoint across train/val/test) for an HONEST number
  - per-epoch checkpoint + resume (single Bash run is capped at 10 min)

Artifacts (only consumed if it beats baseline, after review):
  - models/wav2vec2_finetuned/            (HF save_pretrained of the fine-tuned backbone)
  - models/global_emotion_head_finetuned.pt   (the Linear(768,8) head)
  - models/finetune_metrics.json
Resume state: models/_finetune_ckpt.pt

Run (repeat until it prints DONE; it resumes each time):
    python training/finetune_wav2vec2.py
    python training/finetune_wav2vec2.py --epochs 12 --batch 4 --accum 8 --max-seconds 5
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.feature_config import EMBEDDING_DIM, EMOTION_LABELS, NUM_CLASSES, SAMPLE_RATE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PREPARED = ROOT / "datasets" / "prepared"
CKPT = MODELS_DIR / "_finetune_ckpt.pt"
FREEZE_BOTTOM_LAYERS = 6
SEED = 42


# ── Speaker-independent split ──────────────────────────────────────────────────

def actor_id(dataset: str, stem: str) -> str:
    if dataset == "ravdess":      # 03-01-04-01-01-01-09 → actor = last field
        return f"ravdess_{stem.split('-')[-1]}"
    return f"cremad_{stem.split('_')[0]}"  # 1001_DFA_ANG_XX → actor = first field


def build_speaker_split(ratios=(0.70, 0.15, 0.15)) -> dict:
    """Group cleaned clips by actor, then assign whole actors to splits (no leakage)."""
    items = []
    for dataset in ("ravdess", "cremad"):
        ddir = PREPARED / dataset
        if not ddir.exists():
            continue
        for label_dir in ddir.iterdir():
            if not label_dir.is_dir() or label_dir.name not in EMOTION_LABELS:
                continue
            idx = EMOTION_LABELS.index(label_dir.name)
            for wav in label_dir.glob("*.wav"):
                items.append({"path": str(wav), "label_idx": idx,
                              "actor": actor_id(dataset, wav.stem)})

    actors = sorted({it["actor"] for it in items})
    rng = np.random.RandomState(SEED)
    rng.shuffle(actors)
    n = len(actors)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    split_of = {}
    for a in actors[:n_tr]:
        split_of[a] = "train"
    for a in actors[n_tr:n_tr + n_va]:
        split_of[a] = "val"
    for a in actors[n_tr + n_va:]:
        split_of[a] = "test"

    out = {"train": [], "val": [], "test": []}
    for it in items:
        out[split_of[it["actor"]]].append(it)
    return out


# ── Dataset ────────────────────────────────────────────────────────────────────

def make_dataset(entries, max_samples):
    import torch

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(entries)

        def __getitem__(self, i):
            e = entries[i]
            y, _ = sf.read(e["path"], dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if len(y) > max_samples:
                y = y[:max_samples]
            return torch.from_numpy(np.nan_to_num(y)), e["label_idx"]

    return DS()


def collate(batch):
    import torch

    ys = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    maxlen = max(len(y) for y in ys)
    X = torch.zeros(len(ys), maxlen, dtype=torch.float32)
    mask = torch.zeros(len(ys), maxlen, dtype=torch.long)
    for i, y in enumerate(ys):
        X[i, :len(y)] = y
        mask[i, :len(y)] = 1
    return X, mask, labels


# ── Model: wav2vec2 backbone + mean-pool + Linear(768,8) ───────────────────────

def build_model():
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model

    from api.feature_config import WAV2VEC2_MODEL_NAME

    class EmotionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_NAME)
            self.backbone.feature_extractor._freeze_parameters()  # freeze conv front-end
            # freeze bottom transformer layers
            for layer in self.backbone.encoder.layers[:FREEZE_BOTTOM_LAYERS]:
                for p in layer.parameters():
                    p.requires_grad = False
            self.head = nn.Linear(EMBEDDING_DIM, NUM_CLASSES)

        def forward(self, x, mask):
            out = self.backbone(x, attention_mask=mask).last_hidden_state  # (B,T,768)
            # mean-pool over valid frames (approx: simple mean — wav2vec2 downsamples,
            # but mean over all output frames matches inference encoder)
            pooled = out.mean(dim=1)
            return self.head(pooled), pooled

    return EmotionNet()


# ── Train ──────────────────────────────────────────────────────────────────────

def run(args):
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available — fine-tuning on CPU will be very slow.")

    splits = build_speaker_split()
    print(f"Speaker-independent split: train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])}")

    max_samples = int(args.max_seconds * SAMPLE_RATE)
    train_ds = make_dataset(splits["train"], max_samples)
    val_ds = make_dataset(splits["val"], max_samples)
    test_ds = make_dataset(splits["test"], max_samples)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = build_model().to(device)

    # class weights (inverse freq) from train split
    counts = np.bincount([e["label_idx"] for e in splits["train"]], minlength=NUM_CLASSES).astype(np.float32)
    weights = np.where(counts > 0, counts.sum() / (NUM_CLASSES * counts), 0.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    start_epoch = 1
    best_val = float("inf")
    bad = 0
    if CKPT.exists():
        ck = torch.load(CKPT, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        bad = ck["bad"]
        print(f"Resumed from epoch {ck['epoch']} (best_val={best_val:.4f})")

    def evaluate(dl):
        model.eval()
        tot_loss, correct, total = 0.0, 0, 0
        preds, labels = [], []
        with torch.no_grad():
            for X, mask, y in dl:
                X, mask, y = X.to(device), mask.to(device), y.to(device)
                with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                    logits, _ = model(X, mask)
                    loss = criterion(logits, y)
                tot_loss += loss.item() * len(y)
                p = logits.argmax(1)
                correct += (p == y).sum().item()
                total += len(y)
                preds += p.cpu().tolist()
                labels += y.cpu().tolist()
        return tot_loss / max(1, total), correct / max(1, total), np.array(preds), np.array(labels)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        optimizer.zero_grad()
        for step, (X, mask, y) in enumerate(train_dl):
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits, _ = model(X, mask)
                loss = criterion(logits, y) / args.accum
            scaler.scale(loss).backward()
            if (step + 1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        val_loss, val_acc, _, _ = evaluate(val_dl)
        improved = val_loss < best_val - 1e-4
        if improved:
            best_val = val_loss
            bad = 0
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_val": best_val,
                    "bad": bad}, CKPT)
        print(f"  epoch {epoch:>2}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"({time.time()-t0:.0f}s)  {'*' if improved else ''}", flush=True)
        if bad >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break

    # Final test eval
    test_loss, test_acc, preds, labels = evaluate(test_dl)
    per_class = {}
    for idx, lbl in enumerate(EMOTION_LABELS):
        m = labels == idx
        per_class[lbl] = round(float((preds[m] == idx).mean()), 3) if m.sum() else None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(MODELS_DIR / "wav2vec2_finetuned")
    import torch as _t
    _t.save(model.head.state_dict(), MODELS_DIR / "global_emotion_head_finetuned.pt")
    metrics = {
        "test_accuracy": round(test_acc, 4),
        "baseline_accuracy": 0.6241,
        "per_class_test_accuracy": per_class,
        "split": "speaker-independent",
        "frozen_bottom_layers": FREEZE_BOTTOM_LAYERS,
    }
    (MODELS_DIR / "finetune_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n" + "=" * 56)
    print("  FINE-TUNING DONE")
    print("=" * 56)
    print(f"  Fine-tuned test acc : {test_acc:.4f}   (baseline 0.6241)")
    for lbl, a in per_class.items():
        print(f"    {lbl:<11} {a}")
    print("  Saved: models/wav2vec2_finetuned/, models/global_emotion_head_finetuned.pt")
    print("=" * 56)
    CKPT.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seconds", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
