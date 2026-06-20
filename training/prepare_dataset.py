"""
Prepare (clean + standardize) emotion datasets into a unified manifest.

"Make the dataset better": every source file is cleaned with the SAME pipeline
used at inference time (api/services/audio_processor) so training features match
production features exactly:
    1. load mono @ 16 kHz
    2. VAD-trim silence (librosa.effects.split, top_db=30), concat voiced spans
    3. peak-normalize (rescues quiet recordings instead of dropping them)
Files with < MIN_VOICED_SECONDS of voiced audio after trimming are dropped as
unusable. Cleaned WAVs are written to <out>/<dataset>/<label>/<name>.wav and a
stratified 70/15/15 train/val/test manifest is emitted.

Run:
    python training/prepare_dataset.py --ravdess datasets/ravdess --out datasets/prepared
    python training/prepare_dataset.py --ravdess datasets/ravdess --cremad datasets/cremad/AudioWAV --out datasets/prepared
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.feature_config import (  # noqa: E402
    CREMAD_CODE_TO_IDX,
    EMOTION_LABELS,
    RAVDESS_CODE_TO_IDX,
    SAMPLE_RATE,
)
from training.assess_dataset import cremad_label, ravdess_label  # noqa: E402

MIN_VOICED_SECONDS = 0.5
VAD_TOP_DB = 30
SEED = 42


# ── Cleaning (mirrors api/services/audio_processor._preprocess) ────────────────

def clean_waveform(path: Path) -> np.ndarray | None:
    """Load → VAD-trim → peak-normalize. Returns None if unusable/empty."""
    try:
        y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        return None

    if y is None or len(y) == 0:
        return None

    intervals = librosa.effects.split(y, top_db=VAD_TOP_DB)
    if len(intervals) > 0:
        voiced = np.concatenate([y[s:e] for s, e in intervals])
        if len(voiced) > 0:
            y = voiced

    if len(y) < int(MIN_VOICED_SECONDS * SAMPLE_RATE):
        return None

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    return y.astype(np.float32)


# ── Manifest building ──────────────────────────────────────────────────────────

def collect(name: str, root: Path, label_fn, out_dir: Path) -> tuple[list[dict], dict]:
    wavs = sorted(root.rglob("*.wav"))
    entries: list[dict] = []
    stats = {"scanned": len(wavs), "written": 0, "dropped_empty": 0, "dropped_unlabeled": 0}

    for path in wavs:
        idx = label_fn(path)
        if idx is None:
            stats["dropped_unlabeled"] += 1
            continue

        y = clean_waveform(path)
        if y is None:
            stats["dropped_empty"] += 1
            continue

        label = EMOTION_LABELS[idx]
        dest_dir = out_dir / name / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{path.stem}.wav"
        sf.write(str(dest), y, SAMPLE_RATE)

        entries.append({
            "path": str(dest).replace("\\", "/"),
            "label": label,
            "label_idx": idx,
            "dataset": name,
            "duration": round(len(y) / SAMPLE_RATE, 3),
        })
        stats["written"] += 1

    return entries, stats


def stratified_split(entries: list[dict], ratios=(0.70, 0.15, 0.15)) -> dict:
    by_class: dict[int, list[dict]] = defaultdict(list)
    for e in entries:
        by_class[e["label_idx"]].append(e)

    rng = random.Random(SEED)
    train, val, test = [], [], []
    for idx, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def dist(entries: list[dict]) -> dict:
    c = Counter(e["label"] for e in entries)
    return {lbl: c.get(lbl, 0) for lbl in EMOTION_LABELS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess", type=str, help="Path to RAVDESS root")
    ap.add_argument("--cremad", type=str, help="Path to CREMA-D AudioWAV dir")
    ap.add_argument("--out", type=str, default="datasets/prepared", help="Output dir for cleaned data + manifest")
    args = ap.parse_args()

    if not args.ravdess and not args.cremad:
        ap.error("Provide at least one of --ravdess or --cremad")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_entries: list[dict] = []
    all_stats = {}

    if args.ravdess:
        print("Cleaning RAVDESS ...")
        entries, stats = collect("ravdess", Path(args.ravdess), ravdess_label, out_dir)
        all_entries += entries
        all_stats["ravdess"] = stats
        print(f"  RAVDESS: {stats}")

    if args.cremad:
        print("Cleaning CREMA-D ...")
        entries, stats = collect("cremad", Path(args.cremad), cremad_label, out_dir)
        all_entries += entries
        all_stats["cremad"] = stats
        print(f"  CREMA-D: {stats}")

    splits = stratified_split(all_entries)
    manifest = {
        "emotion_labels": EMOTION_LABELS,
        "sample_rate": SAMPLE_RATE,
        "counts": {
            "total": len(all_entries),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "distribution": {
            "total": dist(all_entries),
            "train": dist(splits["train"]),
            "val": dist(splits["val"]),
            "test": dist(splits["test"]),
        },
        "prepare_stats": all_stats,
        "splits": splits,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n" + "=" * 60)
    print("  PREPARED DATASET MANIFEST")
    print("=" * 60)
    print(f"  Output dir : {out_dir}")
    print(f"  Total clean: {len(all_entries)}")
    print(f"  Split      : train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print("  Class distribution (total):")
    for lbl, n in manifest["distribution"]["total"].items():
        print(f"    {lbl:<11} {n:>5}")
    print(f"\n  Manifest written to {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
