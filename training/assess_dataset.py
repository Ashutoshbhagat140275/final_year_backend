"""
Dataset quality assessment for emotion audio datasets (RAVDESS + CREMA-D).

Answers "is the dataset good or not?" by reporting, per dataset:
  - file count and per-emotion class balance
  - duration distribution (min/max/mean/median/std)
  - sample-rate distribution and channel layout
  - corrupt / unreadable files
  - silent or near-silent files (low mean RMS)
  - clipped files (peak at/over full scale)
  - very-short files (< MIN_USEFUL_SECONDS)

Run:
    python training/assess_dataset.py --ravdess datasets/ravdess
    python training/assess_dataset.py --cremad datasets/cremad/AudioWAV
    python training/assess_dataset.py --ravdess datasets/ravdess --cremad datasets/cremad/AudioWAV --json report.json
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

import sys
# Windows consoles default to cp1252 — force UTF-8 so report glyphs print cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.feature_config import (  # noqa: E402
    CREMAD_CODE_TO_IDX,
    EMOTION_LABELS,
    RAVDESS_CODE_TO_IDX,
)

# Quality thresholds
MIN_USEFUL_SECONDS = 0.5
SILENCE_RMS_THRESHOLD = 0.001
CLIPPING_PEAK_THRESHOLD = 0.999


# ── Label parsing ──────────────────────────────────────────────────────────────

def ravdess_label(path: Path) -> int | None:
    """RAVDESS filename: 03-01-EM-..  → emotion is field index 2 (1-based code)."""
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    try:
        code = int(parts[2])
    except ValueError:
        return None
    return RAVDESS_CODE_TO_IDX.get(code)


def cremad_label(path: Path) -> int | None:
    """CREMA-D filename: 1001_DFA_ANG_XX.wav → emotion is field index 2 (3-letter)."""
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    return CREMAD_CODE_TO_IDX.get(parts[2])


# ── Per-file probe ─────────────────────────────────────────────────────────────

def probe_file(path: Path) -> dict:
    """Return per-file metrics, or {'error': ...} if unreadable."""
    try:
        info = sf.info(str(path))
        y, sr = sf.read(str(path), dtype="float32")
    except Exception as exc:
        return {"error": str(exc)}

    if y.ndim > 1:
        channels = y.shape[1]
        y = y.mean(axis=1)
    else:
        channels = 1

    duration = len(y) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    peak = float(np.max(np.abs(y))) if len(y) else 0.0

    return {
        "duration": duration,
        "sample_rate": sr,
        "channels": channels,
        "rms": rms,
        "peak": peak,
        "format": info.format,
        "subtype": info.subtype,
    }


# ── Dataset-level assessment ───────────────────────────────────────────────────

def assess(name: str, root: Path, label_fn) -> dict:
    wavs = sorted(root.rglob("*.wav"))
    report = {
        "dataset": name,
        "root": str(root),
        "total_files": len(wavs),
        "readable": 0,
        "corrupt": [],
        "unlabeled": [],
        "class_counts": Counter(),
        "durations": [],
        "sample_rates": Counter(),
        "channels": Counter(),
        "silent_files": [],
        "clipped_files": [],
        "short_files": [],
    }

    for path in wavs:
        idx = label_fn(path)
        if idx is None:
            report["unlabeled"].append(path.name)
        else:
            report["class_counts"][EMOTION_LABELS[idx]] += 1

        m = probe_file(path)
        if "error" in m:
            report["corrupt"].append({"file": path.name, "error": m["error"]})
            continue

        report["readable"] += 1
        report["durations"].append(m["duration"])
        report["sample_rates"][m["sample_rate"]] += 1
        report["channels"][m["channels"]] += 1

        if m["rms"] < SILENCE_RMS_THRESHOLD:
            report["silent_files"].append(path.name)
        if m["peak"] >= CLIPPING_PEAK_THRESHOLD:
            report["clipped_files"].append(path.name)
        if m["duration"] < MIN_USEFUL_SECONDS:
            report["short_files"].append(path.name)

    return report


def summarize(report: dict) -> dict:
    d = report["durations"]
    dur_stats = {}
    if d:
        dur_stats = {
            "min": round(min(d), 3),
            "max": round(max(d), 3),
            "mean": round(statistics.mean(d), 3),
            "median": round(statistics.median(d), 3),
            "std": round(statistics.pstdev(d), 3),
        }

    counts = report["class_counts"]
    present = {lbl: counts.get(lbl, 0) for lbl in EMOTION_LABELS}
    nonzero = [v for v in present.values() if v > 0]
    balance_ratio = (min(nonzero) / max(nonzero)) if nonzero else 0.0

    return {
        "dataset": report["dataset"],
        "total_files": report["total_files"],
        "readable": report["readable"],
        "corrupt_count": len(report["corrupt"]),
        "unlabeled_count": len(report["unlabeled"]),
        "classes_present": sum(1 for v in present.values() if v > 0),
        "classes_total": len(EMOTION_LABELS),
        "class_distribution": present,
        "class_balance_ratio": round(balance_ratio, 3),
        "duration_seconds": dur_stats,
        "sample_rates": dict(report["sample_rates"]),
        "channels": dict(report["channels"]),
        "silent_count": len(report["silent_files"]),
        "clipped_count": len(report["clipped_files"]),
        "short_count": len(report["short_files"]),
    }


def verdict(summary: dict) -> list[str]:
    """Plain-language 'is it good' findings."""
    notes = []
    s = summary

    if s["corrupt_count"] == 0:
        notes.append(f"PASS: all {s['readable']} files are readable (0 corrupt).")
    else:
        notes.append(f"WARN: {s['corrupt_count']} corrupt/unreadable files.")

    missing = [lbl for lbl, v in s["class_distribution"].items() if v == 0]
    if missing:
        notes.append(f"NOTE: {s['classes_present']}/{s['classes_total']} classes present. "
                     f"Missing: {missing}.")
    else:
        notes.append(f"PASS: all {s['classes_total']} emotion classes present.")

    br = s["class_balance_ratio"]
    if br >= 0.8:
        notes.append(f"PASS: class balance ratio {br} (well balanced).")
    elif br >= 0.4:
        notes.append(f"NOTE: class balance ratio {br} (mild imbalance — use weighted loss).")
    else:
        notes.append(f"WARN: class balance ratio {br} (strong imbalance — weight or resample).")

    srs = s["sample_rates"]
    if len(srs) == 1:
        notes.append(f"NOTE: uniform sample rate {list(srs)[0]} Hz (will resample to 16k).")
    else:
        notes.append(f"NOTE: mixed sample rates {srs} (all resampled to 16k at train/infer).")

    if s["silent_count"]:
        notes.append(f"WARN: {s['silent_count']} silent/near-silent files — drop these.")
    else:
        notes.append("PASS: no silent files.")

    if s["short_count"]:
        notes.append(f"NOTE: {s['short_count']} files < {MIN_USEFUL_SECONDS}s — drop or pad.")
    else:
        notes.append(f"PASS: no files under {MIN_USEFUL_SECONDS}s.")

    if s["clipped_count"]:
        notes.append(f"NOTE: {s['clipped_count']} clipped files (peak≈1.0) — acceptable after normalize.")

    return notes


def print_report(summary: dict, notes: list[str]) -> None:
    print("\n" + "=" * 64)
    print(f"  DATASET QUALITY REPORT — {summary['dataset']}")
    print("=" * 64)
    print(f"  Total files     : {summary['total_files']}")
    print(f"  Readable        : {summary['readable']}")
    print(f"  Corrupt         : {summary['corrupt_count']}")
    print(f"  Unlabeled       : {summary['unlabeled_count']}")
    print(f"  Classes present : {summary['classes_present']}/{summary['classes_total']}")
    print(f"  Balance ratio   : {summary['class_balance_ratio']}")
    print(f"  Duration (s)    : {summary['duration_seconds']}")
    print(f"  Sample rates    : {summary['sample_rates']}")
    print(f"  Channels        : {summary['channels']}")
    print(f"  Silent / Short / Clipped : "
          f"{summary['silent_count']} / {summary['short_count']} / {summary['clipped_count']}")
    print("\n  Class distribution:")
    for lbl, n in summary["class_distribution"].items():
        bar = "#" * int(n / max(1, summary["total_files"]) * 40)
        print(f"    {lbl:<11} {n:>5}  {bar}")
    print("\n  Verdict:")
    for note in notes:
        print(f"    - {note}")
    print("=" * 64 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess", type=str, help="Path to RAVDESS root (contains Actor_* dirs)")
    ap.add_argument("--cremad", type=str, help="Path to CREMA-D AudioWAV dir")
    ap.add_argument("--json", type=str, help="Optional path to write full JSON report")
    args = ap.parse_args()

    all_summaries = {}

    if args.ravdess:
        rep = assess("RAVDESS", Path(args.ravdess), ravdess_label)
        summ = summarize(rep)
        notes = verdict(summ)
        print_report(summ, notes)
        all_summaries["ravdess"] = {"summary": summ, "verdict": notes,
                                    "corrupt": rep["corrupt"][:20]}

    if args.cremad:
        rep = assess("CREMA-D", Path(args.cremad), cremad_label)
        summ = summarize(rep)
        notes = verdict(summ)
        print_report(summ, notes)
        all_summaries["cremad"] = {"summary": summ, "verdict": notes,
                                   "corrupt": rep["corrupt"][:20]}

    if args.json and all_summaries:
        Path(args.json).write_text(json.dumps(all_summaries, indent=2))
        print(f"Full JSON report written to {args.json}")

    if not args.ravdess and not args.cremad:
        ap.error("Provide at least one of --ravdess or --cremad")


if __name__ == "__main__":
    main()
