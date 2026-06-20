"""
Fit the speaker-feature StandardScaler.

The classical speaker embedding (MFCC/chroma/spectral stats) has wildly different
per-dimension scales, so raw L2-normalization makes every voice ~collinear (cosine≈1).
Standardizing each dimension first restores discriminability. This fits a StandardScaler
on raw speaker features from a sample of the prepared corpus and saves it; speaker_service
applies it before L2.

Run:
    python training/fit_speaker_scaler.py
    python training/fit_speaker_scaler.py --sample 1500
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.services.speaker_service import raw_speaker_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PREPARED = ROOT / "datasets" / "prepared"
OUT = ROOT / "models" / "speaker_feature_scaler.joblib"
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1500)
    args = ap.parse_args()

    wavs = sorted(PREPARED.rglob("*.wav"))
    if not wavs:
        print(f"No clips under {PREPARED} — run prepare_dataset.py first")
        sys.exit(1)

    rng = np.random.RandomState(SEED)
    if len(wavs) > args.sample:
        wavs = [wavs[i] for i in rng.choice(len(wavs), args.sample, replace=False)]

    feats = []
    for i, w in enumerate(wavs):
        try:
            y, sr = librosa.load(str(w), sr=16000, mono=True)
        except Exception:
            continue
        f = raw_speaker_features(y, sr)
        if f is not None:
            feats.append(f)
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(wavs)}")

    X = np.array(feats, dtype=np.float32)
    print(f"Fitting StandardScaler on {X.shape[0]} samples x {X.shape[1]} dims")

    import joblib
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
