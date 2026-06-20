"""
Download the CREMA-D AudioWAV corpus (7442 files) from GitHub.

The files are stored via git-lfs, so the real audio is served from the LFS
media endpoint (media.githubusercontent.com/media/...), NOT raw.githubusercontent
(which returns a 130-byte pointer). This script enumerates the file list via the
git-tree API, then downloads in parallel. It is resumable — already-present,
non-empty files are skipped.

Run:
    python training/download_cremad.py --out datasets/cremad/AudioWAV
    python training/download_cremad.py --out datasets/cremad/AudioWAV --workers 16
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TREE_API = "https://api.github.com/repos/CheyneyComputerScience/CREMA-D/git/trees/master?recursive=1"
MEDIA_BASE = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/"
MIN_VALID_BYTES = 1000  # LFS pointers are ~130 bytes; real clips are tens of KB


def list_files() -> list[str]:
    r = requests.get(TREE_API, timeout=60)
    r.raise_for_status()
    tree = r.json().get("tree", [])
    return [t["path"] for t in tree
            if t["path"].startswith("AudioWAV/") and t["path"].endswith(".wav")]


def download_one(path: str, out_dir: Path) -> tuple[str, str]:
    """Returns (path, status) where status in {ok, skip, error:...}."""
    name = path.split("/", 1)[1]  # strip "AudioWAV/"
    dest = out_dir / name
    if dest.exists() and dest.stat().st_size >= MIN_VALID_BYTES:
        return (name, "skip")
    try:
        r = requests.get(MEDIA_BASE + path, timeout=60)
        r.raise_for_status()
        if len(r.content) < MIN_VALID_BYTES:
            return (name, f"error:too-small({len(r.content)}b)")
        dest.write_bytes(r.content)
        return (name, "ok")
    except Exception as exc:
        return (name, f"error:{exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="datasets/cremad/AudioWAV")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Enumerating CREMA-D file list ...")
    files = list_files()
    print(f"  {len(files)} files to fetch (resumable; existing skipped)")

    ok = skip = err = 0
    errors = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, p, out_dir): p for p in files}
        done = 0
        for fut in as_completed(futures):
            name, status = fut.result()
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                errors.append((name, status))
            if done % 250 == 0 or done == len(files):
                rate = done / max(1e-6, time.time() - t0)
                print(f"  {done}/{len(files)}  ok={ok} skip={skip} err={err}  ({rate:.0f}/s)")

    print(f"\nDone in {time.time()-t0:.0f}s — ok={ok} skip={skip} err={err}")
    if errors:
        print("First errors:")
        for name, status in errors[:10]:
            print(f"  {name}: {status}")
        sys.exit(1 if err > len(files) * 0.05 else 0)  # fail only if >5% errored


if __name__ == "__main__":
    main()
