from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_tracker(sequence: str, detections_dir: Path, tracks_dir: Path, mot_dir: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    sequence_dir = mot_dir / sequence
    detection_file = detections_dir / f"{sequence}.npy"
    output_file = tracks_dir / f"{sequence}.txt"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        os.fspath(ROOT / "deep_sort_app.py"),
        "--sequence_dir",
        os.fspath(sequence_dir),
        "--detection_file",
        os.fspath(detection_file),
        "--output_file",
        os.fspath(output_file),
        "--display",
        "False",
    ]
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tracker using PyTorch detections.")
    parser.add_argument("--sequence", required=True, help="Sequence name, e.g., MOT16-02")
    parser.add_argument("--detections_dir", type=Path, default=ROOT / "resources" / "detections_torch")
    parser.add_argument("--tracks_dir", type=Path, default=ROOT / "tracks_torch")
    parser.add_argument("--mot_dir", type=Path, default=ROOT / "MOT16" / "train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tracker(args.sequence, args.detections_dir, args.tracks_dir, args.mot_dir)


if __name__ == "__main__":
    main()
