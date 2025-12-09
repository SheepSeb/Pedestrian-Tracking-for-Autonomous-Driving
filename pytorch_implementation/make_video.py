from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def make_video(result_dir: Path, mot_dir: Path, output_dir: Path, convert_h264: bool) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        os.fspath(ROOT / "generate_videos.py"),
        "--mot_dir",
        os.fspath(mot_dir),
        "--result_dir",
        os.fspath(result_dir),
        "--output_dir",
        os.fspath(output_dir),
        "--convert_h264",
        str(convert_h264),
    ]
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render videos using PyTorch tracking results.")
    parser.add_argument("--result_dir", type=Path, default=ROOT / "tracks_torch")
    parser.add_argument("--mot_dir", type=Path, default=ROOT / "MOT16" / "train")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "videos_torch")
    parser.add_argument("--convert_h264", action="store_true", help="Also create .mp4 outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_video(args.result_dir, args.mot_dir, args.output_dir, args.convert_h264)


if __name__ == "__main__":
    main()
