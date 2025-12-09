from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_tf_detections(mot_dir: Path, output_dir: Path, model: Path, batch_size: int) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [
        os.fspath(ROOT / "tools" / "generate_detections.py"),
        "--model",
        os.fspath(model),
        "--mot_dir",
        os.fspath(mot_dir),
        "--output_dir",
        os.fspath(output_dir),
    ]
    if batch_size:
        cmd.extend(["--batch_size", str(batch_size)])
    subprocess.run(["python"] + cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detections using the TensorFlow appearance model.")
    parser.add_argument("--mot_dir", type=Path, default=ROOT / "MOT16" / "train", help="Path to MOT16 train/test root.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "resources" / "detections_tf",
        help="Where to write detection .npy files.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models_tf" / "mars-small128.pb",
        help="Path to the TensorFlow frozen graph (.pb).",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_tf_detections(args.mot_dir, args.output_dir, args.model, args.batch_size)


if __name__ == "__main__":
    main()
