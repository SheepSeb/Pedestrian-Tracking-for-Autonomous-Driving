from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from deep_sort.appearance_model_torch import DEFAULT_IMAGE_SIZE

ROOT = Path(__file__).resolve().parents[1]


def run_torch_detections(
    mot_dir: Path,
    output_dir: Path,
    model: str,
    batch_size: int,
    backbone: str,
    image_height: int,
    image_width: int,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [
        "python",
        os.fspath(ROOT / "tools" / "generate_detections.py"),
        "--mot_dir",
        os.fspath(mot_dir),
        "--output_dir",
        os.fspath(output_dir),
    ]
    if model:
        cmd.extend(["--model", model])
    if batch_size:
        cmd.extend(["--batch_size", str(batch_size)])
    if backbone:
        cmd.extend(["--backbone", backbone])
    if image_height:
        cmd.extend(["--image_height", str(image_height)])
    if image_width:
        cmd.extend(["--image_width", str(image_width)])
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detections using the PyTorch appearance model.")
    parser.add_argument("--mot_dir", type=Path, default=ROOT / "MOT16" / "train", help="Path to MOT16 train/test root.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "resources" / "detections_torch",
        help="Where to write detection .npy files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional PyTorch checkpoint path; leave empty to use the default ResNet18 backbone.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction.")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        help="Appearance backbone to use ('resnet18' or DINOv2 variants).",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=DEFAULT_IMAGE_SIZE[0],
        help="Crop height fed to the appearance backbone.",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=DEFAULT_IMAGE_SIZE[1],
        help="Crop width fed to the appearance backbone.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_torch_detections(
        args.mot_dir,
        args.output_dir,
        args.model,
        args.batch_size,
        args.backbone,
        args.image_height,
        args.image_width,
    )


if __name__ == "__main__":
    main()
