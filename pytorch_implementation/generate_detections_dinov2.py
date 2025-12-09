from __future__ import annotations

import argparse
from pathlib import Path

from deep_sort.appearance_model_torch import DINOV2_DEFAULT_IMAGE_SIZE
from pytorch_implementation.generate_detections import run_torch_detections

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate detections using the DINOv2 appearance backbone."
    )
    parser.add_argument(
        "--mot_dir",
        type=Path,
        default=ROOT / "MOT16" / "train",
        help="Path to MOT16 train/test root.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "resources" / "detections_dinov2",
        help="Where to write detection .npy files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional PyTorch checkpoint path; leave empty to use the default DINOv2 weights.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vits14",
        help="DINOv2 backbone variant, e.g., dinov2_vits14 or vit_small_patch14_dinov2.",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=DINOV2_DEFAULT_IMAGE_SIZE[0],
        help="Crop height fed to the DINOv2 backbone (defaults to recommended 518).",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=DINOV2_DEFAULT_IMAGE_SIZE[1],
        help="Crop width fed to the DINOv2 backbone (defaults to recommended 518).",
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
