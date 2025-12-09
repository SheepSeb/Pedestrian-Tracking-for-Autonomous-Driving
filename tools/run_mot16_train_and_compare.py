from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image

from deep_sort.appearance_model_torch import (
    DEFAULT_IMAGE_SIZE,
    load_appearance_model,
    select_device,
)
from tools.train_mot16 import MOT16ReIDDataset

MOT16_URL = "https://motchallenge.net/data/MOT16.zip"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def ensure_mot16(root: Path) -> Path:
    mot_root = root / "MOT16"
    if (mot_root / "train").exists() and (mot_root / "test").exists():
        return mot_root

    zip_path = root / "MOT16.zip"
    if not zip_path.exists():
        print(f"Downloading MOT16 to {zip_path} ...")
        download_file(MOT16_URL, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    return mot_root


def run_training(mot_root: Path, epochs: int = 1, batch_size: int = 64) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train_mot16.py"),
        "--data-root",
        str(mot_root),
        "--output-dir",
        str(Path("checkpoints") / "mot16"),
        "--log-dir",
        str(Path("runs") / "mot16"),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        "auto",
    ]
    print("Running training:", " ".join(cmd))
    subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": "."})


def load_tf_model(tf_model_path: Path):
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "TensorFlow is required for TF comparison but is not installed."
        ) from exc

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(str(tf_model_path), "rb") as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="net")
    sess = tf.compat.v1.Session(graph=graph)
    input_tensor = graph.get_tensor_by_name("net/images:0")
    output_tensor = graph.get_tensor_by_name("net/features:0")
    return sess, input_tensor, output_tensor


def extract_patches(
    image_paths: List[Path],
    boxes: List[Tuple[float, float, float, float]],
    image_size: Tuple[int, int],
) -> List[np.ndarray]:
    patches: List[np.ndarray] = []
    for img_path, box in zip(image_paths, boxes):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            x, y, w, h = box
            crop = img.crop((x, y, x + w, y + h)).resize(
                image_size[::-1], Image.BILINEAR
            )
            patches.append(np.array(crop))
    return patches


def collect_sample_patches(mot_root: Path, max_patches: int = 16) -> List[np.ndarray]:
    dataset = MOT16ReIDDataset(
        root=mot_root, image_size=DEFAULT_IMAGE_SIZE, augment=False
    )
    patches: List[np.ndarray] = []
    for i in range(min(max_patches, len(dataset))):
        img_tensor, _ = dataset[i]
        # Convert tensor back to uint8 image for TF; keep normalized tensor for torch
        arr = img_tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        patches.append(arr)
    return patches


def run_torch_inference(
    patches: List[np.ndarray], checkpoint: Optional[Path] = None
) -> np.ndarray:
    device = select_device("auto")
    model, transform, _ = load_appearance_model(
        model_path=None, device=device, image_size=DEFAULT_IMAGE_SIZE
    )
    if checkpoint and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        encoder_state = state.get("encoder", state)
        model.load_state_dict(encoder_state, strict=False)
    model.eval()
    with torch.no_grad():
        batch = torch.stack([transform(patch) for patch in patches]).to(device)
        feats = model(batch).cpu().numpy()
    return feats


def run_tf_inference(
    patches: List[np.ndarray], tf_model_path: Path
) -> np.ndarray:
    sess, input_tensor, output_tensor = load_tf_model(tf_model_path)
    # BGR expected; convert RGB->BGR
    bgr = [patch[:, :, ::-1] for patch in patches]
    arr = np.stack(bgr).astype(np.uint8)
    feats = sess.run(output_tensor, feed_dict={input_tensor: arr})
    return feats


def compare_embeddings(torch_feats: np.ndarray, tf_feats: np.ndarray) -> None:
    min_len = min(len(torch_feats), len(tf_feats))
    torch_feats = torch_feats[:min_len]
    tf_feats = tf_feats[:min_len]
    cos = np.sum(torch_feats * tf_feats, axis=1)
    print(f"Compared {min_len} embeddings:")
    print(f"  Cosine mean: {cos.mean():.4f}, std: {cos.std():.4f}")
    print(
        f"  Torch norms mean: {np.linalg.norm(torch_feats, axis=1).mean():.4f}, "
        f"TF norms mean: {np.linalg.norm(tf_feats, axis=1).mean():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MOT16, train Torch model for 1 epoch, compare with TensorFlow (optional)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory to store MOT16 and artifacts.",
    )
    parser.add_argument(
        "--tf-model",
        type=Path,
        default=None,
        help="Path to TensorFlow .pb model for comparison (optional).",
    )
    parser.add_argument(
        "--use-checkpoint",
        type=Path,
        default=Path("checkpoints/mot16/epoch_001.pt"),
        help="Torch checkpoint to load for comparison (optional).",
    )
    args = parser.parse_args()

    mot_root = ensure_mot16(args.root)
    run_training(mot_root, epochs=1, batch_size=64)

    if args.tf_model is None:
        print("No TensorFlow model provided; skipping TF comparison.")
        return

    print("Collecting sample patches for comparison ...")
    patches = collect_sample_patches(mot_root, max_patches=16)

    print("Running Torch inference ...")
    torch_feats = run_torch_inference(patches, checkpoint=args.use_checkpoint)

    print("Running TensorFlow inference ...")
    tf_feats = run_tf_inference(patches, args.tf_model)

    compare_embeddings(torch_feats, tf_feats)


if __name__ == "__main__":
    main()
