from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import requests
import torch
from PIL import Image

from deep_sort.appearance_model_torch import (
    DEFAULT_IMAGE_SIZE,
    load_appearance_model,
)


MOT16_URL = "https://motchallenge.net/data/MOT16.zip"


def download_mot16(dest_zip: Path) -> None:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    if dest_zip.exists():
        print(f"[info] MOT16 archive already exists at {dest_zip}")
        return
    print(f"[info] downloading MOT16 from {MOT16_URL} -> {dest_zip}")
    with requests.get(MOT16_URL, stream=True) as r:
        r.raise_for_status()
        with open(dest_zip, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("[info] download complete")


def _zip_is_valid(zip_path: Path) -> bool:
    if not zip_path.exists():
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Attempt to read the first file entry to validate
            test_list = zf.namelist()
            return len(test_list) > 0
    except Exception:
        return False


def extract_mot16(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and (out_dir / "train").exists() and (out_dir / "test").exists():
        print(f"[info] MOT16 already extracted at {out_dir}")
        return
    if not _zip_is_valid(zip_path):
        print(f"[warn] MOT16 archive at {zip_path} is missing or invalid; re-downloading.")
        if zip_path.exists():
            zip_path.unlink()
        download_mot16(zip_path)
    print(f"[info] extracting {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir.parent)
    out_dir.mkdir(exist_ok=True)
    # ensure structure is out_dir/train, out_dir/test
    parent_train = out_dir.parent / "train"
    parent_test = out_dir.parent / "test"
    if not (out_dir / "train").exists() and parent_train.exists():
        parent_train.replace(out_dir / "train")
    if not (out_dir / "test").exists() and parent_test.exists():
        parent_test.replace(out_dir / "test")
    if not (out_dir / "train").exists() or not (out_dir / "test").exists():
        raise FileNotFoundError(
            f"Expected train/test folders under {out_dir}, please check extraction."
        )
    print("[info] extraction complete")


def run_torch_training(repo_root: Path, data_root: Path, output_dir: Path, log_dir: Path) -> Path:
    """Launch the existing torch trainer for 1 epoch."""
    train_script = repo_root / "tools" / "train_mot16.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--log-dir",
        str(log_dir),
        "--epochs",
        "1",
        "--batch-size",
        "64",
        "--device",
        "auto",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    print(f"[info] running torch training: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Torch training failed with exit code {proc.returncode}")
    ckpt = output_dir / "epoch_001.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {ckpt}")
    return ckpt


def _load_torch_encoder_from_ckpt(
    ckpt_path: Path, device: torch.device, image_size: Tuple[int, int]
) -> Tuple[torch.nn.Module, transforms.Compose, torch.device]:
    model, transform, device_resolved = load_appearance_model(
        model_path=None, device=device, image_size=image_size
    )
    checkpoint = torch.load(ckpt_path, map_location=device_resolved)
    state = checkpoint.get("encoder", checkpoint)
    model.load_state_dict(state, strict=False)
    model.to(device_resolved)
    model.eval()
    return model, transform, device_resolved


def _collect_sample_images(data_root: Path, num_images: int = 8) -> List[Path]:
    first_seq = sorted((data_root / "train").glob("MOT16-*/img1/*.jpg"))
    return first_seq[:num_images]


def _compute_embeddings_torch(
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    image_paths: Iterable[Path],
    image_size: Tuple[int, int],
) -> np.ndarray:
    patches = []
    for p in image_paths:
        with Image.open(p) as img:
            img = img.convert("RGB")
            # center crop to person-sized patch approximation
            w, h = img.size
            cx, cy = w // 2, h // 2
            bw, bh = image_size[1], image_size[0]
            x0 = max(0, cx - bw // 2)
            y0 = max(0, cy - bh // 2)
            x1 = min(w, x0 + bw)
            y1 = min(h, y0 + bh)
            crop = img.crop((x0, y0, x1, y1))
            patches.append(transform(crop))
    batch = torch.stack(patches).to(device)
    with torch.no_grad():
        feats = model(batch).cpu().numpy()
    return feats


def try_tensorflow_embeddings(image_paths: Iterable[Path], image_size: Tuple[int, int], model_path: Path) -> np.ndarray:
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        print("[warn] tensorflow not available; skipping TF comparison")
        return np.empty((0, 128), dtype=np.float32)

    graph = tf.Graph()
    with graph.as_default():
        with tf.io.gfile.GFile(str(model_path), "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="net")

    input_var = graph.get_tensor_by_name("net/images:0")
    output_var = graph.get_tensor_by_name("net/features:0")
    sess = tf.compat.v1.Session(graph=graph)

    patches = []
    for p in image_paths:
        with Image.open(p) as img:
            img = img.convert("RGB")
            w, h = img.size
            cx, cy = w // 2, h // 2
            bw, bh = image_size[1], image_size[0]
            x0 = max(0, cx - bw // 2)
            y0 = max(0, cy - bh // 2)
            x1 = min(w, x0 + bw)
            y1 = min(h, y0 + bh)
            crop = img.crop((x0, y0, x1, y1))
            crop = crop.resize((bw, bh))
            patches.append(np.asarray(crop)[:, :, ::-1])  # RGB->BGR for TF model expectation

    batch = np.stack(patches).astype(np.uint8)
    feats = sess.run(output_var, feed_dict={input_var: batch})
    sess.close()
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MOT16, train torch model 1 epoch, compare with TF (optional).")
    parser.add_argument("--data-root", type=Path, default=Path("MOT16"), help="Where MOT16 will be stored.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/mot16"), help="Where to save torch checkpoints.")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/mot16"), help="TensorBoard log directory.")
    parser.add_argument("--tf-model", type=Path, default=None, help="Path to TensorFlow frozen graph (.pb) for comparison.")
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[0], help="Crop height for embedding comparison.")
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[1], help="Crop width for embedding comparison.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    zip_path = repo_root / "MOT16.zip"

    # If no TF model provided, try to use a local default if present.
    if args.tf_model is None:
        default_tf = repo_root / "models_tf" / "mars-small128.pb"
        if default_tf.exists():
            args.tf_model = default_tf

    download_mot16(zip_path)
    extract_mot16(zip_path, repo_root / "MOT16")

    ckpt = args.output_dir / "epoch_001.pt"
    if not ckpt.exists():
        ckpt = run_torch_training(repo_root, args.data_root, args.output_dir, args.log_dir)
    else:
        print(f"[info] using existing checkpoint {ckpt}")

    image_size = (args.image_height, args.image_width)
    sample_images = _collect_sample_images(args.data_root, num_images=8)
    if not sample_images:
        raise RuntimeError("No sample images found for comparison.")

    torch_device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    torch_model, transform, torch_device = _load_torch_encoder_from_ckpt(
        ckpt, torch_device, image_size
    )
    torch_feats = _compute_embeddings_torch(
        torch_model, transform, torch_device, sample_images, image_size
    )
    print(f"[torch] embeddings: shape={torch_feats.shape}, mean_norm={np.linalg.norm(torch_feats, axis=1).mean():.3f}")

    if args.tf_model:
        tf_feats = try_tensorflow_embeddings(sample_images, image_size, args.tf_model)
        if tf_feats.size > 0:
            print(f"[tf] embeddings: shape={tf_feats.shape}, mean_norm={np.linalg.norm(tf_feats, axis=1).mean():.3f}")
    else:
        print("[info] no --tf-model provided; skipping TF comparison.")


if __name__ == "__main__":
    main()
