# vim: expandtab:ts=4:sw=4
from __future__ import annotations

import argparse
import errno
import os
from typing import Callable, Iterable, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from deep_sort.appearance_model_torch import DEFAULT_IMAGE_SIZE, load_appearance_model

# Optional TensorFlow import for legacy .pb models
try:
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None


def extract_image_patch(image: np.ndarray, bbox: np.ndarray, patch_shape: Tuple[int, int]):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.
    """
    bbox = np.array(bbox, dtype=np.float64)
    if patch_shape is not None:
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int64)

    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class TorchImageEncoder:
    """Encode person crops into 128-D embeddings using PyTorch."""

    def __init__(
        self,
        model: torch.nn.Module,
        transform: Callable,
        device: torch.device,
        batch_size: int = 32,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ):
        self.model = model
        self.transform = transform
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size
        projection = getattr(model, "projection", None)
        self.feature_dim = (
            projection.out_features if projection is not None else getattr(model, "feature_dim", 128)
        )

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, self.image_size)
            if patch is None:
                print(f"WARNING: Failed to extract image patch: {box}.")
                patch = np.random.uniform(0.0, 255.0, self.image_size + (3,)).astype(
                    np.uint8
                )
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            image_patches.append(Image.fromarray(patch))
        return self._encode(image_patches)

    def _encode(self, patches: Iterable["Image.Image"]) -> np.ndarray:
        patches = list(patches)
        if not patches:
            return np.zeros((0, self.feature_dim), np.float32)

        outputs = np.zeros((len(patches), self.feature_dim), np.float32)
        with torch.no_grad():
            for start in range(0, len(patches), self.batch_size):
                batch = patches[start : start + self.batch_size]
                batch_tensor = torch.stack([self.transform(p) for p in batch]).to(
                    self.device
                )
                batch_features = self.model(batch_tensor).cpu().numpy()
                end = start + len(batch)
                outputs[start:end] = batch_features
        return outputs


def create_box_encoder(
    model_path: str,
    device: str = "auto",
    batch_size: int = 32,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    backbone: str = "resnet18",
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    # If a TensorFlow frozen graph is provided, use the TF encoder.
    if model_path and model_path.endswith(".pb"):
        if tf is None:
            raise RuntimeError("TensorFlow is not available, cannot load .pb model.")

        class ImageEncoderTF:
            def __init__(self, checkpoint_filename: str):
                self.session = tf.compat.v1.Session()
                with tf.compat.v1.gfile.GFile(checkpoint_filename, "rb") as fh:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(fh.read())
                tf.import_graph_def(graph_def, name="net")
                graph = tf.compat.v1.get_default_graph()

                def _get_tensor(name: str):
                    for candidate in [f"net/{name}:0", f"{name}:0"]:
                        try:
                            return graph.get_tensor_by_name(candidate)
                        except Exception:
                            continue
                    raise KeyError(f"Could not find tensor for name '{name}' in graph.")

                self.input_var = _get_tensor("images")
                self.output_var = _get_tensor("features")
                self.feature_dim = self.output_var.get_shape().as_list()[-1]
                self.image_shape = self.input_var.get_shape().as_list()[1:]

            def __call__(self, data_x, batch_size=32):
                out = np.zeros((len(data_x), self.feature_dim), np.float32)
                # simple batching
                for start in range(0, len(data_x), batch_size):
                    end = start + batch_size
                    batch = data_x[start:end]
                    out[start:end] = self.session.run(self.output_var, feed_dict={self.input_var: batch})
                return out

        encoder_tf = ImageEncoderTF(model_path)
        image_shape = encoder_tf.image_shape

        def encoder(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
            patches = []
            for box in boxes:
                patch = extract_image_patch(image, box, image_shape[:2])
                if patch is None:
                    patch = np.random.uniform(0.0, 255.0, image_shape).astype(np.uint8)
                patches.append(patch)
            patches = np.asarray(patches)
            return encoder_tf(patches, batch_size)

        return encoder

    # Otherwise, use the PyTorch encoder
    model, transform, resolved_device = load_appearance_model(
        model_path=model_path or None,
        device=device,
        image_size=image_size,
        backbone=backbone,
    )
    return TorchImageEncoder(
        model=model,
        transform=transform,
        device=resolved_device,
        batch_size=batch_size,
        image_size=image_size,
    )


def generate_detections(
    encoder: Callable[[np.ndarray, np.ndarray], np.ndarray],
    mot_dir: str,
    output_dir: str,
    detection_dir: str | None = None,
):
    """Generate detections with embeddings using a PyTorch encoder."""
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(f"Failed to create output directory '{output_dir}'")

    for sequence in os.listdir(mot_dir):
        print(f"Processing {sequence}")
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
        }

        detection_file = os.path.join(detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=",")
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int64)
        min_frame_idx = frame_indices.min()
        max_frame_idx = frame_indices.max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print(f"Frame {frame_idx:05d}/{max_frame_idx:05d}")
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print(f"WARNING could not find image for frame {frame_idx}")
                continue
            bgr_image = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [
                np.r_[(row, feature)] for row, feature in zip(rows, features)
            ]

        output_filename = os.path.join(output_dir, f"{sequence}.npy")
        np.save(output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Re-ID feature extractor (PyTorch)")
    parser.add_argument(
        "--model",
        default="",
        help="Optional path to a PyTorch state_dict checkpoint. Defaults to an ImageNet-pretrained ResNet18.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use ('auto', 'cpu', 'cuda', 'mps').",
    )
    parser.add_argument(
        "--backbone",
        default="resnet18",
        help=(
            "Backbone to use for appearance encoding. "
            "Options: 'resnet18' (default) or DINOv2 variants such as "
            "'dinov2_vits14' / 'vit_small_patch14_dinov2'."
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for feature extraction."
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=DEFAULT_IMAGE_SIZE[0],
        help="Height of the resized person crop.",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=DEFAULT_IMAGE_SIZE[1],
        help="Width of the resized person crop.",
    )
    parser.add_argument(
        "--mot_dir",
        help="Path to MOTChallenge directory (train or test)",
        required=True,
    )
    parser.add_argument(
        "--detection_dir",
        help=(
            "Path to custom detections. Defaults to standard MOT detections. "
            "Directory structure should be [sequence]/det/det.txt"
        ),
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory. Will be created if it does not exist.",
        default="detections",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_size = (args.image_height, args.image_width)
    encoder = create_box_encoder(
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
        image_size=image_size,
        backbone=args.backbone,
    )
    generate_detections(
        encoder, args.mot_dir, args.output_dir, args.detection_dir
    )


if __name__ == "__main__":
    main()
