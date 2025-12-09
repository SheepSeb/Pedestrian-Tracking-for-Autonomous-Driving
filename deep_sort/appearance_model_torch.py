from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torchvision import models, transforms

# Default patch size matches the legacy TensorFlow graph: (height, width).
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (128, 64)


def select_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve a torch device with sensible fallbacks."""
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _build_resnet18_backbone(
    weights: models.ResNet18_Weights,
) -> Tuple[nn.Module, int]:
    """Create a ResNet18 backbone without the classification head."""
    backbone = models.resnet18(weights=weights)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


def _build_transform(
    image_size: Tuple[int, int], weights: models.ResNet18_Weights
) -> transforms.Compose:
    mean = weights.meta.get("mean", (0.485, 0.456, 0.406))
    std = weights.meta.get("std", (0.229, 0.224, 0.225))
    return transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class AppearanceModel(nn.Module):
    """Lightweight appearance encoder producing 128-D L2-normalized embeddings."""

    def __init__(self, backbone: nn.Module, feature_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Linear(feature_dim, embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.backbone(inputs)
        embeddings = self.projection(features)
        return nn.functional.normalize(embeddings, p=2, dim=1)


def load_appearance_model(
    model_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    embedding_dim: int = 128,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Tuple[nn.Module, transforms.Compose, torch.device]:
    """Load a PyTorch appearance model and preprocessing transform.

    Parameters
    ----------
    model_path : Optional[str]
        Optional path to a state_dict checkpoint. When omitted, an ImageNet
        pretrained ResNet18 backbone is used.
    device : Optional[str]
        Torch device spec ("cpu", "cuda", "mps", or "auto").
    embedding_dim : int
        Output embedding dimensionality (default: 128).
    image_size : Tuple[int, int]
        Target (height, width) patch size before encoding.
    """
    weights = models.ResNet18_Weights.DEFAULT
    backbone, feature_dim = _build_resnet18_backbone(weights)
    model = AppearanceModel(backbone, feature_dim, embedding_dim=embedding_dim)

    if model_path:
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    resolved_device = select_device(device)
    model.to(resolved_device)
    model.eval()

    transform = _build_transform(image_size, weights)
    return model, transform, resolved_device
