from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.transforms.functional import InterpolationMode

# timm is only needed for DINOv2 variants; keep import lightweight.
try:
    import timm  # type: ignore
except Exception:  # pragma: no cover - optional dependency is handled at runtime.
    timm = None

# Default patch size matches the legacy TensorFlow graph: (height, width).
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (128, 64)
# Recommended square size for DINOv2 backbones (per timm cfg: 518).
DINOV2_DEFAULT_IMAGE_SIZE: Tuple[int, int] = (518, 518)
DINOV2_DEFAULT_VARIANT = "vit_small_patch14_dinov2"


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
    return _build_transform_from_stats(image_size, mean, std, interpolation=InterpolationMode.BILINEAR)


def _build_transform_from_stats(
    image_size: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=interpolation, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _normalize_dinov2_variant(backbone: str) -> str:
    alias_map = {
        "dinov2": DINOV2_DEFAULT_VARIANT,
        "dinov2_small": "vit_small_patch14_dinov2",
        "dinov2_vits14": "vit_small_patch14_dinov2",
        "dinov2_vitb14": "vit_base_patch14_dinov2",
        "dinov2_vitl14": "vit_large_patch14_dinov2",
        "dinov2_vitg14": "vit_giant_patch14_dinov2",
    }
    backbone_key = backbone.lower().replace("-", "_")
    return alias_map.get(backbone_key, backbone_key)


def _build_dinov2_backbone(
    variant: str,
) -> Tuple[nn.Module, int, Tuple[float, float, float], Tuple[float, float, float], Tuple[int, int]]:
    if timm is None:
        raise ImportError(
            "timm is required for DINOv2 backbones. Install it or choose a ResNet backbone."
        )
    normalized_variant = _normalize_dinov2_variant(variant)
    backbone = timm.create_model(
        normalized_variant,
        pretrained=True,
        num_classes=0,
        global_pool="token",
    )
    feature_dim = getattr(backbone, "num_features", None) or getattr(backbone, "embed_dim", None)
    if feature_dim is None:
        raise ValueError(f"Unable to determine feature dimension for backbone '{normalized_variant}'.")

    cfg = getattr(backbone, "pretrained_cfg", {}) or {}
    mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))  # type: ignore[arg-type]
    std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))  # type: ignore[arg-type]
    input_size = cfg.get("input_size")
    if isinstance(input_size, (list, tuple)) and len(input_size) >= 3:
        recommended_size = (int(input_size[1]), int(input_size[2]))
    else:
        recommended_size = DINOV2_DEFAULT_IMAGE_SIZE
    return backbone, int(feature_dim), mean, std, recommended_size


def _maybe_resize_positional_encoding(
    backbone: nn.Module, new_size: Tuple[int, int]
) -> None:
    """Resize positional embeddings when changing ViT input resolution."""
    if not hasattr(backbone, "pos_embed") or not hasattr(backbone, "patch_embed"):
        return
    pos_embed = backbone.pos_embed
    patch_embed = backbone.patch_embed
    if pos_embed is None or patch_embed is None:
        return

    # Expect shape (1, 1 + H*W, C)
    cls_token = pos_embed[:, :1]
    grid_tokens = pos_embed[:, 1:]
    if grid_tokens.numel() == 0:
        return
    old_h, old_w = patch_embed.grid_size
    new_h = new_size[0] // patch_embed.patch_size[0]
    new_w = new_size[1] // patch_embed.patch_size[1]
    if (old_h, old_w) == (new_h, new_w):
        return
    grid_tokens = grid_tokens.reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2)
    grid_tokens = F.interpolate(grid_tokens, size=(new_h, new_w), mode="bicubic", align_corners=False)
    grid_tokens = grid_tokens.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)
    backbone.pos_embed = torch.cat([cls_token, grid_tokens], dim=1)


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
    backbone: str = "resnet18",
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
    backbone : str
        Backbone architecture to use. Supported: "resnet18" (default) or any
        DINOv2 timm variant such as "dinov2_vits14" / "vit_small_patch14_dinov2".
    """
    resolved_device = select_device(device)
    backbone_name = backbone.lower()

    if backbone_name.startswith("dino") or "dinov2" in backbone_name:
        backbone_module, feature_dim, mean, std, recommended_size = _build_dinov2_backbone(backbone_name)
        effective_size = recommended_size
        if image_size != DEFAULT_IMAGE_SIZE and image_size != recommended_size:
            print(
                f"[info] DINOv2 backbone requested custom size {image_size}; "
                f"overriding to recommended {recommended_size} to match model config."
            )
        # Align backbone patch embedding with chosen input size.
        if hasattr(backbone_module, "patch_embed"):
            patch = backbone_module.patch_embed
            if hasattr(patch, "img_size"):
                patch.img_size = effective_size
            if hasattr(patch, "grid_size") and hasattr(patch, "patch_size"):
                patch.grid_size = (
                    effective_size[0] // patch.patch_size[0],
                    effective_size[1] // patch.patch_size[1],
                )
        _maybe_resize_positional_encoding(backbone_module, effective_size)
        transform = _build_transform_from_stats(
            effective_size, mean=mean, std=std, interpolation=InterpolationMode.BICUBIC
        )
        model = AppearanceModel(backbone_module, feature_dim, embedding_dim=embedding_dim)
    elif backbone_name in {"resnet18", "resnet-18"}:
        weights = models.ResNet18_Weights.DEFAULT
        backbone_module, feature_dim = _build_resnet18_backbone(weights)
        transform = _build_transform(image_size, weights)
        model = AppearanceModel(backbone_module, feature_dim, embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose 'resnet18' or a DINOv2 variant.")

    if model_path:
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    model.to(resolved_device)
    model.eval()
    return model, transform, resolved_device
