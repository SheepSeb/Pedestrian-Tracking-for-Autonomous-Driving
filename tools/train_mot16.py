from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from deep_sort.appearance_model_torch import (
    DEFAULT_IMAGE_SIZE,
    load_appearance_model,
    select_device,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Sample:
    image_path: Path
    bbox: Tuple[float, float, float, float]
    label: int


class MOT16ReIDDataset(Dataset):
    """Person re-ID style dataset built from MOT16 train split."""

    def __init__(
        self,
        root: Path,
        image_size: Tuple[int, int],
        augment: bool = True,
        min_height: int = 20,
    ):
        self.root = root
        self.image_size = image_size
        self.augment = augment
        self.min_height = min_height
        self.samples: List[Sample] = []
        self.label_map: Dict[Tuple[str, int], int] = {}
        self._load_samples()
        self.transform = self._build_transform()

    def _build_transform(self) -> Callable:
        ops: List[Callable] = [transforms.Resize(self.image_size, antialias=True)]
        if self.augment:
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                ]
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return transforms.Compose(ops)

    def _load_samples(self) -> None:
        train_dir = self.root / "train"
        if not train_dir.exists():
            raise FileNotFoundError(
                f"MOT16 train directory not found at {train_dir}. "
                "Download MOT16 (https://motchallenge.net/data/MOT16/) "
                "and place it under --data-root."
            )

        for seq_dir in sorted(train_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            gt_path = seq_dir / "gt" / "gt.txt"
            img_dir = seq_dir / "img1"
            if not gt_path.exists() or not img_dir.exists():
                continue

            rows = np.loadtxt(gt_path, delimiter=",")
            for row in rows:
                frame_id = int(row[0])
                track_id = int(row[1])
                if track_id <= 0:
                    continue
                x, y, w, h = row[2:6]
                if h < self.min_height or w <= 1 or h <= 1:
                    continue
                img_path = img_dir / f"{frame_id:06d}.jpg"
                if not img_path.exists():
                    continue
                key = (seq_dir.name, track_id)
                label = self.label_map.setdefault(key, len(self.label_map))
                self.samples.append(Sample(img_path, (x, y, w, h), label))

        if not self.samples:
            raise RuntimeError(
                f"No training samples found under {train_dir}. "
                "Ensure MOT16 is downloaded and annotations are present."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(sample.image_path) as img:
            img = img.convert("RGB")
            x, y, w, h = sample.bbox
            crop = img.crop((x, y, x + w, y + h))
        return self.transform(crop), sample.label


class ReIDModel(nn.Module):
    """Appearance encoder + classifier head for supervised ID training."""

    def __init__(self, encoder: nn.Module, embedding_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(x)
        logits = self.classifier(embeddings)
        return embeddings, logits


def collate_valid(batch: Sequence):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: ReIDModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch in loader:
        if batch is None:
            continue
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            _, logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: ReIDModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch in loader:
        if batch is None:
            continue
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        _, logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


def save_checkpoint(
    model: ReIDModel,
    epoch: int,
    output_dir: Path,
    label_map: Dict[Tuple[str, int], int],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "encoder": model.encoder.state_dict(),
            "classifier": model.classifier.state_dict(),
            "label_map": label_map,
        },
        ckpt_path,
    )
    return ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train appearance model on MOT16")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to MOT16 root directory")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/mot16"), help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/mot16"), help="TensorBoard log directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument(
        "--image-height",
        type=int,
        default=DEFAULT_IMAGE_SIZE[0],
        help="Person crop height after resizing",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=DEFAULT_IMAGE_SIZE[1],
        help="Person crop width after resizing",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto|cpu|cuda|mps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging (assumes local/connected backend)",
    )
    parser.add_argument("--wandb-project", type=str, default="deep-sort-mot16", help="wandb project name")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="offline",
        choices=["offline", "online"],
        help="wandb mode (offline for local/airgapped use)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = select_device(args.device)
    image_size = (args.image_height, args.image_width)

    dataset = MOT16ReIDDataset(args.data_root, image_size=image_size, augment=True)
    num_classes = len(dataset.label_map)

    # 90/10 train/val split
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    def make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=collate_valid,
        )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)

    encoder, _, _ = load_appearance_model(
        model_path=None, device=device, embedding_dim=args.embedding_dim, image_size=image_size
    )
    model = ReIDModel(encoder, args.embedding_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    writer = SummaryWriter(log_dir=args.log_dir)

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            mode=args.wandb_mode,
            config=vars(args),
        )
        wandb.watch(model, log="all", log_freq=100)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        writer.flush()

        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                }
            )

        ckpt_path = save_checkpoint(
            model=model,
            epoch=epoch,
            output_dir=args.output_dir,
            label_map=dataset.label_map,
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} "
            f"- saved {ckpt_path}"
        )

    writer.close()
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
