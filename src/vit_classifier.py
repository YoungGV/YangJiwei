"""Train a small Vision Transformer for image classification on CIFAR-10.

This script is intentionally compact enough for homework practice while still
showing the complete ViT pipeline:

image -> patch embedding -> class token + position embedding -> Transformer
encoder -> classification head.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attention import MultiHeadSelfAttention


@dataclass
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    embed_dim: int = 128
    depth: int = 6
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    @property
    def num_patches(self) -> int:
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class PatchEmbedding(nn.Module):
    """Convert images into patch tokens with a convolutional projection."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.proj = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (batch, channels, height, width)
        patches = self.proj(images)
        # patches: (batch, embed_dim, h/patch, w/patch)
        return patches.flatten(2).transpose(1, 2)
        # tokens: (batch, num_patches, embed_dim)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block used by ViT."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(self.norm1(tokens))
        tokens = tokens + attn_output
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class VisionTransformer(nn.Module):
    """A small ViT classifier suitable for CIFAR-10 experiments."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(
            *[TransformerEncoderBlock(config) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        patch_tokens = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)
        tokens = self.dropout(tokens + self.pos_embed)
        tokens = self.blocks(tokens)
        cls_output = self.norm(tokens[:, 0])
        return self.head(cls_output)


def make_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=test_transform
    )
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, test_loader


def iterate_limited(
    loader: DataLoader,
    max_batches: int | None,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        yield batch


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in iterate_limited(loader, max_batches):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in iterate_limited(loader, max_batches):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_training_log(history: list[dict[str, float]], output_dir: Path) -> Path:
    log_path = output_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "test_loss",
                "test_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(history)
    return log_path


def plot_training_curves(history: list[dict[str, float]], output_dir: Path) -> Path:
    curve_path = output_dir / "training_curve.png"
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["test_loss"] for row in history], marker="o", label="test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], marker="o", label="train")
    axes[1].plot(epochs, [row["test_acc"] for row in history], marker="o", label="test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)
    return curve_path


def denormalize_cifar10(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


@torch.no_grad()
def save_sample_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_images: int,
    use_amp: bool,
) -> Path:
    prediction_path = output_dir / "sample_predictions.png"
    model.eval()
    images, labels = next(iter(loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]

    with torch.autocast(device_type=device.type, enabled=use_amp):
        logits = model(images)
    predictions = logits.argmax(dim=1).cpu()
    images = denormalize_cifar10(images).cpu()

    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = [axes] if num_images == 1 else axes.reshape(-1)

    for idx in range(rows * cols):
        ax = axes[idx]
        ax.axis("off")
        if idx >= len(images):
            continue
        image = images[idx].permute(1, 2, 0).numpy()
        pred_name = CIFAR10_CLASSES[predictions[idx].item()]
        true_name = CIFAR10_CLASSES[labels[idx].item()]
        title_color = "green" if predictions[idx].item() == labels[idx].item() else "red"
        ax.imshow(image)
        ax.set_title(f"pred: {pred_name}\ntrue: {true_name}", color=title_color)

    fig.tight_layout()
    fig.savefig(prediction_path, dpi=160)
    plt.close(fig)
    return prediction_path


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: ViTConfig,
    history: list[dict[str, float]],
    output_dir: Path,
) -> Path:
    checkpoint_path = output_dir / "vit_lightweight.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config.__dict__,
            "history": history,
            "class_names": CIFAR10_CLASSES,
        },
        checkpoint_path,
    )
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT image classification homework")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--num-sample-images", type=int, default=16)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        print(
            "CUDA is not available. Training will run on CPU; "
            "install a CUDA-enabled PyTorch build to use your NVIDIA GPU."
        )

    use_amp = device.type == "cuda" and not args.no_amp
    config = ViTConfig(
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )

    train_loader, test_loader = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = VisionTransformer(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}, amp={use_amp}, config={config}")
    print(
        "training setup: "
        f"epochs={args.epochs}, batch_size={args.batch_size}, "
        f"max_train_batches={args.max_train_batches}, max_test_batches={args.max_test_batches}, "
        f"lr={args.lr}, label_smoothing={args.label_smoothing}"
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.max_train_batches,
            scaler,
            use_amp,
        )
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
            args.max_test_batches,
            use_amp,
        )
        print(
            f"epoch {epoch:02d} | "
            f"lr {current_lr:.6f} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
            f"test loss {test_loss:.4f}, acc {test_acc:.3f}"
        )
        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        scheduler.step()

    log_path = save_training_log(history, args.output_dir)
    curve_path = plot_training_curves(history, args.output_dir)
    prediction_path = save_sample_predictions(
        model,
        test_loader,
        device,
        args.output_dir,
        args.num_sample_images,
        use_amp,
    )
    checkpoint_path = save_checkpoint(
        model, optimizer, scheduler, config, history, args.output_dir
    )
    print("Saved results:")
    print(f"- training log: {log_path}")
    print(f"- training curve: {curve_path}")
    print(f"- sample predictions: {prediction_path}")
    print(f"- model checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
