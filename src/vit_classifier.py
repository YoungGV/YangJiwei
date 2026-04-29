"""用 CIFAR-10 训练一个小型 Vision Transformer 图像分类器。

这个脚本保留了 ViT 的完整流程，适合作业实践和实验报告分析：

图像 -> patch embedding -> class token + position embedding -> Transformer
encoder -> 分类头。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

# 在命令行或 PyCharm 中运行时不弹出窗口，直接把曲线图保存为 png 文件。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attention import MultiHeadSelfAttention


@dataclass
class ViTConfig:
    """集中保存 ViT 模型结构参数，方便命令行修改实验配置。"""

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
        """计算一张图会被切成多少个 patch。"""
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side


# CIFAR-10 的 10 个类别名称，用于保存预测可视化图片。
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
# CIFAR-10 常用均值和标准差。训练时用于归一化，画图时再反归一化回来。
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class PatchEmbedding(nn.Module):
    """把二维图像切成 patch，并映射成 Transformer 能处理的 token 序列。"""

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
        # 输入 images: (batch, channels, height, width)，例如 (B, 3, 32, 32)。
        # kernel_size=stride=patch_size，相当于把图像按不重叠 patch 切块。
        patches = self.proj(images)
        # patches: (batch, embed_dim, h/patch, w/patch)。
        # flatten(2) 展平空间维度，再 transpose 变成 token 序列。
        return patches.flatten(2).transpose(1, 2)
        # 输出 tokens: (batch, num_patches, embed_dim)。


class TransformerEncoderBlock(nn.Module):
    """ViT 使用的 Transformer Encoder block：注意力 + MLP + 残差连接。"""

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
        # Pre-Norm 结构：先 LayerNorm，再做多头注意力，最后残差相加。
        attn_output, _ = self.attn(self.norm1(tokens))
        tokens = tokens + attn_output
        # 第二个子层是 MLP，用来增强每个 token 的非线性表达能力。
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class VisionTransformer(nn.Module):
    """适合 CIFAR-10 实验的小型 ViT 分类模型。"""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        # class token 是一个可学习向量，最终用它代表整张图像做分类。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        # position embedding 用来告诉 Transformer 每个 patch 的空间位置。
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
        """使用截断正态分布初始化可学习参数，使训练更稳定。"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        # 1. 图像切 patch 并映射为 token。
        patch_tokens = self.patch_embed(images)
        # 2. 为 batch 中每张图复制一个 class token。
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 3. 把 class token 拼到 patch token 前面。
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)
        # 4. 加位置编码后送入多层 Transformer Encoder。
        tokens = self.dropout(tokens + self.pos_embed)
        tokens = self.blocks(tokens)
        # 5. 取第 0 个 class token 的输出作为整张图像特征。
        cls_output = self.norm(tokens[:, 0])
        return self.head(cls_output)


def make_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """构造 CIFAR-10 的训练集和测试集 DataLoader。"""

    train_transform = transforms.Compose(
        [
            # 数据增强：随机裁剪、水平翻转、随机擦除，提高泛化能力。
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
        # pin_memory=True 可以配合 non_blocking=True 加快 CPU 到 GPU 的数据拷贝。
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        # worker 常驻和预取数据可以减少 GPU 等待数据的时间。
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
    """遍历 DataLoader；如果设置 max_batches，就只跑指定数量的 batch。"""

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
    """训练一个 epoch，返回平均 loss 和 accuracy。"""

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in iterate_limited(loader, max_batches):
        # non_blocking=True 在 CUDA + pin_memory 时可以异步传输数据。
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # GPU 上启用 AMP 混合精度，可以降低显存占用并加速训练。
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        # GradScaler 用于 AMP，防止半精度训练时梯度下溢。
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
    """在测试集上评估模型，不计算梯度，返回平均 loss 和 accuracy。"""

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
    """把每轮训练结果保存成 CSV 表格，便于写实验报告。"""

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
    """根据 history 绘制 loss 曲线和 accuracy 曲线。"""

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
    """把归一化后的图像还原到 0~1 范围，方便 matplotlib 正常显示。"""

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
    """保存若干测试样本的预测结果图，绿色表示预测正确，红色表示预测错误。"""

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
    """保存模型权重、优化器状态、学习率调度器状态和训练历史。"""

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
    """解析命令行参数，例如训练轮数、batch size、模型大小和输出目录。"""

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
        # 固定输入大小时，benchmark 可以让 cuDNN 自动选择更快的卷积算法。
        torch.backends.cudnn.benchmark = True
        # 允许 Tensor Core 使用较快的矩阵乘法精度，适合 RTX 5060 这类 NVIDIA GPU。
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
    # 余弦退火学习率：前期学习率较大，后期逐渐变小，有助于收敛。
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
        # 每个 epoch 结束后更新一次学习率。
        scheduler.step()

    # 训练结束后自动保存表格、曲线图、预测图和模型权重。
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
