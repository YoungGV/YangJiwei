# Vision Transformer 实践练习作业

本仓库整理了一份面向机械研究生的 Vision Transformer (ViT) 入门实践作业，覆盖题目中的三项要求：

1. 推导自注意力机制和多头注意力机制的实现过程。
2. 用 PyTorch 实现自注意力、多头注意力和 ViT 关键模块。
3. 完成一个基于 ViT 的图像分类任务。

## 文件结构

```text
.
├── docs/
│   └── attention_derivation.md      # 注意力机制公式推导与维度说明
├── src/
│   ├── attention.py                 # SelfAttention 与 MultiHeadSelfAttention 实现
│   └── vit_classifier.py            # ViT 模型与 CIFAR-10 图像分类训练脚本
├── requirements.txt                 # 运行依赖
└── README.md                        # 作业入口说明
```

## 快速开始

建议使用 Python 3.10+。安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

运行注意力模块的张量形状自检：

```bash
python3 src/attention.py
```

运行一个轻量型 ViT 图像分类实验：

```bash
python3 src/vit_classifier.py --epochs 1 --batch-size 128 --max-train-batches 100 --max-test-batches 20
```

如果本机有 GPU，脚本会自动使用 CUDA，并默认开启混合精度训练；否则使用 CPU。第一次运行会自动下载 CIFAR-10 数据集到 `data/` 目录。

默认参数已调整为适合笔记本 GPU 的轻量配置：

- `--patch-size 8`：每张 `32 x 32` 图像只产生 16 个 patch token，注意力计算更少。
- `--embed-dim 64`：降低 token 特征维度。
- `--depth 2`：减少 Transformer block 层数。
- `--num-workers 4`：提高数据加载并行度。

如果显存充足，可以尝试提高 `--batch-size`；如果仍然较慢，可以加上 `--max-train-batches 50` 先快速完成作业演示。

## 作业完成建议

- 先阅读 `docs/attention_derivation.md`，理解 Q、K、V、softmax 和多头拆分/拼接的数学过程。
- 再阅读 `src/attention.py`，对照公式查看代码中的矩阵乘法和维度变化。
- 最后运行 `src/vit_classifier.py`，观察训练损失、测试准确率，并尝试调整 `--embed-dim`、`--depth`、`--num-heads`、`--patch-size` 等参数。
