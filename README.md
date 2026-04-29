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

运行一个适合笔记本 RTX 5060 的 ViT 图像分类实验：

```bash
python3 src/vit_classifier.py
```

如果本机有 GPU，脚本会自动使用 CUDA，并默认开启混合精度训练；否则使用 CPU。第一次运行会自动下载 CIFAR-10 数据集到 `data/` 目录。

训练结束后会自动保存实验结果到 `results/` 目录：

```text
results/
├── training_log.csv          # 每轮训练/测试 loss 和 accuracy
├── training_curve.png        # loss 与 accuracy 曲线图
├── sample_predictions.png    # 测试集样本预测可视化
└── vit_lightweight.pth       # 模型权重和配置
```

也可以通过 `--output-dir my_results` 指定其他输出目录。

默认参数已调整为更高准确率的 GPU 配置：

- `--epochs 20`：训练更充分，预测图会明显更准。
- `--patch-size 4`：每张 `32 x 32` 图像产生 64 个 patch token，保留更多细节。
- `--embed-dim 128`、`--depth 6`：模型容量高于快速演示版。
- 默认使用全量训练集和测试集，不再只训练少量 batch。
- 使用 AdamW、label smoothing、余弦学习率调度和 CUDA 混合精度。

如果只是想快速检查代码能否运行，可以用快速演示命令：

```bash
python3 src/vit_classifier.py --epochs 1 --batch-size 128 --max-train-batches 100 --max-test-batches 20 --patch-size 8 --embed-dim 64 --depth 2 --mlp-ratio 2.0
```

如果显存充足，可以尝试提高 `--batch-size 384`；如果显存不足，则改为 `--batch-size 128`。

## 作业完成建议

- 先阅读 `docs/attention_derivation.md`，理解 Q、K、V、softmax 和多头拆分/拼接的数学过程。
- 再阅读 `src/attention.py`，对照公式查看代码中的矩阵乘法和维度变化。
- 最后运行 `src/vit_classifier.py`，观察训练损失、测试准确率，并尝试调整 `--embed-dim`、`--depth`、`--num-heads`、`--patch-size` 等参数。
