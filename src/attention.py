"""从零实现自注意力和多头自注意力。

本文件尽量让变量名和公式保持一致：
输入 x -> 查询 q、键 k、值 v -> 注意力权重 -> 输出。
直接运行本文件可以检查每一步的张量形状。
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SelfAttention(nn.Module):
    """单头缩放点积自注意力。

    Args:
        embed_dim: 每个 token 的特征维度 D。
        dropout: 加在注意力权重上的 dropout 概率。

    Shape:
        input:  (batch_size, num_tokens, embed_dim)
        output: (batch_size, num_tokens, embed_dim)
    """

    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # 三个线性层分别把输入 token 映射成 Query、Key、Value。
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # 注意力汇聚后的结果再经过一次线性变换，得到最终输出。
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x 的形状为 (B, N, D)：B 是批量大小，N 是 token 数，D 是特征维度。
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # QK^T 得到每个 token 对其他 token 的相关性分数。
        # 除以 sqrt(D) 可以避免点积值过大导致 softmax 梯度变小。
        scale = math.sqrt(self.embed_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        # softmax 后，最后一维的权重和为 1，表示“当前 token 应该关注谁”。
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        # 用注意力权重对 V 加权求和，得到融合上下文信息的新 token 表示。
        context = torch.matmul(attention, v)
        output = self.out_proj(context)
        return output, attention


class MultiHeadSelfAttention(nn.Module):
    """Transformer 和 ViT 中使用的多头自注意力。"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 每个注意力头只处理总特征维度的一部分。
        self.head_dim = embed_dim // num_heads

        # 一次线性变换同时生成 Q、K、V，输出维度为 3 * embed_dim。
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = x.shape

        # qkv: (B, N, 3D)
        qkv = self.qkv_proj(x)
        # reshape 后拆出 Q/K/V 和多个注意力头：
        # (B, N, 3, H, d_head) -> (3, B, H, N, d_head)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        # 每个头独立计算注意力，attention 形状为 (B, H, N, N)。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 每个头得到 (B, H, N, d_head)，再拼回 (B, N, D)。
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)
        output = self.out_proj(context)
        return output, attention


def _demo() -> None:
    # 构造一个很小的随机输入，用于演示输出和注意力权重的形状。
    torch.manual_seed(7)
    batch_size, num_tokens, embed_dim, num_heads = 2, 5, 32, 4
    x = torch.randn(batch_size, num_tokens, embed_dim)

    single_head = SelfAttention(embed_dim)
    single_output, single_attention = single_head(x)
    print("SelfAttention output:", tuple(single_output.shape))
    print("SelfAttention weights:", tuple(single_attention.shape))

    multi_head = MultiHeadSelfAttention(embed_dim, num_heads)
    multi_output, multi_attention = multi_head(x)
    print("MultiHeadSelfAttention output:", tuple(multi_output.shape))
    print("MultiHeadSelfAttention weights:", tuple(multi_attention.shape))


if __name__ == "__main__":
    _demo()
