"""Self-attention and multi-head self-attention from first principles.

The implementation keeps tensor names close to the mathematical notation:
input x -> queries q, keys k, values v -> attention weights -> output.
Run this file directly to verify tensor shapes.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SelfAttention(nn.Module):
    """Single-head scaled dot-product self-attention.

    Args:
        embed_dim: Feature dimension D of each token.
        dropout: Dropout probability applied to attention probabilities.

    Shape:
        input:  (batch_size, num_tokens, embed_dim)
        output: (batch_size, num_tokens, embed_dim)
    """

    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scale = math.sqrt(self.embed_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)
        output = self.out_proj(context)
        return output, attention


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention used by Transformer and ViT blocks."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)
        output = self.out_proj(context)
        return output, attention


def _demo() -> None:
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
