# 自注意力机制与多头注意力机制推导

本文面向刚开始学习 Vision Transformer (ViT) 的同学，按“公式 - 维度 - 代码对应关系”的顺序整理自注意力与多头注意力的实现过程。

## 1. 自注意力机制

### 1.1 输入表示

设一个样本包含 `N` 个 token，每个 token 的特征维度为 `D`。在 ViT 中，token 通常来自图像 patch 的线性投影：

```text
X in R^(N x D)
```

如果按批量训练，输入张量为：

```text
X in R^(B x N x D)
```

其中：

- `B`：batch size
- `N`：序列长度，即 patch token 数量加上可选的 class token
- `D`：每个 token 的嵌入维度

### 1.2 Q、K、V 的线性变换

自注意力先把同一个输入 `X` 映射成三组向量：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

其中：

```text
W_Q, W_K, W_V in R^(D x d_k)
Q, K, V in R^(B x N x d_k)
```

直观理解：

- `Q` (Query)：当前 token 想“查找什么信息”。
- `K` (Key)：每个 token 能被匹配的“索引特征”。
- `V` (Value)：真正被加权汇聚的信息。

### 1.3 注意力分数

对每个 token 的 query，与所有 token 的 key 做点积，得到相关性分数：

```text
S = Q K^T
```

考虑 batch 后：

```text
Q:      B x N x d_k
K^T:    B x d_k x N
S:      B x N x N
```

`S[i, j]` 表示第 `i` 个 token 对第 `j` 个 token 的关注强度。

### 1.4 缩放与 softmax

当 `d_k` 较大时，点积结果的方差会变大，softmax 容易进入梯度很小的饱和区。因此使用缩放：

```text
A = softmax(S / sqrt(d_k))
```

其中：

```text
A in R^(B x N x N)
```

`A` 的最后一个维度和为 1，可以看成每个 token 对所有 token 的注意力权重分布。

### 1.5 加权求和

最后用注意力权重加权 value：

```text
Y = A V
```

维度为：

```text
A: B x N x N
V: B x N x d_k
Y: B x N x d_k
```

完整公式：

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

## 2. 多头注意力机制

单个注意力头只能在一个表示子空间中学习 token 关系。多头注意力把特征维度拆成多个子空间，使模型可以同时学习不同类型的关系，例如：

- 局部纹理关系
- 边缘或轮廓关系
- 物体不同区域之间的长距离关系
- 对机械零件图像中的孔、边、角、磨损区域等结构的关联

### 2.1 拆分多个头

设总嵌入维度为 `D`，头数为 `H`，则每个头的维度为：

```text
d_head = D / H
```

输入：

```text
X in R^(B x N x D)
```

经过一次线性层得到：

```text
QKV in R^(B x N x 3D)
```

再 reshape 成：

```text
Q, K, V in R^(B x H x N x d_head)
```

### 2.2 每个头独立计算注意力

第 `h` 个头：

```text
head_h = softmax(Q_h K_h^T / sqrt(d_head)) V_h
```

所有头并行计算后得到：

```text
heads in R^(B x H x N x d_head)
```

### 2.3 拼接与输出投影

把多个头拼接回总维度：

```text
Concat(head_1, ..., head_H) in R^(B x N x D)
```

再经过输出线性层：

```text
MultiHead(X) = Concat(head_1, ..., head_H) W_O
```

其中：

```text
W_O in R^(D x D)
```

## 3. ViT 中的注意力流程

ViT 将图像处理成 token 序列：

1. 输入图像 `B x C x H x W`。
2. 切成大小为 `P x P` 的 patch。
3. 每个 patch 展平成 `C * P * P` 维向量。
4. 线性映射到 `D` 维，得到 patch embeddings。
5. 加入 class token 与位置编码。
6. 输入多层 Transformer Encoder。
7. 使用 class token 的最终表示做分类。

若图像大小为 `32 x 32`，patch 大小为 `4 x 4`，则 patch 数量为：

```text
N_patch = (32 / 4) * (32 / 4) = 64
```

加上 class token 后：

```text
N = 65
```

## 4. 与代码的对应关系

`src/attention.py` 中的核心代码与公式对应如下：

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
attn = torch.softmax(scores, dim=-1)
out = torch.matmul(attn, v)
```

这三行分别对应：

1. `QK^T / sqrt(d_k)`
2. `softmax`
3. `Attention(Q, K, V) = A V`

## 5. 实验报告可写内容

完成实验后，可以在报告中回答：

1. 自注意力中为什么要除以 `sqrt(d_k)`？
2. 多头注意力相比单头注意力有什么优势？
3. patch size 变大或变小时，token 数量和计算量如何变化？
4. ViT 与 CNN 在图像建模方式上有什么不同？
5. 在 CIFAR-10 实验中，训练轮数、embedding 维度、层数、头数对准确率有什么影响？
