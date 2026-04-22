# Transformer 架构深度解析

> 2017 年 Google 发表的 "Attention Is All You Need" 彻底改变了 NLP 乃至整个 AI 领域。本文从原理到实现，拆解 Transformer 的每一个核心组件。

## 为什么需要 Transformer

在 Transformer 之前，序列建模的主流方案是 RNN/LSTM：

- **顺序计算**：必须逐步处理序列，无法并行，训练极慢
- **长距离依赖衰减**：即使有 LSTM 的门控机制，超过几百个 token 后信息仍会丢失
- **梯度问题**：长序列上梯度消失/爆炸难以根治

Transformer 用纯注意力机制替代了循环结构，实现了：
1. 完全并行计算 → 训练速度提升数十倍
2. 任意位置之间直接建立联系 → 长距离依赖不再衰减
3. 架构简洁统一 → 易于扩展到超大规模

## 整体架构

```
输入 → [Encoder × N] → 编码表示
                              ↓
输出(shifted) → [Decoder × N] → 预测下一个 token
```

原始 Transformer 是 Encoder-Decoder 结构（用于翻译等 seq2seq 任务）。后来演化出三大变体：

| 变体 | 代表模型 | 用途 |
|------|---------|------|
| Encoder-only | BERT | 理解类任务（分类、NER） |
| Decoder-only | GPT 系列 | 生成类任务（对话、写作） |
| Encoder-Decoder | T5, BART | seq2seq 任务（翻译、摘要） |

当前大模型主流是 **Decoder-only** 架构。

## 核心组件详解

### 1. 自注意力机制（Self-Attention）

这是 Transformer 的灵魂。对于输入序列中的每个 token，计算它与所有其他 token 的关联程度。

**计算过程：**

```
输入 X (序列长度 n × 维度 d)
  ↓
通过三个线性变换得到 Q, K, V
  Q = X · W_Q    (Query: 我在找什么)
  K = X · W_K    (Key: 我能提供什么)
  V = X · W_V    (Value: 我的实际内容)
  ↓
计算注意力分数
  Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

**为什么除以 √d_k？** 当维度 d_k 很大时，Q·K^T 的值会很大，导致 softmax 输出趋近 one-hot（梯度接近 0）。除以 √d_k 是一种缩放，保持梯度稳定。

**直觉理解：** 想象你在读一句话 "小猫坐在垫子上，它很开心"。当模型处理"它"这个 token 时，注意力机制会让"它"重点关注"小猫"，因为它们的 Q-K 匹配度最高。

### 2. 多头注意力（Multi-Head Attention）

单个注意力头只能捕捉一种关联模式。多头注意力让模型同时关注不同类型的关系：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

其中 head_i = Attention(Q · W_Qi, K · W_Ki, V · W_Vi)
```

例如 8 个头可能分别学到：
- Head 1: 语法依赖（主语-谓语）
- Head 2: 指代关系（代词-先行词）
- Head 3: 相邻词关系
- Head 4: 语义相似性
- ...

GPT-3 使用 96 个注意力头，维度 12288，每个头维度 128。

### 3. 位置编码（Positional Encoding）

注意力机制本身是置换不变的——"猫吃鱼"和"鱼吃猫"在纯注意力下没有区别。位置编码注入了顺序信息。

**原始方案：正弦位置编码**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**现代方案：RoPE（旋转位置编码）**

LLaMA、Qwen 等模型采用 RoPE，核心思想是将位置信息编码为旋转角度，使得两个 token 的注意力分数只取决于它们的相对距离：

```
q_m · k_n = f(q, m) · f(k, n) = g(q, k, m-n)
```

RoPE 的优势：
- 天然支持相对位置
- 可通过 NTK-aware 插值等方法扩展到更长上下文
- 计算高效，无需额外参数

### 4. 前馈网络（FFN）

每个 Transformer 层中，注意力之后是一个两层前馈网络：

```
FFN(x) = W_2 · activation(W_1 · x + b_1) + b_2
```

激活函数的演进：
- 原始 Transformer: ReLU
- GPT 系列: GELU
- LLaMA: SwiGLU（效果更好，但参数多 50%）

FFN 的参数量通常占整个模型的 2/3。它被认为是模型的"知识存储器"——事实性知识主要编码在 FFN 的权重中。

### 5. 层归一化（Layer Normalization）

稳定训练的关键组件。有两种放置方式：

- **Post-LN**（原始 Transformer）：`x + LayerNorm(SubLayer(x))` — 训练不稳定，需要 warmup
- **Pre-LN**（GPT-2 起）：`x + SubLayer(LayerNorm(x))` — 训练更稳定，现代模型标配

LLaMA 进一步使用 **RMSNorm**（去掉均值中心化，只做缩放），计算更快：

```
RMSNorm(x) = x / RMS(x) · γ
RMS(x) = √(mean(x²))
```

### 6. 残差连接（Residual Connection）

每个子层都有残差连接：`output = x + SubLayer(x)`

作用：
- 缓解梯度消失，让梯度直接回传
- 允许模型堆叠到上百层
- 低层特征可以直接传递到高层

## KV Cache：推理加速的关键

自回归生成时，每生成一个新 token 都需要对之前所有 token 做注意力计算。KV Cache 缓存已计算的 K、V 矩阵，避免重复计算：

```
无 KV Cache: 生成第 n 个 token 需要计算 n 次注意力
有 KV Cache: 只需计算新 token 与缓存 KV 的注意力

时间复杂度: O(n²) → O(n)
```

代价是显存占用。对于 LLaMA-70B，单个请求的 KV Cache 在 4096 长度下约占 2.5GB。这也是为什么长上下文模型需要 GQA（Grouped Query Attention）等优化。

## 注意力的变体与优化

| 方法 | 核心思想 | 代表模型 |
|------|---------|---------|
| MHA | 标准多头注意力 | GPT-3 |
| MQA | 所有头共享 K、V | PaLM |
| GQA | K、V 分组共享 | LLaMA 2/3 |
| Sliding Window | 只关注局部窗口 | Mistral |
| Flash Attention | IO 感知的精确注意力 | 几乎所有现代模型 |

**GQA** 是当前最流行的方案：将 Q 头分成若干组，每组共享一套 K、V。在 MHA 和 MQA 之间取得平衡——既节省 KV Cache 显存，又保持模型质量。

## 从论文到现代 LLM 的演进

原始 Transformer（2017）到今天的 LLM，关键改进包括：

```
Transformer (2017)
  ├─ 位置编码: 正弦 → RoPE / ALiBi
  ├─ 归一化: Post-LN → Pre-LN + RMSNorm
  ├─ 激活函数: ReLU → GELU → SwiGLU
  ├─ 注意力: MHA → GQA + Flash Attention
  ├─ 上下文长度: 512 → 128K+ (YaRN, NTK 插值)
  └─ 规模: 6500万参数 → 数千亿参数
```

架构本身的变化其实不大——Transformer 的核心设计经受住了时间考验。更多的进步来自规模扩展（Scaling Laws）和训练方法的改进。

## 关键数字速查

| 指标 | GPT-3 | LLaMA 2-70B | GPT-4 (推测) |
|------|-------|-------------|-------------|
| 参数量 | 175B | 70B | ~1.8T (MoE) |
| 层数 | 96 | 80 | - |
| 注意力头数 | 96 | 64 | - |
| 隐藏维度 | 12288 | 8192 | - |
| 上下文长度 | 2048 | 4096 | 128K |
| 注意力类型 | MHA | GQA | GQA |

## 小结

Transformer 的核心洞察很简单：**让序列中的每个位置都能直接关注其他所有位置**。这个看似暴力的方法，配合规模扩展，产生了惊人的涌现能力。理解 Transformer 是理解整个现代 AI 技术栈的基础。

---

下一篇：[预训练与微调：从 GPT 到 LLaMA](02-pretraining-finetuning.md)
