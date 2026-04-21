# 第5-6周：Transformer 与 LLM 原理深入

> 本阶段目标：精读 Transformer 论文，理解 GPT 架构细节，掌握 LLM 训练流程（预训练→SFT→RLHF）。

---

## 1. Transformer 架构详解

### 1.1 论文背景

2017 年 Google 发表 "Attention Is All You Need"，提出 Transformer 架构，彻底取代了 RNN/LSTM 在 NLP 中的主导地位。

**核心创新：**
- 完全基于注意力机制，抛弃循环结构
- 支持并行计算，训练效率大幅提升
- 通过位置编码保留序列顺序信息

### 1.2 整体架构

```
输入 → [Encoder × N] → [Decoder × N] → 输出

Encoder 层：
  Multi-Head Attention → Add & Norm → FFN → Add & Norm

Decoder 层：
  Masked Multi-Head Attention → Add & Norm
  → Cross Attention → Add & Norm
  → FFN → Add & Norm
```

**GPT 系列只使用 Decoder 部分（Decoder-only 架构）。**

### 1.3 位置编码（Positional Encoding）

Transformer 没有循环结构，需要显式注入位置信息。

**原始正弦位置编码：**

```python
import torch
import math

def sinusoidal_position_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
    return pe

pe = sinusoidal_position_encoding(100, 512)
print(f"位置编码形状: {pe.shape}")  # [100, 512]
```

**RoPE（旋转位置编码）—— 现代 LLM 的主流选择：**

```python
def rotary_position_embedding(x, seq_len):
    """RoPE: 通过旋转矩阵编码相对位置"""
    d = x.shape[-1]
    theta = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
    positions = torch.arange(seq_len).float()
    angles = positions.unsqueeze(1) * theta.unsqueeze(0)  # [seq_len, d/2]
    
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    
    # 将 x 拆分为偶数和奇数维度
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # 旋转
    out1 = x1 * cos_vals - x2 * sin_vals
    out2 = x1 * sin_vals + x2 * cos_vals
    
    return torch.stack([out1, out2], dim=-1).flatten(-2)
```

RoPE 的优势：
- 天然编码相对位置关系
- 支持长度外推（处理比训练时更长的序列）
- LLaMA、Qwen、Mistral 等主流模型均采用

### 1.4 因果注意力（Causal Attention）

GPT 使用因果注意力掩码，确保每个 token 只能看到它之前的 token：

```python
import torch
import torch.nn.functional as F

def causal_self_attention(x, W_q, W_k, W_v):
    """
    x: [batch, seq_len, d_model]
    """
    Q, K, V = W_q(x), W_k(x), W_v(x)
    d_k = Q.size(-1)
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # 因果掩码：上三角设为 -inf
    seq_len = x.size(1)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)
```

### 1.5 Feed-Forward Network（FFN）

每个 Transformer 层中的前馈网络：

```python
import torch.nn as nn

# 标准 FFN
class FFN(nn.Module):
    def __init__(self, d_model=4096, d_ff=11008):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))

# SwiGLU FFN（LLaMA 等现代模型使用）
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model=4096, d_ff=11008):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

### 1.6 Layer Normalization

```python
# Pre-Norm（现代 LLM 主流）vs Post-Norm（原始 Transformer）

# RMSNorm（LLaMA 使用，比 LayerNorm 更高效）
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

---

## 2. GPT 架构：从 GPT-1 到 GPT-4

### 2.1 架构演进

| 模型 | 参数量 | 层数 | 隐藏维度 | 注意力头 | 训练数据 |
|------|--------|------|----------|----------|----------|
| GPT-1 | 117M | 12 | 768 | 12 | ~5GB 文本 |
| GPT-2 | 1.5B | 48 | 1600 | 25 | 40GB (WebText) |
| GPT-3 | 175B | 96 | 12288 | 96 | 570GB |
| GPT-4 | 未公开 | - | - | - | 未公开 |

### 2.2 Decoder-only 架构为何成为主流

```
Encoder-only (BERT):  双向注意力，适合理解任务（分类、NER）
Encoder-Decoder (T5): 编码-解码，适合 seq2seq 任务（翻译、摘要）
Decoder-only (GPT):   单向注意力，适合生成任务，也能做理解任务
```

Decoder-only 胜出的原因：
- 统一的生成范式：所有任务都可以转化为"续写"
- 训练效率高：每个 token 都参与损失计算
- 涌现能力：规模增大后出现 few-shot、推理等能力

### 2.3 从零实现一个 Mini GPT

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPTConfig:
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 512
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool())
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn.masked_fill_(self.mask[:T, :T], float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))
        return self.proj((attn @ v).transpose(1, 2).reshape(B, T, C))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(tok_emb + pos_emb))
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# 测试
config = MiniGPTConfig()
model = MiniGPT(config)
params = sum(p.numel() for p in model.parameters())
print(f"Mini GPT 参数量: {params / 1e6:.1f}M")
```


---

## 3. LLM 训练流程全景

现代 LLM 的训练分为三个阶段：

```
阶段1: 预训练 (Pre-training)
  大规模无标注文本 → Next Token Prediction → 基座模型 (Base Model)

阶段2: 监督微调 (SFT - Supervised Fine-Tuning)
  高质量指令数据 → 指令跟随训练 → Chat 模型

阶段3: 对齐 (Alignment)
  人类偏好数据 → RLHF / DPO → 对齐后的模型
```

### 3.1 预训练：Next Token Prediction

预训练的目标极其简单——预测下一个 token：

```python
# 预训练损失函数
def pretraining_loss(model, input_ids):
    """
    input_ids: [batch, seq_len]
    目标：用前 n-1 个 token 预测后 n-1 个 token
    """
    inputs = input_ids[:, :-1]   # [batch, seq_len-1]
    targets = input_ids[:, 1:]   # [batch, seq_len-1]
    
    logits, _ = model(inputs)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    return loss
```

**预训练数据规模参考：**

| 模型 | 训练 Token 数 | 数据来源 |
|------|--------------|----------|
| LLaMA-2 7B | 2T tokens | 公开网页、代码、书籍 |
| Qwen-2 7B | 7T tokens | 多语言网页、代码 |
| Mistral 7B | 未公开 | 未公开 |

### 3.2 Scaling Laws（缩放定律）

OpenAI 的研究发现，模型性能与三个因素呈幂律关系：

```
Loss ∝ 1/N^α × 1/D^β × 1/C^γ

N = 模型参数量
D = 训练数据量
C = 计算量 (FLOPs)
```

**Chinchilla 定律（DeepMind）：**
- 最优训练策略：训练 token 数 ≈ 20 × 参数量
- 7B 模型最优训练量 ≈ 140B tokens
- 实际上现在的模型都远超这个比例（over-training）

### 3.3 SFT（监督微调）

将基座模型训练为能够遵循指令的 Chat 模型：

```python
# SFT 数据格式示例（Alpaca 格式）
sft_example = {
    "instruction": "将以下英文翻译为中文",
    "input": "Large language models are transforming AI.",
    "output": "大语言模型正在改变人工智能。"
}

# ShareGPT 多轮对话格式
sharegpt_example = {
    "conversations": [
        {"from": "human", "value": "什么是机器学习？"},
        {"from": "gpt", "value": "机器学习是人工智能的一个分支..."},
        {"from": "human", "value": "它和深度学习有什么区别？"},
        {"from": "gpt", "value": "深度学习是机器学习的一个子集..."}
    ]
}
```

**SFT 的关键：只对 assistant 回复部分计算损失**

```python
def sft_loss(model, input_ids, labels):
    """
    labels 中，用户输入部分标记为 -100（忽略），
    只对模型回复部分计算交叉熵损失
    """
    logits, _ = model(input_ids)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100
    )
    return loss
```

### 3.4 RLHF（基于人类反馈的强化学习）

```
步骤1: 收集人类偏好数据
  对同一个问题，模型生成多个回答，人类标注哪个更好

步骤2: 训练 Reward Model
  学习人类偏好，给回答打分

步骤3: PPO 优化
  用 Reward Model 的分数作为奖励，通过 PPO 算法优化策略模型
```

**Reward Model 训练：**

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids):
        hidden = self.base(input_ids).last_hidden_state
        reward = self.reward_head(hidden[:, -1, :])  # 取最后一个 token
        return reward

# 偏好损失：chosen 的奖励应高于 rejected
def preference_loss(reward_chosen, reward_rejected):
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

### 3.5 DPO（Direct Preference Optimization）

DPO 是 RLHF 的简化替代方案，无需训练 Reward Model：

```python
def dpo_loss(policy_model, ref_model, chosen_ids, rejected_ids, beta=0.1):
    """
    直接从偏好数据优化策略，无需 Reward Model
    """
    # 计算策略模型的 log prob
    pi_chosen = get_log_prob(policy_model, chosen_ids)
    pi_rejected = get_log_prob(policy_model, rejected_ids)
    
    # 计算参考模型的 log prob
    with torch.no_grad():
        ref_chosen = get_log_prob(ref_model, chosen_ids)
        ref_rejected = get_log_prob(ref_model, rejected_ids)
    
    # DPO 损失
    logits = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    loss = -F.logsigmoid(logits).mean()
    return loss
```

**DPO vs RLHF：**

| 方面 | RLHF | DPO |
|------|------|-----|
| 复杂度 | 高（需要 RM + PPO） | 低（直接优化） |
| 稳定性 | PPO 训练不稳定 | 更稳定 |
| 效果 | 经过验证，效果好 | 接近 RLHF |
| 资源需求 | 需要多个模型同时在显存 | 只需策略+参考模型 |

---

## 4. 关键概念补充

### 4.1 KV Cache

推理加速的核心技术——缓存已计算的 Key 和 Value：

```python
# 无 KV Cache：每生成一个 token 都要重新计算所有位置的 K, V
# 有 KV Cache：缓存之前的 K, V，只计算新 token 的

class AttentionWithKVCache(nn.Module):
    def forward(self, x, kv_cache=None):
        q, k, v = self.compute_qkv(x)
        
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=-2)
            v = torch.cat([prev_v, v], dim=-2)
        
        new_cache = (k, v)
        output = self.attention(q, k, v)
        return output, new_cache
```

### 4.2 GQA（Grouped Query Attention）

减少 KV Cache 显存占用：

```
MHA:  每个 head 都有独立的 Q, K, V
MQA:  所有 head 共享一组 K, V
GQA:  将 head 分组，每组共享 K, V（折中方案）
```

LLaMA-2 70B、Qwen-2 等模型使用 GQA。

### 4.3 模型参数量估算

```python
def estimate_params(vocab_size, d_model, n_layers, d_ff):
    embedding = vocab_size * d_model
    per_layer = (
        4 * d_model * d_model +   # QKV + Output projection
        3 * d_model * d_ff +       # SwiGLU FFN (gate, up, down)
        2 * d_model                # 2x RMSNorm
    )
    total = embedding + n_layers * per_layer + vocab_size * d_model  # + lm_head
    return total

# LLaMA-2 7B 估算
params = estimate_params(32000, 4096, 32, 11008)
print(f"估算参数量: {params / 1e9:.1f}B")  # ≈ 6.7B
```

---

## 6. 补充：Flash Attention 原理

Flash Attention 是推理和训练加速的关键技术，通过 IO 感知的算法设计大幅减少显存访问。

### 6.1 标准 Attention 的瓶颈

```
标准 Attention 流程：
1. 计算 S = Q @ K^T          → 写入 HBM（显存）
2. 计算 P = softmax(S)       → 写入 HBM
3. 计算 O = P @ V            → 写入 HBM

问题：S 和 P 的大小为 [seq_len × seq_len]，序列长时显存爆炸
例如 seq_len=8192 → S 矩阵占 256MB (FP16)
```

### 6.2 Flash Attention 核心思想

```
关键洞察：GPU 计算很快，但显存读写（IO）是瓶颈

Flash Attention 策略：
- 将 Q, K, V 分块（tiling），每次只处理一小块
- 在 SRAM（片上高速缓存）中完成 softmax 计算
- 永远不在显存中存储完整的 S 和 P 矩阵
- 使用 online softmax 技巧实现分块 softmax

结果：
- 显存从 O(N²) 降到 O(N)
- 速度提升 2-4x（减少 HBM 访问）
```

### 6.3 使用 Flash Attention

```python
# PyTorch 2.0+ 内置支持
import torch
import torch.nn.functional as F

q = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.float16)

# 自动使用 Flash Attention（PyTorch 2.0+）
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# 在 HuggingFace 模型中启用
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",  # 显式启用
    device_map="auto"
)
```

---

## 7. 补充：Speculative Decoding（投机解码）

自回归生成的瓶颈是每次只能生成一个 token。Speculative Decoding 通过"猜测+验证"打破这个限制。

### 7.1 核心思想

```
传统解码：大模型逐 token 生成，每步都要完整前向传播

Speculative Decoding：
1. 用小模型（draft model）快速生成 K 个候选 token
2. 用大模型（target model）一次性验证这 K 个 token
3. 接受匹配的 token，从第一个不匹配处重新生成

关键：大模型验证 K 个 token 的成本 ≈ 生成 1 个 token（并行计算）
```

### 7.2 简化实现

```python
import torch

def speculative_decode(target_model, draft_model, tokenizer, prompt, K=5, max_tokens=100):
    """Speculative Decoding 简化实现"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(target_model.device)
    generated = input_ids.clone()
    
    for _ in range(max_tokens // K):
        # 1. Draft model 快速生成 K 个 token
        draft_ids = generated.clone()
        for _ in range(K):
            with torch.no_grad():
                logits = draft_model(draft_ids).logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            draft_ids = torch.cat([draft_ids, next_token], dim=1)
        
        draft_tokens = draft_ids[:, generated.shape[1]:]  # K 个候选 token
        
        # 2. Target model 一次性验证
        with torch.no_grad():
            target_logits = target_model(draft_ids).logits
        
        # 3. 逐个比较，找到第一个不匹配的位置
        n_accepted = 0
        for i in range(K):
            pos = generated.shape[1] + i - 1
            target_token = target_logits[:, pos, :].argmax(dim=-1)
            if target_token.item() == draft_tokens[:, i].item():
                n_accepted += 1
            else:
                # 用 target model 的预测替换
                generated = torch.cat([generated, target_token.unsqueeze(0)], dim=1)
                break
        else:
            # 全部接受
            generated = torch.cat([generated, draft_tokens], dim=1)
        
        if tokenizer.eos_token_id in generated[0, input_ids.shape[1]:]:
            break
    
    return tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
```

### 7.3 加速效果

```
Draft Model: Qwen2-0.5B, Target Model: Qwen2-7B
接受率 ~70-80% 时，加速比约 2-3x

适用场景：
- 代码生成（token 可预测性高，接受率高）
- 翻译任务
- 不适合创意写作（随机性大，接受率低）
```

> **延伸阅读**：[Leviathan et al. - Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

---

## 8. 学习检查清单

- [ ] 能画出 Transformer 的完整架构图
- [ ] 理解因果注意力掩码的作用
- [ ] 了解 RoPE 位置编码的优势
- [ ] 能解释 Pre-training → SFT → RLHF/DPO 三阶段流程
- [ ] 理解 Scaling Laws 的基本含义
- [ ] 了解 KV Cache 和 GQA 的作用
- [ ] 能从零实现一个简单的 GPT 模型
- [ ] 理解 Flash Attention 为什么能加速（IO 感知）
- [ ] 了解 Speculative Decoding 的"猜测+验证"思路

---

## 9. 推荐学习材料

### 必读论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) - GPT-3 论文
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - LLaMA 论文
- [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) - RLHF 论文
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) - DPO 论文
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - 缩放定律

### 视频
- [Andrej Karpathy - Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) - 2小时从零实现 GPT
- [Andrej Karpathy - Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU) - 复现 GPT-2
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) - NLP 经典课程
- [李宏毅 - Transformer 讲解](https://www.youtube.com/watch?v=ugWDIIOHtPA) - 中文讲解

### 博客文章
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 图解 Transformer
- [Lilian Weng - The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) - Transformer 变体综述
- [Lilian Weng - LLM Training](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) - LLM 训练详解
- [Chip Huyen - RLHF](https://huyenchip.com/2023/05/02/rlhf.html) - RLHF 详解

### 代码实践
- [nanoGPT - Andrej Karpathy](https://github.com/karpathy/nanoGPT) - 最简 GPT 实现
- [minGPT - Andrej Karpathy](https://github.com/karpathy/minGPT) - 教学用 GPT 实现
- [llm.c - Andrej Karpathy](https://github.com/karpathy/llm.c) - 纯 C 实现 GPT-2 训练

---

## 10. 下一步

理解了 Transformer 和 LLM 的原理后，下一阶段我们将进入 [开源模型生态](week07-08-opensource-ecosystem.md)，学习如何使用 Hugging Face 生态系统加载、推理和部署开源模型。