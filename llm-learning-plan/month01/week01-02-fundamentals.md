# 第1-2周：NLP 基础与大模型入门

> 本阶段目标：理解 NLP 核心概念，掌握 Tokenization、Embedding、Attention 机制，搭建好开发环境。

---

## 1. 为什么要学习大模型？

大语言模型（Large Language Model, LLM）是当前 AI 领域最重要的突破之一。从 GPT-4 到 Claude，从 LLaMA 到 Qwen，这些模型正在改变软件开发、内容创作、科学研究等各个领域。

学习大模型，你将获得：
- 理解当前最前沿的 AI 技术
- 具备开发 AI 应用的能力
- 掌握模型训练和优化的核心技能

---

## 2. NLP 基础概念

### 2.1 什么是自然语言处理（NLP）

NLP 是让计算机理解、生成和处理人类语言的技术。大模型是 NLP 发展的最新阶段。

**NLP 发展简史：**
```
规则方法 → 统计方法 → 深度学习 → 预训练模型 → 大语言模型
(1950s)    (1990s)    (2013)      (2018 BERT)   (2020 GPT-3)
```

### 2.2 Tokenization（分词）

Tokenization 是将文本转换为模型可处理的数字序列的过程。

**主流分词方法：**

| 方法 | 说明 | 代表 |
|------|------|------|
| Word-level | 按单词切分 | 早期 NLP |
| Character-level | 按字符切分 | 中文处理 |
| Subword (BPE) | 子词切分，平衡词表大小和覆盖率 | GPT 系列 |
| SentencePiece | 语言无关的子词分词 | LLaMA、Qwen |

**代码示例：**

```python
from transformers import AutoTokenizer

# 加载 GPT-2 的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "大语言模型正在改变世界"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"原文: {text}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")

# 查看词表大小
print(f"词表大小: {tokenizer.vocab_size}")
```

**关键理解：**
- Token 不等于单词，一个单词可能被拆成多个 token
- 中文通常每个字对应 1-2 个 token
- 词表大小影响模型参数量和表达能力

### 2.3 Embedding（词嵌入）

Embedding 将离散的 token 映射为连续的向量空间，使得语义相近的词在向量空间中距离更近。

```python
import torch
import torch.nn as nn

# 简单的 Embedding 层
vocab_size = 50000
embedding_dim = 768

embedding = nn.Embedding(vocab_size, embedding_dim)

# 输入 token id，输出向量
token_id = torch.tensor([1024])
vector = embedding(token_id)
print(f"Token 1024 的向量维度: {vector.shape}")  # [1, 768]
```

**Word2Vec 到 Contextual Embedding 的演进：**
- Word2Vec / GloVe：静态词向量，一个词只有一个向量
- ELMo：根据上下文动态生成词向量
- BERT / GPT：通过 Transformer 生成深层上下文表示

### 2.4 Attention 机制

Attention 是 Transformer 的核心，让模型能够"关注"输入序列中最相关的部分。

**Self-Attention 的直觉理解：**

当模型处理"小猫坐在垫子上，它很舒服"这句话时，"它"需要关注"小猫"才能理解指代关系。Attention 机制就是让每个 token 能够"看到"并"关注"序列中的其他 token。

**数学公式：**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**代码实现：**

```python
import torch
import torch.nn.functional as F

def self_attention(x, d_model=512):
    """
    x: [batch_size, seq_len, d_model]
    """
    # 线性变换得到 Q, K, V
    W_q = torch.nn.Linear(d_model, d_model)
    W_k = torch.nn.Linear(d_model, d_model)
    W_v = torch.nn.Linear(d_model, d_model)
    
    Q = W_q(x)  # [batch, seq_len, d_model]
    K = W_k(x)
    V = W_v(x)
    
    # 计算注意力分数
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Softmax 归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

**Multi-Head Attention（多头注意力）：**

将注意力分成多个"头"，每个头关注不同的语义子空间：

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 线性变换并拆分为多头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(output)
```

---

## 3. 环境搭建

### 3.1 Python 环境

```bash
# 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建虚拟环境
conda create -n llm python=3.11
conda activate llm
```

### 3.2 PyTorch 安装

```bash
# CUDA 12.1 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证安装
python -c "import torch; print(torch.cuda.is_available())"
```

### 3.3 核心库安装

```bash
pip install transformers datasets tokenizers accelerate
pip install jupyter notebook ipywidgets
pip install numpy pandas matplotlib
```

### 3.4 GPU 环境选择

| 平台 | 价格 | 适合阶段 |
|------|------|----------|
| [Google Colab](https://colab.research.google.com/) | 免费 / Pro $10/月 | 学习、小实验 |
| [AutoDL](https://www.autodl.com/) | ¥1-5/小时 | 微调、训练 |
| [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud) | $1.1/小时 (A100) | 大规模训练 |
| 本地 RTX 4090 | 一次性 ¥14000+ | 长期使用 |

### 3.5 验证环境

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# 测试加载一个小模型
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
print(f"GPT-2 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
```

---

## 4. 动手练习

### 练习 1：探索不同 Tokenizer

```python
from transformers import AutoTokenizer

models = ["gpt2", "meta-llama/Llama-2-7b-hf", "Qwen/Qwen2-7B"]
text = "Hello, 大语言模型！Let's learn about LLMs."

for model_name in models:
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        tokens = tok.tokenize(text)
        print(f"\n{model_name}:")
        print(f"  Token 数量: {len(tokens)}")
        print(f"  Tokens: {tokens[:10]}...")
    except Exception as e:
        print(f"  {model_name}: 需要登录 HuggingFace 获取权限")
```

### 练习 2：可视化 Attention

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

text = "The cat sat on the mat because it was comfortable"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# 获取最后一层的注意力权重
attention = outputs.attentions[-1]  # [batch, heads, seq_len, seq_len]
print(f"Attention shape: {attention.shape}")

# 可视化第一个 head 的注意力
import matplotlib.pyplot as plt

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
attn_matrix = attention[0, 0].detach().numpy()

plt.figure(figsize=(10, 10))
plt.imshow(attn_matrix, cmap="viridis")
plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.yticks(range(len(tokens)), tokens)
plt.colorbar()
plt.title("Attention Weights (Head 0, Last Layer)")
plt.tight_layout()
plt.savefig("attention_visualization.png")
plt.show()
```

### 练习 3：从零实现简单的 Bigram 语言模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        emb = self.embedding(x)  # [batch, seq_len, embed_dim]
        logits = self.lm_head(emb)  # [batch, seq_len, vocab_size]
        return logits
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :]  # 取最后一个位置
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

# 使用示例
vocab_size = 256  # 字符级别
model = BigramLM(vocab_size)
start = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(start, max_new_tokens=20)
print(f"生成的 token ids: {generated[0].tolist()}")
```

---

## 5. 本周学习检查清单

- [ ] 能解释 Tokenization 的作用和 BPE 算法的基本思想
- [ ] 理解 Embedding 如何将离散 token 映射为连续向量
- [ ] 能手写 Self-Attention 的计算过程
- [ ] 理解 Multi-Head Attention 的意义
- [ ] 成功搭建 Python + PyTorch + Transformers 环境
- [ ] 能加载并运行 GPT-2 模型
- [ ] 完成至少 2 个动手练习

---

## 6. 推荐学习材料

### 视频
- [3Blue1Brown - But what is a GPT? Visual intro to Transformers](https://www.youtube.com/watch?v=wjZofJX0v4M) - 最直观的 Transformer 可视化讲解
- [3Blue1Brown - Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) - Attention 机制可视化
- [Andrej Karpathy - Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) - 从零实现 GPT，强烈推荐
- [李宏毅 - 2024 生成式 AI 导论](https://www.youtube.com/playlist?list=PLJV_el3uVTsPz6CTopeRp2L2t4aL_KgiI) - 中文讲解，通俗易懂

### 文章
- [The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/) - 图解 Transformer，经典必读
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - 图解 GPT-2 架构
- [Attention Is All You Need 论文](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) - 官方 NLP 入门课程

### 书籍
- [动手学深度学习 (d2l.ai)](https://d2l.ai/) - 第10-11章 注意力机制与 Transformer
- [Speech and Language Processing - Jurafsky](https://web.stanford.edu/~jurafsky/slp3/) - NLP 经典教材，免费在线

### 工具
- [Tokenizer Playground](https://platform.openai.com/tokenizer) - OpenAI 在线 Tokenizer 可视化
- [Google Colab](https://colab.research.google.com/) - 免费 GPU 环境

---

## 7. 补充：BPE 算法详细实现

BPE（Byte Pair Encoding）是现代 LLM 最常用的分词算法。理解其原理对调试 tokenization 问题至关重要。

### 7.1 BPE 算法步骤

```
1. 将所有单词拆分为字符序列，统计词频
2. 统计所有相邻字符对的出现频率
3. 合并频率最高的字符对，生成新 token
4. 重复步骤 2-3，直到达到目标词表大小
```

### 7.2 从零实现 BPE

```python
from collections import Counter, defaultdict

class SimpleBPE:
    """从零实现 BPE 分词器"""
    
    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.merges = []  # 合并规则列表
        self.vocab = {}
    
    def _get_pair_counts(self, word_freqs):
        """统计所有相邻 token 对的频率"""
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def _merge_pair(self, pair, word_freqs):
        """将指定的 token 对合并"""
        new_word_freqs = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        return new_word_freqs
    
    def train(self, text):
        """训练 BPE 分词器"""
        # 统计词频，每个单词拆分为字符
        words = text.split()
        word_counts = Counter(words)
        word_freqs = {}
        for word, count in word_counts.items():
            # 用空格分隔每个字符，末尾加 </w> 标记词边界
            word_freqs[" ".join(list(word)) + " </w>"] = count
        
        # 初始词表 = 所有字符 + </w>
        self.vocab = set()
        for word in word_freqs:
            for token in word.split():
                self.vocab.add(token)
        
        # 迭代合并
        while len(self.vocab) < self.vocab_size:
            pairs = self._get_pair_counts(word_freqs)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            self.merges.append(best_pair)
            self.vocab.add("".join(best_pair))
            word_freqs = self._merge_pair(best_pair, word_freqs)
            
            if len(self.merges) % 50 == 0:
                print(f"合并 {len(self.merges)} 次, 词表大小: {len(self.vocab)}")
        
        print(f"训练完成! 词表大小: {len(self.vocab)}, 合并规则: {len(self.merges)}")
    
    def tokenize(self, word):
        """使用学到的合并规则对单词进行分词"""
        tokens = list(word) + ["</w>"]
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens = tokens[:i] + ["".join(pair)] + tokens[i + 2:]
                else:
                    i += 1
        return tokens

# 使用示例
corpus = "low low low low low lower lower newest newest newest widest widest"
bpe = SimpleBPE(vocab_size=30)
bpe.train(corpus)

print("\n分词结果:")
for word in ["low", "lower", "newest", "widest"]:
    print(f"  {word} → {bpe.tokenize(word)}")
print(f"\n合并规则: {bpe.merges[:10]}")
```

### 7.3 使用 HuggingFace tokenizers 库训练自定义 Tokenizer

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def train_bpe_tokenizer(files, vocab_size=32000, save_path="my_tokenizer.json"):
    """训练一个生产级 BPE 分词器"""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        min_frequency=2,
        show_progress=True,
    )
    
    tokenizer.train(files, trainer)
    tokenizer.save(save_path)
    print(f"词表大小: {tokenizer.get_vocab_size()}")
    
    # 测试
    test = "大语言模型正在改变世界 Hello World!"
    encoded = tokenizer.encode(test)
    print(f"原文: {test}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")
    return tokenizer

# 准备训练语料（保存为文本文件）
with open("corpus.txt", "w") as f:
    f.write("这里放入你的训练语料...\n" * 1000)

tokenizer = train_bpe_tokenizer(["corpus.txt"], vocab_size=8000)
```

### 7.4 对比不同 Tokenizer 的效率

```python
from transformers import AutoTokenizer

text_zh = "大语言模型通过海量数据训练，能够理解和生成自然语言。"
text_en = "Large language models are trained on massive datasets to understand and generate natural language."

for name in ["gpt2", "Qwen/Qwen2-7B", "meta-llama/Llama-2-7b-hf"]:
    try:
        tok = AutoTokenizer.from_pretrained(name)
        zh_tokens = tok.encode(text_zh)
        en_tokens = tok.encode(text_en)
        print(f"{name:40s} | 中文: {len(zh_tokens):3d} tokens | 英文: {len(en_tokens):3d} tokens | 词表: {tok.vocab_size}")
    except Exception:
        print(f"{name:40s} | 需要权限")
```

> **关键洞察**：词表设计直接影响模型的多语言效率。Qwen 的词表对中文更友好（每个汉字约 1 token），而 LLaMA-2 的词表偏向英文（中文每字可能 2-3 tokens）。

---

## 8. 下一步

完成本阶段学习后，你应该对 NLP 的基础概念有了清晰的理解。下一阶段我们将进入 [应用开发实战](week03-04-app-development.md)，学习如何调用大模型 API 构建实际应用。
