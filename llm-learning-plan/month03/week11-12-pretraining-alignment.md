# 第11-12周：完整训练流程与对齐

> 本阶段目标：掌握预训练数据处理、分布式训练（DeepSpeed/FSDP）、对齐训练（SFT→DPO）全流程，完成毕业项目。

---

## 1. 预训练数据处理

### 1.1 数据处理 Pipeline

```
原始数据采集 → 去重 → 质量过滤 → 敏感内容过滤 → Tokenization → 打包成训练格式
```

### 1.2 数据去重

```python
from datasketch import MinHash, MinHashLSH

def minhash_dedup(documents, threshold=0.8, num_perm=128):
    """基于 MinHash LSH 的近似去重"""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_docs = []
    
    for i, doc in enumerate(documents):
        mh = MinHash(num_perm=num_perm)
        # 用 n-gram 生成 MinHash
        for ngram in [doc[j:j+5] for j in range(len(doc)-4)]:
            mh.update(ngram.encode("utf-8"))
        
        # 查询是否有近似重复
        if not lsh.query(mh):
            lsh.insert(f"doc_{i}", mh)
            unique_docs.append(doc)
    
    print(f"去重前: {len(documents)}, 去重后: {len(unique_docs)}")
    return unique_docs
```

### 1.3 数据质量过滤

```python
import re

def quality_filter(text, min_length=100, max_length=100000):
    """基础质量过滤规则"""
    # 长度过滤
    if len(text) < min_length or len(text) > max_length:
        return False
    
    # 特殊字符比例
    special_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
    if special_ratio > 0.3:
        return False
    
    # 重复行比例
    lines = text.split('\n')
    unique_lines = set(lines)
    if len(unique_lines) / max(len(lines), 1) < 0.5:
        return False
    
    # 语言检测（可选）
    # 确保主要是目标语言
    
    return True
```

### 1.4 Tokenization 与数据打包

```python
from transformers import AutoTokenizer
import numpy as np

def tokenize_and_pack(texts, tokenizer, max_length=2048):
    """将文本 tokenize 并打包成固定长度的训练样本"""
    all_token_ids = []
    
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        all_token_ids.extend(ids)
    
    # 打包成固定长度
    total = len(all_token_ids)
    n_samples = total // max_length
    packed = np.array(all_token_ids[:n_samples * max_length]).reshape(n_samples, max_length)
    
    print(f"总 token 数: {total:,}, 生成 {n_samples} 个训练样本")
    return packed
```

### 1.5 数据配比

预训练数据通常包含多种来源，需要合理配比：

```python
data_mix = {
    "web_text": 0.50,      # 网页文本
    "books": 0.15,          # 书籍
    "code": 0.15,           # 代码
    "academic": 0.10,       # 学术论文
    "conversation": 0.05,   # 对话数据
    "math": 0.05,           # 数学数据
}
```

---

## 2. 分布式训练

### 2.1 为什么需要分布式训练？

```
7B 模型训练显存需求估算：
- 模型参数 (FP16): 14GB
- 梯度 (FP16): 14GB
- 优化器状态 (Adam, FP32): 56GB
- 激活值: 取决于 batch size 和序列长度
总计: ~100GB+ → 单卡放不下
```

### 2.2 分布式训练策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| Data Parallel (DP) | 每张卡完整模型，数据分片 | 模型能放进单卡 |
| ZeRO (DeepSpeed) | 分片优化器/梯度/参数 | 最常用 |
| FSDP (PyTorch) | 类似 ZeRO-3，PyTorch 原生 | PyTorch 生态 |
| Tensor Parallel (TP) | 将单层拆分到多卡 | 超大模型 |
| Pipeline Parallel (PP) | 不同层放不同卡 | 超大模型 |

### 2.3 DeepSpeed ZeRO

```python
# deepspeed_config.json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8
    },
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

**ZeRO 三个阶段：**

```
ZeRO-1: 分片优化器状态 → 节省 ~4x 显存
ZeRO-2: + 分片梯度 → 节省 ~8x 显存
ZeRO-3: + 分片模型参数 → 节省 ~N 倍显存（N=GPU数）
```

### 2.4 使用 DeepSpeed 训练

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps=1000,
    bf16=True,
    logging_steps=10,
    save_steps=1000,
    deepspeed="deepspeed_config.json",  # 启用 DeepSpeed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

```bash
# 多卡启动
deepspeed --num_gpus=4 train.py
# 或
torchrun --nproc_per_node=4 train.py
```

### 2.5 FSDP（PyTorch 原生方案）

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Hugging Face Trainer 中使用 FSDP
training_args = TrainingArguments(
    output_dir="./output",
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
    },
    bf16=True,
    # ... 其他参数
)
```

### 2.6 训练监控

```python
# 使用 Weights & Biases 监控
# pip install wandb
import wandb

wandb.init(project="llm-pretraining", name="7b-run-1")

training_args = TrainingArguments(
    report_to="wandb",
    logging_steps=10,
    # ...
)

# 关键监控指标：
# - train/loss: 训练损失，应持续下降
# - train/learning_rate: 学习率变化
# - train/grad_norm: 梯度范数，过大说明不稳定
# - system/gpu_memory: GPU 显存使用
```


---

## 3. 对齐训练全流程

### 3.1 完整对齐 Pipeline

```
Base Model → SFT → DPO/RLHF → 对齐后的 Chat Model

阶段1 SFT: 学会遵循指令
阶段2 DPO: 学会人类偏好（安全、有帮助、诚实）
```

### 3.2 SFT 训练（使用 TRL 库）

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")

def format_chat(example):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = dataset.map(format_chat)

sft_config = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    max_seq_length=2048,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./sft_model")
```

### 3.3 构建偏好数据

DPO 需要偏好对数据（chosen vs rejected）：

```python
# 偏好数据格式
preference_example = {
    "prompt": "如何学习编程？",
    "chosen": "学习编程建议从 Python 入手，它语法简洁适合初学者。可以通过以下步骤：\n1. 学习基础语法\n2. 做小项目练手\n3. 阅读优秀代码\n4. 参与开源项目",
    "rejected": "编程很简单，随便学学就行了。"
}
```

**构建偏好数据的方法：**

```python
def generate_preference_pairs(model, tokenizer, prompts, n_samples=2):
    """为每个 prompt 生成多个回答，人工标注偏好"""
    pairs = []
    for prompt in prompts:
        responses = []
        for _ in range(n_samples):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.9)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(response)
        
        pairs.append({
            "prompt": prompt,
            "response_a": responses[0],
            "response_b": responses[1],
            "preference": None  # 需要人工标注
        })
    return pairs
```

### 3.4 DPO 训练

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 加载 SFT 后的模型作为策略模型和参考模型
model = AutoModelForCausalLM.from_pretrained("./sft_model", torch_dtype=torch.bfloat16)
ref_model = AutoModelForCausalLM.from_pretrained("./sft_model", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("./sft_model")

# 加载偏好数据
dataset = load_dataset("json", data_files="preference_data.jsonl", split="train")

# DPO 数据需要 prompt, chosen, rejected 三个字段
def format_dpo(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

dataset = dataset.map(format_dpo)

dpo_config = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,       # DPO 学习率要比 SFT 小很多
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    beta=0.1,                 # DPO 温度参数
    max_length=2048,
    max_prompt_length=1024,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./dpo_model")
```

### 3.5 RLHF（PPO）训练（进阶）

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# PPO 需要一个带 Value Head 的模型
model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft_model")

# 还需要一个 Reward Model
reward_model = AutoModelForSequenceClassification.from_pretrained("./reward_model")

ppo_config = PPOConfig(
    learning_rate=1e-6,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
    kl_penalty="kl",
    init_kl_coef=0.2,  # KL 散度惩罚系数
)

ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    tokenizer=tokenizer,
)

# PPO 训练循环
for batch in dataloader:
    # 1. 生成回答
    response_tensors = ppo_trainer.generate(batch["input_ids"], max_new_tokens=256)
    
    # 2. 计算奖励
    rewards = reward_model(response_tensors)
    
    # 3. PPO 更新
    stats = ppo_trainer.step(batch["input_ids"], response_tensors, rewards)
```

---

## 4. 模型评估

### 4.1 自动评估 Benchmark

| Benchmark | 评估能力 | 说明 |
|-----------|----------|------|
| MMLU | 知识广度 | 57 个学科的选择题 |
| C-Eval | 中文知识 | 中文多学科评估 |
| HumanEval | 代码能力 | Python 编程题 |
| GSM8K | 数学推理 | 小学数学应用题 |
| HellaSwag | 常识推理 | 句子补全 |
| TruthfulQA | 真实性 | 检测模型是否编造信息 |

### 4.2 使用 lm-evaluation-harness 评估

```bash
# 安装
pip install lm-eval

# 评估
lm_eval --model hf \
    --model_args pretrained=./dpo_model,dtype=float16 \
    --tasks mmlu,gsm8k,hellaswag \
    --batch_size 8 \
    --output_path ./eval_results
```

### 4.3 简单评估脚本

```python
import json

def evaluate_qa(model, tokenizer, test_data):
    """简单的问答评估"""
    correct = 0
    total = len(test_data)
    
    for item in test_data:
        messages = [{"role": "user", "content": item["question"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        if item["answer"].lower() in response.lower():
            correct += 1
    
    accuracy = correct / total
    print(f"准确率: {accuracy:.2%} ({correct}/{total})")
    return accuracy
```

### 4.4 人工评估

自动指标无法完全反映模型质量，人工评估仍然重要：

```
评估维度：
1. 有帮助性 (Helpfulness): 回答是否解决了用户的问题
2. 准确性 (Accuracy): 信息是否正确
3. 安全性 (Safety): 是否拒绝有害请求
4. 流畅性 (Fluency): 语言是否自然流畅
5. 格式 (Format): 输出格式是否符合要求
```

---

## 5. 毕业项目：领域模型完整训练

### 5.1 项目目标

从数据准备到 SFT + DPO，完成一个领域模型的完整训练流程。

### 5.2 项目结构

```
domain-llm-project/
├── data/
│   ├── raw/                 # 原始数据
│   ├── processed/           # 处理后的数据
│   ├── sft_data.jsonl       # SFT 训练数据
│   └── preference_data.jsonl # DPO 偏好数据
├── scripts/
│   ├── prepare_data.py      # 数据处理
│   ├── train_sft.py         # SFT 训练
│   ├── train_dpo.py         # DPO 训练
│   └── evaluate.py          # 评估
├── configs/
│   ├── sft_config.yaml
│   └── dpo_config.yaml
└── README.md
```

### 5.3 数据准备脚本

```python
"""scripts/prepare_data.py"""
import json, random

def prepare_sft_data(raw_path, output_path, val_ratio=0.1):
    """准备 SFT 数据"""
    with open(raw_path) as f:
        data = [json.loads(line) for line in f]
    
    # 清洗
    cleaned = [d for d in data if len(d["output"]) > 20 and len(d["instruction"]) > 5]
    
    # 打乱并划分
    random.shuffle(cleaned)
    split = int(len(cleaned) * (1 - val_ratio))
    train_data, val_data = cleaned[:split], cleaned[split:]
    
    for name, subset in [("train", train_data), ("val", val_data)]:
        path = output_path.replace(".jsonl", f"_{name}.jsonl")
        with open(path, "w") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{name}: {len(subset)} 条")

if __name__ == "__main__":
    prepare_sft_data("data/raw/domain_qa.jsonl", "data/sft_data.jsonl")
```

### 5.4 SFT 训练脚本

```python
"""scripts/train_sft.py"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import LoraConfig

# 配置
BASE_MODEL = "Qwen/Qwen2-7B-Instruct"
OUTPUT_DIR = "./output/sft"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)

dataset = load_dataset("json", data_files={
    "train": "data/sft_data_train.jsonl",
    "validation": "data/sft_data_val.jsonl"
})

def format_fn(example):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = dataset.map(format_fn)

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUTPUT_DIR, num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, bf16=True, logging_steps=10,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, max_seq_length=2048,
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
print("SFT 训练完成！")
```

### 5.5 DPO 训练脚本

```python
"""scripts/train_dpo.py"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import PeftModel
from datasets import load_dataset

SFT_MODEL = "./output/sft/final"
BASE_MODEL = "Qwen/Qwen2-7B-Instruct"
OUTPUT_DIR = "./output/dpo"

tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL)

# 加载 SFT 模型
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, SFT_MODEL)
model = model.merge_and_unload()

ref_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
ref_model = PeftModel.from_pretrained(ref_model, SFT_MODEL).merge_and_unload()

dataset = load_dataset("json", data_files="data/preference_data.jsonl", split="train")

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=DPOConfig(
        output_dir=OUTPUT_DIR, num_train_epochs=1,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=5e-7, beta=0.1, bf16=True,
        logging_steps=10, save_strategy="epoch",
        max_length=2048, max_prompt_length=1024,
    ),
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
print("DPO 训练完成！")
```

### 5.6 评估脚本

```python
"""scripts/evaluate.py"""
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_and_evaluate(model_path, test_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    
    with open(test_path) as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"评估 {model_path}，共 {len(test_data)} 个测试样本\n")
    
    for item in test_data[:10]:  # 展示前10个
        messages = [{"role": "user", "content": item["instruction"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"Q: {item['instruction']}")
        print(f"A: {response}")
        print(f"Expected: {item['output'][:100]}...")
        print("=" * 60)

if __name__ == "__main__":
    # 对比 SFT 和 DPO 模型
    for path in ["./output/sft/final", "./output/dpo/final"]:
        load_and_evaluate(path, "data/sft_data_val.jsonl")
```


---

## 6. 补充：GRPO/ORPO 等新对齐方法

DPO 之后，社区涌现了更多简化对齐的方法。

### 6.1 ORPO（Odds Ratio Preference Optimization）

ORPO 的核心创新：将 SFT 和偏好对齐合并为一步训练，无需单独的 SFT 阶段。

```python
from trl import ORPOTrainer, ORPOConfig

# ORPO 数据格式与 DPO 相同：prompt + chosen + rejected
orpo_config = ORPOConfig(
    output_dir="./orpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    beta=0.1,           # odds ratio 权重
    bf16=True,
    max_length=2048,
    max_prompt_length=1024,
    logging_steps=10,
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

**ORPO vs DPO：**
- 不需要参考模型（ref_model），节省显存
- 不需要单独的 SFT 阶段
- 一步完成 SFT + 对齐

### 6.2 GRPO（Group Relative Policy Optimization）

DeepSeek 提出的方法，用于 DeepSeek-R1 的训练。核心思想：对同一 prompt 生成一组回答，用组内相对奖励替代绝对奖励。

```python
# GRPO 简化伪代码
def grpo_step(model, prompts, reward_fn, group_size=8):
    """
    1. 对每个 prompt 生成 group_size 个回答
    2. 用 reward function 打分
    3. 组内归一化奖励（减均值除标准差）
    4. 用归一化后的奖励做策略梯度更新
    """
    all_losses = []
    for prompt in prompts:
        # 生成一组回答
        responses = [model.generate(prompt) for _ in range(group_size)]
        rewards = [reward_fn(prompt, r) for r in responses]
        
        # 组内归一化
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r)**2 for r in rewards) / len(rewards)) ** 0.5
        normalized = [(r - mean_r) / (std_r + 1e-8) for r in rewards]
        
        # 策略梯度损失
        for response, norm_reward in zip(responses, normalized):
            log_prob = model.log_prob(prompt, response)
            all_losses.append(-log_prob * norm_reward)
    
    return sum(all_losses) / len(all_losses)
```

**GRPO 的优势：**
- 不需要 Critic/Value 模型（PPO 需要）
- 不需要参考模型（DPO 需要）
- 特别适合有明确 reward function 的场景（如数学、代码）

### 6.3 对齐方法对比

| 方法 | 需要 RM | 需要 Ref Model | 训练稳定性 | 适用场景 |
|------|---------|---------------|-----------|----------|
| PPO (RLHF) | ✅ | ✅ | 较差 | 通用对齐 |
| DPO | ❌ | ✅ | 好 | 通用对齐 |
| ORPO | ❌ | ❌ | 好 | SFT+对齐一步到位 |
| GRPO | ❌ | ❌ | 好 | 有明确奖励函数 |

---

## 7. 补充：模型蒸馏

将大模型的能力"蒸馏"到小模型中，实现部署友好的高性能小模型。

### 7.1 知识蒸馏原理

```
Teacher Model (大模型，如 70B) → 生成 soft labels / 训练数据
Student Model (小模型，如 7B)  → 学习 teacher 的输出分布

蒸馏损失 = α × CE(student_logits, hard_labels) + (1-α) × KL(student_logits, teacher_logits)
```

### 7.2 基于输出的蒸馏（最常用）

```python
"""用大模型生成高质量数据，训练小模型"""
from openai import OpenAI
import json

client = OpenAI()

def distill_from_teacher(prompts, teacher_model="gpt-4o", output_path="distill_data.jsonl"):
    """从 teacher 模型蒸馏数据"""
    with open(output_path, "w") as f:
        for prompt in prompts:
            response = client.chat.completions.create(
                model=teacher_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            item = {
                "instruction": prompt,
                "output": response.choices[0].message.content
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"蒸馏完成: {len(prompts)} 条数据 → {output_path}")

# 然后用蒸馏数据 SFT 训练小模型
```

### 7.3 基于 Logits 的蒸馏

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """
    student_logits: 学生模型输出
    teacher_logits: 教师模型输出（需要提前计算并缓存）
    labels: 真实标签
    """
    # Hard label loss
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Soft label loss (KL divergence)
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)
    
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

### 7.4 实际蒸馏策略

```
方法1: 数据蒸馏（最简单）
  - 用 GPT-4 / Claude 生成大量高质量回答
  - 用这些数据 SFT 训练小模型
  - 代表：Alpaca, Vicuna, OpenHermes

方法2: 在线蒸馏
  - 训练时实时用 teacher 生成 soft labels
  - 计算成本高但效果好

方法3: 渐进式蒸馏
  - 70B → 13B → 7B → 3B 逐级蒸馏
  - 每级差距不大，蒸馏效果更好
```

---

## 8. 学习检查清单

- [ ] 理解预训练数据处理流程（去重、过滤、打包）
- [ ] 了解 DeepSpeed ZeRO 的三个阶段
- [ ] 能配置并启动分布式训练
- [ ] 理解 SFT → DPO 的完整对齐流程
- [ ] 能构建偏好数据集
- [ ] 使用 TRL 库完成 DPO 训练
- [ ] 了解常用评估 Benchmark
- [ ] 了解 GRPO/ORPO 等新对齐方法的核心思想
- [ ] 理解模型蒸馏的基本策略
- [ ] 完成毕业项目：领域模型的完整训练

---

## 9. 推荐学习材料

### 论文
- [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) - 缩放定律与最优训练策略
- [InstructGPT](https://arxiv.org/abs/2203.02155) - RLHF 对齐方法
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - DPO 论文
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) - 最新训练实践

### 工具与框架
- [DeepSpeed 文档](https://www.deepspeed.ai/) - 微软分布式训练框架
- [TRL 文档](https://huggingface.co/docs/trl) - Hugging Face 对齐训练库
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - 模型评估框架
- [Weights & Biases](https://wandb.ai/) - 训练监控平台

### 教程与博客
- [Hugging Face - RLHF 教程](https://huggingface.co/blog/rlhf) - RLHF 入门
- [Hugging Face - DPO 教程](https://huggingface.co/blog/dpo-trl) - DPO 实践
- [Lilian Weng - LLM Training](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) - 训练技术综述
- [Sebastian Raschka - LLM Training](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) - RLHF 与替代方案

### 视频
- [Andrej Karpathy - Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU) - 完整预训练复现
- [Yannic Kilcher - RLHF 讲解](https://www.youtube.com/watch?v=2MBJOuVq380)

---

## 10. 三个月学习总结

恭喜你完成了整个学习计划！回顾一下你掌握的技能：

```
✅ 第1个月：基础 + 应用
   - NLP 基础概念、Tokenization、Embedding、Attention
   - API 调用、Prompt Engineering、RAG、LangChain

✅ 第2个月：深入原理
   - Transformer 架构、GPT 系列演进
   - Hugging Face 生态、模型部署（vLLM/Ollama）

✅ 第3个月：模型训练
   - LoRA/QLoRA 微调
   - 预训练数据处理、分布式训练
   - SFT + DPO 对齐全流程
```

### 后续学习方向

- **多模态模型**：LLaVA、Qwen-VL 等视觉语言模型
- **长上下文**：RoPE 外推、Ring Attention
- **推理优化**：Speculative Decoding、量化部署
- **Agent 框架**：AutoGPT、CrewAI、LangGraph
- **模型安全**：Red Teaming、对抗攻击与防御
