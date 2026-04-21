# 第9-10周：模型微调技术

> 本阶段目标：掌握全参数微调与参数高效微调（LoRA/QLoRA），学会构建指令数据集，完成单卡微调 7B 模型实战。

---

## 1. 微调概述

### 1.1 为什么需要微调？

预训练模型是通用的，微调让模型适应特定任务或领域：

```
基座模型 (Base Model)
  ↓ 微调
领域专家模型（医疗问答、代码生成、法律咨询...）
```

**微调 vs Prompt Engineering vs RAG：**

| 方法 | 适用场景 | 成本 | 效果 |
|------|----------|------|------|
| Prompt Engineering | 通用任务，快速迭代 | 低 | 受限于模型能力 |
| RAG | 需要外部知识 | 中 | 知识准确但生成风格不变 |
| 微调 | 需要改变模型行为/风格 | 高 | 深度定制 |

### 1.2 微调方法分类

```
全参数微调 (Full Fine-tuning)
  - 更新所有参数
  - 效果最好，但显存需求大
  - 7B 模型需要 ~60GB 显存 (FP16 + Adam)

参数高效微调 (PEFT)
  ├── LoRA: 低秩分解，只训练小矩阵
  ├── QLoRA: 量化 + LoRA，单卡可微调 7B
  ├── Prefix Tuning: 在输入前添加可训练前缀
  └── Adapter: 在层间插入小型网络
```

---

## 2. LoRA 原理与实践

### 2.1 LoRA 核心思想

LoRA (Low-Rank Adaptation) 的核心假设：微调时的权重变化是低秩的。

```
原始权重: W (d × d)
LoRA 分解: ΔW = A × B，其中 A (d × r), B (r × d), r << d

推理时: W' = W + ΔW = W + A × B
```

**参数量对比（以 7B 模型为例）：**
- 全参数微调：~7B 参数
- LoRA (r=16)：~20M 参数（仅原始的 0.3%）

### 2.2 LoRA 代码实现原理

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """LoRA 层的简化实现"""
    def __init__(self, original_linear, r=16, alpha=32):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad_(False)  # 冻结原始权重
        
        d_in = original_linear.in_features
        d_out = original_linear.out_features
        
        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(d_in, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, d_out))
        self.scaling = alpha / r
    
    def forward(self, x):
        # 原始输出 + LoRA 增量
        original_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out
```

### 2.3 使用 PEFT 库进行 LoRA 微调

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # 秩
    lora_alpha=32,                 # 缩放因子
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # 应用 LoRA 的层
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 20,971,520 || all params: 7,636,013,056 || trainable%: 0.2747
```

### 2.4 LoRA 超参数选择指南

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `r` | 8-64 | 秩越大表达能力越强，但参数越多 |
| `lora_alpha` | 2×r | 通常设为 r 的 2 倍 |
| `target_modules` | 所有线性层 | 覆盖越多效果越好 |
| `lora_dropout` | 0.05-0.1 | 防止过拟合 |

---

## 3. QLoRA：单卡微调 7B 模型

QLoRA = 4-bit 量化 + LoRA，让单张 24GB 显卡就能微调 7B 模型。

### 3.1 QLoRA 配置

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 量化
    bnb_4bit_compute_dtype=torch.float16, # 计算时用 FP16
    bnb_4bit_use_double_quant=True        # 双重量化，进一步压缩
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# 准备模型用于 k-bit 训练
model = prepare_model_for_kbit_training(model)

# 应用 LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
```

### 3.2 显存对比

| 方法 | 7B 模型显存需求 |
|------|----------------|
| 全参数 FP16 + Adam | ~60GB |
| LoRA FP16 | ~16GB |
| QLoRA 4-bit | ~6GB |

---

## 4. 数据准备

### 4.1 指令数据集格式

**Alpaca 格式：**
```json
{
    "instruction": "将以下文本翻译为英文",
    "input": "今天天气真好",
    "output": "The weather is really nice today."
}
```

**ShareGPT 格式（多轮对话）：**
```json
{
    "conversations": [
        {"from": "human", "value": "帮我写一个Python函数计算斐波那契数列"},
        {"from": "gpt", "value": "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"},
        {"from": "human", "value": "能优化一下性能吗？"},
        {"from": "gpt", "value": "可以用动态规划：\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a + b\n    return b\n```"}
    ]
}
```

### 4.2 数据构建实践

```python
import json

def create_instruction_dataset(raw_data, output_path):
    """将原始数据转换为指令微调格式"""
    dataset = []
    for item in raw_data:
        dataset.append({
            "instruction": item["question"],
            "input": item.get("context", ""),
            "output": item["answer"]
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"生成 {len(dataset)} 条训练数据 → {output_path}")

# 数据质量检查
def validate_dataset(path):
    issues = []
    with open(path) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            if len(item["output"]) < 10:
                issues.append(f"第{i}行: output 过短")
            if len(item["instruction"]) < 5:
                issues.append(f"第{i}行: instruction 过短")
    print(f"检查完成，发现 {len(issues)} 个问题")
    return issues
```

### 4.3 数据质量原则

- **质量 > 数量**：1000 条高质量数据 > 10000 条低质量数据
- **多样性**：覆盖不同任务类型和难度
- **一致性**：输出风格统一
- **去重**：避免重复样本导致过拟合
- **长度均衡**：避免全是短回答或全是长回答

---

## 5. 完整微调实战

### 5.1 使用 Transformers Trainer

```python
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset

# 加载数据
dataset = load_dataset("json", data_files="train_data.jsonl", split="train")

# 数据预处理
def preprocess(examples):
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        if input_text:
            prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n{output}"
        else:
            prompt = f"### 指令:\n{instruction}\n\n### 回答:\n{output}"
        texts.append(prompt)
    
    tokenized = tokenizer(texts, truncation=True, max_length=1024, padding=False)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# 训练参数
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 等效 batch_size = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

# 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
)

trainer.train()

# 保存 LoRA 权重
model.save_pretrained("./lora_weights")
tokenizer.save_pretrained("./lora_weights")
```

### 5.2 使用 LLaMA-Factory（推荐，更简单）

```bash
# 安装
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# Web UI 方式（最简单）
llamafactory-cli webui

# 命令行方式
llamafactory-cli train examples/train_qlora/qwen2_lora_sft.yaml
```

**LLaMA-Factory 配置文件示例：**

```yaml
# qwen2_qlora_sft.yaml
model_name_or_path: Qwen/Qwen2-7B-Instruct
stage: sft
finetuning_type: lora

# LoRA 参数
lora_rank: 16
lora_alpha: 32
lora_target: all

# 数据
dataset: my_dataset  # 在 data/dataset_info.json 中注册
template: qwen

# 训练参数
output_dir: output/qwen2-7b-qlora
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05

# 量化
quantization_bit: 4

# 其他
fp16: true
logging_steps: 10
save_steps: 500
```

### 5.3 合并 LoRA 权重

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基座模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct", torch_dtype=torch.float16
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, "./lora_weights")

# 合并权重
merged_model = model.merge_and_unload()

# 保存完整模型
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

### 5.4 微调后评估

```python
def evaluate_finetuned(model, tokenizer, test_prompts):
    model.eval()
    results = []
    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response})
        print(f"Q: {prompt}\nA: {response}\n{'='*50}")
    return results
```

---

## 6. 常见问题与调优

### 6.1 过拟合

```
症状：训练 loss 持续下降，但模型回答变得死板/重复
解决：
- 减少训练轮数
- 增加 dropout
- 增加数据多样性
- 使用更小的学习率
```

### 6.2 灾难性遗忘

```
症状：微调后模型在原有能力上退化
解决：
- 使用 LoRA 而非全参数微调
- 混入通用数据（10-20%）
- 降低学习率
```

### 6.3 学习率选择

```
全参数微调：1e-5 ~ 5e-5
LoRA：1e-4 ~ 5e-4
QLoRA：1e-4 ~ 2e-4
```

---

## 7. 补充：Unsloth 加速微调

[Unsloth](https://github.com/unslothai/unsloth) 通过手写 CUDA kernel 优化，实现 2x 训练加速 + 60% 显存节省，且无精度损失。

### 7.1 安装与使用

```bash
pip install unsloth
```

```python
from unsloth import FastLanguageModel
import torch

# 加载模型（自动应用优化）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 添加 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",  # Unsloth 优化的 gradient checkpointing
)

# 后续训练流程与标准 HuggingFace Trainer 完全一致
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./output", num_train_epochs=3,
        per_device_train_batch_size=4, learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    ),
)
trainer.train()

# 导出为多种格式
model.save_pretrained_merged("./merged", tokenizer)       # HuggingFace 格式
model.save_pretrained_gguf("./gguf", tokenizer, "q4_k_m") # GGUF 格式
```

### 7.2 Unsloth vs 标准训练对比

| 指标 | 标准 QLoRA | Unsloth QLoRA |
|------|-----------|---------------|
| 训练速度 | 1x | ~2x |
| 显存占用 | ~6GB (7B) | ~4GB (7B) |
| 精度损失 | 无 | 无 |
| 支持模型 | 所有 | LLaMA/Qwen/Mistral 等主流 |

---

## 8. 补充：数据合成技术

当领域数据不足时，可以用强大的 LLM 生成训练数据（Self-Instruct / Evol-Instruct）。

### 8.1 基础数据合成

```python
from openai import OpenAI

client = OpenAI()

def generate_training_data(topic, n=10):
    """用 GPT-4 生成指令微调数据"""
    prompt = f"""请为"{topic}"领域生成 {n} 条高质量的问答训练数据。

要求：
- 问题多样化，覆盖不同难度
- 回答详细准确，200-500字
- 输出 JSON 数组格式

格式：
[{{"instruction": "问题", "output": "回答"}}]"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.8,
    )
    return response.choices[0].message.content

# 生成数据
import json
data = json.loads(generate_training_data("Python 编程", n=5))
for item in data:
    print(f"Q: {item['instruction'][:60]}...")
```

### 8.2 Evol-Instruct：渐进式难度提升

```python
def evolve_instruction(instruction, evolution_type="deepening"):
    """将简单指令进化为更复杂的版本"""
    evolution_prompts = {
        "deepening": "请将以下指令改写为更深入、更有挑战性的版本，要求更详细的分析：",
        "widening": "请将以下指令扩展，增加额外的约束条件或子任务：",
        "reasoning": "请改写以下指令，使其需要多步推理才能回答：",
    }
    
    prompt = f"""{evolution_prompts[evolution_type]}

原始指令：{instruction}

进化后的指令（只输出新指令，不要回答）："""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content

# 示例
simple = "解释什么是递归"
evolved = evolve_instruction(simple, "reasoning")
print(f"原始: {simple}")
print(f"进化: {evolved}")
```

### 8.3 数据质量过滤

```python
def score_data_quality(instruction, output):
    """用 LLM 评估生成数据的质量"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"""请评估以下训练数据的质量（1-5分）：

指令：{instruction}
回答：{output}

评分标准：准确性、完整性、清晰度
只输出数字分数："""}],
        temperature=0,
    )
    try:
        return int(response.choices[0].message.content.strip())
    except ValueError:
        return 3

# 过滤低质量数据
filtered = [d for d in data if score_data_quality(d["instruction"], d["output"]) >= 4]
```

> **注意**：合成数据应与少量人工标注数据混合使用，纯合成数据可能导致模型学到 GPT-4 的偏见和风格。

---

## 9. 学习检查清单

- [ ] 理解 LoRA 的低秩分解原理
- [ ] 能解释 QLoRA 如何节省显存
- [ ] 能构建符合格式的指令数据集
- [ ] 使用 PEFT 库完成一次 LoRA 微调
- [ ] 使用 LLaMA-Factory 完成一次微调
- [ ] 能合并 LoRA 权重并部署
- [ ] 了解微调中的常见问题和解决方案

---

## 10. 推荐学习材料

### 论文
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - LoRA 原始论文
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - QLoRA 论文
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) - 数据质量的重要性

### 工具与框架
- [PEFT 文档](https://huggingface.co/docs/peft) - Hugging Face 参数高效微调库
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 一站式微调框架
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - 另一个流行的微调框架
- [Unsloth](https://github.com/unslothai/unsloth) - 2x 加速微调

### 教程
- [Hugging Face PEFT 教程](https://huggingface.co/docs/peft/tutorial/peft_model_config)
- [LLaMA-Factory 使用指南](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md)
- [Sebastian Raschka - Finetuning LLMs](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) - 微调实用技巧

### 数据集资源
- [Alpaca 数据集](https://github.com/tatsu-lab/stanford_alpaca)
- [Open Assistant 数据集](https://huggingface.co/datasets/OpenAssistant/oasst2)
- [ShareGPT 数据集](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)

---

## 11. 下一步

掌握微调技术后，最后阶段我们将学习 [完整训练流程与对齐](week11-12-pretraining-alignment.md)，包括从零预训练、分布式训练和 RLHF/DPO 对齐全流程。
