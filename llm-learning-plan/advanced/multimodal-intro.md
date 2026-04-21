# 进阶：多模态大模型入门

> 目标：理解视觉语言模型（VLM）的架构原理，掌握开源多模态模型的使用和微调。

---

## 1. 多模态模型概览

### 1.1 什么是多模态大模型？

多模态大模型能同时处理文本、图像、音频等多种输入，实现跨模态理解和生成。

```
典型架构：
  图像 → Vision Encoder (ViT) → Projection Layer → LLM → 文本输出
  文本 → Tokenizer ────────────────────────────────↗
```

### 1.2 主流多模态模型

| 模型 | 开发者 | 架构 | 特点 |
|------|--------|------|------|
| GPT-4o | OpenAI | 原生多模态 | 闭源，能力最强 |
| Claude 3.5 | Anthropic | 原生多模态 | 闭源，文档理解强 |
| LLaVA | 学术界 | ViT + LLM | 开源先驱，架构简洁 |
| Qwen-VL | 阿里 | ViT + Qwen | 中文多模态最强之一 |
| InternVL | 上海AI Lab | InternViT + LLM | 开源性能领先 |
| LLaMA 3.2 Vision | Meta | ViT + LLaMA | Meta 官方多模态 |

---

## 2. VLM 架构原理

### 2.1 LLaVA 架构（最经典）

```
                    ┌─────────────┐
  Image ──→ ViT ──→│  Projection  │──→ Image Tokens ──┐
                    │   (MLP)      │                    ├──→ LLM ──→ Output
  Text  ──→ Tokenizer ──→ Text Tokens ────────────────┘
                    └─────────────┘
```

```python
import torch
import torch.nn as nn

class SimpleLLaVA(nn.Module):
    """LLaVA 架构简化实现"""
    def __init__(self, vision_encoder, llm, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.vision_encoder = vision_encoder  # 如 CLIP ViT-L
        self.llm = llm
        # 关键：将视觉特征映射到 LLM 的嵌入空间
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
    
    def forward(self, images, text_input_ids):
        # 1. 提取视觉特征
        with torch.no_grad():
            vision_features = self.vision_encoder(images)  # [B, N_patches, vision_dim]
        
        # 2. 投影到 LLM 空间
        image_tokens = self.projection(vision_features)  # [B, N_patches, llm_dim]
        
        # 3. 拼接图像 token 和文本 token
        text_embeds = self.llm.get_input_embeddings()(text_input_ids)
        combined = torch.cat([image_tokens, text_embeds], dim=1)
        
        # 4. 送入 LLM
        outputs = self.llm(inputs_embeds=combined)
        return outputs
```

### 2.2 训练流程

```
阶段1: 预训练对齐（Pretrain）
  - 冻结 Vision Encoder 和 LLM
  - 只训练 Projection Layer
  - 数据：图文对（image-caption pairs）
  - 目标：让 projection 学会将视觉特征映射到 LLM 能理解的空间

阶段2: 指令微调（Instruction Tuning）
  - 冻结 Vision Encoder
  - 训练 Projection + LLM（或 LoRA）
  - 数据：视觉问答、图像描述等指令数据
  - 目标：让模型能遵循多模态指令
```

---

## 3. 使用开源多模态模型

### 3.1 使用 Qwen-VL

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "https://example.com/photo.jpg"},
        {"type": "text", "text": "描述这张图片的内容，并分析其构图特点"}
    ]
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=512)
response = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
print(response)
```

### 3.2 使用 LLaVA

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import requests

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

image = Image.open(requests.get("https://example.com/image.jpg", stream=True).raw)
prompt = "<image>\nWhat is shown in this image? Describe in detail."

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=300)
print(processor.decode(output[0], skip_special_tokens=True))
```

---

## 4. 多模态模型微调

### 4.1 使用 LLaMA-Factory 微调 VLM

```yaml
# qwen2vl_lora_sft.yaml
model_name_or_path: Qwen/Qwen2-VL-7B-Instruct
stage: sft
finetuning_type: lora

lora_rank: 16
lora_alpha: 32
lora_target: all

dataset: my_vl_dataset
template: qwen2_vl

output_dir: output/qwen2vl-lora
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1e-4
bf16: true
```

### 4.2 多模态数据格式

```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data/images/001.jpg"},
                {"type": "text", "text": "这张X光片显示了什么异常？"}
            ]
        },
        {
            "role": "assistant",
            "content": "根据X光片显示..."
        }
    ]
}
```

---

## 5. 实战项目：图像问答系统

```python
"""简单的多模态问答服务"""
from fastapi import FastAPI, UploadFile, Form
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch, io

app = FastAPI()
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

@app.post("/ask")
async def ask_about_image(image: UploadFile, question: str = Form(...)):
    img = Image.open(io.BytesIO(await image.read()))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": question}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=512)
    answer = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return {"answer": answer}
```

---

## 6. 推荐学习材料

### 论文
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) - 开源 VLM 先驱
- [Qwen-VL 技术报告](https://arxiv.org/abs/2308.12966) - 阿里多模态模型
- [InternVL](https://arxiv.org/abs/2312.14238) - 开源多模态 SOTA

### 教程
- [Hugging Face VLM 教程](https://huggingface.co/docs/transformers/model_doc/llava)
- [LLaVA 官方仓库](https://github.com/haotian-liu/LLaVA)

### 工具
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 支持多模态微调
- [Ollama](https://ollama.com/) - 支持 `ollama run llava` 本地运行
