# 🚀 大模型（LLM）3个月学习计划

> 从应用开发到原理理解，再到独立训练，系统性掌握大语言模型。

## 📋 学习路线总览

| 阶段 | 周次 | 主题 | 目标 |
|------|------|------|------|
| **第一个月：基础+应用** | 第1-2周 | [基础知识](month01/week01-02-fundamentals.md) | 掌握 NLP 基础概念、搭建开发环境 |
| | 第3-4周 | [应用开发实战](month01/week03-04-app-development.md) | 掌握 API 调用、Prompt Engineering、RAG |
| **第二个月：深入原理** | 第5-6周 | [Transformer 与 LLM 原理](month02/week05-06-transformer-theory.md) | 理解 Transformer 架构与 LLM 训练流程 |
| | 第7-8周 | [开源模型生态](month02/week07-08-opensource-ecosystem.md) | 熟悉 Hugging Face 生态与模型部署 |
| **第三个月：模型训练** | 第9-10周 | [微调技术](month03/week09-10-finetuning.md) | 掌握 LoRA/QLoRA 微调实操 |
| | 第11-12周 | [完整训练流程与对齐](month03/week11-12-pretraining-alignment.md) | 完成从预训练到对齐的全流程 |

---

## 🔥 进阶专题（3个月后）

完成基础学习后，可根据兴趣方向深入以下专题：

| 专题 | 内容 | 前置要求 |
|------|------|----------|
| [多模态大模型入门](advanced/multimodal-intro.md) | VLM 架构原理、Qwen-VL/LLaVA 使用与微调 | 第2-3个月 |
| [Agent 框架实战](advanced/agent-frameworks.md) | LangGraph、CrewAI、工具集成、多 Agent 协作 | 第1个月 |
| [推理优化与部署](advanced/inference-optimization.md) | 量化(AWQ/GPTQ)、vLLM 高级配置、KV Cache 优化 | 第2个月 |

---

## 📝 各阶段补充内容索引

本计划在核心内容基础上，还补充了以下前沿话题：

- **BPE 算法从零实现** → [第1-2周](month01/week01-02-fundamentals.md#7-补充bpe-算法详细实现)
- **多模态 API (GPT-4V/Claude Vision)** → [第3-4周](month01/week03-04-app-development.md#9-补充多模态-api-调用)
- **Structured Output** → [第3-4周](month01/week03-04-app-development.md#10-补充structured-output结构化输出)
- **Flash Attention 原理** → [第5-6周](month02/week05-06-transformer-theory.md#6-补充flash-attention-原理)
- **Speculative Decoding** → [第5-6周](month02/week05-06-transformer-theory.md#7-补充speculative-decoding投机解码)
- **模型安全与 Red Teaming** → [第7-8周](month02/week07-08-opensource-ecosystem.md#6-补充模型安全与-red-teaming-基础)
- **GGUF 量化实操** → [第7-8周](month02/week07-08-opensource-ecosystem.md#7-补充gguf-量化实操)
- **Unsloth 加速微调** → [第9-10周](month03/week09-10-finetuning.md#7-补充unsloth-加速微调)
- **数据合成技术** → [第9-10周](month03/week09-10-finetuning.md#8-补充数据合成技术)
- **GRPO/ORPO 新对齐方法** → [第11-12周](month03/week11-12-pretraining-alignment.md#6-补充grpoorpo-等新对齐方法)
- **模型蒸馏** → [第11-12周](month03/week11-12-pretraining-alignment.md#7-补充模型蒸馏)

---

## 🛠 环境准备

```bash
# 基础环境
conda create -n llm python=3.11
conda activate llm

# 核心依赖
pip install torch transformers datasets accelerate
pip install langchain openai chromadb faiss-cpu
pip install peft trl bitsandbytes
pip install vllm deepspeed
```

## 💻 硬件建议

- **第1个月（应用开发）**：CPU 即可，主要使用 API
- **第2个月（原理学习）**：Google Colab 免费 GPU 足够
- **第3个月（微调/训练）**：至少 1×24GB GPU（RTX 4090 / A100），推荐 [AutoDL](https://www.autodl.com/) 或 [Lambda](https://lambdalabs.com/) 租用

## 💡 学习建议

1. **动手优先**：每个章节的代码都要亲自跑一遍，不要只看不练
2. **循序渐进**：不要跳过基础直接看进阶，前面的知识是后面的基础
3. **项目驱动**：每个阶段完成至少一个实战项目
4. **关注前沿**：关注 Hugging Face Blog、arXiv 上的最新论文
5. **社区参与**：加入 r/LocalLLaMA、Hugging Face 论坛等社区交流

## 📚 核心参考资源

- [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [3Blue1Brown - Transformer 可视化](https://www.youtube.com/watch?v=wjZofJX0v4M)
- [李宏毅 - 机器学习课程](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)
- [动手学深度学习 d2l.ai](https://d2l.ai/)
- [Hugging Face 官方文档](https://huggingface.co/docs)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [vLLM 文档](https://docs.vllm.ai/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)

---

## 📁 目录结构

```
llm-learning-plan/
├── README.md                          ← 你在这里
├── month01/
│   ├── week01-02-fundamentals.md      # NLP 基础 + BPE 实现
│   └── week03-04-app-development.md   # 应用开发 + 多模态 API
├── month02/
│   ├── week05-06-transformer-theory.md # Transformer + Flash Attention
│   └── week07-08-opensource-ecosystem.md # 开源生态 + 安全评估
├── month03/
│   ├── week09-10-finetuning.md        # 微调 + Unsloth + 数据合成
│   └── week11-12-pretraining-alignment.md # 对齐 + GRPO + 蒸馏
└── advanced/
    ├── multimodal-intro.md            # 多模态入门
    ├── agent-frameworks.md            # Agent 框架实战
    └── inference-optimization.md      # 推理优化与部署
```
