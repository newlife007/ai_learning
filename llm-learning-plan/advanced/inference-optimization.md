# 进阶：推理优化与部署

> 目标：掌握 LLM 推理优化的核心技术（量化、KV Cache 优化、批处理策略），能够为生产环境选择合适的部署方案。

---

## 1. 推理性能瓶颈分析

### 1.1 LLM 推理的两个阶段

```
Prefill 阶段（计算密集）：
  - 处理整个输入 prompt
  - 一次性计算所有 token 的 KV Cache
  - 瓶颈：计算量（FLOPs）

Decode 阶段（访存密集）：
  - 逐 token 生成
  - 每步只计算一个 token，但需要读取整个 KV Cache
  - 瓶颈：显存带宽（Memory Bandwidth）
```

### 1.2 关键性能指标

| 指标 | 含义 | 优化方向 |
|------|------|----------|
| TTFT (Time to First Token) | 首 token 延迟 | 优化 Prefill |
| TPS (Tokens Per Second) | 生成速度 | 优化 Decode |
| Throughput | 吞吐量（tokens/s 全系统） | 批处理优化 |
| 显存占用 | 模型 + KV Cache | 量化、GQA |

### 1.3 显存占用估算

```python
def estimate_memory(params_b, seq_len, batch_size, n_layers, d_model, n_kv_heads, dtype_bytes=2):
    """估算推理显存需求"""
    # 模型权重
    model_mem = params_b * 1e9 * dtype_bytes / 1e9  # GB
    
    # KV Cache: 2 * n_layers * batch * seq_len * n_kv_heads * head_dim * dtype
    head_dim = d_model // 32  # 假设 32 heads
    kv_cache = 2 * n_layers * batch_size * seq_len * n_kv_heads * head_dim * dtype_bytes / 1e9
    
    print(f"模型权重: {model_mem:.1f} GB")
    print(f"KV Cache: {kv_cache:.1f} GB")
    print(f"总计: {model_mem + kv_cache:.1f} GB")
    return model_mem + kv_cache

# Qwen2-7B, batch=8, seq_len=4096
estimate_memory(7, 4096, 8, 32, 4096, 4)
```

---

## 2. 量化技术

### 2.1 量化方法对比

| 方法 | 精度 | 速度 | 适用场景 |
|------|------|------|----------|
| FP16 | 基准 | 基准 | 显存充足时 |
| INT8 (W8A8) | 接近无损 | 1.5-2x | 通用加速 |
| INT4 (W4A16) | 轻微损失 | 2-3x | 显存受限 |
| GPTQ | 好 | 快 | 离线量化，GPU 推理 |
| AWQ | 好 | 快 | 保护重要权重 |
| GGUF | 灵活 | 中等 | CPU/混合推理 |

### 2.2 使用 AWQ 量化

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "Qwen/Qwen2-7B-Instruct"
quant_path = "qwen2-7b-awq"

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
}

# 加载并量化
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.quantize(tokenizer, quant_config=quant_config)

# 保存
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### 2.3 使用 GPTQ 量化

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

quantization_config = GPTQConfig(
    bits=4,
    dataset="c4",          # 校准数据集
    group_size=128,
    desc_act=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
)
model.save_pretrained("qwen2-7b-gptq")
```

---

## 3. 推理引擎深入

### 3.1 vLLM 高级配置

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    tensor_parallel_size=2,          # 2卡张量并行
    gpu_memory_utilization=0.9,      # GPU 显存利用率
    max_model_len=8192,              # 最大序列长度
    quantization="awq",              # 使用 AWQ 量化模型
    enforce_eager=False,             # 使用 CUDA Graph 加速
)

# 批量推理
prompts = ["问题1", "问题2", "问题3"]
params = SamplingParams(temperature=0.7, max_tokens=512, top_p=0.9)
outputs = llm.generate(prompts, params)

for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Output: {output.outputs[0].text[:100]}...")
    print(f"Tokens/s: {len(output.outputs[0].token_ids) / output.metrics.finished_time:.1f}")
```

### 3.2 vLLM 生产部署

```bash
# 启动 API 服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --max-num-seqs 256 \
    --port 8000

# 性能测试
python -m vllm.entrypoints.openai.api_server --help
```

### 3.3 TensorRT-LLM（NVIDIA 官方）

```python
# TensorRT-LLM 提供最极致的 NVIDIA GPU 推理性能
# 安装较复杂，适合追求极致性能的生产环境

# 基本流程：
# 1. 将模型转换为 TensorRT 引擎
# 2. 使用 Triton Inference Server 部署

# 转换命令示例
# python convert_checkpoint.py --model_dir ./model --output_dir ./trt_ckpt --dtype float16
# trtllm-build --checkpoint_dir ./trt_ckpt --output_dir ./trt_engine --gemm_plugin float16
```

---

## 4. KV Cache 优化

### 4.1 PagedAttention（vLLM 核心）

```
传统方式：为每个请求预分配最大长度的 KV Cache → 显存浪费严重

PagedAttention：
- 将 KV Cache 分成固定大小的"页"（如 16 tokens/页）
- 按需分配，用完释放
- 类似操作系统的虚拟内存管理
- 显存利用率从 ~50% 提升到 ~95%
```

### 4.2 Prefix Caching

```python
# 多个请求共享相同的 system prompt 时，可以缓存其 KV Cache

# vLLM 启用 prefix caching
# --enable-prefix-caching

# 效果：如果 100 个请求都有相同的 2000 token system prompt
# 无缓存：每个请求都要计算 2000 token 的 KV
# 有缓存：只计算一次，后续请求直接复用
```

### 4.3 KV Cache 量化

```python
# 将 KV Cache 从 FP16 量化为 INT8/FP8，减少 50% 显存
# vLLM 支持：
# --kv-cache-dtype fp8

# 对质量影响很小，但能显著增加可服务的并发数
```

---

## 5. 批处理策略

### 5.1 Static Batching vs Continuous Batching

```
Static Batching（传统）：
  - 等一批请求凑齐后一起处理
  - 所有请求必须等最长的那个完成
  - GPU 利用率低

Continuous Batching（vLLM/TGI）：
  - 请求随到随处理
  - 某个请求完成后立即插入新请求
  - GPU 利用率接近 100%
```

### 5.2 吞吐量优化配置

```bash
# vLLM 吞吐量优化
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B-Instruct \
    --max-num-seqs 512 \           # 最大并发序列数
    --max-num-batched-tokens 32768 # 每批最大 token 数
```

---

## 6. 部署架构选择

### 6.1 不同规模的部署方案

| 规模 | 方案 | 配置 |
|------|------|------|
| 个人/Demo | Ollama | 单机 CPU/GPU |
| 小团队 (QPS<10) | vLLM 单实例 | 1-2 GPU |
| 中等 (QPS 10-100) | vLLM + 负载均衡 | 多实例 |
| 大规模 (QPS 100+) | TRT-LLM + Triton + K8s | GPU 集群 |

### 6.2 简单的负载均衡部署

```yaml
# docker-compose.yml
services:
  vllm-1:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              count: 1
    command: --model Qwen/Qwen2-7B-Instruct --port 8000
    
  vllm-2:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              count: 1
    command: --model Qwen/Qwen2-7B-Instruct --port 8000

  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

---

## 7. 性能基准测试

```python
"""benchmark.py - 推理性能测试"""
import time, asyncio
from openai import AsyncOpenAI

async def benchmark(base_url, model, n_requests=50, max_tokens=256):
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    
    async def single_request():
        start = time.time()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "写一首关于春天的诗"}],
            max_tokens=max_tokens,
        )
        elapsed = time.time() - start
        tokens = response.usage.completion_tokens
        return {"time": elapsed, "tokens": tokens, "tps": tokens / elapsed}
    
    # 并发请求
    tasks = [single_request() for _ in range(n_requests)]
    results = await asyncio.gather(*tasks)
    
    avg_tps = sum(r["tps"] for r in results) / len(results)
    avg_latency = sum(r["time"] for r in results) / len(results)
    total_tokens = sum(r["tokens"] for r in results)
    total_time = max(r["time"] for r in results)
    
    print(f"并发数: {n_requests}")
    print(f"平均延迟: {avg_latency:.2f}s")
    print(f"平均 TPS: {avg_tps:.1f} tokens/s (per request)")
    print(f"系统吞吐: {total_tokens / total_time:.1f} tokens/s")

asyncio.run(benchmark("http://localhost:8000/v1", "Qwen/Qwen2-7B-Instruct"))
```

---

## 8. 推荐学习材料

### 论文
- [FlashAttention](https://arxiv.org/abs/2205.14135) - IO 感知的精确注意力
- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) - 高效 KV Cache 管理
- [Speculative Decoding](https://arxiv.org/abs/2211.17192) - 投机解码加速
- [AWQ](https://arxiv.org/abs/2306.00978) - 激活感知权重量化

### 工具
- [vLLM](https://docs.vllm.ai/) - 高性能推理引擎
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA 官方
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU/边缘推理
- [SGLang](https://github.com/sgl-project/sglang) - 结构化生成加速

### 博客
- [vLLM Blog](https://blog.vllm.ai/) - 推理优化最新进展
- [Hugging Face - Optimizing LLM Inference](https://huggingface.co/docs/transformers/llm_optims)
