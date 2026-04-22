# 推荐系统在线服务深度解析

> 离线训练好的模型只是起点，真正的挑战在于如何在毫秒级延迟内为亿级用户提供个性化推荐。本文深入剖析推荐系统在线服务的技术难点、架构设计、性能优化，并结合主流开源方案的源码进行分析。

## 🎯 在线服务的核心挑战

```
离线训练 vs 在线服务:

  离线训练:                        在线服务:
  ├── 可以跑几小时甚至几天         ├── 必须在 100ms 内返回结果
  ├── 批量处理                     ├── 逐请求处理 (QPS 可达数万)
  ├── 容忍偶尔失败                 ├── 99.9% 可用性要求
  ├── 关注模型效果                 ├── 效果 + 延迟 + 吞吐 + 稳定性
  └── 资源可以弹性扩展             └── 资源预算固定，成本敏感
```

### 五大技术难点

```
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  ① 延迟约束                                             │
│     用户能感知的延迟阈值 ~200ms                          │
│     推荐服务需要在 50-100ms 内完成全部计算               │
│     包括: 特征获取 + 召回 + 排序 + 重排                  │
│                                                          │
│  ② 吞吐压力                                             │
│     峰值 QPS 可达数万甚至数十万                          │
│     每个请求需要对数百个候选物品打分                      │
│     总计算量 = QPS × 候选数 × 模型复杂度                 │
│                                                          │
│  ③ 特征一致性                                            │
│     训练时用的特征 和 服务时用的特征 必须一致             │
│     "训练-服务偏差" (Training-Serving Skew) 是常见 bug   │
│                                                          │
│  ④ 模型更新                                             │
│     新模型上线不能中断服务                               │
│     需要热加载、灰度发布、快速回滚                       │
│                                                          │
│  ⑤ 长尾问题                                             │
│     P99 延迟远高于平均延迟                               │
│     一个慢请求可能拖垮整个服务链路                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 🏗️ 在线服务整体架构

```
                        用户请求
                           │
                           ▼
                   ┌───────────────┐
                   │   API Gateway  │  限流、鉴权、路由
                   └───────┬───────┘
                           │
                           ▼
                   ┌───────────────┐
                   │  推荐编排服务   │  Orchestrator
                   │  (业务逻辑层)  │  协调各阶段调用
                   └───┬───┬───┬───┘
                       │   │   │
            ┌──────────┘   │   └──────────┐
            ▼              ▼              ▼
    ┌──────────────┐ ┌──────────┐ ┌──────────────┐
    │  召回服务     │ │ 特征服务  │ │  排序服务     │
    │  Recall      │ │ Feature  │ │  Ranking     │
    │  Service     │ │ Service  │ │  Service     │
    └──────┬───────┘ └────┬─────┘ └──────┬───────┘
           │              │              │
    ┌──────┴───────┐ ┌────┴─────┐ ┌──────┴───────┐
    │ 向量索引      │ │ Redis /  │ │ 模型推理服务  │
    │ FAISS/Milvus │ │ Feature  │ │ TF Serving / │
    │              │ │ Store    │ │ Triton       │
    └──────────────┘ └──────────┘ └──────────────┘
```

### 请求处理时序

```
时间轴 (总预算 100ms):

  0ms ──┬── 请求到达，解析参数
        │
  5ms ──┼── 并行发起: 获取用户特征 + 多路召回
        │   ┌─ 特征服务: 查 Redis 获取用户画像     (~15ms)
        │   ├─ 召回路1: 向量召回 (FAISS ANN)       (~10ms)
        │   ├─ 召回路2: ItemCF 召回 (Redis 查表)   (~5ms)
        │   ├─ 召回路3: 热门召回 (Redis 查表)      (~3ms)
        │   └─ 召回路4: 标签召回                   (~5ms)
        │
 20ms ──┼── 召回结果合并去重 (~1000 候选)
        │
 25ms ──┼── 批量获取候选物品特征 (Redis MGET)      (~10ms)
        │
 35ms ──┼── 特征拼接 + 粗排模型打分                (~15ms)
        │   筛选到 ~200 候选
        │
 50ms ──┼── 精排模型推理 (TF Serving / Triton)     (~30ms)
        │   对 200 个候选精确打分
        │
 80ms ──┼── 重排: 多样性、去重、业务规则            (~10ms)
        │
 90ms ──┼── 组装结果，返回 top-20
        │
100ms ──┴── 响应返回给用户
```

## 📦 模型推理服务

模型推理是在线服务中计算最密集的环节。

### TensorFlow Serving

```
TF Serving 是 Google 开源的模型服务框架，推荐系统中使用最广泛。

架构:
┌─────────────────────────────────────────┐
│            TF Serving                    │
│                                          │
│  ┌──────────────┐  ┌──────────────┐     │
│  │ Model Manager│  │ Servable     │     │
│  │ 管理模型版本 │  │ 可服务的模型  │     │
│  └──────┬───────┘  └──────────────┘     │
│         │                                │
│  ┌──────┴───────┐  ┌──────────────┐     │
│  │ Source       │  │ Loader       │     │
│  │ 监控模型文件 │  │ 加载/卸载模型│     │
│  └──────────────┘  └──────────────┘     │
│                                          │
│  请求 → Batching → Session.Run → 响应   │
└─────────────────────────────────────────┘
```

#### 核心源码解析：请求批处理（Batching）

TF Serving 的批处理是性能优化的关键。源码位于 `tensorflow_serving/batching/`：

```cpp
// batching_session.cc 核心逻辑简化
//
// 关键思想: 将多个独立请求合并为一个大 batch 送入 GPU
// GPU 擅长并行计算，batch 越大吞吐越高

// 1. 请求到达，加入等待队列
Status BatchingSession::Run(const RunOptions& run_options,
                            const std::vector<std::pair<string, Tensor>>& inputs,
                            const std::vector<string>& output_tensor_names,
                            std::vector<Tensor>* outputs) {
  // 创建一个 BatchTask，包含本次请求的输入和输出指针
  auto task = std::make_unique<BatchTask>();
  task->inputs = inputs;
  task->output_tensor_names = output_tensor_names;
  
  // 加入 batch 调度器的队列
  // 调度器会等待一小段时间收集更多请求，然后一起处理
  batch_scheduler_->Schedule(&task);
  
  // 阻塞等待本次请求的结果
  task->done.WaitForNotification();
  *outputs = task->outputs;
}

// 2. 调度器凑够一个 batch 后触发处理
void BatchingSession::ProcessBatch(std::unique_ptr<Batch<BatchTask>> batch) {
  // 将 batch 中所有请求的输入 tensor 沿 batch 维度拼接
  // 例如: 8 个请求各有 1 个样本 → 拼成 batch_size=8 的 tensor
  std::vector<std::pair<string, Tensor>> merged_inputs;
  MergeInputTensors(*batch, &merged_inputs);
  
  // 一次 GPU 推理处理整个 batch
  std::vector<Tensor> merged_outputs;
  wrapped_session_->Run(merged_inputs, output_names, &merged_outputs);
  
  // 将输出拆分回各个请求
  SplitOutputTensors(merged_outputs, batch.get());
  
  // 通知所有等待的请求
  for (auto& task : *batch) {
    task.done.Notify();
  }
}
```

#### 批处理参数调优

```
关键参数 (--batching_parameters_file):

  max_batch_size: 128
    # 一个 batch 最多包含多少请求
    # 太大: 延迟增加 (等待凑 batch)
    # 太小: GPU 利用率低

  batch_timeout_micros: 5000  (5ms)
    # 最多等待多久凑 batch
    # 太大: 延迟增加
    # 太小: batch 太小，吞吐低

  num_batch_threads: 4
    # 并行处理 batch 的线程数

  max_enqueued_batches: 100
    # 队列最大长度，超过则拒绝请求 (背压保护)

调优策略:
  高吞吐场景 (离线批量打分):
    max_batch_size=256, batch_timeout=50ms
  
  低延迟场景 (在线推荐):
    max_batch_size=32, batch_timeout=2ms
```

#### 模型热更新源码解析

```cpp
// model_life_cycle.cc 简化逻辑
//
// TF Serving 通过文件系统轮询检测新模型版本

// Source 组件定期扫描模型目录
// /models/ranking_model/
//   ├── 1/  (版本1, 当前服务中)
//   │   ├── saved_model.pb
//   │   └── variables/
//   └── 2/  (版本2, 新上传的)
//       ├── saved_model.pb
//       └── variables/

void FileSystemStoragePathSource::PollFileSystemForConfig() {
  // 扫描模型目录，发现新版本
  std::vector<ServableData<StoragePath>> versions;
  for (auto& child : GetChildren(base_path)) {
    int64 version;
    if (ParseVersion(child, &version)) {
      versions.push_back({version, JoinPath(base_path, child)});
    }
  }
  // 通知 Manager 加载新版本
  aspired_versions_callback_(servable_name, versions);
}

// Manager 执行版本切换
// 策略: 先加载新版本，确认可用后再卸载旧版本
// 保证切换过程中服务不中断
void BasicManager::ManageServableVersions(versions) {
  // 1. 加载新版本到内存 (新模型 warm up)
  LoadServable(new_version);
  
  // 2. 新版本就绪后，将流量切换到新版本
  UpdateServingMap(new_version);
  
  // 3. 卸载旧版本，释放资源
  UnloadServable(old_version);
}
```

### NVIDIA Triton Inference Server

```
Triton 是 NVIDIA 开源的高性能推理服务器，支持多框架、多模型。

优势对比 TF Serving:
┌──────────────┬──────────────────┬──────────────────┐
│              │ TF Serving       │ Triton           │
├──────────────┼──────────────────┼──────────────────┤
│ 框架支持     │ 仅 TensorFlow    │ TF/PyTorch/ONNX  │
│              │                  │ TensorRT/自定义   │
│ 模型编排     │ 不支持           │ Ensemble 模型     │
│ 动态批处理   │ 支持             │ 支持 + 更灵活     │
│ GPU 利用率   │ 一般             │ 更优 (MPS/MIG)   │
│ 监控         │ 基础             │ Prometheus 集成   │
└──────────────┴──────────────────┴──────────────────┘
```

#### Triton 的 Ensemble 模型（模型编排）

推荐系统的在线推理通常是多步骤的流水线，Triton 的 Ensemble 功能可以将多个模型串联：

```
推荐场景的 Ensemble Pipeline:

  用户特征 ──→ ┌──────────────┐
               │ 特征预处理    │ (Python Backend)
  物品特征 ──→ │ 模型 Step 1  │
               └──────┬───────┘
                      │
                      ▼
               ┌──────────────┐
               │ 双塔召回模型  │ (TensorRT)
               │ 模型 Step 2  │
               └──────┬───────┘
                      │
                      ▼
               ┌──────────────┐
               │ 精排模型      │ (TensorRT)
               │ 模型 Step 3  │
               └──────┬───────┘
                      │
                      ▼
                  推荐结果
```

Ensemble 配置文件：

```
# ensemble_model/config.pbtxt
name: "recommendation_pipeline"
platform: "ensemble"
input [
  { name: "USER_ID"  data_type: TYPE_INT64  dims: [1] },
  { name: "ITEM_IDS" data_type: TYPE_INT64  dims: [-1] }
]
output [
  { name: "SCORES" data_type: TYPE_FP32 dims: [-1] }
]
ensemble_scheduling {
  step [
    {
      model_name: "feature_preprocessing"
      model_version: -1
      input_map { key: "USER_ID"  value: "USER_ID" }
      input_map { key: "ITEM_IDS" value: "ITEM_IDS" }
      output_map { key: "FEATURES" value: "preprocessed_features" }
    },
    {
      model_name: "ranking_model"
      model_version: -1
      input_map { key: "FEATURES" value: "preprocessed_features" }
      output_map { key: "SCORES" value: "SCORES" }
    }
  ]
}
```

#### Triton 动态批处理源码解析

Triton 的动态批处理比 TF Serving 更灵活，源码位于 `src/core/dynamic_batch_scheduler.cc`：

```cpp
// dynamic_batch_scheduler.cc 核心逻辑简化
//
// Triton 的批处理支持 "preferred batch sizes"
// 例如设置 preferred_batch_size: [8, 16, 32]
// 调度器会尽量凑到这些大小，GPU 计算效率最高

void DynamicBatchScheduler::Enqueue(std::unique_ptr<InferenceRequest>& request) {
  // 请求加入队列
  queue_.push_back(std::move(request));
  
  // 检查是否可以组成一个 batch
  // 条件: 队列大小 >= preferred_batch_size 或 等待超时
  if (queue_.size() >= preferred_batch_sizes_.back() ||
      TimeSinceOldestRequest() >= max_queue_delay_) {
    // 从队列中取出请求组成 batch
    auto batch = FormBatch();
    // 送入模型执行
    model_instance_->Execute(std::move(batch));
  }
}

// 关键优化: Triton 支持 "sequence batching"
// 对于有状态的模型 (如 DIEN 的 GRU)，
// 同一用户的多次请求会被路由到同一个模型实例
// 保持隐藏状态的连续性
```

#### Triton 性能调优配置

```
# model_config.pbtxt
name: "deepfm_ranking"
platform: "tensorrt_plan"    # 使用 TensorRT 加速

instance_group [
  {
    count: 2                 # 每个 GPU 上运行 2 个模型实例
    kind: KIND_GPU
    gpus: [0, 1]             # 使用 GPU 0 和 1
  }
]

dynamic_batching {
  preferred_batch_size: [8, 16, 32]   # 偏好的 batch 大小
  max_queue_delay_microseconds: 3000  # 最大等待 3ms
  
  # 优先级队列: 高优先级请求优先处理
  priority_levels: 2
  default_priority_level: 1
}

# 模型预热: 启动时用假数据跑几次，避免首次请求慢
model_warmup [
  {
    name: "warmup_requests"
    batch_size: 32
    inputs {
      key: "features"
      value: { data_type: TYPE_FP32  dims: [200]  zero_data: true }
    }
  }
]
```

## 🔍 特征服务（Feature Service）

特征服务是在线推荐中最容易出问题、也最影响效果的环节。

### 训练-服务偏差（Training-Serving Skew）

```
这是推荐系统中最隐蔽的 bug:

  训练时:
    特征 "用户7天点击数" = 从 Hive 离线计算 = 42

  服务时:
    特征 "用户7天点击数" = 从 Redis 实时查询 = 38
    (因为实时统计和离线统计的口径不同)

  结果: 模型在线效果比离线差很多，但很难定位原因

  根因:
    - 特征计算逻辑不一致 (离线用 SQL，在线用代码)
    - 数据延迟不同 (离线 T+1，在线实时)
    - 缺失值处理不同 (离线填 0，在线填 -1)
```

### Feature Store 架构

```
┌─────────────────────────────────────────────────────┐
│                   Feature Store                      │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │  特征定义层 (Feature Registry)               │    │
│  │  统一定义特征的计算逻辑、数据类型、更新频率  │    │
│  │  训练和服务共用同一份定义 → 消除偏差         │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ┌──────────────┐        ┌──────────────┐           │
│  │ 离线存储      │        │ 在线存储      │           │
│  │ (Hive/S3)    │ ─同步→ │ (Redis/DDB)  │           │
│  │ 训练时读取    │        │ 服务时读取    │           │
│  └──────────────┘        └──────────────┘           │
│                                                      │
│  ┌──────────────────────────────────────────┐       │
│  │ 流式计算层 (Flink / Spark Streaming)     │       │
│  │ 实时特征计算 → 写入在线存储               │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

### Feast：开源 Feature Store

> GitHub: https://github.com/feast-dev/feast  (Star: 5.5K+)

Feast 是最流行的开源 Feature Store，专门解决训练-服务偏差问题。

#### 核心概念

```python
# feature_repo/features.py
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from datetime import timedelta

# 1. 定义实体
user = Entity(name="user_id", value_type=ValueType.INT64, description="用户ID")

# 2. 定义数据源
user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",
)

# 3. 定义特征视图 — 训练和服务共用这份定义
user_stats_fv = FeatureView(
    name="user_stats",
    entities=["user_id"],
    ttl=timedelta(days=1),          # 特征有效期
    features=[
        Feature(name="click_count_7d", dtype=ValueType.INT64),
        Feature(name="avg_watch_time", dtype=ValueType.FLOAT),
        Feature(name="favorite_category", dtype=ValueType.STRING),
    ],
    online=True,                     # 同步到在线存储
    source=user_stats_source,
)
```

#### Feast 源码解析：在线特征获取

```python
# sdk/python/feast/infra/online_stores/redis.py 简化
#
# Feast 的在线特征获取核心逻辑

class RedisOnlineStore(OnlineStore):
    def online_read(
        self,
        config: RepoConfig,
        table: FeatureView,
        entity_keys: List[EntityKeyProto],
    ) -> List[Tuple[Optional[datetime], Optional[Dict[str, ValueProto]]]]:
        
        # 1. 构造 Redis key
        # key 格式: {project}:{feature_view}:{entity_key}
        # 例如: "myproject:user_stats:user_12345"
        redis_keys = []
        for entity_key in entity_keys:
            redis_key = _redis_key(config.project, table.name, entity_key)
            redis_keys.append(redis_key)
        
        # 2. 批量查询 Redis (MGET，一次网络往返获取所有特征)
        with self._get_client(config) as client:
            values = client.mget(redis_keys)
        
        # 3. 反序列化 protobuf → Python 对象
        results = []
        for value in values:
            if value is None:
                results.append((None, None))  # 特征缺失
            else:
                feature_values = FeatureRowProto()
                feature_values.ParseFromString(value)
                results.append((feature_values.event_timestamp, 
                               dict(feature_values.fields)))
        
        return results
```

#### Feast 在推荐服务中的使用

```python
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

# 在线服务: 获取用户特征 (毫秒级)
features = store.get_online_features(
    features=[
        "user_stats:click_count_7d",
        "user_stats:avg_watch_time",
        "user_stats:favorite_category",
    ],
    entity_rows=[{"user_id": 12345}],
).to_dict()

# 训练时: 获取历史特征 (point-in-time join，避免数据穿越)
training_df = store.get_historical_features(
    entity_df=entity_df,  # 包含 user_id 和 event_timestamp
    features=[
        "user_stats:click_count_7d",
        "user_stats:avg_watch_time",
    ],
).to_df()
# point-in-time join 确保训练时只用到事件发生前的特征
# 避免 "未来信息泄露"
```

### 特征获取性能优化

```
┌─────────────────────────────────────────────────┐
│  优化手段              效果                      │
├─────────────────────────────────────────────────┤
│  Redis Pipeline/MGET   减少网络往返              │
│  (批量获取)            100 个 key: 1 次 vs 100 次│
│                                                  │
│  本地缓存              热门物品特征缓存在内存    │
│  (LRU Cache)           命中率 >80% 时效果显著    │
│                                                  │
│  特征预计算             复杂交叉特征离线算好      │
│                        服务时直接查表             │
│                                                  │
│  特征压缩              Embedding 量化 FP32→INT8  │
│                        减少网络传输和内存占用     │
│                                                  │
│  异步预取              预测用户下一次请求         │
│                        提前加载特征到缓存         │
└─────────────────────────────────────────────────┘
```

## 🔎 向量检索服务（召回阶段核心）

向量召回是在线服务中延迟最敏感的环节之一——需要在 5-10ms 内从亿级物品中找到 top-K。

### FAISS：Meta 开源的向量检索库

> GitHub: https://github.com/facebookresearch/faiss  (Star: 32K+)

```
FAISS 不是一个服务，而是一个 C++/Python 库。
需要自己封装成服务 (通常用 gRPC/HTTP)。

核心优势:
  - GPU 加速检索
  - 丰富的索引类型
  - 极致的性能优化
```

#### FAISS 索引选择指南

```
数据规模 → 推荐索引:

  < 10 万:   IndexFlatL2 (暴力搜索，精确)
  10-100 万: IndexIVFFlat (倒排索引 + 精确距离)
  100-1000万: IndexIVFPQ (倒排 + 乘积量化，省内存)
  > 1000 万: IndexIVFPQ + OPQ (旋转优化的乘积量化)

  需要 GPU 加速: GpuIndexIVFFlat / GpuIndexIVFPQ
```

#### FAISS 核心源码解析：IVF 索引

```python
# IVF (Inverted File Index) 的核心思想:
# 1. 将向量空间划分为 nlist 个聚类 (Voronoi cells)
# 2. 每个向量归属到最近的聚类中心
# 3. 查询时只搜索最近的 nprobe 个聚类

import faiss
import numpy as np

d = 128          # 向量维度
nlist = 1024     # 聚类数量
nprobe = 32      # 查询时搜索的聚类数

# 构建索引
quantizer = faiss.IndexFlatL2(d)           # 聚类中心用精确搜索
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# 训练 (学习聚类中心)
index.train(training_vectors)  # 需要一批训练数据
index.add(all_item_vectors)    # 添加所有物品向量

# 查询
index.nprobe = nprobe  # 搜索 32 个最近的聚类
distances, indices = index.search(user_vector, k=100)  # 返回 top-100
```

```cpp
// faiss/IndexIVF.cpp 搜索核心逻辑简化
//
// search() 的两阶段过程:

void IndexIVF::search(idx_t n, const float* x, idx_t k,
                      float* distances, idx_t* labels) {
    
    // 阶段 1: 找到查询向量最近的 nprobe 个聚类中心
    // 用 quantizer (通常是 IndexFlat) 做精确搜索
    std::vector<idx_t> coarse_ids(n * nprobe);
    std::vector<float> coarse_dis(n * nprobe);
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_ids.data());
    
    // 阶段 2: 在这 nprobe 个聚类内搜索最近邻
    // 只需要遍历 nprobe/nlist 比例的数据
    // 例如 nprobe=32, nlist=1024 → 只搜索 3.1% 的数据
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_ids[i * nprobe + j];
            // 在该聚类的倒排列表中搜索
            scan_list(list_no, x + i * d, k, 
                     distances + i * k, labels + i * k);
        }
    }
}

// nprobe 是精度和速度的权衡旋钮:
//   nprobe=1:   最快，但可能遗漏好结果
//   nprobe=nlist: 等价于暴力搜索，最准但最慢
//   通常取 nlist 的 1-5%
```

#### FAISS 在推荐中的部署模式

```
模式 1: 嵌入推荐服务进程 (小规模)
  推荐服务启动时加载 FAISS 索引到内存
  直接在进程内调用 faiss.search()
  优点: 无网络开销  缺点: 索引更新需要重启

模式 2: 独立的向量检索服务 (中大规模)
  FAISS 索引封装为 gRPC 服务
  推荐服务通过 RPC 调用
  优点: 独立扩缩容  缺点: 增加网络延迟 (~2ms)

模式 3: 使用 Milvus 等向量数据库 (大规模)
  Milvus 底层使用 FAISS/HNSW
  提供分布式、持久化、动态更新能力
  优点: 功能完整  缺点: 运维复杂度高
```

### Milvus：分布式向量数据库

> GitHub: https://github.com/milvus-io/milvus  (Star: 32K+)

```
Milvus 在 FAISS 基础上提供了完整的数据库能力:

┌─────────────────────────────────────────────┐
│                  Milvus                      │
│                                              │
│  FAISS 没有的:              Milvus 提供:     │
│  ├── 分布式               ├── 分片 + 副本   │
│  ├── 持久化               ├── WAL + S3      │
│  ├── 动态增删             ├── 实时插入/删除  │
│  ├── 标量过滤             ├── 混合查询      │
│  ├── 多租户               ├── Collection 隔离│
│  └── 监控运维             └── Prometheus     │
└─────────────────────────────────────────────┘
```

#### Milvus 在推荐召回中的使用

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# 连接 Milvus
connections.connect("default", host="localhost", port="19530")

# 定义 Collection (类似数据库的表)
fields = [
    FieldSchema(name="item_id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
]
schema = CollectionSchema(fields, description="item embeddings")
collection = Collection("item_vectors", schema)

# 创建索引
index_params = {
    "metric_type": "IP",        # 内积 (向量已归一化时等价于余弦)
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index("embedding", index_params)

# 向量召回 + 标量过滤 (混合查询)
results = collection.search(
    data=[user_embedding],       # 用户向量
    anns_field="embedding",
    param={"nprobe": 32},
    limit=200,                   # 返回 top-200
    expr='category in ["电子", "数码"]',  # 只在特定类别中召回
    output_fields=["item_id", "category"],
)
```

## ⚡ 在线服务性能优化

### 1. 模型优化

```
┌─────────────────────────────────────────────────────┐
│  技术              原理                 效果         │
├─────────────────────────────────────────────────────┤
│  TensorRT 加速     算子融合 + 内核优化   推理快 2-5x │
│  模型量化          FP32 → FP16/INT8     内存减半     │
│  模型蒸馏          大模型 → 小模型       参数减少 10x │
│  算子融合          多个小算子合并        减少内存拷贝 │
│  剪枝              去掉不重要的连接      参数减少     │
└─────────────────────────────────────────────────────┘
```

#### TensorRT 优化示例

```python
# 将 TensorFlow/PyTorch 模型转换为 TensorRT 引擎
import tensorrt as trt

# 1. 从 ONNX 模型构建 TensorRT 引擎
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, logger)

with open("ranking_model.onnx", "rb") as f:
    parser.parse(f.read())

# 2. 配置优化选项
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

# 启用 FP16 推理 (精度损失极小，速度提升 ~2x)
config.set_flag(trt.BuilderFlag.FP16)

# 设置动态 batch size
profile = builder.create_optimization_profile()
profile.set_shape("input", min=(1, 200), opt=(32, 200), max=(128, 200))
config.add_optimization_profile(profile)

# 3. 构建引擎
engine = builder.build_serialized_network(network, config)

# 构建后的引擎可以直接部署到 Triton
with open("ranking_model.plan", "wb") as f:
    f.write(engine)
```

### 2. 请求级优化

```
┌─────────────────────────────────────────────────────┐
│                                                      │
│  并行化:                                             │
│    多路召回并行执行 (不互相依赖)                      │
│    特征获取与召回并行 (用户特征 + 召回同时进行)       │
│                                                      │
│  超时控制:                                           │
│    每个阶段设置独立超时                               │
│    召回超时 15ms: 超时的召回路直接跳过                │
│    精排超时 40ms: 超时则用粗排分数兜底                │
│                                                      │
│  降级策略:                                           │
│    精排服务不可用 → 降级为粗排结果                    │
│    特征服务不可用 → 使用默认特征值                    │
│    全部不可用 → 返回热门推荐 (兜底)                  │
│                                                      │
│  缓存策略:                                           │
│    用户特征: LRU 缓存，TTL 5 分钟                    │
│    物品特征: LRU 缓存，TTL 1 小时                    │
│    热门推荐: 全量缓存，TTL 10 分钟                   │
│    个性化结果: 不缓存 (每次都需要实时计算)            │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 3. Embedding 表优化

推荐模型中 Embedding 表通常占模型参数的 90% 以上：

```
问题:
  用户 Embedding: 1 亿用户 × 128 维 × 4 字节 = 48 GB
  物品 Embedding: 1000 万物品 × 128 维 × 4 字节 = 4.8 GB
  总计: ~53 GB → 单机放不下

解决方案:

  1. 哈希 Embedding (Hash Embedding)
     将 1 亿用户 ID 哈希到 100 万个桶
     参数量减少 100x，但有哈希冲突
     
  2. 混合存储
     热门用户/物品: Embedding 在 GPU 显存中
     长尾用户/物品: Embedding 在 CPU 内存或 SSD 中
     HugeCTR 的 Hierarchical Parameter Server 就是这个思路

  3. 量化压缩
     FP32 → FP16: 内存减半，精度损失极小
     FP32 → INT8: 内存减 4x，需要校准

  4. 特征交叉替代
     不为每个 ID 学独立 Embedding
     而是用 ID 的属性特征 (类别、标签) 的 Embedding 组合
     → 参数量大幅减少，且新 ID 也能有表示
```

### 4. 系统级优化

```
连接池:
  Redis 连接池: 避免每次请求都建立 TCP 连接
  gRPC 连接池: 复用到模型服务的连接
  
  # 典型配置
  redis_pool_size: 50        # 每个实例 50 个连接
  grpc_channel_pool: 10      # 每个下游服务 10 个 channel

序列化优化:
  JSON → Protobuf: 序列化速度快 5-10x，体积小 3-5x
  Protobuf → FlatBuffers: 零拷贝反序列化

内存优化:
  对象池: 复用请求/响应对象，减少 GC 压力
  Arena 分配: Protobuf Arena 减少内存碎片
  
CPU 亲和性:
  将推荐服务绑定到特定 CPU 核心
  减少上下文切换和缓存失效
```

## 📊 监控与可观测性

```
┌─────────────────────────────────────────────────────┐
│                 监控三板斧                            │
│                                                      │
│  Metrics (指标):                                     │
│  ├── 延迟: P50, P95, P99, P999                      │
│  ├── 吞吐: QPS, 成功率                              │
│  ├── 资源: CPU, 内存, GPU 利用率                     │
│  ├── 业务: CTR, 空结果率, 召回数量                   │
│  └── 模型: 预测分数分布, 特征缺失率                  │
│                                                      │
│  Logging (日志):                                     │
│  ├── 请求日志: 每次推荐的输入输出                    │
│  ├── 特征快照: 记录推荐时使用的特征值                │
│  │   (用于离线分析和样本拼接)                        │
│  └── 错误日志: 异常、超时、降级事件                  │
│                                                      │
│  Tracing (链路追踪):                                 │
│  ├── 每个请求的完整调用链路                          │
│  ├── 各阶段耗时分解                                 │
│  └── 定位慢请求的瓶颈环节                           │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 关键告警规则

```yaml
# Prometheus 告警规则示例
groups:
  - name: recommendation_service
    rules:
      # P99 延迟超过 200ms
      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(rec_request_duration_seconds_bucket[5m])) > 0.2
        for: 5m
        labels: { severity: warning }

      # 错误率超过 1%
      - alert: HighErrorRate
        expr: rate(rec_request_errors_total[5m]) / rate(rec_request_total[5m]) > 0.01
        for: 3m
        labels: { severity: critical }

      # 空结果率超过 5% (推荐系统可能有问题)
      - alert: HighEmptyRate
        expr: rate(rec_empty_results_total[5m]) / rate(rec_request_total[5m]) > 0.05
        for: 10m
        labels: { severity: warning }

      # 特征缺失率突增 (Feature Store 可能有问题)
      - alert: FeatureMissing
        expr: rate(feature_miss_total[5m]) / rate(feature_request_total[5m]) > 0.1
        for: 5m
        labels: { severity: critical }
```

## ✅ 在线服务最佳实践清单

```
┌─────────────────────────────────────────────────────┐
│                                                      │
│  架构设计:                                           │
│  ☐ 各阶段解耦为独立服务，可独立扩缩容               │
│  ☐ 多路召回并行执行，设置独立超时                    │
│  ☐ 每个环节都有降级方案和兜底策略                    │
│  ☐ 无状态设计，支持水平扩展                          │
│                                                      │
│  特征服务:                                           │
│  ☐ 使用 Feature Store 统一管理特征定义               │
│  ☐ 训练和服务使用相同的特征计算逻辑                  │
│  ☐ 批量获取特征 (MGET)，减少网络往返                │
│  ☐ 监控特征缺失率，设置默认值                        │
│                                                      │
│  模型服务:                                           │
│  ☐ 使用 TF Serving / Triton 等成熟框架              │
│  ☐ 开启动态批处理，提升 GPU 利用率                   │
│  ☐ 模型热更新，不中断服务                            │
│  ☐ TensorRT / ONNX Runtime 加速推理                 │
│  ☐ 模型预热，避免冷启动延迟                          │
│                                                      │
│  向量检索:                                           │
│  ☐ 根据数据规模选择合适的索引类型                    │
│  ☐ 定期重建索引 (新物品加入后)                       │
│  ☐ nprobe/ef_search 参数调优，平衡精度和速度         │
│                                                      │
│  性能优化:                                           │
│  ☐ 端到端延迟 < 100ms (P99 < 200ms)                │
│  ☐ 各阶段设置超时，防止级联故障                      │
│  ☐ 连接池复用，减少连接建立开销                      │
│  ☐ Protobuf 替代 JSON 做内部通信                    │
│                                                      │
│  可靠性:                                             │
│  ☐ 全链路监控 + 告警                                │
│  ☐ 记录特征快照，支持问题回溯                        │
│  ☐ 灰度发布新模型，观察指标后再全量                  │
│  ☐ 定期压测，确认容量上限                            │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## 🗺️ 开源方案选型总结

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ 组件          │ 推荐方案      │ 备选方案      │ 适用场景      │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 模型推理     │ Triton       │ TF Serving   │ 多框架用Triton│
│              │              │              │ 纯TF用Serving│
│ 特征存储     │ Feast+Redis  │ 自建         │ 中小规模Feast │
│              │              │              │ 大规模自建    │
│ 向量检索     │ Milvus       │ FAISS        │ 需持久化Milvus│
│              │              │              │ 纯内存FAISS  │
│ 消息队列     │ Kafka        │ Pulsar       │ 行为日志收集  │
│ 流式计算     │ Flink        │ Spark Stream │ 实时特征计算  │
│ 监控         │ Prometheus   │ DataDog      │ 指标+告警     │
│ 链路追踪     │ Jaeger       │ Zipkin       │ 延迟分析      │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

## 小结

推荐系统的在线服务是一个复杂的系统工程，核心挑战在于在严格的延迟约束下完成大量计算。关键要点：

1. **分层架构**：召回→粗排→精排→重排，每层在速度和精度间权衡
2. **特征一致性**：用 Feature Store 消除训练-服务偏差，这是最容易被忽视但影响最大的问题
3. **模型推理优化**：动态批处理、TensorRT 加速、模型热更新是三大核心技术
4. **向量检索**：FAISS 的 IVF 索引通过聚类将搜索范围缩小到 1-5%，实现毫秒级亿级检索
5. **全链路可观测**：监控每个环节的延迟、错误率和业务指标，问题发生时能快速定位

选择开源方案时，Triton + Feast + Milvus 是一套经过验证的组合，覆盖了模型推理、特征服务和向量检索三大核心环节。

---

上一篇：[开源推荐框架详解（下）：工业级方案](10-opensource-industrial.md) | 返回：[目录](README.md)
