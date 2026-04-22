# 开源推荐框架详解（上）：经典框架

> 学习推荐系统最好的方式是动手实践。本文详细介绍最流行的开源推荐框架，帮你快速上手实验和开发。

## 🗺️ 开源推荐框架全景

```
┌─────────────────────────────────────────────────────┐
│                                                      │
│  学术研究 / 快速实验                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Surprise │ │ RecBole  │ │ LensKit  │            │
│  └──────────┘ └──────────┘ └──────────┘            │
│                                                      │
│  深度学习推荐模型                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │DeepCTR   │ │FuxiCTR   │ │ BARS     │            │
│  └──────────┘ └──────────┘ └──────────┘            │
│                                                      │
│  工业级 / 生产环境                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Merlin   │ │ TFRec    │ │EasyRec   │            │
│  └──────────┘ └──────────┘ └──────────┘            │
│                                                      │
│  端到端平台                                          │
│  ┌──────────┐ ┌──────────────┐                      │
│  │ Gorse    │ │ Microsoft    │                      │
│  │          │ │ Recommenders │                      │
│  └──────────┘ └──────────────┘                      │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## 1️⃣ Surprise — 入门首选

> GitHub: https://github.com/NicolasHug/Surprise

```
定位: 经典推荐算法的 Python 库，适合入门学习
Star: 6.5K+
语言: Python
```

### 支持的算法

```
┌──────────────────────────────────────────┐
│ 基线算法:                                │
│   NormalPredictor, BaselineOnly          │
│                                          │
│ 近邻方法:                                │
│   KNNBasic, KNNWithMeans, KNNWithZScore  │
│   KNNBaseline                            │
│                                          │
│ 矩阵分解:                                │
│   SVD, SVD++, NMF                        │
│                                          │
│ 其他:                                    │
│   SlopeOne, CoClustering                 │
└──────────────────────────────────────────┘
```

### 快速上手

```python
from surprise import Dataset, SVD, accuracy
from surprise.model_selection import cross_validate, train_test_split

# 加载内置的 MovieLens 100K 数据集
data = Dataset.load_builtin('ml-100k')

# 使用 SVD 算法
algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)

# 5 折交叉验证
results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# 输出: RMSE ≈ 0.934, MAE ≈ 0.737

# 训练并预测
trainset, testset = train_test_split(data, test_size=0.2)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# 为特定用户推荐
uid = '196'
# 获取用户未评分的物品
trainset_inner = algo.trainset
anti_testset = trainset_inner.build_anti_testset()
user_predictions = [p for p in algo.test(anti_testset) if p.uid == uid]
user_predictions.sort(key=lambda x: x.est, reverse=True)
top_10 = user_predictions[:10]
for pred in top_10:
    print(f"  物品 {pred.iid}: 预测评分 {pred.est:.2f}")
```

### 适用场景

```
✅ 学习经典推荐算法的原理
✅ 快速验证想法和做实验
✅ 小规模数据集上的推荐

❌ 不支持深度学习模型
❌ 不适合大规模数据
❌ 不支持隐式反馈（只支持评分数据）
```

## 2️⃣ RecBole — 最全面的推荐算法库

> GitHub: https://github.com/RUCAIBox/RecBole

```
定位: 统一的推荐算法框架，覆盖 90+ 种模型
Star: 3.5K+
语言: Python (PyTorch)
机构: 中国人民大学
```

### 支持的模型（90+）

```
┌──────────────────────────────────────────────────┐
│ 通用推荐 (General):                               │
│   BPR, NeuMF, LightGCN, NGCF, SASRec, BERT4Rec  │
│                                                   │
│ 序列推荐 (Sequential):                            │
│   GRU4Rec, Caser, SASRec, BERT4Rec, FDSA         │
│                                                   │
│ CTR 预估 (Context-aware):                         │
│   FM, DeepFM, Wide&Deep, DCN, DIN, AutoInt        │
│                                                   │
│ 知识图谱推荐 (Knowledge-based):                    │
│   KGAT, KGNN-LS, RippleNet, KGCN                 │
│                                                   │
│ 对话推荐 (Conversational):                        │
│   ReDial, KBRD, KGSF                             │
└──────────────────────────────────────────────────┘
```

### 快速上手

```python
from recbole.quick_start import run_recbole

# 一行代码运行实验
result = run_recbole(
    model='BPR',           # 模型名称
    dataset='ml-100k',     # 数据集
    config_dict={
        'epochs': 50,
        'learning_rate': 0.001,
        'embedding_size': 64,
        'train_batch_size': 2048,
    }
)
```

### 配置文件方式

```yaml
# config.yaml
model: DeepFM
dataset: ml-1m

# 训练配置
epochs: 100
learning_rate: 0.001
train_batch_size: 4096
eval_batch_size: 4096

# 模型配置
embedding_size: 64
mlp_hidden_size: [256, 128, 64]
dropout_prob: 0.2

# 评估配置
eval_args:
  split: {'RS': [0.8, 0.1, 0.1]}
  order: RO
  mode: full
metrics: ['Recall', 'NDCG', 'Hit', 'MRR']
topk: [5, 10, 20]
```

```python
from recbole.quick_start import run_recbole

run_recbole(config_file_list=['config.yaml'])
```

### 自定义数据集

```
数据集目录结构:
  my_dataset/
  ├── my_dataset.inter    # 交互数据 (必须)
  ├── my_dataset.user     # 用户特征 (可选)
  └── my_dataset.item     # 物品特征 (可选)

my_dataset.inter 格式:
  user_id:token  item_id:token  rating:float  timestamp:float
  user_001       item_123       4.0           1609459200
  user_001       item_456       5.0           1609545600
  user_002       item_123       3.0           1609632000
```

### 适用场景

```
✅ 学术研究，快速复现论文
✅ 算法对比实验
✅ 支持 90+ 种模型，覆盖面最广
✅ 统一的数据格式和评估框架

❌ 不适合直接用于生产环境
❌ 大规模数据性能有限
❌ 自定义模型需要了解框架内部结构
```

## 3️⃣ DeepCTR — 深度 CTR 模型集合

> GitHub: https://github.com/shenweichen/DeepCTR

```
定位: CTR 预估深度学习模型的统一实现
Star: 7.5K+
语言: Python (TensorFlow/PyTorch)
```

### 支持的模型

```
┌──────────────────────────────────────────────────┐
│ 经典模型:                                         │
│   FM, FFM, FNN, PNN, Wide&Deep                   │
│                                                   │
│ 注意力模型:                                       │
│   DIN, DIEN, AFM, AutoInt, InterHAt              │
│                                                   │
│ 交叉网络:                                         │
│   DeepFM, DCN, DCN-V2, xDeepFM                  │
│                                                   │
│ 多任务模型:                                       │
│   SharedBottom, ESMM, MMOE, PLE                  │
│                                                   │
│ 序列模型:                                         │
│   DIN, DIEN, BST, DSIN                           │
└──────────────────────────────────────────────────┘
```

### 快速上手

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

# 准备数据
data = pd.read_csv('criteo_sample.csv')

# 定义特征列
sparse_features = ['C1', 'C2', 'C3']  # 类别特征
dense_features = ['I1', 'I2', 'I3']   # 数值特征

# 特征编码
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

# 定义特征列
fixlen_feature_columns = [
    SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=8)
    for feat in sparse_features
] + [
    DenseFeat(feat, 1) for feat in dense_features
]

# 构建模型
model = DeepFM(
    linear_feature_columns=fixlen_feature_columns,
    dnn_feature_columns=fixlen_feature_columns,
    dnn_hidden_units=(256, 128, 64),
    dnn_dropout=0.2,
    task='binary'
)

model.compile('adam', 'binary_crossentropy', metrics=['AUC'])

# 训练
feature_names = get_feature_names(fixlen_feature_columns)
train_input = {name: data[name].values for name in feature_names}
model.fit(train_input, data['label'].values, batch_size=256, epochs=10, validation_split=0.2)
```

### 适用场景

```
✅ 快速实现和对比各种 CTR 模型
✅ API 设计简洁，上手容易
✅ 同时支持 TensorFlow 和 PyTorch
✅ 适合从研究到小规模生产的过渡

❌ 不包含召回模型
❌ 大规模分布式训练支持有限
❌ 特征工程需要自己处理
```

## 4️⃣ Microsoft Recommenders — 微软推荐工具箱

> GitHub: https://github.com/microsoft/recommenders

```
定位: 推荐系统最佳实践和示例代码集合
Star: 19K+
语言: Python
机构: 微软
```

### 特色

```
不只是一个算法库，更是一个学习资源:

  ├── 算法实现 (经典 + 深度学习)
  ├── 完整的 Jupyter Notebook 教程
  ├── 数据处理工具
  ├── 评估工具
  └── Azure 云部署示例

支持的算法:
  经典: SVD, SAR, LightGCN, BPR
  深度: Wide&Deep, DeepFM, DKN, xDeepFM
  序列: SASRec, SLi-Rec, GRU4Rec
  其他: LightGBM, Vowpal Wabbit, RBM
```

### 快速上手

```python
from recommenders.datasets import movielens
from recommenders.models.sar import SAR
from recommenders.evaluation.python_evaluation import (
    map_at_k, ndcg_at_k, precision_at_k, recall_at_k
)

# 加载数据
data = movielens.load_pandas_df(size='100k',
    header=['userID', 'itemID', 'rating', 'timestamp'])

# 划分数据
from recommenders.datasets.python_splitters import python_chrono_split
train, test = python_chrono_split(data, ratio=0.75)

# SAR (Smart Adaptive Recommendations) 模型
model = SAR(
    col_user='userID', col_item='itemID', col_rating='rating',
    col_timestamp='timestamp', similarity_type='jaccard',
    time_decay_coefficient=30, timedecay_formula=True
)
model.fit(train)

# 预测
top_k = model.recommend_k_items(test, top_k=10, remove_seen=True)

# 评估
eval_map = map_at_k(test, top_k, k=10)
eval_ndcg = ndcg_at_k(test, top_k, k=10)
eval_precision = precision_at_k(test, top_k, k=10)
eval_recall = recall_at_k(test, top_k, k=10)

print(f"MAP@10: {eval_map:.4f}")
print(f"NDCG@10: {eval_ndcg:.4f}")
print(f"Precision@10: {eval_precision:.4f}")
print(f"Recall@10: {eval_recall:.4f}")
```

### 适用场景

```
✅ 学习推荐系统的最佳入口（丰富的 Notebook 教程）
✅ 完整的评估工具
✅ 从实验到 Azure 部署的完整链路
✅ 社区活跃，文档完善

❌ 算法实现不如专门框架全面
❌ 部分功能依赖 Azure
```

## 5️⃣ Gorse — 开箱即用的推荐服务

> GitHub: https://github.com/gorse-io/gorse

```
定位: 开箱即用的推荐系统服务，提供 REST API
Star: 8.5K+
语言: Go
```

### 架构

```
┌─────────────────────────────────────────────┐
│                 Gorse                        │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Master   │  │ Worker   │  │ Server   │  │
│  │ 协调调度 │  │ 模型训练 │  │ REST API │  │
│  └──────────┘  └──────────┘  └──────────┘  │
│       ↕              ↕              ↕        │
│  ┌──────────────────────────────────────┐   │
│  │  存储层: Redis + MySQL/PostgreSQL    │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘

支持的推荐方式:
  - 协同过滤 (ItemCF, UserCF)
  - 矩阵分解 (BPR)
  - 基于内容的推荐
  - 热门推荐
  - 最新推荐
```

### 快速部署

```yaml
# docker-compose.yml
version: "3"
services:
  gorse:
    image: zhenghaoz/gorse-in-one
    restart: unless-stopped
    ports:
      - 8086:8086   # REST API
      - 8088:8088   # Dashboard
    environment:
      GORSE_CACHE_STORE: redis://redis:6379
      GORSE_DATA_STORE: mysql://gorse:password@tcp(mysql:3306)/gorse
    depends_on:
      - redis
      - mysql
```

### REST API 使用

```bash
# 插入用户
curl -X POST http://localhost:8086/api/user \
  -H 'Content-Type: application/json' \
  -d '{"UserId": "user_001", "Labels": ["tech", "music"]}'

# 插入物品
curl -X POST http://localhost:8086/api/item \
  -H 'Content-Type: application/json' \
  -d '{"ItemId": "item_123", "Labels": ["electronics"], "Categories": ["phone"]}'

# 插入反馈
curl -X POST http://localhost:8086/api/feedback \
  -H 'Content-Type: application/json' \
  -d '[{"FeedbackType": "click", "UserId": "user_001", "ItemId": "item_123"}]'

# 获取推荐
curl http://localhost:8086/api/recommend/user_001?n=10
```

### 适用场景

```
✅ 中小型项目快速搭建推荐系统
✅ 不需要写代码，REST API 开箱即用
✅ 自带 Dashboard 监控
✅ 支持 Docker 一键部署

❌ 不支持深度学习模型
❌ 大规模数据性能有限
❌ 自定义能力有限
```

## 📊 经典框架对比

```
┌──────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│              │ Surprise │ RecBole  │ DeepCTR  │ MS Rec   │ Gorse    │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 定位         │ 入门学习 │ 算法研究 │ CTR 模型 │ 最佳实践 │ 即用服务 │
│ 模型数量     │ ~10      │ 90+      │ ~30      │ ~20      │ ~5       │
│ 深度学习     │ ❌       │ ✅       │ ✅       │ ✅       │ ❌       │
│ 上手难度     │ ⭐       │ ⭐⭐     │ ⭐⭐     │ ⭐⭐     │ ⭐       │
│ 生产可用     │ ❌       │ ❌       │ ⚠️       │ ⚠️       │ ✅       │
│ 文档质量     │ 好       │ 好       │ 好       │ 优秀     │ 好       │
│ 适合场景     │ 课程作业 │ 论文复现 │ CTR 实验 │ 系统学习 │ 小型项目 │
└──────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

## 🎯 选择建议

```
你是谁？                          推荐框架

刚接触推荐系统的学生              → Surprise + MS Recommenders
要复现论文做研究                  → RecBole
要实验 CTR 模型                  → DeepCTR / FuxiCTR
要快速搭建一个推荐服务            → Gorse
要在生产环境部署                  → 看下一篇（工业级方案）
```

## 小结

经典开源框架各有侧重：Surprise 适合入门，RecBole 适合研究，DeepCTR 适合 CTR 模型实验，MS Recommenders 适合系统学习，Gorse 适合快速搭建服务。对于需要生产级部署的场景，请看下一篇的工业级方案。

---

上一篇：[推荐系统评估指标与 A/B 测试](08-evaluation.md) | 下一篇：[开源推荐框架详解（下）：工业级方案](10-opensource-industrial.md)
