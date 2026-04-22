# Embedding 在推荐中的应用

> Embedding 是将离散的 ID（用户 ID、物品 ID）或高维稀疏特征映射为低维稠密向量的技术。它是连接传统推荐和深度学习推荐的桥梁，也是现代推荐系统最核心的技术之一。

## 💡 为什么需要 Embedding

```
传统表示方法 — One-Hot 编码:

  物品 ID = 12345，总共 100 万个物品
  → [0, 0, ..., 0, 1, 0, ..., 0]  (100 万维，只有一个 1)

  问题:
  ❌ 维度太高 (100 万维)
  ❌ 极度稀疏 (99.9999% 是 0)
  ❌ 无法表达物品之间的关系 (任意两个 one-hot 向量正交)

Embedding 表示:

  物品 ID = 12345
  → [0.12, -0.34, 0.56, ..., 0.78]  (128 维，每个值都有意义)

  优势:
  ✅ 低维稠密 (128 维)
  ✅ 语义丰富 (相似物品的向量相近)
  ✅ 可以做向量运算 (加减、相似度计算)
```

## 🎯 Word2Vec → Item2Vec

### Word2Vec 的启发

```
NLP 中的 Word2Vec:
  "国王" - "男人" + "女人" ≈ "女王"

  训练方法: 用词的上下文来学习词向量
  "我 喜欢 [吃] 苹果" → "吃" 的上下文是 "喜欢" 和 "苹果"
  → 经常出现在相似上下文中的词，向量相近
```

### Item2Vec：把物品当"词"

```
核心类比:
  NLP:  句子 = [词1, 词2, 词3, ...]
  推荐: 用户行为序列 = [物品1, 物品2, 物品3, ...]

  用户的浏览序列就像一个"句子"
  物品就像"词"
  用 Word2Vec 的方法训练物品向量

用户 Alice 的行为序列:
  [iPhone手机壳, AirPods, MacBook, iPad保护套, Apple Watch]

  → "iPhone手机壳" 和 "AirPods" 经常在同一序列中出现
  → 它们的 Embedding 向量会很接近
  → 推荐时: 买了 iPhone 手机壳 → 推荐 AirPods
```

### Skip-Gram 模型

```
训练过程:

  行为序列: [A, B, C, D, E]
  窗口大小: 2

  训练样本 (中心词, 上下文词):
    (A, B), (A, C)
    (B, A), (B, C), (B, D)
    (C, A), (C, B), (C, D), (C, E)
    ...

  目标: 给定中心物品，预测上下文物品
  
  max Σ log P(context | center)
  P(context | center) = softmax(v_context · v_center)
```

### 负采样优化

```
Softmax 需要对所有物品计算 → 太慢

负采样: 只对少量负样本计算
  正样本: (iPhone手机壳, AirPods) → 应该相似
  负样本: (iPhone手机壳, 洗衣液) → 应该不相似 (随机采样)

  目标: max log σ(v_pos · v_center) + Σ log σ(-v_neg · v_center)
```

## 🏗️ 双塔模型（Two-Tower Model）

推荐系统中最重要的 Embedding 模型，广泛用于召回阶段。

```
┌──────────────┐          ┌──────────────┐
│  用户塔       │          │  物品塔       │
│  (User Tower)│          │  (Item Tower)│
│              │          │              │
│  用户 ID     │          │  物品 ID     │
│  年龄/性别   │          │  类别/标签   │
│  历史行为    │          │  标题 emb    │
│  ...         │          │  ...         │
│      ↓       │          │      ↓       │
│  [MLP 网络]  │          │  [MLP 网络]  │
│      ↓       │          │      ↓       │
│  用户向量    │          │  物品向量    │
│  (128维)     │          │  (128维)     │
└──────┬───────┘          └──────┬───────┘
       │                         │
       └────────┐  ┌─────────────┘
                ↓  ↓
         相似度 = cos(u, v)
                ↓
         预测: 用户是否会点击该物品
```

### 为什么叫"双塔"

```
两个独立的网络（塔），分别编码用户和物品:

  优势:
  ✅ 物品向量可以离线预计算并建索引
  ✅ 在线只需计算用户向量 + ANN 检索
  ✅ 检索速度极快 (~5ms 从亿级物品中找 top-K)

  劣势:
  ❌ 用户和物品的交互只在最后一步 (内积)
  ❌ 无法建模复杂的交叉特征
  → 所以双塔主要用于召回，不用于精排
```

### 训练方法

```
正样本: 用户实际点击/购买的 (user, item) 对
负样本: 用户没有交互的 (user, item) 对

负采样策略:
  1. 随机负采样: 随机选物品作为负样本
  2. In-batch 负采样: 同一 batch 内其他用户的正样本作为负样本
  3. 难负样本: 选择与正样本相似但用户没点击的物品

损失函数:
  Softmax Cross-Entropy 或 Sampled Softmax
```

## 🔄 Graph Embedding

用户和物品的关系可以构成图，用图方法学习 Embedding。

### DeepWalk / Node2Vec

```
用户-物品二部图:

  User1 ── Item_A
  User1 ── Item_B
  User2 ── Item_B
  User2 ── Item_C
  User3 ── Item_A
  User3 ── Item_C

随机游走生成序列:
  User1 → Item_A → User3 → Item_C → User2 → Item_B → User1

  → 把随机游走序列当作"句子"
  → 用 Word2Vec 训练节点 Embedding
  → 经常在同一游走路径上的节点，向量相近
```

### Node2Vec 的改进

```
Node2Vec 引入两个参数控制游走策略:

  p (Return Parameter): 控制回到上一个节点的概率
  q (In-Out Parameter): 控制探索远处节点的概率

  p 小: 倾向于局部探索 (BFS-like) → 捕捉社区结构
  q 小: 倾向于远处探索 (DFS-like) → 捕捉全局角色

  ┌─────────────────────────────────┐
  │  BFS (p小):                     │
  │  A ─ B ─ A ─ C ─ A ─ D         │
  │  → 学到 A 的邻居关系            │
  │                                 │
  │  DFS (q小):                     │
  │  A ─ B ─ E ─ F ─ G ─ H         │
  │  → 学到 A 在图中的结构角色      │
  └─────────────────────────────────┘
```

## 🏢 工业实践中的 Embedding

### YouTube 的深度推荐（2016）

```
YouTube 的经典论文 "Deep Neural Networks for YouTube Recommendations"

召回阶段:
  用户特征:
    - 观看历史 (视频 ID 的 Embedding 平均)
    - 搜索历史 (搜索词的 Embedding 平均)
    - 地理位置、年龄、性别

  → MLP → 用户 Embedding (256维)
  → 在视频 Embedding 索引中做 ANN 检索

关键技巧:
  1. 用观看历史的 Embedding 平均值作为用户表示
  2. 训练时用 Softmax 分类 (预测下一个观看的视频)
  3. 服务时用 ANN 近似 Softmax
```

### Airbnb 的 Listing Embedding（2018）

```
Airbnb 的搜索排序 Embedding:

  特殊之处: 考虑了预订行为
  
  浏览序列: [房源A, 房源B, 房源C, ..., 房源X(预订)]
  
  训练时:
    - 普通浏览: 用 Skip-Gram 学习相似性
    - 最终预订: 额外加强预订房源与序列中所有房源的关联
    
  效果:
    - 相似风格的房源 Embedding 相近
    - 同一城市的房源 Embedding 相近
    - 价格相近的房源 Embedding 相近
    → 这些关系是自动学到的，不需要人工定义
```

## 📐 Embedding 的可视化

```
将高维 Embedding 用 t-SNE 降维到 2D 可视化:

  ●●● 科幻电影 (聚在一起)
     ●●●
  
              ▲▲▲ 爱情电影 (聚在一起)
             ▲▲▲
  
  ■■■ 动作电影
    ■■■
                    ◆◆◆ 纪录片
                   ◆◆◆

  → 同类物品自动聚类
  → 类别之间的距离反映了它们的关系
     (科幻和动作较近，科幻和爱情较远)
```

## 🔑 Embedding 的关键技巧

### 1. Embedding 维度选择

```
维度太低: 表达能力不足，不同物品的向量挤在一起
维度太高: 过拟合，计算和存储成本高

经验法则:
  物品数量 < 1 万:    维度 32-64
  物品数量 1 万-100 万: 维度 64-128
  物品数量 > 100 万:   维度 128-256

Google 的建议: d ≈ 物品数量的四次方根
  100 万物品 → d ≈ 100万^0.25 ≈ 31 → 取 32
```

### 2. 预训练 Embedding

```
先用无监督方法 (Item2Vec) 预训练 Embedding
再在下游任务 (CTR 预估) 中微调

好处:
  - 利用全量行为数据学习通用表示
  - 下游任务训练更快、效果更好
  - 缓解数据稀疏问题
```

### 3. 多模态 Embedding 融合

```
物品的 Embedding 可以融合多种信息:

  ID Embedding:    基于行为学习的向量
  文本 Embedding:  标题/描述的语义向量 (BERT)
  图像 Embedding:  封面图的视觉向量 (ResNet/ViT)
  
  融合方式:
    拼接: [ID_emb; Text_emb; Image_emb]
    加权: α × ID_emb + β × Text_emb + γ × Image_emb
    注意力: Attention([ID_emb, Text_emb, Image_emb])
```

## 小结

Embedding 是现代推荐系统的基石技术：
- **Item2Vec** 将 NLP 的思想引入推荐，用行为序列学习物品向量
- **双塔模型** 是工业界最重要的召回方案，支持毫秒级向量检索
- **Graph Embedding** 利用图结构学习更丰富的关系
- Embedding 的质量直接决定了推荐系统的效果上限

---

上一篇：[推荐系统的工程架构](04-architecture.md) | 下一篇：[深度学习推荐模型演进](06-deep-learning-models.md)
