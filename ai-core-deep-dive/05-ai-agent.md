# AI Agent：从工具调用到自主决策

> LLM 本身只能生成文本。但当它能调用工具、规划步骤、自我反思时，就从"聊天机器人"进化为"AI Agent"——一个能自主完成复杂任务的智能体。

## 什么是 AI Agent

```
传统 LLM:  用户提问 → 模型回答 → 结束

AI Agent:  用户给目标 → 模型规划 → 调用工具 → 观察结果 
           → 反思调整 → 继续执行 → ... → 完成目标
```

Agent 的核心能力：
1. **规划（Planning）**：将复杂任务分解为可执行的步骤
2. **工具使用（Tool Use）**：调用 API、执行代码、搜索网络
3. **记忆（Memory）**：记住之前的操作和结果
4. **反思（Reflection）**：评估自己的输出，发现并修正错误

## 工具调用（Function Calling）

这是 Agent 最基础的能力。模型不直接回答，而是决定调用哪个工具：

```
用户: "北京今天天气怎么样？"

模型思考: 我需要查询实时天气，应该调用天气 API

模型输出:
{
  "function": "get_weather",
  "arguments": {"city": "北京", "date": "today"}
}

系统执行函数，返回结果:
{"temperature": 22, "condition": "晴", "humidity": 45}

模型最终回答: "北京今天晴天，气温 22°C，湿度 45%。"
```

### 工具定义格式

OpenAI 风格的工具定义：

```json
{
  "type": "function",
  "function": {
    "name": "search_database",
    "description": "搜索产品数据库",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "搜索关键词"},
        "category": {"type": "string", "enum": ["电子", "服装", "食品"]}
      },
      "required": ["query"]
    }
  }
}
```

模型通过训练学会了根据工具描述选择合适的工具，并生成正确的参数。

### 并行工具调用

现代模型支持一次调用多个工具：

```
用户: "比较北京和上海的天气"

模型输出:
[
  {"function": "get_weather", "arguments": {"city": "北京"}},
  {"function": "get_weather", "arguments": {"city": "上海"}}
]
```

## Agent 架构模式

### ReAct（Reasoning + Acting）

最经典的 Agent 模式，交替进行推理和行动：

```
Thought: 用户想知道最新的 AI 论文，我需要搜索 ArXiv
Action: search_arxiv(query="large language model", date="2024")
Observation: 找到 5 篇相关论文...

Thought: 我需要获取排名最高的那篇论文的摘要
Action: get_paper_abstract(paper_id="2401.xxxxx")
Observation: 这篇论文提出了...

Thought: 现在我有足够信息回答用户了
Answer: 最新的 AI 论文中，最值得关注的是...
```

ReAct 的优势是推理过程透明，便于调试。

### Plan-and-Execute

先制定完整计划，再逐步执行：

```
Plan:
  1. 搜索用户需要的数据
  2. 下载并解析数据
  3. 进行数据分析
  4. 生成可视化图表
  5. 撰写分析报告

Execute:
  Step 1: [执行搜索] → 成功
  Step 2: [下载数据] → 成功
  Step 3: [分析数据] → 发现异常值，需要调整计划
  Re-plan: 增加数据清洗步骤...
```

适合复杂的多步骤任务。

### 多 Agent 协作

多个专业化 Agent 协同工作：

```
用户需求: "帮我写一篇技术博客"

Researcher Agent → 搜索相关资料
Writer Agent → 撰写初稿
Reviewer Agent → 审核并提出修改建议
Editor Agent → 最终润色
```

框架如 AutoGen、CrewAI 支持这种模式。

## 记忆系统

Agent 需要记忆来处理长期任务：

```
短期记忆 (Short-term):
  - 当前对话上下文
  - 最近几步的工具调用结果
  - 实现: 直接放在 prompt 中

长期记忆 (Long-term):
  - 用户偏好、历史交互
  - 之前任务的经验总结
  - 实现: 向量数据库 + 检索
```

### 记忆管理策略

| 策略 | 说明 |
|------|------|
| 滑动窗口 | 只保留最近 N 轮对话 |
| 摘要压缩 | 用 LLM 将历史对话压缩为摘要 |
| 实体记忆 | 提取并维护关键实体信息 |
| 向量检索 | 将历史存入向量库，按需检索 |

## 反思与自我修正

让 Agent 评估自己的输出并改进：

```
生成代码 → 执行 → 报错
  ↓
反思: "错误是 IndexError，说明数组越界，我需要添加边界检查"
  ↓
修正代码 → 再次执行 → 成功
```

Reflexion 框架将这种模式系统化：
1. 执行任务
2. 评估结果
3. 生成反思（总结失败原因）
4. 将反思加入记忆
5. 重新尝试

## 主流 Agent 框架

| 框架 | 特点 |
|------|------|
| LangChain/LangGraph | 最流行，生态丰富，支持复杂工作流 |
| AutoGen | 微软，多 Agent 对话框架 |
| CrewAI | 角色扮演式多 Agent 协作 |
| Semantic Kernel | 微软，企业级 Agent 框架 |
| Dify | 低代码 Agent 构建平台 |

### LangGraph 示例架构

```
用户输入 → [路由节点] → 需要工具? 
                          ├─ 是 → [工具调用节点] → [结果处理节点] → 回到路由
                          └─ 否 → [生成回答节点] → 输出
```

LangGraph 用图（Graph）来定义 Agent 的工作流，节点是操作，边是条件跳转。

## MCP（Model Context Protocol）

Anthropic 提出的开放协议，标准化 Agent 与外部工具/数据源的连接：

```
Agent ←→ MCP Client ←→ MCP Server ←→ 外部服务
                                       ├─ 数据库
                                       ├─ API
                                       ├─ 文件系统
                                       └─ ...
```

MCP 的价值：一次实现工具接口，所有支持 MCP 的 Agent 都能使用。类似于 USB 协议统一了外设接口。

## 当前挑战

1. **可靠性**：Agent 在复杂任务上的成功率仍然不够高
2. **成本**：多步推理意味着多次 LLM 调用，成本高
3. **安全性**：Agent 能执行代码、调用 API，权限控制至关重要
4. **评估**：如何衡量 Agent 的能力？SWE-bench、WebArena 等基准在探索
5. **幻觉传播**：一步的错误可能在后续步骤中被放大

## 实践建议

1. **从简单开始**：先实现单工具调用，再逐步增加复杂度
2. **工具描述要清晰**：模型选择工具的准确性高度依赖工具描述的质量
3. **设置护栏**：限制最大步骤数、敏感操作需人工确认
4. **结构化输出**：用 JSON Schema 约束模型的工具调用格式
5. **可观测性**：记录每一步的推理和行动，便于调试

## 小结

AI Agent 代表了 LLM 应用的下一个阶段——从被动回答到主动执行。工具调用让模型能与外部世界交互，规划和反思让模型能处理复杂任务。虽然当前 Agent 的可靠性仍有提升空间，但这个方向的潜力巨大。

---

上一篇：[RAG：检索增强生成](04-rag.md) | 下一篇：[多模态大模型](06-multimodal.md)
