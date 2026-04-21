# 进阶：Agent 框架实战

> 目标：理解 LLM Agent 的核心架构，掌握 LangGraph、CrewAI 等主流框架，构建能自主规划和使用工具的 AI Agent。

---

## 1. Agent 核心概念

### 1.1 什么是 LLM Agent？

Agent = LLM + 记忆 + 工具 + 规划能力

```
用户输入 → Agent（LLM 大脑）
              ├── 思考：分析任务，制定计划
              ├── 行动：调用工具（搜索、代码执行、API...）
              ├── 观察：获取工具返回结果
              └── 循环：直到任务完成
```

### 1.2 Agent 架构模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| ReAct | 交替推理和行动 | 通用任务 |
| Plan-and-Execute | 先规划再执行 | 复杂多步任务 |
| Multi-Agent | 多个 Agent 协作 | 团队协作场景 |
| Reflexion | 自我反思改进 | 需要迭代优化 |

---

## 2. LangGraph：构建有状态的 Agent

LangGraph 是 LangChain 团队推出的 Agent 框架，基于图（Graph）定义 Agent 的工作流。

### 2.1 基础概念

```
Node（节点）：执行具体操作（调用 LLM、工具等）
Edge（边）：定义节点间的流转逻辑
State（状态）：在节点间传递的共享数据
```

### 2.2 构建 ReAct Agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4o")

@tool
def search(query: str) -> str:
    """搜索互联网获取最新信息"""
    # 实际项目中接入搜索 API
    return f"搜索结果：关于 '{query}' 的最新信息..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

@tool
def python_repl(code: str) -> str:
    """执行 Python 代码并返回结果"""
    import io, contextlib
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(code)
    return output.getvalue()

# 创建 Agent
agent = create_react_agent(llm, tools=[search, calculator, python_repl])

# 使用
result = agent.invoke({"messages": [("user", "帮我计算前100个素数的和")]})
print(result["messages"][-1].content)
```

### 2.3 自定义 Graph Agent

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str

def should_continue(state: AgentState) -> str:
    """决定是否继续执行"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

def call_model(state: AgentState):
    """调用 LLM"""
    response = llm.bind_tools([search, calculator]).invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState):
    """执行工具调用"""
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool_fn = {"search": search, "calculator": calculator}[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        results.append({"role": "tool", "content": result, "tool_call_id": tool_call["id"]})
    return {"messages": results}

# 构建图
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

app = graph.compile()

# 运行
result = app.invoke({"messages": [("user", "搜索今天的新闻，然后总结要点")]})
```

---

## 3. CrewAI：多 Agent 协作

CrewAI 让你定义多个有不同角色的 Agent，像团队一样协作完成任务。

### 3.1 安装与基础使用

```bash
pip install crewai crewai-tools
```

```python
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# 定义 Agent（角色）
researcher = Agent(
    role="研究员",
    goal="深入研究给定主题，收集全面准确的信息",
    backstory="你是一位经验丰富的研究员，擅长从多个角度分析问题",
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="技术作家",
    goal="将研究结果整理为清晰易懂的文章",
    backstory="你是一位技术写作专家，擅长将复杂概念用简单语言表达",
    llm=llm,
    verbose=True,
)

# 定义任务
research_task = Task(
    description="研究 {topic} 的最新进展，包括关键技术、主要玩家和未来趋势",
    expected_output="一份包含关键发现的研究报告",
    agent=researcher,
)

writing_task = Task(
    description="基于研究报告，撰写一篇 1000 字的技术博客文章",
    expected_output="一篇结构清晰、通俗易懂的技术文章",
    agent=writer,
)

# 组建团队
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True,
)

# 执行
result = crew.kickoff(inputs={"topic": "大模型推理优化技术"})
print(result)
```

---

## 4. 工具集成实战

### 4.1 常用工具类型

```python
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import subprocess, json

# 网络搜索
search_tool = DuckDuckGoSearchRun()

# 文件操作
@tool
def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path) as f:
        return f.read()

@tool
def write_file(path: str, content: str) -> str:
    """写入文件"""
    with open(path, "w") as f:
        f.write(content)
    return f"已写入 {path}"

# Shell 命令
@tool
def run_command(command: str) -> str:
    """执行 shell 命令"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
    return result.stdout or result.stderr

# API 调用
@tool
def call_api(url: str, method: str = "GET") -> str:
    """调用 HTTP API"""
    import requests
    resp = requests.request(method, url, timeout=10)
    return resp.text[:2000]
```

### 4.2 构建代码助手 Agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

system_prompt = """你是一个编程助手 Agent。你可以：
1. 编写和执行 Python 代码
2. 读写文件
3. 运行 shell 命令

请一步步完成用户的编程任务。每次执行代码后检查结果，如有错误则修复。"""

coding_agent = create_react_agent(
    llm,
    tools=[python_repl, read_file, write_file, run_command],
    state_modifier=system_prompt,
)

result = coding_agent.invoke({
    "messages": [("user", "创建一个 Python 脚本，读取当前目录下所有 .py 文件，统计总代码行数")]
})
```

---

## 5. Agent 设计最佳实践

### 5.1 常见问题与解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Agent 陷入循环 | 工具返回不明确 | 设置最大迭代次数，改进工具描述 |
| 工具调用错误 | 参数理解偏差 | 提供更详细的 tool description 和示例 |
| 任务偏离 | 上下文丢失 | 使用 memory 保持上下文 |
| 响应太慢 | 多次 LLM 调用 | 并行工具调用，使用更快的模型 |

### 5.2 设计原则

```
1. 工具描述要精确：description 是 Agent 选择工具的唯一依据
2. 限制工具数量：太多工具会让 Agent 困惑（建议 < 10 个）
3. 设置安全边界：限制文件访问范围、命令执行权限
4. 添加人工确认：关键操作前要求用户确认
5. 实现优雅降级：工具失败时有备选方案
```

---

## 6. 推荐学习材料

### 框架文档
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) - 官方文档
- [CrewAI 文档](https://docs.crewai.com/) - 多 Agent 框架
- [AutoGen](https://microsoft.github.io/autogen/) - 微软多 Agent 框架

### 论文
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761) - 让 LLM 学会使用工具
- [Voyager](https://arxiv.org/abs/2305.16291) - Minecraft 中的 LLM Agent

### 博客
- [Lilian Weng - LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [LangChain Blog - Agent 教程](https://blog.langchain.dev/)
