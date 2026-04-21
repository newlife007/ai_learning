# 第3-4周：大模型应用开发实战

> 本阶段目标：掌握大模型 API 调用、Prompt Engineering 技巧、RAG 架构，并完成一个知识问答系统项目。

---

## 1. 大模型 API 调用

### 1.1 OpenAI API

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "用一句话解释什么是大语言模型"}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

**关键参数说明：**

| 参数 | 作用 | 常用值 |
|------|------|--------|
| `temperature` | 控制随机性，越高越有创意 | 0-1，默认 0.7 |
| `max_tokens` | 最大输出 token 数 | 根据需求设置 |
| `top_p` | 核采样，与 temperature 二选一 | 0.9-1.0 |
| `frequency_penalty` | 降低重复内容 | 0-2 |
| `stream` | 流式输出 | True/False |

### 1.2 流式输出

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "写一首关于AI的诗"}],
    stream=True
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

### 1.3 调用开源模型 API（兼容 OpenAI 格式）

很多开源模型部署后提供 OpenAI 兼容接口：

```python
# 调用本地部署的模型（如 vLLM、Ollama）
client = OpenAI(
    base_url="http://localhost:8000/v1",  # 本地服务地址
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="qwen2-7b",
    messages=[{"role": "user", "content": "你好"}]
)
```

### 1.4 Claude API

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "解释 Transformer 架构"}]
)

print(message.content[0].text)
```

---

## 2. Prompt Engineering

Prompt Engineering 是与大模型高效交互的核心技能。好的 prompt 能让模型输出质量提升数倍。

### 2.1 基础原则

```
1. 明确具体：告诉模型你要什么，而不是不要什么
2. 提供上下文：给模型足够的背景信息
3. 指定格式：明确输出格式（JSON、列表、表格等）
4. 分步引导：复杂任务拆解为步骤
```

### 2.2 Few-shot Prompting（少样本提示）

通过提供示例来引导模型输出：

```python
prompt = """请对以下评论进行情感分类。

示例：
评论：这个产品太好用了，强烈推荐！
情感：正面

评论：质量很差，用了一天就坏了。
情感：负面

评论：一般般，没什么特别的。
情感：中性

请分类以下评论：
评论：包装精美，但功能不如预期。
情感："""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)
```

### 2.3 Chain-of-Thought（思维链）

让模型"一步步思考"，显著提升推理能力：

```python
# 直接提问 - 容易出错
prompt_direct = "一个房间有23个人，至少两人生日相同的概率是多少？"

# CoT 提示 - 准确率大幅提升
prompt_cot = """一个房间有23个人，至少两人生日相同的概率是多少？

请一步步思考：
1. 先计算所有人生日都不同的概率
2. 然后用 1 减去这个概率
3. 给出最终答案"""
```

### 2.4 ReAct（推理+行动）

结合推理和工具调用的模式：

```python
system_prompt = """你是一个能使用工具的AI助手。

可用工具：
- search(query): 搜索互联网
- calculator(expression): 计算数学表达式

请按以下格式回答：
Thought: 我需要思考...
Action: 工具名(参数)
Observation: 工具返回结果
... (重复直到得到答案)
Answer: 最终答案"""
```

### 2.5 结构化输出

```python
prompt = """分析以下文本，以 JSON 格式输出结果。

文本：苹果公司今天发布了新款 iPhone，售价 7999 元起。

请输出以下格式的 JSON：
{
    "company": "公司名",
    "product": "产品名",
    "price": "价格",
    "sentiment": "正面/负面/中性"
}"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"}  # 强制 JSON 输出
)
```

---

## 3. LangChain 框架

LangChain 是构建 LLM 应用的主流框架，提供了链式调用、记忆管理、工具集成等能力。

### 3.1 基础使用

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)

messages = [
    SystemMessage(content="你是一个Python专家"),
    HumanMessage(content="如何用Python读取CSV文件？")
]

response = llm.invoke(messages)
print(response.content)
```

### 3.2 Prompt Template

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，请用{style}的风格回答问题。"),
    ("user", "{question}")
])

chain = prompt | llm

response = chain.invoke({
    "role": "历史学家",
    "style": "通俗易懂",
    "question": "为什么罗马帝国会衰落？"
})
print(response.content)
```

### 3.3 输出解析

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="电影名称")
    rating: float = Field(description="评分 1-10")
    summary: str = Field(description="一句话总结")

parser = JsonOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "分析用户提到的电影。{format_instructions}"),
    ("user", "{input}")
])

chain = prompt | llm | parser

result = chain.invoke({
    "input": "我昨天看了《星际穿越》，太震撼了！",
    "format_instructions": parser.get_format_instructions()
})
print(result)  # {'title': '星际穿越', 'rating': 9.5, 'summary': '...'}
```

### 3.4 对话记忆

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    ("placeholder", "{history}"),
    ("user", "{input}")
])

chain = prompt | llm

with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 多轮对话
config = {"configurable": {"session_id": "user_001"}}
print(with_history.invoke({"input": "我叫小明"}, config=config).content)
print(with_history.invoke({"input": "我叫什么名字？"}, config=config).content)  # 能记住
```

---

## 4. RAG（检索增强生成）

RAG 是当前最实用的 LLM 应用模式，通过检索外部知识来增强模型的回答能力。

### 4.1 RAG 架构概览

```
用户提问 → 问题向量化 → 向量数据库检索 → 获取相关文档 → 拼接 Prompt → LLM 生成回答
                ↑                    ↑
          Embedding 模型        预先索引的文档
```

### 4.2 文档加载与切分

```python
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载文档
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 文本切分 - 关键步骤
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每个块的最大字符数
    chunk_overlap=50,     # 块之间的重叠字符数
    separators=["\n\n", "\n", "。", ".", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"文档被切分为 {len(chunks)} 个块")
```

**切分策略的重要性：**
- `chunk_size` 太大：检索不精确，包含无关信息
- `chunk_size` 太小：丢失上下文，语义不完整
- `chunk_overlap`：保证块之间的语义连贯性

### 4.3 向量化与存储

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 使用 OpenAI Embedding 模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 创建向量数据库
vectorstore = FAISS.from_documents(chunks, embeddings)

# 保存到本地
vectorstore.save_local("faiss_index")

# 加载已有索引
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
```

**主流向量数据库对比：**

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| FAISS | Meta 开源，纯本地，速度快 | 小规模、本地开发 |
| Chroma | 轻量级，API 友好 | 原型开发 |
| Milvus | 分布式，高性能 | 生产环境 |
| Pinecone | 全托管云服务 | 不想运维 |

### 4.4 检索与生成

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 返回最相关的3个文档
)

# RAG Prompt
rag_prompt = ChatPromptTemplate.from_template("""
基于以下参考资料回答用户的问题。如果资料中没有相关信息，请说明你不确定。

参考资料：
{context}

用户问题：{question}

回答：""")

# 构建 RAG 链
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 使用
answer = rag_chain.invoke("什么是 Transformer 的自注意力机制？")
print(answer)
```

### 4.5 使用开源 Embedding 模型（免费替代方案）

```python
from langchain_huggingface import HuggingFaceEmbeddings

# 使用 BGE 中文 Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 后续使用方式与 OpenAI Embedding 完全一致
vectorstore = FAISS.from_documents(chunks, embeddings)
```

---

## 5. 实战项目：知识问答系统

将上述知识整合，构建一个完整的 RAG 知识问答系统。

### 5.1 项目结构

```
rag-qa-system/
├── app.py              # 主应用
├── ingest.py           # 文档导入脚本
├── requirements.txt
└── data/               # 存放文档
```

### 5.2 文档导入（ingest.py）

```python
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_documents(data_dir="data", index_dir="faiss_index"):
    # 加载目录下所有 txt 文件
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"加载了 {len(documents)} 个文档")

    # 切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"切分为 {len(chunks)} 个块")

    # 向量化并存储
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)
    print(f"索引已保存到 {index_dir}")

if __name__ == "__main__":
    ingest_documents()
```

### 5.3 问答应用（app.py）

```python
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_qa_chain(index_dir="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
你是一个知识问答助手。请根据以下参考资料回答问题。
如果资料中没有相关信息，请诚实地说"我在知识库中没有找到相关信息"。

参考资料：
{context}

问题：{question}
回答：""")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def format_docs(docs):
        return "\n---\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever

def main():
    chain, retriever = create_qa_chain()
    print("知识问答系统已启动！输入 'quit' 退出。\n")

    while True:
        question = input("你的问题: ").strip()
        if question.lower() == "quit":
            break

        # 显示检索到的文档
        docs = retriever.invoke(question)
        print(f"\n📚 检索到 {len(docs)} 个相关文档片段")

        # 生成回答
        answer = chain.invoke(question)
        print(f"\n🤖 回答: {answer}\n")

if __name__ == "__main__":
    main()
```

### 5.4 requirements.txt

```
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
langchain-huggingface>=0.1.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
pypdf>=4.0.0
```

---

## 6. 进阶话题

### 6.1 Function Calling / Tool Use

让 LLM 调用外部工具：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto"
)

# 检查是否调用了工具
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"调用工具: {tool_call.function.name}")
    print(f"参数: {tool_call.function.arguments}")
```

### 6.2 Agent 模式

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """搜索互联网获取信息"""
    return f"搜索结果：关于 '{query}' 的最新信息..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

tools = [search_web, calculator]
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手，可以使用工具来回答问题。"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "计算 (123 + 456) * 789 等于多少"})
```

---

## 7. 学习检查清单

- [ ] 能使用 OpenAI API 进行基本对话
- [ ] 掌握 Few-shot、CoT、ReAct 等 Prompt 技巧
- [ ] 理解 RAG 架构的完整流程
- [ ] 能使用 LangChain 构建链式应用
- [ ] 了解向量数据库的基本原理和使用
- [ ] 完成知识问答系统实战项目
- [ ] 了解 Function Calling 和 Agent 的基本概念

---

## 8. 推荐学习材料

### 官方文档
- [OpenAI API 文档](https://platform.openai.com/docs) - API 使用指南
- [LangChain 文档](https://python.langchain.com/docs/introduction/) - 框架完整文档
- [LangChain 教程](https://python.langchain.com/docs/tutorials/) - 官方教程合集

### 视频教程
- [LangChain 官方 YouTube](https://www.youtube.com/@LangChain) - 最新功能演示
- [freeCodeCamp - RAG Tutorial](https://www.youtube.com/watch?v=sVcwVQRHIc8) - RAG 完整教程

### 文章
- [Prompt Engineering Guide](https://www.promptingguide.ai/zh) - Prompt 工程完整指南（中文）
- [RAG 综述论文](https://arxiv.org/abs/2312.10997) - Retrieval-Augmented Generation 综述
- [Lilian Weng - LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) - Agent 架构详解

### 工具
- [FAISS 文档](https://faiss.ai/) - Meta 向量检索库
- [Chroma 文档](https://docs.trychroma.com/) - 轻量级向量数据库
- [Ollama](https://ollama.com/) - 本地运行开源模型

---

---

## 9. 补充：多模态 API 调用

### 9.1 GPT-4V 视觉理解

```python
from openai import OpenAI
import base64

client = OpenAI()

# 方式1：通过 URL
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片的内容"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]
)

# 方式2：通过 Base64 编码本地图片
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "这张图里有什么？请用中文回答。"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encode_image('photo.png')}",
                "detail": "high"  # low/high/auto
            }}
        ]
    }],
    max_tokens=500
)
print(response.choices[0].message.content)
```

### 9.2 Claude Vision

```python
import anthropic, base64

client = anthropic.Anthropic()

with open("diagram.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
            {"type": "text", "text": "请解释这个架构图的设计思路"}
        ]
    }]
)
print(message.content[0].text)
```

### 9.3 多模态应用场景

| 场景 | 示例 Prompt | 适用模型 |
|------|-------------|----------|
| OCR + 理解 | "提取图中表格数据为 JSON" | GPT-4o, Claude |
| 代码截图 → 代码 | "将截图中的代码转为可运行的 Python" | GPT-4o |
| 图表分析 | "分析这个折线图的趋势" | GPT-4o, Claude |
| UI 设计稿 → 代码 | "根据设计稿生成 HTML/CSS" | GPT-4o |

---

## 10. 补充：Structured Output（结构化输出）

OpenAI 的 Structured Output 功能保证模型输出严格符合 JSON Schema，不再需要祈祷模型输出合法 JSON。

### 10.1 使用 response_format 强制 JSON Schema

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class ExtractedInfo(BaseModel):
    company: str
    product: str
    price: float
    currency: str
    sentiment: str

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "从文本中提取结构化信息"},
        {"role": "user", "content": "苹果公司发布了 iPhone 16 Pro，售价 8999 元，用户反响热烈。"}
    ],
    response_format=ExtractedInfo,
)

result = response.choices[0].message.parsed
print(f"公司: {result.company}, 产品: {result.product}, 价格: {result.price}")
```

### 10.2 复杂嵌套结构

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Step(BaseModel):
    explanation: str
    output: str

class MathSolution(BaseModel):
    steps: List[Step]
    final_answer: str

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是数学老师，请一步步解题"},
        {"role": "user", "content": "解方程 2x + 5 = 13"}
    ],
    response_format=MathSolution,
)

solution = response.choices[0].message.parsed
for i, step in enumerate(solution.steps, 1):
    print(f"步骤{i}: {step.explanation} → {step.output}")
print(f"最终答案: {solution.final_answer}")
```

> **实用建议**：Structured Output 非常适合构建数据管道——从非结构化文本中批量提取信息，直接入库，无需手动解析。

---

## 11. 下一步

完成应用开发实战后，你已经能够构建实用的 LLM 应用了。下一阶段我们将深入 [Transformer 与 LLM 原理](../month02/week05-06-transformer-theory.md)，理解这些模型背后的工作机制。
