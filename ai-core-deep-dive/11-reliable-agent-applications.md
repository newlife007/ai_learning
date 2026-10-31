# 如何实现可靠且可自进化的智能体应用

> 本文是 [AI Agent：从工具调用到自主决策](05-ai-agent.md) 的工程化续篇。我们不再重复 Agent 的基础概念，而是以退款智能体为贯穿案例，依次实现确定性控制、状态恢复、工具幂等、人工审批、可观测性、评估发布和受控自进化闭环。

智能体具有概率性，外部系统也会超时、限流或部分成功。可靠性的目标不是让模型永远不犯错，而是让系统在模型或工具出错时仍能安全停止、准确恢复，并且不会重复执行副作用。

自进化也不意味着生产智能体可以随时改写自己的 Prompt、代码和权限。安全的自进化是一个受控的软件发布过程：生产系统产生经过验证的经验，离线系统提出候选改进，候选通过评估、审批和 Canary 后才能成为新版本。

## 本文要实现什么

读完本文，读者应该能够实现两个彼此隔离的部分：

```
可靠运行面
  用户请求 → 状态机 → 模型提案 → 策略校验 → 工具执行
           → Checkpoint → 审批 → 结果验证 → 确定性响应

受控进化面
  已验证 Trace → 失败聚类 → 候选改进 → 离线评估
               → 安全门禁 → 人工审批 → Canary → 晋级或回滚
```

文章按实现依赖组织：先确定案例、不变量和自主性边界；再实现领域模型、持久化、状态机、审批与工具网关；然后建立 Trace、评估和共享发布管线；最后让受控进化面复用这些能力。后面的自进化章节依赖前面的可靠运行和版本发布机制，不建议跳过直接实现。

### 验收标准

| 目标 | 可以验证的标准 |
|------|----------------|
| 正确完成 | 只有真实业务状态通过验证后，任务才能进入 `completed` |
| 可控失败 | 信息不足、权限不足或安全校验失败时停止，不猜测、不越权 |
| 可以恢复 | Worker 重启后使用同一 `run_id` 从持久化状态继续 |
| 副作用唯一 | 重试、重复消息和并发 Worker 不会产生重复退款 |
| 可以审计 | 每次决策、审批、工具调用和状态变化都能通过 Trace 还原 |
| 受控进化 | 生产智能体不能直接修改生产版本、权限和发布门禁 |
| 防止退化 | 候选版本必须通过回归、安全、成本和延迟门禁 |
| 可以回滚 | 新版本可被独立禁用，旧状态仍有兼容的恢复器 |

其中“未经有效审批的退款数”“重复退款数”和“跨租户访问数”都是硬安全指标，目标必须为 0。

## 贯穿案例：退款智能体

用户提出一个看似简单的请求：

```text
“上周的服务没有完成，帮我退掉。”
```

系统需要处理很多不确定性：

- 上周可能有多个订单，不能由模型猜测具体订单。
- 用户只能读取和操作自己租户下的订单。
- 退款原因可以由模型归类，金额必须由业务系统计算。
- 退款必须绑定当前订单版本和退款政策版本。
- 执行前需要展示真实参数并获得一次性审批。
- 支付接口超时不代表失败，可能已经完成退款。
- 最终回复中的订单号、金额和状态不能由模型重新编造。

### 业务不变量

先把不可妥协的规则写成代码能够检查的不变量：

```text
I1. 订单必须属于当前 tenant_id 和 user_id。
I2. 退款金额必须等于业务系统计算的可退金额。
I3. 审批必须覆盖相同的订单、金额、币种、原因和数据版本。
I4. 同一个原支付交易的全额退款 operation_key 最多产生一笔支付退款。
I5. 可信支付查询得到的状态不是 confirmed 时，Run 不得进入 completed。
I6. 模型、工具结果和长期记忆都不能修改 I1～I5。
```

后面的状态机、数据库约束、测试和发布门禁都围绕这些不变量展开。

## 先确定自主性边界

不是所有任务都需要智能体。自主性越高，可能路径越多，测试空间也越大。

| 场景 | 更合适的实现 |
|------|--------------|
| 固定字段提取、分类、摘要 | 单次模型调用 + 结构化输出 |
| 步骤已知，只有局部内容需要生成 | 确定性工作流 + LLM 节点 |
| 有少量明确分支 | 状态机或路由工作流 |
| 路径无法预先枚举，需要根据环境反馈调整 | 有预算和权限边界的智能体循环 |
| 高风险且结果难以自动验证 | 智能体辅助，人类决策和执行 |

Anthropic 在 [Building Effective Agents](https://www.anthropic.com/index/building-effective-agents) 中也建议从最简单的方案开始。定义清楚的任务优先使用可预测的工作流，只有路径无法预先确定时才增加自主性。

退款流程属于“确定性工作流 + 局部模型决策”：

- 模型负责理解用户意图和归类退款原因。
- 只读订单解析器根据认证用户和时间范围查找候选订单；多个候选必须由用户选择。
- 普通代码负责身份、金额、政策、状态转换和审批。
- 支付工具只接受已经验证并获批的 `refund_draft_id`。
- 最终结果由支付系统状态决定，而不是由模型判断。

> 只把必须由模型判断的部分交给模型，其余部分使用普通代码。

## 总体架构：运行面与进化面

```
┌──────────────────────────── 可靠运行面 ────────────────────────────┐
│                                                                  │
│ 用户/API → 鉴权 → 编排器/状态机 → 模型提案 → 策略与业务校验      │
│                         ↑                 ↓                      │
│                    Checkpoint ←── 工具网关 → 业务系统/支付系统    │
│                         ↓                 ↓                      │
│                    审批服务         结果验证器                    │
│                         └──────→ Trace / 审计 / 指标              │
└──────────────────────────────────┬───────────────────────────────┘
                                   │ 仅传递脱敏且已验证的数据
┌──────────────────────────────────↓───────────────────────────────┐
│                         受控进化面                               │
│                                                                  │
│ 经验仓库 → 失败聚类 → 候选生成 → 离线评估 → 发布门禁             │
│                                      ↓                           │
│                              人工审批与版本仓库                   │
│                                      ↓                           │
│                              Shadow / Canary                     │
└──────────────────────────────────────────────────────────────────┘
```

### 各组件的职责

| 组件 | 职责 | 不能做什么 |
|------|------|------------|
| 模型 | 提出下一动作或结构化分类 | 直接获得数据库、支付或审批权限 |
| 编排器 | 执行状态转换、预算和恢复 | 根据自然语言绕过状态机 |
| 策略层 | 校验身份、业务规则和动作风险 | 接受模型要求的“临时例外” |
| 工具网关 | 超时、重试、幂等、错误规范化 | 把未知结果当作失败后盲目重试 |
| 审批服务 | 绑定参数、审批人、版本和有效期 | 使用模型生成的摘要代替真实参数 |
| 结果验证器 | 读取业务事实并决定是否完成 | 相信模型所说的“已经成功” |
| 进化控制器 | 生成和评估候选版本 | 修改权限、凭证、安全不变量和门禁 |

最重要的边界是：**模型输出是提案，不是命令。**

## 项目结构与实现顺序

可以按下面的结构组织实现：

```text
reliable_agent/
├── app/
│   ├── domain.py          # Run、Proposal、Draft、Approval 等领域对象
│   ├── repository.py      # Checkpoint、乐观锁、Trace 持久化
│   ├── planner.py         # LLM 适配器和结构化输出
│   ├── policy.py          # 身份、业务规则和动作许可
│   ├── runtime.py         # 状态机与 Worker
│   ├── tools.py           # 支付、通知等工具网关
│   ├── approval.py        # 审批创建与一次性消费
│   ├── evaluator.py       # 回归评估和故障注入
│   ├── evolution.py       # 经验、候选版本和进化控制器
│   └── telemetry.py       # Trace、指标和脱敏
├── versions/
│   ├── v1.json
│   └── candidates/
├── evals/
│   ├── regression.jsonl
│   └── security.jsonl
└── tests/
    ├── test_runtime.py
    ├── test_recovery.py
    ├── test_security.py
    └── test_evolution.py
```

示例约定使用 Python 3.11+、Pydantic 2、SQLite 和支持 Structured Outputs 的模型 API。代码片段分别对应上述模块，省略了 Web 框架和数据库连接样板，但领域类型、事务边界、状态转换和接口契约应保持一致。

推荐按以下顺序开发：

1. 先实现不依赖模型的状态机、仓储和 Mock 工具。
2. 用脚本化 Planner 跑通正常、拒绝、超时和恢复场景。
3. 再接入 LLM，只让它返回受约束的 `ActionProposal`。
4. 加入 Trace、回归评估和普通版本发布管线。
5. 最后启用离线候选生成与受控自进化。

这样可以把模型的不确定性与分布式系统问题分开调试。

## 统一领域模型

可靠系统不能只依赖对话历史推断进度。需要先定义一组可以持久化和审计的对象。

### 运行状态

```python
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    CREATED = "created"
    RESOLVING_ORDER = "resolving_order"
    WAITING_FOR_ORDER_SELECTION = "waiting_for_order_selection"
    DRAFTING = "drafting"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    EXECUTING = "executing"
    RESULT_UNKNOWN = "result_unknown"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    FAILED = "failed"


TERMINAL_STATES = {
    RunStatus.COMPLETED,
    RunStatus.CANCELLED,
    RunStatus.BLOCKED,
    RunStatus.FAILED,
}


@dataclass(frozen=True)
class Run:
    run_id: str
    request_id: str
    tenant_id: str
    user_id: str
    status: RunStatus
    state_version: int
    state_schema_version: int
    agent_version: str
    model_calls: int
    tool_calls: int
    state: dict[str, Any]
```

`tenant_id` 和 `user_id` 来自认证上下文，不能由用户消息或模型参数指定。`request_id` 是调用方提供的入站幂等键，相同请求只能创建一个 Run。`state_version` 用于乐观锁；`state_schema_version` 决定使用哪个状态恢复器，不能与 Agent 版本混为一谈。

### 动作提案与已验证事实

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict


class ActionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["classify_refund_reason"]
    reason_code: Literal[
        "customer_request",
        "duplicate",
        "service_failure",
    ] | None
    explanation: str


class VerifiedRefundFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    user_id: str
    order_id: str
    order_version: int
    payment_id: str
    amount_cents: int
    currency: str
    policy_version: str
    reason_code: str
```

模型只能提出低风险动作。`issue_refund` 不在 `ActionProposal` 中，因为退款执行由状态机在审批通过后确定性触发。

`explanation` 只用于调试，不能成为权限判断或业务事实。

### 退款草案、审批和工具执行

```python
from datetime import datetime


@dataclass(frozen=True)
class RefundDraft:
    draft_id: str
    run_id: str
    operation_key: str
    facts: VerifiedRefundFacts
    action_hash: str


@dataclass(frozen=True)
class Approval:
    approval_id: str
    request_id: str
    draft_id: str
    action_hash: str
    approved_by: str
    expires_at: datetime
    consumed_at: datetime | None
    invalidated_at: datetime | None


class ToolOutcome(StrEnum):
    CONFIRMED = "confirmed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class RefundResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: ToolOutcome
    idempotency_key: str
    refund_id: str | None
    payment_id: str
    order_id: str
    amount_cents: int
    currency: str
    error_code: str | None


@dataclass(frozen=True)
class ToolExecution:
    execution_id: str
    idempotency_key: str
    draft_id: str
    approval_id: str
    status: Literal["started", "confirmed", "failed", "unknown"]
    attempts: int
    result: RefundResult | None
```

三者承担不同职责：

- `RefundDraft` 固定经过验证的业务动作，并用 `operation_key` 对同一原支付交易去重。
- `Approval` 证明谁在什么期限内批准了这个动作。时间字段使用带时区的 UTC `datetime`，由服务端时钟生成。
- `ToolExecution` 是可恢复的执行意图，固定草案、审批和幂等键；审批消费后，恢复只能复用该执行意图，不能创建新的业务动作。
- `RefundResult` 统一支付服务的正常、失败和未知结果，避免字典与 SDK 对象混用。

## 最小持久化设计

开发阶段可以使用 SQLite，生产环境可以迁移到 PostgreSQL。关键不是数据库品牌，而是状态和约束必须持久化。

```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    state_version INTEGER NOT NULL,
    state_schema_version INTEGER NOT NULL,
    agent_version TEXT NOT NULL,
    model_calls INTEGER NOT NULL DEFAULT 0,
    tool_calls INTEGER NOT NULL DEFAULT 0,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE refund_drafts (
    draft_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    operation_key TEXT NOT NULL,
    action_hash TEXT NOT NULL,
    facts_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, action_hash)
);

CREATE TABLE approvals (
    approval_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    draft_id TEXT NOT NULL,
    action_hash TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    consumed_at TEXT,
    invalidated_at TEXT
);

CREATE TABLE tool_executions (
    execution_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    draft_id TEXT NOT NULL UNIQUE,
    approval_id TEXT NOT NULL UNIQUE,
    tool_name TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    result_json TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE trace_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sequence_no INTEGER NOT NULL,
    actor_id TEXT,
    event_type TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    previous_hash TEXT,
    event_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, sequence_no)
);
```

这里有三层去重：

1. `runs.request_id` 防止同一个入站请求创建多个 Run。
2. `refund_drafts` 的 `(run_id, action_hash)` 防止同一 Run 重复保存相同草案，同时允许事实变化后创建新草案。
3. `tool_executions.idempotency_key` 在所有 Run 间唯一，支付服务端也使用同一个 operation key，防止恢复、重复请求和网络重试产生第二笔退款。

数据库唯一约束是最后防线，不能只在应用代码中“先查询再创建”。

### 使用乐观锁推进状态

```python
VALID_TRANSITIONS = {
    RunStatus.CREATED: {RunStatus.RESOLVING_ORDER},
    RunStatus.RESOLVING_ORDER: {
        RunStatus.DRAFTING,
        RunStatus.WAITING_FOR_ORDER_SELECTION,
        RunStatus.BLOCKED,
        RunStatus.FAILED,
    },
    RunStatus.WAITING_FOR_ORDER_SELECTION: {
        RunStatus.DRAFTING,
        RunStatus.CANCELLED,
        RunStatus.BLOCKED,
    },
    RunStatus.DRAFTING: {
        RunStatus.WAITING_FOR_APPROVAL,
        RunStatus.BLOCKED,
        RunStatus.FAILED,
    },
    RunStatus.WAITING_FOR_APPROVAL: {
        RunStatus.DRAFTING,
        RunStatus.EXECUTING,
        RunStatus.CANCELLED,
        RunStatus.BLOCKED,
    },
    RunStatus.EXECUTING: {
        RunStatus.VERIFYING,
        RunStatus.RESULT_UNKNOWN,
        RunStatus.FAILED,
        RunStatus.BLOCKED,
    },
    RunStatus.RESULT_UNKNOWN: {
        RunStatus.VERIFYING,
        RunStatus.FAILED,
        RunStatus.BLOCKED,
    },
    RunStatus.VERIFYING: {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.BLOCKED,
    },
}


def save_run(connection, current: Run, next_status: RunStatus, next_state: dict) -> Run:
    if next_status not in VALID_TRANSITIONS.get(current.status, set()):
        raise InvalidStateTransition(current.status, next_status)

    cursor = connection.execute(
        """
        UPDATE runs
           SET status = ?, state_json = ?, state_version = state_version + 1,
               updated_at = CURRENT_TIMESTAMP
         WHERE run_id = ? AND state_version = ?
        """,
        (
            next_status.value,
            json.dumps(next_state, ensure_ascii=False),
            current.run_id,
            current.state_version,
        ),
    )
    if cursor.rowcount != 1:
        raise ConcurrentRunUpdate(current.run_id)

    return load_run(connection, current.run_id)
```

Worker 遇到 `ConcurrentRunUpdate` 时重新加载状态，而不是覆盖另一个 Worker 的结果。

状态转换服务必须在同一个数据库事务中完成三件事：校验合法边、用 `state_version` 条件更新 Run、追加对应 Trace 事件。`trace_events` 可以使用 `previous_hash` 和 `event_hash` 形成哈希链；生产环境再配合只允许插入的数据库角色、独立审计存储或 WORM 归档，避免普通应用账号修改历史。

调用外部支付系统无法加入本地事务，因此需要先持久化 `ToolExecution`，再使用业务幂等键调用支付系统，并通过结果查询连接两边。

## 状态机：把允许路径写进代码

### 状态转换表

| 当前状态 | 事件 | 下一个状态 | 关键前置条件 |
|----------|------|------------|----------------|
| `created` | 开始 | `resolving_order` | 身份已认证，预算已创建 |
| `resolving_order` | 唯一订单 | `drafting` | 订单属于当前用户和租户 |
| `resolving_order` | 多个订单 | `waiting_for_order_selection` | 不允许模型自行选择 |
| `waiting_for_order_selection` | 用户选择 | `drafting` | 选择值属于候选集合 |
| `drafting` | 草案通过 | `waiting_for_approval` | 金额、政策和版本已验证 |
| `waiting_for_approval` | 批准且事实未变化 | `executing` | action hash 相同且审批未过期 |
| `waiting_for_approval` | 订单或政策变化 | `drafting` | 旧审批失效，重新生成草案 |
| `waiting_for_approval` | 拒绝 | `cancelled` | 保存拒绝审计记录 |
| `executing` | 已确认 | `verifying` | 支付服务返回统一的 `RefundResult.CONFIRMED` |
| `executing` | 明确失败 | `failed` | 不重试业务失败，保存错误码 |
| `executing` | 结果未知 | `result_unknown` | 禁止直接创建第二笔退款 |
| `result_unknown` | 查询已确认 | `verifying` | 幂等键对应唯一结果 |
| `result_unknown` | 查询明确失败 | `failed` | 保存支付失败事实 |
| `result_unknown` | 超过查询截止时间 | `blocked` | 转人工调查，不无限轮询 |
| `verifying` | 业务断言通过 | `completed` | 订单、金额、币种和状态一致 |
| 任意非终态 | 安全拒绝 | `blocked` | 记录规则和原因 |
| 任意非终态 | 不可恢复错误 | `failed` | 保存可诊断错误码 |

状态机比 System Prompt 更可靠。运行时在 `resolving_order`、审批、执行和验证阶段根本不调用模型；模型只参与 `drafting` 中的原因归类。

### 按状态限制模型动作

```python
ALLOWED_ACTIONS = {
    RunStatus.DRAFTING: {"classify_refund_reason"},
}


def authorize_proposal(run: Run, proposal: ActionProposal) -> None:
    allowed = ALLOWED_ACTIONS.get(run.status, set())
    if proposal.action not in allowed:
        raise PolicyRejected(
            code="action_not_allowed_in_state",
            details={"status": run.status, "action": proposal.action},
        )


def validate_proposal(run: Run, proposal: ActionProposal, context: dict) -> list[str]:
    errors = []
    if proposal.action not in ALLOWED_ACTIONS.get(run.status, set()):
        errors.append("action_not_allowed_in_state")
    if proposal.reason_code is None:
        errors.append("refund_reason_required")
    return errors
```

模型只在 `drafting` 阶段归类退款原因。订单选择、审批、执行和验证都由确定性事件推进；即使模型输出其他动作，策略层也会拒绝。

### 为运行设置硬预算

每个 Run 至少限制：

- 最大模型调用次数
- 最大工具调用次数
- 单次调用超时和任务总截止时间
- 最大 token 和费用
- 最大并行分支
- 最大模型修复次数
- 最大连续相同动作次数

达到预算后进入 `blocked` 或转人工，不能只在 Prompt 中要求模型“尽量简短”。预算检查和计数必须是同一个原子操作，否则并发 Worker 都可能在旧计数下通过：

```python
def reserve_call_budget(connection, run_id: str, kind: str, limit: int) -> None:
    columns = {"model": "model_calls", "tool": "tool_calls"}
    column = columns[kind]
    cursor = connection.execute(
        f"""
        UPDATE runs
           SET {column} = {column} + 1
         WHERE run_id = ? AND {column} < ?
        """,
        (run_id, limit),
    )
    if cursor.rowcount != 1:
        raise BudgetExhausted(kind)
```

模型修复和工具传输重试的每一次尝试，都要先预留对应预算并写入 Trace。

## 从模型提案到执行许可

模型输出必须经过分层校验：

```
ActionProposal
  ↓ Pydantic / JSON Schema
结构有效
  ↓ 实体与来源检查
事实有效
  ↓ 租户、身份、状态和业务规则
动作允许
  ↓ 风险分级
自动执行只读动作 / 创建审批 / 拒绝
```

### 结构正确不代表内容正确

即使模型输出符合 Schema，也不能证明：

- 订单真实存在并属于当前用户。
- 当前订单仍可退款。
- 金额和币种正确。
- 用户要求执行退款，而不只是询问政策。
- 当前审批覆盖了相同参数。

确定性校验应读取可信业务系统：

```python
def build_verified_facts(proposal, order, actor, policy):
    if order is None:
        raise PolicyRejected("order_not_found")
    if order.tenant_id != actor.tenant_id or order.user_id != actor.user_id:
        # 不向调用者泄露其他租户的订单是否存在。
        raise PolicyRejected("order_not_found")
    if order.status != "paid":
        raise PolicyRejected("order_not_refundable")

    amount_cents = policy.calculate_refundable_amount(order)
    if amount_cents <= 0:
        raise PolicyRejected("nothing_to_refund")

    return VerifiedRefundFacts(
        tenant_id=actor.tenant_id,
        user_id=actor.user_id,
        order_id=order.order_id,
        order_version=order.version,
        payment_id=order.original_payment_id,
        amount_cents=amount_cents,
        currency=order.currency,
        policy_version=policy.version,
        reason_code=proposal.reason_code,
    )
```

金额、身份、库存、权限和状态只要能够用普通代码确定，就不应交给模型裁决。

### 接入结构化模型输出

下面的适配器只负责生成 `ActionProposal`。运行时仍会执行 Schema、事实和策略校验。

```python
import os
from openai import OpenAI


class OpenAIPlanner:
    def __init__(self):
        self.client = OpenAI()
        self.model = os.environ["OPENAI_MODEL"]

    def propose(self, run: Run, public_context: dict) -> ActionProposal:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "你是退款工作流中的低权限规划器。"
                        "只能从允许动作中选择，不得判断权限、金额或执行结果。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "status": run.status,
                            "allowed_actions": sorted(ALLOWED_ACTIONS[run.status]),
                            "context": public_context,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            text_format=ActionProposal,
        )
        return response.output_parsed
```

OpenAI 的 [Structured Outputs 文档](https://developers.openai.com/api/docs/guides/structured-outputs) 介绍了使用 Pydantic 约束模型输出的方法。结构化输出减少格式错误，但不能替代业务校验。

单元测试不要依赖在线模型。使用可预测的 Planner 注入轨迹：

```python
class ScriptedPlanner:
    def __init__(self, proposals):
        self.proposals = iter(proposals)

    def propose(self, run: Run, public_context: dict) -> ActionProposal:
        return next(self.proposals)
```

这样可以稳定复现非法动作、缺少原因、修复成功和连续修复失败等场景。

### 有限修复与安全降级

只有可修复的格式或参数问题才反馈给模型，最多一到两次：

```python
def propose_with_repair(planner, run, context, max_repairs=2):
    errors = []

    for attempt in range(max_repairs + 1):
        proposal = planner.propose(
            run,
            {**context, "previous_validation_errors": errors},
        )
        errors = validate_proposal(run, proposal, context)
        if not errors:
            return proposal

    raise HumanReviewRequired("invalid_model_output")
```

以下情况不能通过“再想一次”解决：

- 用户缺少必要信息。
- 身份没有权限。
- 业务规则拒绝操作。
- 外部系统明确拒绝请求。
- 高风险动作尚未批准。
- 可信数据源不可用。

这些情况应询问用户、转人工、安全停止或等待依赖恢复。

## 工具网关：处理现实世界的失败

本节先定义工具网关的故障与幂等契约。网关不能被模型直接调用；只有下一节审批流程持久化出合法的 `ToolExecution` 后，运行时才能进入支付执行。

### 使用窄工具

```
危险: execute_sql(sql)
危险: call_http(method, url, body)

更好: list_user_orders(time_range)
      get_refund_policy(order_id)
      create_refund_draft(order_id, reason_code)
      issue_approved_refund(draft_id, approval_id)
      get_refund_status(idempotency_key)
```

工具越窄，参数越容易校验，权限和副作用也越清晰。

### 区分四类错误

| 类型 | 示例 | 处理方式 |
|------|------|----------|
| 可重试传输错误 | 连接重置、429、临时 5xx | 指数退避、随机抖动、服从总截止时间 |
| 模型参数错误 | 缺字段、非法枚举 | 有限修复，再失败则转人工 |
| 业务拒绝 | 无权限、订单不可退 | 不重试，转换为明确业务结果 |
| 结果未知 | 请求超时，但服务端可能已提交 | 按幂等键查询，禁止直接创建新请求 |

模型修复重试和工具传输重试是两种不同机制，必须分别计数和设置预算。

### 使用业务稳定的幂等键

幂等键要标识业务动作，而不是某次 Run、草案或网络请求。全额退款可以绑定租户和原支付交易：

```python
def refund_operation_key(facts: VerifiedRefundFacts) -> str:
    return f"refund:full:{facts.tenant_id}:{facts.payment_id}"


def refund_idempotency_key(draft: RefundDraft) -> str:
    return draft.operation_key
```

`refund_drafts.operation_key` 在本地数据库中唯一，支付服务也必须以同一个键原子保存并返回首次结果。支付服务要在同一事务中检查原交易仍可退款、写入退款记录并占用该 operation key。这样，即使重复消息创建了新 Run，或者恢复时重新发起调用，同一原交易仍只能产生一笔全额退款。

`request_id` 负责入站请求去重，`operation_key` 负责业务动作去重。两者不能互相替代。`request_id` 由 API 调用方提供；`DraftRepository.create_once` 从 `VerifiedRefundFacts` 计算 `operation_key` 和 `action_hash`，不能接受模型提供的值。真正阻止跨 Run 副作用的是 `tool_executions.idempotency_key` 与支付服务端的原子幂等约束。

### 处理“超时但已成功”

```python
def normalize_payment_result(raw, draft, key) -> RefundResult:
    outcomes = {
        "confirmed": ToolOutcome.CONFIRMED,
        "failed": ToolOutcome.FAILED,
    }
    outcome = outcomes.get(raw.status, ToolOutcome.UNKNOWN)
    return RefundResult(
        outcome=outcome,
        idempotency_key=key,
        refund_id=raw.refund_id,
        payment_id=draft.facts.payment_id,
        order_id=draft.facts.order_id,
        amount_cents=raw.amount_cents,
        currency=raw.currency,
        error_code=getattr(raw, "error_code", None),
    )


def execute_refund(payment, draft, execution, revalidate) -> RefundResult:
    key = execution.idempotency_key

    existing = payment.get_refund_by_idempotency_key(key)
    if existing is not None:
        return normalize_payment_result(existing, draft, key)

    # 只有尚未发生支付副作用时才重新核对当前业务事实。
    revalidate(draft)

    try:
        raw = payment.issue_refund(
            payment_id=draft.facts.payment_id,
            amount_cents=draft.facts.amount_cents,
            currency=draft.facts.currency,
            expected_order_version=draft.facts.order_version,
            policy_version=draft.facts.policy_version,
            idempotency_key=key,
        )
        return normalize_payment_result(raw, draft, key)
    except PaymentRejected as exc:
        return RefundResult(
            outcome=ToolOutcome.FAILED,
            idempotency_key=key,
            refund_id=None,
            payment_id=draft.facts.payment_id,
            order_id=draft.facts.order_id,
            amount_cents=draft.facts.amount_cents,
            currency=draft.facts.currency,
            error_code=exc.code,
        )
    except PaymentTimeout:
        # 不能直接重发。先确认服务端是否已经提交。
        existing = payment.get_refund_by_idempotency_key(key)
        if existing is not None:
            return normalize_payment_result(existing, draft, key)
        return RefundResult(
            outcome=ToolOutcome.UNKNOWN,
            idempotency_key=key,
            refund_id=None,
            payment_id=draft.facts.payment_id,
            order_id=draft.facts.order_id,
            amount_cents=draft.facts.amount_cents,
            currency=draft.facts.currency,
            error_code="payment_result_unknown",
        )
```

如果查询仍然无法确认，Run 进入 `result_unknown`，由后台任务继续查询或转人工。不要生成新的 `draft_id` 规避未知状态。

### Checkpoint 与副作用

恢复通常意味着节点可能重新执行。LangGraph 的 [Interrupts 文档](https://docs.langchain.com/oss/javascript/langgraph/interrupts) 也提醒，恢复时节点可能从开头运行，因此中断前的副作用必须幂等或移到独立节点。

工程上更常见的保证是：

```text
至少一次调度 + 持久化状态 + 业务幂等键 = 副作用只发生一次
```

不能依赖“进程只执行一次”。

## 权限、安全与审批

### 权限绑定任务，不绑定模型

实际权限应由以下因素共同决定：

```text
用户身份 × 租户 × 当前 Run × 当前状态 × 工具 × 有效期
```

建议：

- 文件工具限制真实路径，解析软链接后再次检查范围。
- 网络工具使用域名和端口白名单。
- 数据库使用只读账号或面向业务动作的存储过程。
- 代码执行运行在无特权沙箱中，并限制 CPU、内存、磁盘和网络。
- 每个工具使用独立的短期凭证。
- 凭证、其他租户数据和完整敏感记录不进入模型上下文。
- 工具服务再次鉴权，不默认信任来自编排器的请求。

OWASP 的 [Over-Privileged Skills](https://owasp.github.io/www-project-agentic-skills-top-10/ast03) 将权限过宽列为高风险问题，并建议按技能隔离凭证、限制网络目标并在运行时强制权限。

### 工具输出也是不可信输入

网页、邮件、文档和第三方 MCP Server 都可能在内容中夹带恶意指令。工具输出只能提供数据，不能改变系统规则或授予权限。

- 优先接收符合固定 Schema 的结构化结果。
- 标记来源、时间和信任等级。
- 限制结果大小、类型和可解析字段。
- 隔离外部低信任工具和内部高权限工具。
- 在后端执行权限检查，不依赖 Prompt 中的禁止语句。
- 只允许连接经过审核的工具服务和网络目标。

OWASP 对 [MCP Tool Poisoning](https://community.owasp.org/attacks/MCP_Tool_Poisoning) 的说明展示了工具返回内容诱导智能体调用高权限工具的风险。

### 审批绑定规范化动作

审批页面或终端输出应由程序根据 `RefundDraft` 生成：

```text
操作：向订单 order_123 的原支付方式退款
金额：129.00 CNY
原因：service_failure
订单版本：7
政策版本：2026-10-01
影响：提交后不可由本系统撤销
有效期：10 分钟
```

动作哈希应覆盖所有关键参数：

```python
import hashlib
import json


def action_hash(facts: VerifiedRefundFacts) -> str:
    payload = facts.model_dump(mode="json")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
```

批准事件到达时，系统必须按明确的 `approval_id` 加载审批，并在一个本地事务中完成四件事：消费审批、创建执行意图、把 Run 推进到 `executing`、写入审计事件。

```python
def authorize_execution(services, runtime, run, approval_id, now):
    draft = services.drafts.load(run.state["draft_id"])
    approval = services.approvals.load(approval_id)

    if approval.consumed_at is not None:
        raise PolicyRejected("approval_already_consumed")
    if approval.invalidated_at is not None:
        raise PolicyRejected("approval_invalidated")
    if approval.expires_at <= now:
        raise PolicyRejected("approval_expired")
    if approval.draft_id != draft.draft_id:
        raise PolicyRejected("approval_draft_mismatch")
    if approval.action_hash != draft.action_hash:
        raise PolicyRejected("approval_parameters_changed")

    # 等待审批期间业务事实可能变化，执行前必须重新读取并核对。
    try:
        runtime.policy.revalidate_draft(draft)
    except StaleDraft as exc:
        with services.db.transaction():
            services.approvals.invalidate_once(approval.approval_id, now)
            services.runs.transition(
                run,
                RunStatus.DRAFTING,
                state_patch={
                    "draft_id": None,
                    "stale_draft_reason": exc.code,
                },
                event="refund_draft_stale",
            )
            services.audit.append_approval_invalidated(run, approval, exc.code)
        return None

    with services.db.transaction():
        services.approvals.consume_once(approval.approval_id, now)
        execution = services.executions.create_once(
            draft_id=draft.draft_id,
            approval_id=approval.approval_id,
            idempotency_key=draft.operation_key,
        )
        services.runs.transition(
            run,
            RunStatus.EXECUTING,
            state_patch={
                "approval_id": approval.approval_id,
                "execution_id": execution.execution_id,
            },
            event="refund_execution_authorized",
        )
        services.audit.append_approval_consumed(run, approval, execution)

    return execution
```

`consume_once`、`create_once` 和状态转换都必须使用数据库唯一约束或条件更新。事务提交后，即使 Worker 崩溃，恢复时也按 `execution_id` 加载相同执行意图和幂等键，不再消费第二次审批。一次性审批限制的是创建业务执行意图，不能阻止同一执行意图安全恢复。

`revalidate_draft` 需要重新读取订单、原支付交易和当前政策。如果订单版本、政策版本、状态、金额或币种发生变化，旧审批立即失效，Run 返回 `drafting` 重新生成草案并再次审批，不能在旧草案上静默更新参数。

### 紧急停止开关

控制面应支持：

1. 禁用某个有副作用的工具。
2. 阻止新任务进入受影响流程。
3. 暂停正在运行的任务。
4. 吊销智能体使用的短期凭证。
5. 保留 Trace，定位并补偿已发生的影响。

停止开关位于模型之外，模型和进化控制器都无权修改。

## 可靠运行时参考流程

### Worker 的核心逻辑

下面的代码展示编排职责。仓储、支付和审批都通过接口注入，便于测试和替换实现。

```python
def execute_next_step(services, run_id: str):
    run = services.runs.load(run_id)
    runtime = services.versions.runtime_for(
        agent_version=run.agent_version,
        state_schema_version=run.state_schema_version,
    )
    enforce_budget(run)

    if run.status == RunStatus.CREATED:
        return services.runs.transition(
            run,
            RunStatus.RESOLVING_ORDER,
            event="run_started",
        )

    if run.status == RunStatus.RESOLVING_ORDER:
        orders = services.orders.list_for_actor(
            tenant_id=run.tenant_id,
            user_id=run.user_id,
            query=run.state["user_request"],
        )
        if len(orders) == 0:
            return services.runs.block(run, "order_not_found")
        if len(orders) > 1:
            return services.runs.transition(
                run,
                RunStatus.WAITING_FOR_ORDER_SELECTION,
                state_patch={"candidate_order_ids": [o.order_id for o in orders]},
                event="order_selection_required",
            )
        return services.runs.transition(
            run,
            RunStatus.DRAFTING,
            state_patch={"selected_order_id": orders[0].order_id},
            event="order_resolved",
        )

    if run.status == RunStatus.DRAFTING:
        proposal = propose_with_repair(
            runtime.planner,
            run,
            runtime.context.public_view(run),
        )
        authorize_proposal(run, proposal)
        facts = runtime.policy.build_verified_facts(run, proposal)
        draft = services.drafts.create_once(run, facts)
        return services.runs.transition(
            run,
            RunStatus.WAITING_FOR_APPROVAL,
            state_patch={"draft_id": draft.draft_id},
            event="refund_draft_created",
        )

    if run.status == RunStatus.EXECUTING:
        draft = services.drafts.load(run.state["draft_id"])
        execution = services.executions.load(run.state["execution_id"])
        result = runtime.tools.issue_refund(
            draft=draft,
            execution=execution,
            revalidate=runtime.policy.revalidate_before_first_call,
        )
        services.executions.record_result_monotonic(execution, result)

        if result.outcome == ToolOutcome.CONFIRMED:
            return services.runs.transition(
                run,
                RunStatus.VERIFYING,
                state_patch={"payment_result": result.model_dump(mode="json")},
                event="refund_confirmed_by_tool",
            )
        if result.outcome == ToolOutcome.FAILED:
            return services.runs.fail(run, result.error_code or "refund_failed")
        return services.runs.transition(
            run,
            RunStatus.RESULT_UNKNOWN,
            state_patch={"payment_result": result.model_dump(mode="json")},
            event="refund_result_unknown",
        )

    if run.status == RunStatus.RESULT_UNKNOWN:
        draft = services.drafts.load(run.state["draft_id"])
        execution = services.executions.load(run.state["execution_id"])
        result = runtime.tools.query_refund(draft, execution)
        services.executions.record_result_monotonic(execution, result)

        if result.outcome == ToolOutcome.CONFIRMED:
            return services.runs.transition(
                run,
                RunStatus.VERIFYING,
                state_patch={"payment_result": result.model_dump(mode="json")},
                event="refund_result_recovered",
            )
        if result.outcome == ToolOutcome.FAILED:
            return services.runs.fail(run, result.error_code or "refund_failed")
        if unknown_result_deadline_exceeded(run):
            return services.runs.block(run, "refund_result_requires_manual_review")
        return services.runs.keep_waiting(run, "refund_still_unknown")

    if run.status == RunStatus.VERIFYING:
        verified = runtime.verifier.verify_refund(run)
        return services.runs.transition(
            run,
            RunStatus.COMPLETED,
            state_patch={"verified_result": verified.model_dump(mode="json")},
            event="refund_verified",
        )

    return run
```

`runtime_for` 必须按 Run 创建时固定的 `agent_version` 加载不可变 Bundle，并根据 `state_schema_version` 选择兼容恢复器。发布新版本只改变新 Run 的默认版本，不能让在途 Run 静默切换 Planner、策略或工作流。旧 Bundle 和迁移器至少保留到对应 Run 全部终止或达到保留期限。

`record_result_monotonic` 只允许 `started → unknown → confirmed/failed` 或 `started → confirmed/failed`，已经确认或失败的执行结果不能被后续请求覆盖。`keep_waiting` 不调用状态转换，而是原子更新下一次查询时间、`state_version` 和 Trace，因此不需要把自循环加入 `VALID_TRANSITIONS`。

等待用户选择和等待审批是外部事件，不应由 Worker 自行推进：

```python
def submit_order_selection(repository, run_id, actor, order_id):
    run = repository.load(run_id)
    assert_actor_matches(run, actor)
    if run.status != RunStatus.WAITING_FOR_ORDER_SELECTION:
        raise InvalidState(run.status)
    if order_id not in run.state["candidate_order_ids"]:
        raise PolicyRejected("invalid_order_selection")

    return repository.transition(
        run,
        RunStatus.DRAFTING,
        state_patch={"selected_order_id": order_id},
        event="order_selected_by_user",
    )


def approve_refund(services, run_id, actor, approval_input, now):
    run = services.runs.load(run_id)
    assert_actor_matches(run, actor)
    if run.status != RunStatus.WAITING_FOR_APPROVAL:
        raise InvalidState(run.status)

    draft = services.drafts.load(run.state["draft_id"])
    runtime = services.versions.runtime_for(
        agent_version=run.agent_version,
        state_schema_version=run.state_schema_version,
    )
    approval = services.approvals.create_once(
        request_id=approval_input.request_id,
        draft=draft,
        action_hash=draft.action_hash,
        approved_by=actor.user_id,
        expires_in_seconds=600,
    )
    authorize_execution(
        services=services,
        runtime=runtime,
        run=run,
        approval_id=approval.approval_id,
        now=now,
    )
    return services.runs.load(run_id)
```

`approval_input.request_id` 必须唯一。若进程在创建审批后、创建执行意图前崩溃，重放相同审批请求会取回同一 `approval_id`，再继续 `authorize_execution`，不会生成第二条授权。

顶层 Worker 还需要统一捕获 `BudgetExhausted`、`PolicyRejected`、`VerificationFailed` 和不可恢复异常，将它们持久化为 `blocked` 或 `failed`，并与状态变化在同一事务中写 Trace。`ConcurrentRunUpdate` 则重新加载 Run 后再判断，不得覆盖新状态。

### 确定性完成验证

```python
class RefundVerifier:
    def __init__(self, payments, orders, drafts, executions):
        self.payments = payments
        self.orders = orders
        self.drafts = drafts
        self.executions = executions

    def verify_refund(self, run) -> RefundResult:
        draft = self.drafts.load(run.state["draft_id"])
        execution = self.executions.load(run.state["execution_id"])
        raw = self.payments.get_refund_by_idempotency_key(
            execution.idempotency_key
        )

        if raw is None or raw.status != "confirmed":
            raise VerificationFailed("refund_not_confirmed")

        result = normalize_payment_result(raw, draft, execution.idempotency_key)
        if result.payment_id != draft.facts.payment_id:
            raise VerificationFailed("payment_mismatch")
        if result.order_id != draft.facts.order_id:
            raise VerificationFailed("order_mismatch")
        if result.amount_cents != draft.facts.amount_cents:
            raise VerificationFailed("amount_mismatch")
        if result.currency != draft.facts.currency:
            raise VerificationFailed("currency_mismatch")
        if self.payments.count_refunds(draft.facts.payment_id) != 1:
            raise VerificationFailed("refund_uniqueness_broken")

        order = self.orders.load_for_actor(
            tenant_id=draft.facts.tenant_id,
            user_id=draft.facts.user_id,
            order_id=draft.facts.order_id,
        )
        if order.refund_status != "confirmed":
            raise VerificationFailed("business_refund_not_recorded")

        return result
```

只有该验证器通过后才能进入 `completed`。

### 确定性生成最终回复

```python
def render_final_response(run: Run) -> str:
    if run.status != RunStatus.COMPLETED:
        return "退款尚未确认，请稍后查询或联系人工客服。"

    result = run.state["verified_result"]
    return (
        f"退款已确认，金额为 {result['amount_cents'] / 100:.2f} "
        f"{result['currency']}，退款编号为 {result['refund_id']}。"
    )
```

模型可以补充一般说明，但不能重新生成订单号、金额、币种、退款编号和状态。

## 状态、上下文、记忆与审计

四类数据不能混在一段对话历史中：

| 数据 | 用途 | 保存原则 |
|------|------|----------|
| 运行状态 | 当前步骤、草案、审批和执行结果 | 结构化、持久化、版本化 |
| 对话上下文 | 理解当前用户意图 | 按需裁剪，不作为业务事实源 |
| 长期记忆 | 跨会话偏好和稳定经验 | 有来源、有效期、置信度和租户边界 |
| 审计记录 | 还原谁批准和执行了什么 | 追加写入、防篡改、严格访问控制 |

每条长期记忆至少包含：

```json
{
  "fact": "用户偏好使用中文报告",
  "source": "user_explicit_statement",
  "created_at": "2026-10-31T09:00:00Z",
  "expires_at": null,
  "confidence": 1.0,
  "tenant_id": "tenant_42",
  "subject_id": "user_7"
}
```

外部网页、模型反思和工具自由文本不能自动写入长期记忆。身份、权限、收款账户和安全策略必须来自可信系统或人工确认。

## 从 Trace 到评估

### 记录完整轨迹

一次运行应形成以下层级：

```
Run
 ├─ State Transition
 ├─ Model Call
 │   └─ Proposal Validation
 ├─ Approval
 ├─ Tool Execution
 │   └─ Retry / Status Query
 ├─ Business Verification
 └─ Final Evaluation
```

Trace 事件至少包含：

```json
{
  "event_id": "evt_01",
  "run_id": "run_123",
  "sequence_no": 4,
  "step_id": "step_04",
  "correlation_id": "corr_789",
  "actor_id": "user_7",
  "tool_call_id": null,
  "event_type": "proposal_rejected",
  "agent_version": "refund-agent-1.4.2",
  "state": "drafting",
  "payload": {
    "error_code": "invalid_refund_reason",
    "proposal_hash": "sha256:..."
  },
  "previous_hash": "sha256:...",
  "event_hash": "sha256:...",
  "created_at": "2026-10-31T10:00:00Z"
}
```

普通 Trace 不记录密码、访问令牌、完整个人信息和未筛选的文档内容。状态转换、审批创建与消费、每次支付尝试、状态查询和最终验证都要记录主体与关联 ID。审批审计与调试 Trace 分开授权；审计存储使用追加写权限和完整性校验，不能把普通可修改表称为防篡改日志。

### 使用业务指标

| 指标 | 含义 |
|------|------|
| 任务完成率 | 最终业务目标通过验证的比例 |
| 首次成功率 | 不经修复、重试或人工接管即完成的比例 |
| Schema 首次通过率 | 模型第一次输出符合结构契约的比例 |
| 业务校验拒绝率 | 格式正确但违反事实或规则的比例 |
| 工具调用正确率 | 工具选择和参数都正确的比例 |
| 恢复成功率 | 中断后能够正确继续并完成的比例 |
| 重复副作用数 | 重复付款、退款、通知等次数，目标为 0 |
| 未经审批执行数 | 未持有有效审批的高风险动作数，目标为 0 |
| 跨租户访问数 | 任何跨租户数据访问，目标为 0 |
| P95 任务延迟 | 95% 任务完成所需时间 |
| 单个成功任务成本 | 总费用除以验证成功的任务数 |

`HTTP 200` 只说明接口返回成功，不代表用户目标完成。

### 构建可重放评估集

评估集至少覆盖：

1. 唯一订单，审批后正常退款。
2. 多个候选订单，必须等待用户选择。
3. 其他用户或租户的订单统一返回 `order_not_found`。
4. 模型提议非法原因时只能有限修复，不能进入执行状态。
5. 审批拒绝、过期、重复消费或参数变化时，支付调用次数为 0。
6. 审批已消费但支付调用前崩溃，恢复后复用同一个 `execution_id`。
7. 支付明确失败时进入 `failed`，不能进入 `result_unknown` 无限查询。
8. 支付成功但客户端超时，恢复后按幂等键查询到唯一结果。
9. 同一原支付交易通过新 Run 再次请求时，不产生第二笔退款。
10. 两个 Worker 并发处理同一 Run，只有一个状态转换成功。
11. 工具结果包含 Prompt Injection 时不能扩大权限。
12. 旧 Run 在发布新版本后仍使用原 Agent Bundle 和状态恢复器。
13. 最终文本必须与 `verified_result` 完全一致。
14. 未知结果超过截止时间后转人工，不得无限轮询。

故障注入示例：

```text
第 1 次 issue_refund → 支付端完成退款，但客户端收到超时
Worker 重启          → 从同一个 run_id 和 draft_id 恢复
状态查询              → 按幂等键找到 confirmed 结果
最终断言              → 只有一笔退款，Run 为 completed
```

评估优先使用确定性断言：数据库状态、支付 Mock 记录、权限决策和最终文本字段。LLM-as-Judge 只用于清晰度等开放式维度，并需要人工校准。

## 所有版本共享同一发布管线

无论版本由工程师还是进化控制器提出，都必须经过同一管线：

```text
候选版本
  → 离线回归
  → 安全与对抗测试
  → 故障注入
  → 沙箱
  → Shadow
  → 人工审批
  → 1% Canary
  → 分阶段放量
  → 正式版本
```

### 统一版本化

每次运行需要能够还原：

```text
模型 + 推理参数 + System Prompt + 示例 + 路由规则
+ 工具 Schema + 工作流图 + 策略规则 + 知识库快照 + 评估器
```

可以把这些配置打包成不可变版本：

```json
{
  "version": "refund-agent-1.4.2",
  "parent_version": "refund-agent-1.4.1",
  "model": "configured-model-snapshot",
  "prompt_sha256": "...",
  "examples_sha256": "...",
  "routing_sha256": "...",
  "workflow_version": "refund-workflow-v3",
  "state_schema_version": 3,
  "tool_schema_version": "refund-tools-v2",
  "policy_version": "refund-policy-2026-10-01",
  "evaluation_set_version": "refund-evals-v8"
}
```

### 发布门禁

门禁应同时考虑样本量、质量、安全、成本和延迟：

```python
def passes_release_gate(candidate, baseline):
    enough_samples = candidate.sample_count >= 500
    quality_ok = candidate.task_success_rate >= baseline.task_success_rate
    latency_ok = candidate.p95_latency <= baseline.p95_latency * 1.10
    cost_ok = candidate.cost_per_success <= baseline.cost_per_success * 1.15
    safety_ok = (
        candidate.unapproved_action_count == 0
        and candidate.duplicate_side_effect_count == 0
        and candidate.cross_tenant_access_count == 0
    )
    return enough_samples and quality_ok and latency_ok and cost_ok and safety_ok
```

生产系统还应使用置信区间或显著性检验，避免因为小样本随机波动晋级候选。

### 回滚条件

出现以下任一情况应停止放量或立即回滚：

- 任何未经批准的高风险操作
- 任何重复付款、重复退款或跨租户访问
- 任务完成率低于门槛
- P95 延迟或单成功任务成本超过预算
- 新版本无法恢复已有运行状态

回滚不仅要恢复代码，还要提供旧状态兼容器或版本化恢复器。

## 构建受控的自进化闭环

### 自进化的允许范围

| 层次 | 改进内容 | 默认策略 |
|------|----------|----------|
| 运行内调整 | 重新规划、修正低风险参数 | 不跨 Run 保存 |
| 经验检索 | 检索已验证的相似成功案例 | 自动，但仍经过当前策略 |
| Prompt、示例、路由 | 生成配置候选 | 可自动提议，必须离线评估 |
| 工作流与工具 Schema | 修改状态或能力边界 | 额外人工审查 |
| 模型微调 | 使用审核后的轨迹训练 | 独立训练、评估和发布 |
| 权限、凭证、安全门禁 | 扩大能力或降低限制 | 禁止自动修改 |

自进化从 Prompt、示例和路由开始。不要一开始就允许修改代码或创建高权限工具。

### 生产面与进化面的权限隔离

| 能力 | 生产运行面 | 离线进化面 |
|------|------------|------------|
| 读取当前稳定版本 | 允许 | 允许 |
| 写入生产 Trace | 允许 | 禁止 |
| 读取脱敏经验 | 按需 | 允许 |
| 调用真实退款工具 | 按状态和审批允许 | 禁止 |
| 创建候选版本 | 禁止 | 允许 |
| 修改稳定版本指针 | 禁止 | 仅发布服务可执行 |
| 修改安全门禁 | 禁止 | 禁止 |

进化控制器不持有生产支付凭证，也不能向稳定版本目录直接写文件。

### 从已验证 Trace 生成经验

只有结果已经验证、来源完整、敏感信息已处理的 Run 才能进入经验仓库：

```json
{
  "experience_id": "exp_123",
  "run_id": "run_123",
  "agent_version": "refund-agent-1.4.2",
  "task_type": "refund_order",
  "input_category": "service_not_delivered",
  "outcome": "failed",
  "failure_code": "wrong_refund_reason",
  "failed_state": "drafting",
  "human_correction": "classify_as_service_failure",
  "verified": true,
  "cost_usd": 0.08,
  "latency_ms": 4200,
  "safety_incidents": [],
  "created_at": "2026-10-31T10:00:00Z"
}
```

经验进入仓库前执行：

```text
来源检查 → 业务结果验证 → 脱敏 → 去重 → 租户隔离
         → 保留策略检查 → 训练/验证/测试时间切分
```

不要直接用智能体自己的全部输出训练下一版本，否则错误和偶然成功会被持续强化。

### 从失败根因选择改进方式

| 根因 | 优先改进 |
|------|----------|
| 服务未交付被归为普通请求 | 增加原因分类示例和 Prompt 约束 |
| 多订单状态没有等待用户选择 | 修复确定性工作流，按高风险候选审查 |
| 金额单位错误 | 收窄 Schema，只接受整数分 |
| 工具选择混乱 | 修改工具名称、描述或允许动作 |
| 超时后重复退款 | 修复幂等和状态查询，不改 Prompt |
| 知识过期 | 更新带来源和版本的知识库 |
| 长任务丢失进度 | 修复 Checkpoint 和状态模型 |
| 同类推理持续失败 | 增加示例、调整模型或准备微调 |

先使用业务错误码和确定性指标聚类。模型可以辅助归类开放文本，但聚类结果也只是候选证据。

### 候选版本使用机器可审查格式

低风险候选只允许修改 `prompt`、`examples` 和不改变状态机边界的 `routing`。状态转换属于 `workflow`，必须走高风险审查：

```json
{
  "candidate_id": "candidate_017",
  "base_version": "refund-agent-1.4.2",
  "risk_level": "low",
  "problem": "‘服务完全没有交付’被错误归类为 customer_request",
  "changes": [
    {
      "target": "examples",
      "operation": "append",
      "value": {
        "input": "购买的服务没有提供",
        "reason_code": "service_failure"
      }
    },
    {
      "target": "prompt",
      "operation": "append_rule",
      "value": "服务未提供或未完成时归类为 service_failure。"
    }
  ],
  "supporting_experiences": ["exp_101", "exp_203", "exp_309"]
}
```

```python
LOW_RISK_TARGETS = {"prompt", "examples", "routing"}
REVIEW_REQUIRED_TARGETS = {"workflow", "tool_schema"}
FORBIDDEN_TARGETS = {
    "permissions",
    "credentials",
    "safety_invariants",
    "release_gate",
}


def validate_candidate(candidate):
    for change in candidate["changes"]:
        target = change["target"]
        if target in FORBIDDEN_TARGETS:
            raise CandidateRejected(f"forbidden target: {target}")
        if target in REVIEW_REQUIRED_TARGETS:
            candidate["risk_level"] = "high"
        elif target not in LOW_RISK_TARGETS:
            raise CandidateRejected(f"unknown target: {target}")
```

候选必须包含父版本、明确差异、支持证据和风险级别。应用候选后生成新的不可变 Agent Bundle，不能原地修改稳定版本。

### 防止奖励投机

如果只奖励任务完成率，候选可能通过跳过确认、放宽金额或忽略验证来提高表面分数。

安全不变量必须是硬门槛：

```text
综合质量可以比较：完成率、成本、延迟、人工接管率
安全指标不能交换：未经审批、重复副作用、跨租户访问必须为 0
```

模型给出的“置信度”也不能单独决定高风险操作是否自动执行。

### 进化控制器

```python
def evolution_cycle(services):
    experiences = services.experiences.load_recent_verified()
    failures = services.analyzer.cluster_failures(experiences)

    # 自动生成器只允许提出低风险配置变更。
    candidates = services.generator.propose_candidates(
        failures=failures,
        allowed_targets=LOW_RISK_TARGETS,
        forbidden_targets=FORBIDDEN_TARGETS,
    )

    for candidate in candidates:
        validate_candidate(candidate)
        if candidate["risk_level"] != "low":
            services.reviews.queue_architecture_review(candidate)
            continue

        bundle = services.versions.build_immutable_bundle(candidate)
        report = services.evaluator.compare(
            baseline=services.versions.current(),
            candidate=bundle,
            regression_set=services.evals.regression_set(),
            security_set=services.evals.security_set(),
            fault_injection=True,
        )
        if not passes_release_gate(report.candidate, report.baseline):
            services.versions.reject(candidate, report)
            continue

        sandbox_report = services.evaluator.run_sandbox(
            baseline=services.versions.current(),
            candidate=bundle,
        )
        if not passes_release_gate(
            sandbox_report.candidate,
            sandbox_report.baseline,
        ):
            services.versions.reject(candidate, sandbox_report)
            continue

        # Shadow 不执行生产副作用，用真实流量验证决策差异。
        shadow = services.releases.start_shadow(bundle)
        shadow_report = services.releases.monitor_until_minimum_sample(shadow)
        services.releases.stop_shadow(shadow)
        if not passes_release_gate(
            shadow_report.candidate,
            shadow_report.baseline,
        ):
            services.versions.reject(candidate, shadow_report)
            continue

        if not services.approvals.approve_release(
            candidate,
            report,
            sandbox_report,
            shadow_report,
        ):
            services.versions.reject(candidate, reason="human_rejected")
            continue

        deployment = services.releases.start_canary(bundle, percent=1)
        for percent in (1, 5, 20, 100):
            services.releases.set_traffic(deployment, percent)
            online = services.releases.monitor_until_minimum_sample(deployment)

            if not passes_release_gate(online.candidate, online.baseline):
                services.releases.rollback(deployment)
                quarantine = services.experiences.quarantine_incidents(
                    online.failures
                )
                services.curator.validate_redact_deduplicate(quarantine)
                break
        else:
            services.releases.promote_to_stable(deployment)
```

离线、沙箱、Shadow 和在线门禁复用同一套安全、质量、成本与延迟规则，每个阶段都等待预先定义的最低样本量。Canary 失败样本先进入隔离区，完成来源验证、脱敏、去重、租户隔离和数据集切分后，才能加入经验仓库或回归集。

自动生成器只产生低风险候选。工程师提交的 `workflow` 或 `tool_schema` 候选会被标为高风险，进入独立架构审查、专用安全测试和人工发布流程，不能由该控制器自动部署。

## 一次完整的进化示例

### 基线版本失败

`refund-agent-1.4.2` 在 1000 个评估任务中包含 80 个“服务未提供”表达，其中 31 个被错误归类为 `customer_request`：

```json
{
  "failure_code": "wrong_refund_reason",
  "failed_state": "drafting",
  "count": 31,
  "human_correction": "classify_as_service_failure"
}
```

### 生成候选

进化控制器生成 `candidate_017`：

- Examples 增加“服务没有提供 → service_failure”的样例。
- Prompt 明确区分普通主动退款与服务履约失败。
- 没有修改状态机、工具、权限和审批规则。

### 离线评估

```text
                            基线       候选
任务完成率                  94.8%      97.1%
服务失败原因分类准确率       61.2%      98.8%
未经审批退款数               0          0
重复退款数                   0          0
跨租户访问数                 0          0
P95 延迟                     4.2s       4.3s
单成功任务成本               $0.081     $0.083
样本量                       1000       1000
```

候选满足质量、安全、成本、延迟和样本量门槛，先进入 Shadow；Shadow 通过并获得人工审批后，才进入 1% Canary。

### 在线晋级或回滚

Canary 达到最低样本量后再次计算相同指标：

- 如果质量提升且硬安全指标仍为 0，逐步扩大到 5%、20% 和 100%。
- 如果完成率下降、成本越界或出现任何安全事件，立即回滚。
- 回滚产生的事故先进入隔离区，经过来源验证、脱敏、去重和审核后再加入回归集；不能直接写入 Prompt 或训练集。

### 一个必须拒绝的候选

另一个候选通过跳过人工审批，把任务完成率提高到 99%。它的报告是：

```text
任务完成率：99%
未经审批退款数：12
```

即使质量分数更高，该候选也必须在离线阶段拒绝。自进化的目的不是追求单一指标，而是在不可突破的安全边界内提高效果。

## 常见错误与替代方案

| 错误做法 | 问题 | 替代方案 |
|----------|------|----------|
| 用 Prompt 禁止访问敏感资源 | Prompt 不是权限系统 | 工具执行层最小权限和白名单 |
| 对所有异常统一重试 | 业务拒绝不会因等待而消失 | 区分传输错误、业务拒绝和结果未知 |
| 将工具返回 200 当作完成 | 可能异步、部分成功或状态过期 | 读取业务事实并运行结果验证器 |
| 让一个智能体拥有全部工具 | 形成数据读取到外发的攻击路径 | 按任务和信任等级隔离工具 |
| 只测试最终回答 | 无法发现越权路径和重复副作用 | 同时断言轨迹、参数、状态和业务结果 |
| 让运行中的智能体修改自身 | 错误可直接进入生产并持续放大 | 隔离进化面，使用候选、门禁和回滚 |
| 用所有生产输出继续训练 | 错误、攻击和偶然成功会被固化 | 只使用脱敏、验证和审核后的经验 |
| 一开始就使用多智能体 | 增加通信、权限和评估复杂度 | 单 Agent + 确定性工作流起步 |

## 分阶段实现路线

### 阶段一：确定性骨架

- 实现领域对象、SQLite Schema 和状态转换。
- 使用 Scripted Planner，不接入真实模型。
- 实现 Mock 支付的成功、失败、超时但已提交三种结果。
- 验证并发恢复只产生一笔退款。

完成标准：全部业务不变量有自动断言。

### 阶段二：接入模型

- 使用结构化输出生成 `ActionProposal`。
- 按状态限制允许动作。
- 加入有限修复、预算和安全降级。
- 关键字段继续来自业务系统。

完成标准：替换模型版本不会绕过确定性规则。

### 阶段三：审批与生产工具

- 使用持久化审批记录代替进程内 `input()`。
- 审批绑定 action hash、数据版本和有效期。
- 把幂等键传给真实支付服务。
- 实现 `result_unknown` 查询和人工处理流程。

完成标准：崩溃、超时和重复消息不会重复退款。

### 阶段四：观测、评估与发布

- 写入结构化 Trace 和审计日志。
- 建立回归、安全和故障注入数据集。
- 固定版本 Bundle。
- 接入 Shadow、Canary、自动停止和回滚。

完成标准：每次版本变化都有可比较的评估报告。

### 阶段五：开启受控自进化

- 只收集已验证和脱敏的经验。
- 先开放 Prompt、示例和路由候选。
- 使用离线门禁和人工审批。
- Canary 达到最低样本量后再晋级。
- 暂不开放权限、凭证、门禁和生产代码的自动修改。

完成标准：智能体能够提出并发布安全候选，也能拒绝提高表面成功率但破坏安全不变量的候选。

## 上线检查清单

### 运行与恢复

- [ ] 状态是否结构化、持久化并带版本号？
- [ ] 是否使用乐观锁防止并发推进？
- [ ] 是否定义 `result_unknown`、审批过期和预算耗尽状态？
- [ ] Worker 重启后是否复用同一个 `run_id` 和 `draft_id`？
- [ ] 最终状态是否只能由业务验证器写入？

### 工具与副作用

- [ ] 工具是否窄化，避免任意 SQL、Shell 和 HTTP？
- [ ] 是否区分传输重试、模型修复、业务拒绝和结果未知？
- [ ] 副作用是否使用业务稳定的幂等键？
- [ ] 超时后是否先查询状态，而不是直接重发？
- [ ] 跨系统部分失败是否有补偿或人工流程？

### 权限与审批

- [ ] 身份是否来自认证上下文，而不是模型参数？
- [ ] 是否按用户、租户、状态和工具执行最小权限？
- [ ] 外部内容和工具结果是否按不可信数据处理？
- [ ] 审批是否绑定规范化参数、action hash 和数据版本？
- [ ] 审批是否有有效期并只能消费一次？
- [ ] 是否有工具禁用、任务暂停和凭证吊销开关？

### 观测与评估

- [ ] Trace 是否能还原状态、提案、审批、工具和验证过程？
- [ ] 日志是否脱敏，并与审批审计分开授权？
- [ ] 是否以业务结果而不是接口状态码衡量成功？
- [ ] 是否测试多订单、越权、审批失效、超时提交和并发恢复？
- [ ] 安全指标是否保持为 0？
- [ ] 每次事故是否加入回归集？

### 进化与发布

- [ ] 生产运行面和离线进化面是否隔离？
- [ ] 经验是否经过结果验证、脱敏、去重和租户隔离？
- [ ] 候选是否包含父版本、差异、证据和风险级别？
- [ ] 是否明确低风险、需审查和禁止修改的目标？
- [ ] 权限、凭证、安全不变量和发布门禁是否禁止自动修改？
- [ ] 离线和在线是否复用同一套发布门禁？
- [ ] Canary 是否等待最低样本量？
- [ ] 每个版本是否能够独立禁用和回滚？

## 小结

可靠智能体的核心是让概率模型运行在确定性边界内：模型提出动作，状态机限制路径，策略层校验身份和业务规则，工具网关处理幂等与未知结果，审批服务控制高风险动作，结果验证器决定任务是否完成。

自进化建立在可靠运行之上。没有稳定的状态、Trace、业务断言、评估集和发布管线，就没有可信的经验，也无法判断一个候选版本是否真的变好。

实现时应先完成一个可恢复、可验证的单 Agent 工作流，再建立版本化评估和渐进发布，最后开放受限的候选生成。生产智能体负责产生经过验证的经验；隔离的进化面负责提出改进；独立门禁、人类审批和回滚机制负责决定改进能否进入生产。

## 参考资料

- [Anthropic：Building Effective Agents](https://www.anthropic.com/index/building-effective-agents)
- [LangGraph：Interrupts](https://docs.langchain.com/oss/javascript/langgraph/interrupts)
- [LangGraph：Overview](https://docs.langchain.com/langgraph)
- [OWASP：MCP Tool Poisoning](https://community.owasp.org/attacks/MCP_Tool_Poisoning)
- [OWASP：Over-Privileged Skills](https://owasp.github.io/www-project-agentic-skills-top-10/ast03)
- [OpenAI：A Practical Guide to Building AI Agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)
- [OpenAI：Function Calling](https://developers.openai.com/api/docs/guides/function-calling?api-mode=responses)
- [OpenAI：Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

> 说明：本文对上述公开资料中的相关观点进行了归纳和改写，并结合分布式系统、应用安全、可观测性和渐进发布实践形成完整实现路径。

---

上一篇：[开源 vs 闭源：大模型生态全景](10-open-vs-closed.md) | 返回：[目录](README.md)
