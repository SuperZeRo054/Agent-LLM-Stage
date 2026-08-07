# Agent-LLM-Stage：基于 LangGraph 的 LLM 评测 Agent

## 1. 一句话定义

这是一个**评测 Agent**，而不是聊天机器人或普通批处理脚本：用户提交“用哪些模型、哪份测试集、按什么规则评测”的任务后，Agent 自动规划并执行一次完整评测，最后给出可追溯的对比报告。

它的核心能力是：

```text
理解评测任务 -> 生成执行计划 -> 并发调用多个 LLM
-> 规则校验与 LLM Judge 盲评 -> 汇总指标 -> 解释结果并生成报告
```

测试集和具体业务场景暂不预设，后续独立设计。系统先把评测流程、数据契约和可恢复执行能力做完整。

## 2. 为什么选 LangGraph

LangChain 适合统一接入不同模型和封装 Prompt/Structured Output；LangGraph 适合这个项目的核心问题：让一个长流程 Agent 拥有明确状态、条件分支、失败恢复和人工审批点。

- `StateGraph`：把评测步骤定义为清晰、可测试的节点。
- Checkpointer：每个节点完成后保存状态；中断后可从最后成功节点恢复。
- 条件边：配置或测试集不合法时停止；没有可评分结果时跳过 Judge，生成失败报告。
- `interrupt()`：在高成本运行或测试集尚未确认时，暂停并等待人工确认。

Agent 的“智能”应受边界约束：LLM 可以规划任务、语义评分、解释报告，但不能随意修改测试集、模型配置或历史结果。模型调用、指标计算、落库和重试必须保持确定性。

## 3. Agent 职责边界

| 能力 | 实现方式 | 是否由 LLM 决策 |
| --- | --- | --- |
| 理解自然语言评测请求 | Planner 节点 + 结构化输出 | 是，受 Schema 限制 |
| 校验可执行性 | Pydantic/规则 | 否 |
| 创建 `case x model` 任务 | 确定性函数 | 否 |
| 并发调用被测模型 | `asyncio` + Adapter | 否 |
| 格式、关键点、安全规则评分 | 确定性规则 | 否 |
| 回答质量评分 | LLM Judge | 是，盲评且结构化 |
| 统计延迟、成本、失败率 | 确定性函数 | 否 |
| 解释差异和生成摘要 | Analyst 节点 | 是，只读结果 |

这个划分是项目的重点。若把所有步骤交给 LLM 决定，结果就不可复现，也不能称为可靠的评测系统。

## 4. LangGraph 状态图

```text
START
  -> intake_request           # 读取自然语言请求或 YAML 配置
  -> plan_evaluation          # LLM 生成结构化评测计划
  -> validate_plan            # 校验模型、数据集、预算、评分配置
  -> [need_approval?] --- 是 -> human_approval (interrupt)
                           否 -> build_tasks
  -> build_tasks              # case x model 任务矩阵
  -> execute_models           # 限流、重试、并发执行
  -> validate_outputs         # 结果解析、失败归档
  -> [has_candidates?] --- 否 -> generate_failure_report -> END
                        是 -> score_results
  -> score_results            # 规则评分 + Judge 盲评
  -> aggregate_metrics        # 指标聚合、评分方差、异常标记
  -> analyze_results          # LLM 根据证据生成解释
  -> generate_report
  -> END
```

MVP 不需要多 Agent。这里是**一个 Agent、一个 StateGraph**；每个节点是职责单一的工具或受限 LLM 调用。将来只有出现真正独立的长任务，例如自动扩充测试集或自动归因，才考虑拆成子图。

## 5. 输入、输出与验收标准

### 5.1 用户输入

用户可通过 CLI/API 提交 YAML，后续再支持自然语言输入：

```yaml
run_name: model-comparison-v1
dataset: datasets/dataset_v1.jsonl
models:
  - provider: provider_a
    model: model_a
    concurrency: 3
  - provider: provider_b
    model: model_b
    concurrency: 3
judge:
  model: judge_model
  repeats: 2
budget_limit_usd: 10
```

自然语言请求只能被 Planner 转换为上述 Schema；任何缺失、超预算或不允许的配置都必须进入人工确认，不能由 Agent 猜测填充。

### 5.2 系统输出

- 一个不可变的 `run_id`；
- 每个 `case x model` 的输入、输出、耗时、token 用量、成本、重试和错误；
- 规则评分、Judge 评分、评分证据和评分方差；
- Markdown/HTML 报告；
- 可从任一 checkpoint 继续或回放的运行记录。

### 5.3 MVP 验收

| 项目 | MVP 目标 |
| --- | --- |
| 被测模型 | 3 个模型或 3 组配置 |
| 测试集 | 后续单独设计，数据符合 JSONL Schema |
| 评分 | 规则评分 + 1 个独立 Judge，双次评分 |
| 指标 | 质量分、P50/P95 延迟、token 成本、失败率、评分方差 |
| 恢复 | 在执行或评分节点失败后可从 checkpoint 重跑 |
| 交付 | CLI、Supabase、Markdown 报告、Docker Compose |

## 6. 推荐技术栈

| 层级 | 选择 | 用途 |
| --- | --- | --- |
| Agent 编排 | LangGraph `StateGraph` | 状态机、路由、checkpoint、interrupt |
| 模型抽象 | LangChain Core | 统一 Chat Model、Prompt、结构化输出 |
| 并发 | Python `asyncio`、`httpx` | 请求并发、限流、超时、重试 |
| 数据模型 | Pydantic | 输入计划、测试样本、模型结果、Judge 结果 |
| 数据库 | Supabase PostgreSQL + SQLAlchemy | 业务审计数据、认证、RLS；开发与部署使用同一 PostgreSQL |
| 图状态 | LangGraph PostgreSQL Checkpointer | checkpoint、暂停恢复和运行回放 |
| 服务层 | FastAPI + Typer | API、CLI 和运行状态查询 |
| 报告 | Jinja2 + Plotly | HTML 图表；MVP 可先输出 Markdown |
| 工程质量 | pytest、Ruff、Docker Compose | 测试、规范和可复现部署 |

依赖版本应在 `pyproject.toml` 中锁定。不要对不受信任的配置或 checkpoint 进行反序列化；持久化相关依赖须持续更新并做安全公告检查。

### 6.1 Supabase 数据库设计

Supabase 托管的是 PostgreSQL，因此项目从第一天起就使用 PostgreSQL，不存在从 SQLite 迁移的步骤。建议分成两类数据：

| 数据域 | 表/存储 | 说明 |
| --- | --- | --- |
| 业务数据 | `evaluation_runs`、`test_cases`、`model_results`、`judge_results`、`reports` | 评测任务、结果、评分和报告元数据 |
| Agent 状态 | LangGraph PostgreSQL checkpoint 表 | 节点状态、`thread_id`、interrupt 与恢复信息 |
| 大对象 | Supabase Storage | 大型报告、导出文件；数据库只存路径和摘要 |

业务表通过 SQLAlchemy 访问 Supabase PostgreSQL；前端或用户侧访问时使用 Supabase Auth 和 Row Level Security（RLS）限制为 `user_id = auth.uid()`。服务端使用专用数据库连接或服务端密钥，密钥只放在环境变量，绝不返回给浏览器。

`run_id` 是业务运行标识，LangGraph 的 `thread_id` 建议直接使用同一个值。这样可从业务运行记录定位到对应 checkpoint，但不应让用户直接写入 checkpoint 表。

## 7. 目录结构

```text
Agent-LLM-Stage/
  app/
    graph/
      state.py               # EvaluationState、reducer
      graph.py               # StateGraph 与条件边
      nodes.py               # 9 个节点函数
      routes.py              # 路由函数
    domain/
      schemas.py             # Pydantic Schema
    adapters/
      base.py                # ModelAdapter 协议
      provider_a.py
      provider_b.py
    services/
      executor.py            # asyncio、限流、重试
      scoring.py             # 规则评分、Judge 调用
      metrics.py             # 统计和异常标记
      reports.py             # Markdown/HTML
    repositories/
      runs.py                # 运行记录、原始结果
    api/
    cli.py
  datasets/                  # 后续设计的 JSONL 测试集
  configs/
    run.example.yaml
  prompts/
    planner_v1.txt
    judge_v1.txt
    analyst_v1.txt
  tests/
  docker-compose.yml
  pyproject.toml
  README.md
```

## 8. 状态、工具与数据契约

### 8.1 Agent State

```python
from typing import TypedDict

class EvaluationState(TypedDict):
    run_id: str
    request: str | None
    plan: dict | None
    cases: list[dict]
    tasks: list[dict]
    model_results: list[dict]
    score_results: list[dict]
    metrics: dict | None
    errors: list[dict]
    report_path: str | None
    approval_required: bool
    status: str
```

State 中只存运行必需、可序列化的数据。完整 Prompt、原始模型响应和日志保存到数据库或文件，并通过 ID 回链，避免 checkpoint 过大。

### 8.2 工具契约

| 工具/服务 | 输入 | 输出 | 约束 |
| --- | --- | --- | --- |
| `load_dataset` | 数据集路径 | 校验后的 samples | 路径白名单、Schema 校验 |
| `run_model_batch` | model config、tasks | results | 每供应商独立限流；幂等 request key |
| `score_batch` | 匿名回答、rubric | score results | 不传被测模型身份 |
| `calculate_metrics` | results、scores | metrics | 纯函数、可单测 |
| `write_report` | metrics、result refs | report path | 不改写原始结果 |

### 8.3 最小数据结构

测试集在后续设计时应满足这个最小结构：

```json
{
  "id": "case-001",
  "category": "待定义",
  "input": "待设计的测试问题",
  "context": [],
  "reference_answer": "待设计的参考答案",
  "rubric": {
    "correctness": "待定义",
    "completeness": "待定义",
    "safety": "待定义"
  }
}
```

每条模型结果保存 `model_id`、配置快照、Prompt 版本、回答、开始/结束时间、token、成本、重试次数、状态和错误。Judge 结果额外保存匿名候选 ID、评分 Prompt 版本、各维度分数、证据、风险标记和原始 JSON。

## 9. 节点实现方案

### 9.1 `plan_evaluation`

输入是用户请求，输出必须是 Pydantic `EvaluationPlan`。Planner 只允许选择已注册的模型、数据集和评分模板；缺少关键信息时返回 `requires_clarification`，不能发起评测。

### 9.2 `validate_plan` 与人工审批

检查数据集、模型白名单、并发上限、预计调用数和预算。满足以下任一条件则调用 `interrupt()`：预算超过阈值、测试集未锁定、用户要求的模型未配置、预计耗时过长。恢复时以同一 `thread_id` 继续执行。

### 9.3 `execute_models`

生成 `case x model` 矩阵，再使用 `asyncio` 并发执行。为每个供应商单独设 Semaphore 和指数退避；单个任务失败写入结果，不中断整个 run。每个任务使用稳定的 `request_key = run_id + case_id + model_id`，以支持断点续跑时去重。

### 9.4 `score_results`

先运行确定性规则，再对合格候选进行 LLM Judge 盲评。Judge 只接收任务、参考答案、评分量表和匿名回答；每条回答评分两次，记录均值和方差。Judge 不能给自己的模型响应打分，或至少要在报告中标记潜在偏差。

### 9.5 `analyze_results`

Analyst 节点只能读取已聚合的指标和具体样本证据。它输出“观察到的差异、支持该结论的数据、不能下结论的原因”，不能自行生成不存在的指标或推荐未经评测的模型。

## 10. Graph 骨架

```python
from langgraph.graph import START, END, StateGraph

builder = StateGraph(EvaluationState)
builder.add_node("intake_request", intake_request)
builder.add_node("plan_evaluation", plan_evaluation)
builder.add_node("validate_plan", validate_plan)
builder.add_node("human_approval", human_approval)
builder.add_node("build_tasks", build_tasks)
builder.add_node("execute_models", execute_models)
builder.add_node("validate_outputs", validate_outputs)
builder.add_node("score_results", score_results)
builder.add_node("aggregate_metrics", aggregate_metrics)
builder.add_node("analyze_results", analyze_results)
builder.add_node("generate_report", generate_report)

builder.add_edge(START, "intake_request")
builder.add_edge("intake_request", "plan_evaluation")
builder.add_edge("plan_evaluation", "validate_plan")
builder.add_conditional_edges("validate_plan", route_after_validation)
builder.add_edge("human_approval", "build_tasks")
builder.add_edge("build_tasks", "execute_models")
builder.add_edge("execute_models", "validate_outputs")
builder.add_conditional_edges("validate_outputs", route_after_output_validation)
builder.add_edge("score_results", "aggregate_metrics")
builder.add_edge("aggregate_metrics", "analyze_results")
builder.add_edge("analyze_results", "generate_report")
builder.add_edge("generate_report", END)

graph = builder.compile(checkpointer=checkpointer)
```

节点必须幂等：失败后重跑不应产生重复扣费、重复写入或覆盖原始结果。尤其 `interrupt()` 前的外部副作用必须可安全重放。

## 11. 评分与报告原则

1. 规则评分和 LLM Judge 分开保存，不能只保留总分。
2. 被测模型名称对 Judge 隐藏，候选展示顺序随机化。
3. Judge 输出使用严格 JSON Schema 校验；解析失败有限重试并留痕。
4. 小样本报告只描述“在本测试集上的表现”，不宣称通用能力领先。
5. 报告必须给出样本数、运行配置、总成本、失败率、评分方差和低分样本回放。

## 12. 三阶段开发计划

### 阶段一：Graph 骨架与模拟执行

- 建立 Pydantic Schema、`EvaluationState` 和 StateGraph；
- 用 Fake Adapter 模拟三个模型，验证节点路由、失败分支和 checkpoint 恢复；
- 完成 CLI：`run`、`status`、`resume`、`report`；
- 为路由函数、状态更新和指标函数编写单元测试。

### 阶段二：真实模型与评分闭环

- 接入 2-3 个真实模型 Adapter；
- 加入限流、超时、幂等 request key 和成本记录；
- 接入规则评分和 LLM Judge 双次盲评；
- 在测试集确定后跑通第一份真实报告。

### 阶段三：可信度与可展示性

- 配置 Supabase PostgreSQL 业务表、RLS 与 LangGraph PostgreSQL Checkpointer；
- 增加 `interrupt()` 审批、失败恢复和样本回放；
- 生成 HTML 报告与图表；
- Docker Compose、README、演示视频和 GitHub Actions 回归集。

## 13. 面试可展示物

- LangGraph 状态图和每个节点的职责说明；
- 一次完整 run 的输入配置、checkpoint、原始输出与最终报告；
- 一次中断后恢复、且不重复扣费的演示；
- Judge 评分与人工复核存在差异的真实案例；
- 已知局限：测试集覆盖范围、Judge 偏差、模型版本变化。

## 14. 完成后的简历表述

> 基于 LangGraph 构建 LLM 评测 Agent，将任务规划、配置校验、多模型并发执行、盲评 LLM Judge、指标聚合与报告生成编排为可 checkpoint 恢复的状态图；基于 Supabase PostgreSQL 实现运行审计、结果持久化与权限隔离，支持幂等执行，并输出质量得分、P95 延迟、token 成本、失败率及低分样本回放报告。

模型数量、样本数量、自动化率和性能指标必须在项目完成后按真实运行结果填写，并能在仓库和报告中复现。
