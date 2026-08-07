"""LangGraph 状态定义（文档 8.1）。

设计要点：
- 所有字段可 JSON 序列化，便于写入 checkpoint（内存或 PostgreSQL）。
- 列表字段不使用 add reducer，而采用"全量替换"语义：节点返回完整列表。
  这样失败重跑时不会重复累加，配合 executor 的 request_key 去重实现幂等。
- 时间相关数据用 unix epoch（float），避免 datetime 序列化问题。
"""

from __future__ import annotations

from typing import TypedDict


class EvaluationState(TypedDict, total=False):
    """评测 Agent 的全局状态。total=False 使节点可返回部分更新。"""

    # 标识
    run_id: str

    # 输入
    request: str | None  # 自然语言请求（阶段二）；配置驱动时为 None
    config: dict | None  # 解析后的 RunConfig（intake_request 产出）

    # 计划
    plan: dict | None  # EvaluationPlan.model_dump()

    # 数据与任务
    cases: list[dict]  # TestCase.model_dump()
    tasks: list[dict]  # TaskItem.model_dump()，case × model 矩阵

    # 执行与评分结果
    model_results: list[dict]  # ModelResult.model_dump()
    score_results: list[dict]  # ScoreResult.model_dump()

    # 聚合
    metrics: dict | None  # Metrics.model_dump()
    analysis: dict | None  # Analyst 节点输出（observed/evidence/limitations）

    # 错误与审批
    errors: list[dict]  # ErrorRecord.model_dump()
    approval_required: bool
    approval_decision: str | None  # "approved" | "rejected" | None（resume 时注入）

    # 产物与进度
    report_path: str | None
    status: str  # RunStatus 值
    current_node: str  # 当前节点名，供 status 查询
