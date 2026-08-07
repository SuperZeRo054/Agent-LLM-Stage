"""条件边路由函数（文档第 4 节）。

- route_after_validation：校验失败 -> 失败报告；需审批 -> 人工；否则 -> 构建任务。
- route_after_approval：approved -> 构建任务；rejected -> 失败报告。
- route_after_output_validation：有可评分候选 -> 评分；否则 -> 失败报告。
"""

from __future__ import annotations

from app.domain.schemas import TaskStatus
from app.graph.state import EvaluationState


def route_after_validation(state: EvaluationState) -> str:
    if state.get("errors"):
        return "generate_failure_report"
    if state.get("approval_required"):
        return "human_approval"
    return "build_tasks"


def route_after_approval(state: EvaluationState) -> str:
    if state.get("approval_decision") == "approved":
        return "build_tasks"
    return "generate_failure_report"


def route_after_output_validation(state: EvaluationState) -> str:
    results = state.get("model_results") or []
    has_candidates = any(
        r.get("status") == TaskStatus.success.value and (r.get("answer") or "").strip()
        for r in results
    )
    return "score_results" if has_candidates else "generate_failure_report"
