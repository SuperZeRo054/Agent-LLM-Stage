"""StateGraph 组装（文档第 10 节）。

一个 Agent、一个 StateGraph；12 个节点（含文档第 4 节的 generate_failure_report 分支）。
条件边处理：校验失败/审批/无可评分候选。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.routes import (
    route_after_approval,
    route_after_output_validation,
    route_after_validation,
)
from app.graph.state import EvaluationState


def build_graph(checkpointer=None):
    """构建并编译 StateGraph。checkpointer 为 None 时不启用持久化（仅测试用）。"""
    builder = StateGraph(EvaluationState)

    builder.add_node("intake_request", nodes.intake_request)
    builder.add_node("plan_evaluation", nodes.plan_evaluation)
    builder.add_node("validate_plan", nodes.validate_plan)
    builder.add_node("human_approval", nodes.human_approval)
    builder.add_node("build_tasks", nodes.build_tasks)
    builder.add_node("execute_models", nodes.execute_models)
    builder.add_node("validate_outputs", nodes.validate_outputs)
    builder.add_node("score_results", nodes.score_results)
    builder.add_node("aggregate_metrics", nodes.aggregate_metrics)
    builder.add_node("analyze_results", nodes.analyze_results)
    builder.add_node("generate_report", nodes.generate_report)
    builder.add_node("generate_failure_report", nodes.generate_failure_report)

    builder.add_edge(START, "intake_request")
    builder.add_edge("intake_request", "plan_evaluation")
    builder.add_edge("plan_evaluation", "validate_plan")
    builder.add_conditional_edges("validate_plan", route_after_validation)
    builder.add_conditional_edges("human_approval", route_after_approval)
    builder.add_edge("build_tasks", "execute_models")
    builder.add_edge("execute_models", "validate_outputs")
    builder.add_conditional_edges("validate_outputs", route_after_output_validation)
    builder.add_edge("score_results", "aggregate_metrics")
    builder.add_edge("aggregate_metrics", "analyze_results")
    builder.add_edge("analyze_results", "generate_report")
    builder.add_edge("generate_report", END)
    builder.add_edge("generate_failure_report", END)

    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()
