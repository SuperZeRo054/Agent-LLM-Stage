"""StateGraph 节点函数（文档第 4、9 节）。

每个节点职责单一；LLM 节点（plan/judge/analyst）在阶段一为确定性实现，
结构兼容阶段二真实 LLM。确定性步骤（执行、指标、落库）保持可复现。
节点幂等：失败重跑不产生重复扣费/写入（executor 的 request_key 去重 + repo upsert）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from langgraph.types import interrupt

from app.adapters.base import get_adapter, registered_providers
from app.domain.schemas import (
    ErrorRecord,
    EvaluationPlan,
    JudgeConfig,
    Metrics,
    ModelConfig,
    ModelResult,
    RunConfig,
    RunStatus,
    ScoreResult,
    TaskItem,
    TaskStatus,
    TestCase,
)
from app.graph.state import EvaluationState
from app.prompts import load_prompt
from app.repositories.runs import RunRepository
from app.services.executor import execute_batch
from app.services.metrics import calculate_metrics
from app.services.reports import generate_markdown_report
from app.services.scoring import score_batch
from app.utils import load_dataset, new_run_id, parse_json_lenient

logger = logging.getLogger(__name__)

# 预算超过该阈值即使 auto_approve 也需人工确认
BUDGET_APPROVAL_THRESHOLD = 50.0


def _err(node: str, message: str, **ctx) -> dict:
    return ErrorRecord(node=node, message=message, context=ctx).model_dump(mode="json")


# --------------------------------------------------------------------------- #
# 1. intake_request
# --------------------------------------------------------------------------- #
def intake_request(state: EvaluationState) -> dict:
    """读取 YAML 配置（或阶段二的自然语言请求），产出 run_id 与 RunConfig。"""
    config_dict = state.get("config")
    if config_dict is None:
        request = state.get("request")
        if request:
            # 自然语言请求需 --config 提供 planner 配置
            return {
                "errors": [_err("intake_request", "自然语言请求需通过 --config 提供 planner 配置")],
                "status": RunStatus.failed.value,
                "current_node": "intake_request",
            }
        return {
            "errors": [_err("intake_request", "缺少 config 与 request")],
            "status": RunStatus.failed.value,
            "current_node": "intake_request",
        }
    config = RunConfig(**config_dict)
    run_id = state.get("run_id") or new_run_id(config.run_name)
    return {
        "run_id": run_id,
        "config": config.model_dump(mode="json"),
        "status": RunStatus.planning.value,
        "current_node": "intake_request",
    }


# --------------------------------------------------------------------------- #
# 2. plan_evaluation
# --------------------------------------------------------------------------- #
PLAN_PARSE_RETRIES = 2


def _list_datasets() -> list[str]:
    """列出 datasets/ 下可用测试集文件名，供 Planner 选择。"""
    d = Path("datasets")
    if not d.exists():
        return []
    return sorted(p.name for p in d.glob("*.jsonl"))


def _parse_plan_json(raw: str, config: RunConfig) -> EvaluationPlan:
    """将 Planner 返回的 JSON 解析为 EvaluationPlan。失败抛 ValueError。

    使用 parse_json_lenient 容错解析（剥离代码块 + json_repair），
    以应对 Planner 在字符串字段中混入未转义引号等瑕疵。
    """
    data = parse_json_lenient(raw)
    if not isinstance(data, dict):
        raise ValueError("Planner 返回非对象 JSON")

    if data.get("requires_clarification"):
        return EvaluationPlan(
            run_name=data.get("run_name") or config.run_name,
            dataset=data.get("dataset") or "",
            models=[ModelConfig(**m) for m in data.get("models", [])],
            judge=JudgeConfig(**data["judge"]) if data.get("judge") else config.judge,
            budget_limit_usd=data.get("budget_limit_usd", config.budget_limit_usd),
            requires_clarification=True,
            missing_fields=list(data.get("missing_fields", [])),
        )

    models = [ModelConfig(**m) for m in data.get("models", [])]
    if not models:
        raise ValueError("Planner 未产出 models")
    return EvaluationPlan(
        run_name=data.get("run_name") or config.run_name,
        dataset=data.get("dataset") or "",
        models=models,
        judge=JudgeConfig(**data["judge"]) if data.get("judge") else config.judge,
        budget_limit_usd=data.get("budget_limit_usd", config.budget_limit_usd),
        requires_clarification=False,
        missing_fields=[],
    )


async def _plan_with_llm(request: str, config: RunConfig) -> EvaluationPlan:
    """自然语言请求 -> 结构化 EvaluationPlan（阶段二 LLM Planner）。"""
    system = load_prompt("planner_v1")
    user_prompt = (
        f"【用户请求】\n{request}\n\n"
        f"【可用 provider】\n{registered_providers()}\n\n"
        f"【可用数据集】\n{_list_datasets()}\n\n"
        "请输出符合 EvaluationPlan Schema 的 JSON。"
    )
    adapter = get_adapter(config.planner)  # type: ignore[arg-type]
    last_error = ""
    for attempt in range(PLAN_PARSE_RETRIES + 1):
        try:
            out = await adapter.invoke(user_prompt, [system], json_mode=True)
        except Exception as e:  # noqa: BLE001
            last_error = f"{type(e).__name__}: {e}"
            continue
        try:
            return _parse_plan_json(out.answer, config)
        except (ValueError, TypeError) as e:
            last_error = str(e)
            logger.warning("Planner JSON 解析失败(第%d次): %s", attempt + 1, e)
            continue
    # 全部失败：返回需澄清的空计划，由 validate_plan 转失败报告
    logger.warning("Planner 降级: %s", last_error)
    return EvaluationPlan(
        run_name=config.run_name,
        dataset="",
        models=[],
        judge=config.judge,
        budget_limit_usd=config.budget_limit_usd,
        requires_clarification=True,
        missing_fields=[f"planner_failed: {last_error}"],
    )


async def plan_evaluation(state: EvaluationState) -> dict:
    """将请求转换为结构化 EvaluationPlan。

    YAML 配置 -> 确定性构造；自然语言 request -> LLM Planner 结构化输出。
    """
    config = RunConfig(**state["config"])
    request = state.get("request")

    if request:
        if not config.planner:
            return {
                "errors": [_err("plan_evaluation", "自然语言请求需在配置中提供 planner")],
                "status": RunStatus.failed.value,
                "current_node": "plan_evaluation",
            }
        plan = await _plan_with_llm(request, config)
        return {
            "plan": plan.model_dump(mode="json"),
            "status": RunStatus.planning.value,
            "current_node": "plan_evaluation",
        }

    # 确定性：直接从配置构造
    plan = EvaluationPlan(
        run_name=config.run_name,
        dataset=config.dataset,
        models=config.models,
        judge=config.judge,
        budget_limit_usd=config.budget_limit_usd,
        requires_clarification=False,
        missing_fields=[],
        estimated_calls=0,
    )
    return {
        "plan": plan.model_dump(mode="json"),
        "status": RunStatus.planning.value,
        "current_node": "plan_evaluation",
    }


# --------------------------------------------------------------------------- #
# 3. validate_plan
# --------------------------------------------------------------------------- #
def validate_plan(state: EvaluationState) -> dict:
    """校验数据集、模型白名单、并发、预算与预计调用数；判定是否需人工审批。"""
    plan = EvaluationPlan(**state["plan"])
    config = RunConfig(**state["config"])
    errors: list[dict] = list(state.get("errors") or [])
    cases: list[TestCase] = []

    # Planner 标记需澄清 -> 失败报告（列出 missing_fields）
    if plan.requires_clarification:
        missing = ", ".join(plan.missing_fields) or "未知字段"
        errors.append(_err("validate_plan", f"评测计划需澄清，缺少: {missing}"))
        return {
            "plan": plan.model_dump(mode="json"),
            "cases": [],
            "errors": errors,
            "approval_required": False,
            "status": RunStatus.failed.value,
            "current_node": "validate_plan",
        }

    # 数据集加载（路径白名单 + Schema 校验）
    try:
        cases = load_dataset(plan.dataset)
    except Exception as e:  # noqa: BLE001
        errors.append(_err("validate_plan", f"数据集加载失败: {e}"))

    # 被测模型非空
    if not plan.models:
        errors.append(_err("validate_plan", "计划未包含任何被测模型"))

    # 模型白名单
    providers = registered_providers()
    for m in plan.models:
        if m.provider not in providers:
            errors.append(
                _err("validate_plan", f"模型 {m.model} 的 provider '{m.provider}' 未注册", available=providers)
            )

    # 预计调用数
    estimated = len(cases) * len(plan.models)
    plan.estimated_calls = estimated

    # 审批条件（文档 9.2）：预算超阈值 / 非 auto_approve
    approval_required = (not config.auto_approve) or (plan.budget_limit_usd > BUDGET_APPROVAL_THRESHOLD)

    status = RunStatus.awaiting_approval.value if approval_required else RunStatus.executing.value
    return {
        "plan": plan.model_dump(mode="json"),
        "cases": [c.model_dump(mode="json") for c in cases],
        "errors": errors,
        "approval_required": approval_required,
        "status": status,
        "current_node": "validate_plan",
    }


# --------------------------------------------------------------------------- #
# 4. human_approval
# --------------------------------------------------------------------------- #
def human_approval(state: EvaluationState) -> dict:
    """高成本运行前暂停，等待人工确认（interrupt）。resume 时注入审批决定。"""
    decision = interrupt(
        {
            "run_id": state.get("run_id"),
            "message": "评测计划已就绪，请确认是否执行（approved / rejected）",
            "estimated_calls": EvaluationPlan(**state["plan"]).estimated_calls,
        }
    )
    status = RunStatus.executing.value if decision == "approved" else RunStatus.failed.value
    return {
        "approval_decision": decision,
        "status": status,
        "current_node": "human_approval",
    }


# --------------------------------------------------------------------------- #
# 5. build_tasks
# --------------------------------------------------------------------------- #
def build_tasks(state: EvaluationState) -> dict:
    """生成 case × model 任务矩阵，每项带稳定 request_key。"""
    plan = EvaluationPlan(**state["plan"])
    cases = [TestCase(**c) for c in state["cases"]]
    run_id = state["run_id"]
    tasks: list[dict] = []
    for case in cases:
        for m in plan.models:
            request_key = f"{run_id}:{case.id}:{m.provider}:{m.model}"
            tasks.append(
                TaskItem(
                    request_key=request_key,
                    run_id=run_id,
                    case_id=case.id,
                    model_id=m.model,
                    provider=m.provider,
                ).model_dump(mode="json")
            )
    return {
        "tasks": tasks,
        "status": RunStatus.executing.value,
        "current_node": "build_tasks",
    }


# --------------------------------------------------------------------------- #
# 6. execute_models
# --------------------------------------------------------------------------- #
async def execute_models(state: EvaluationState) -> dict:
    """限流、重试、并发执行被测模型（文档 9.3）。"""
    plan = EvaluationPlan(**state["plan"])
    cases = [TestCase(**c) for c in state["cases"]]
    tasks = [TaskItem(**t) for t in state["tasks"]]
    repo = RunRepository(state["run_id"])
    results = await execute_batch(tasks, cases, plan.models, repo)
    return {
        "model_results": [r.model_dump(mode="json") for r in results],
        "status": RunStatus.scoring.value,
        "current_node": "execute_models",
    }


# --------------------------------------------------------------------------- #
# 7. validate_outputs
# --------------------------------------------------------------------------- #
def validate_outputs(state: EvaluationState) -> dict:
    """结果解析、失败归档；无可评分候选时由路由转失败报告。"""
    results = [ModelResult(**r) for r in state.get("model_results", [])]
    failed = [r for r in results if r.status == TaskStatus.failed]
    # 失败已由 executor 写入 repo；此处仅在 errors 中留痕（去重）
    errors: list[dict] = list(state.get("errors") or [])
    existing_msgs = {e.get("message") for e in errors}
    for r in failed:
        msg = f"模型调用失败 [{r.model_id}/{r.case_id}]: {r.error}"
        if msg not in existing_msgs:
            errors.append(_err("validate_outputs", msg, model_id=r.model_id, case_id=r.case_id))
            existing_msgs.add(msg)

    has_candidates = any(r.status == TaskStatus.success and r.answer.strip() for r in results)
    status = RunStatus.scoring.value if has_candidates else RunStatus.failed.value
    return {"errors": errors, "status": status, "current_node": "validate_outputs"}


# --------------------------------------------------------------------------- #
# 8. score_results
# --------------------------------------------------------------------------- #
async def score_results(state: EvaluationState) -> dict:
    """规则评分 + Judge 双次盲评（文档 9.4）。"""
    plan = EvaluationPlan(**state["plan"])
    cases = [TestCase(**c) for c in state["cases"]]
    results = [ModelResult(**r) for r in state["model_results"]]
    scores = await score_batch(results, cases, plan.judge, plan.judge.model)
    repo = RunRepository(state["run_id"])
    for s in scores:
        repo.upsert_score_result(s)
    return {
        "score_results": [s.model_dump(mode="json") for s in scores],
        "status": RunStatus.aggregating.value,
        "current_node": "score_results",
    }


# --------------------------------------------------------------------------- #
# 9. aggregate_metrics
# --------------------------------------------------------------------------- #
def aggregate_metrics(state: EvaluationState) -> dict:
    """指标聚合、评分方差、异常标记（纯函数）。"""
    plan = EvaluationPlan(**state["plan"])
    results = [ModelResult(**r) for r in state["model_results"]]
    scores = [ScoreResult(**s) for s in state.get("score_results", [])]
    metrics = calculate_metrics(state["run_id"], results, scores, plan.budget_limit_usd)
    return {
        "metrics": metrics.model_dump(mode="json"),
        "status": RunStatus.analyzing.value,
        "current_node": "aggregate_metrics",
    }


# --------------------------------------------------------------------------- #
# 10. analyze_results
# --------------------------------------------------------------------------- #
def _deterministic_analysis(metrics: Metrics, judge_provider: str = "fake") -> dict:
    """阶段一确定性 Analyst：只读指标与样本证据（文档 9.5）。阶段二替换为 LLM。"""
    observed: list[str] = []
    evidence: list[str] = []
    if metrics.models:
        best = max(metrics.models, key=lambda m: m.quality_score)
        worst = min(metrics.models, key=lambda m: m.quality_score)
        observed.append(
            f"{best.model_id} 在本测试集上质量分最高（{best.quality_score:.2f}），"
            f"{worst.model_id} 最低（{worst.quality_score:.2f}）。"
        )
        evidence.append(f"样本数 {metrics.sample_count}，共 {sum(m.count for m in metrics.models)} 次调用。")
        for m in metrics.models:
            if m.failure_rate > 0:
                observed.append(f"{m.model_id} 存在失败调用，失败率 {m.failure_rate:.0%}。")
            evidence.append(
                f"{m.model_id}: P95 {m.p95_latency_ms:.0f}ms，成本 ${m.total_cost_usd:.4f}，"
                f"评分方差 {m.quality_variance:.2f}。"
            )
        if metrics.budget_exceeded:
            observed.append("总成本超过预算。")
    if judge_provider == "fake":
        judge_limit = "Judge 为确定性 fake judge，存在已知偏差，需人工复核校准。"
    else:
        judge_limit = f"Judge 为真实 LLM（{judge_provider}），评分仍存在模型自身偏差，需人工复核校准。"
    limitations = [
        "仅描述被测模型在本测试集上的表现，不构成通用能力领先的结论。",
        judge_limit,
        "测试集覆盖范围有限，结论不可外推到其他场景或模型版本。",
    ]
    return {"observed_differences": observed, "supporting_evidence": evidence, "limitations": limitations}


def analyze_results(state: EvaluationState) -> dict:
    """LLM Analyst 只读已聚合指标生成解释（阶段一确定性）。"""
    metrics = Metrics(**state["metrics"])
    config = RunConfig(**state["config"])
    analysis = _deterministic_analysis(metrics, config.judge.provider)
    return {
        "analysis": analysis,
        "status": RunStatus.completed.value,
        "current_node": "analyze_results",
    }


# --------------------------------------------------------------------------- #
# 11. generate_report
# --------------------------------------------------------------------------- #
def generate_report(state: EvaluationState) -> dict:
    """生成 Markdown 报告并落盘。不改写原始结果。"""
    config = RunConfig(**state["config"])
    metrics = Metrics(**state["metrics"])
    results = [ModelResult(**r) for r in state["model_results"]]
    scores = [ScoreResult(**s) for s in state.get("score_results", [])]
    cases = [TestCase(**c) for c in state["cases"]]
    analysis = state.get("analysis")

    md = generate_markdown_report(
        state["run_id"], config, metrics, results, scores, cases, analysis
    )
    repo = RunRepository(state["run_id"])
    path = repo.save_report(md)
    repo.save_meta(
        {
            "run_id": state["run_id"],
            "run_name": config.run_name,
            "status": RunStatus.completed.value,
            "report_path": str(path),
            "total_cost_usd": metrics.total_cost_usd,
            "sample_count": metrics.sample_count,
        }
    )
    return {
        "report_path": str(path),
        "status": RunStatus.completed.value,
        "current_node": "generate_report",
    }


# --------------------------------------------------------------------------- #
# 12. generate_failure_report（文档第 4 节：无可评分候选 / 校验失败）
# --------------------------------------------------------------------------- #
def generate_failure_report(state: EvaluationState) -> dict:
    """无可评分候选、校验失败或用户拒绝时生成失败报告。"""
    repo = RunRepository(state["run_id"])
    errors = state.get("errors") or []
    decision = state.get("approval_decision")
    lines = [
        "# 评测失败报告",
        "",
        f"- **run_id**: `{state.get('run_id')}`",
        f"- **status**: {state.get('status')}",
        "",
        "## 错误",
        "",
    ]
    if errors:
        for e in errors:
            lines.append(f"- [{e.get('node')}] {e.get('message')}")
    elif decision == "rejected":
        lines.append("- 用户拒绝执行评测（human_approval 返回 rejected）。")
    else:
        lines.append("- 无可评分候选（所有模型调用失败或无成功结果）。")
    lines += ["", "本次评测未产生有效结果，请检查配置、数据集与模型可用性后重试。", ""]
    content = "\n".join(lines)
    path = repo.save_report(content)
    repo.save_meta({"run_id": state.get("run_id"), "status": RunStatus.failed.value, "report_path": str(path)})
    return {
        "report_path": str(path),
        "status": RunStatus.failed.value,
        "current_node": "generate_failure_report",
    }
