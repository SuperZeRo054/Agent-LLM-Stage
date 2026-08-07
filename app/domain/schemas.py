"""Pydantic 数据契约：输入计划、测试样本、模型结果、Judge 结果、指标。

对应文档第 8.3 节"最小数据结构"。所有模型均可 JSON 序列化，以便写入 LangGraph checkpoint。
时间字段使用 Unix epoch（float）而非 datetime 对象，确保 checkpoint 可序列化。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 评分量表与测试样本
# --------------------------------------------------------------------------- #
class Rubric(BaseModel):
    """单条测试用例的评分量表。"""

    correctness: str = "待定义"
    completeness: str = "待定义"
    safety: str = "待定义"


class TestCase(BaseModel):
    """测试集最小结构（文档 8.3）。"""

    id: str
    category: str = "未分类"
    input: str
    context: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    rubric: Rubric = Field(default_factory=Rubric)


# --------------------------------------------------------------------------- #
# 配置与计划
# --------------------------------------------------------------------------- #
class LLMEndpointConfig(BaseModel):
    """LLM 端点连接配置（被测模型 / Judge / Planner 共用）。

    阶段二新增字段均为可选，向后兼容阶段一的 fake 配置。
    - base_url：OpenAI 兼容端点；None 时用 SDK 默认。
    - api_key_env：读取密钥的环境变量名；None 时按 provider 推导
      （provider_a -> PROVIDER_A_API_KEY，judge -> JUDGE_API_KEY，planner -> PLANNER_API_KEY）。
    - pricing：{"input": x, "output": y} 每 1M token USD；None 则不计成本只记 token。
    """

    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int | None = None
    timeout: float = Field(default=60.0, gt=0)
    pricing: dict[str, float] | None = None


class ModelConfig(LLMEndpointConfig):
    """被测模型配置。"""

    concurrency: int = Field(default=3, ge=1, le=64)


class JudgeConfig(LLMEndpointConfig):
    """Judge 配置。阶段一默认 fake。"""

    provider: str = "fake"
    model: str = "judge"
    repeats: int = Field(default=2, ge=1, le=10)


class PlannerConfig(LLMEndpointConfig):
    """Planner 配置（阶段二：自然语言请求 -> 结构化 EvaluationPlan）。"""


class RunConfig(BaseModel):
    """用户提交的 YAML 配置（文档 5.1）。"""

    run_name: str
    # dataset/models 可空：自然语言 Planner 场景由 LLM 产出，YAML 场景直接提供
    dataset: str = ""
    models: list[ModelConfig] = Field(default_factory=list)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    planner: PlannerConfig | None = None
    budget_limit_usd: float = Field(default=10.0, ge=0)
    auto_approve: bool = False


class EvaluationPlan(BaseModel):
    """Planner 节点输出的结构化评测计划。

    缺少关键信息时 requires_clarification=True，并给出 missing_fields，不得猜测填充。
    models 非空校验由 validate_plan 完成（Planner 产出 requires_clarification 时允许空）。
    """

    run_name: str
    dataset: str = ""
    models: list[ModelConfig] = Field(default_factory=list)
    judge: JudgeConfig
    budget_limit_usd: float
    requires_clarification: bool = False
    missing_fields: list[str] = Field(default_factory=list)
    estimated_calls: int = 0  # case 数 × 模型数，由 validate_plan 填充


# --------------------------------------------------------------------------- #
# 执行任务与结果
# --------------------------------------------------------------------------- #
class TaskStatus(StrEnum):
    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"
    skipped = "skipped"


class TaskItem(BaseModel):
    """case × model 任务矩阵中的一项。request_key 用于幂等去重。"""

    request_key: str  # run_id + case_id + model_id
    run_id: str
    case_id: str
    model_id: str
    provider: str
    status: TaskStatus = TaskStatus.pending


class ModelResult(BaseModel):
    """单条 case × model 的执行结果（文档 8.3）。"""

    request_key: str
    run_id: str
    case_id: str
    model_id: str
    provider: str
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str = "v1"
    answer: str = ""
    started_at: float = 0.0  # unix epoch
    ended_at: float = 0.0
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    status: TaskStatus = TaskStatus.success
    error: str = ""


class RiskFlag(StrEnum):
    none = "none"
    self_recognition = "self_recognition"
    hallucination = "hallucination"
    unsafe_content = "unsafe_content"
    other = "other"


class JudgeResult(BaseModel):
    """单次 Judge 盲评结果。不包含被测模型身份，仅含匿名候选 ID。"""

    anonymous_id: str
    request_key: str
    scores: dict[str, float]  # correctness / completeness / safety (0-10)
    evidence: str = ""
    risk_flag: RiskFlag = RiskFlag.none
    raw_rationale: str = ""
    prompt_version: str = "judge_v1"
    repeat_index: int = 0
    # 阶段二：真实 Judge 调用的 token 与成本（fake judge 保持默认 0）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class ScoreResult(BaseModel):
    """一条 case × model 的最终评分：规则分 + Judge 均值/方差。"""

    request_key: str
    run_id: str
    case_id: str
    model_id: str
    anonymous_id: str
    rule_scores: dict[str, float] = Field(default_factory=dict)
    judge_scores: list[JudgeResult] = Field(default_factory=list)
    judge_mean: float = 0.0
    judge_variance: float = 0.0
    quality_score: float = 0.0  # 规则与 Judge 加权后的综合分
    risk_flag: RiskFlag = RiskFlag.none


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
class ModelMetrics(BaseModel):
    """单个被测模型的聚合指标。"""

    model_id: str
    provider: str
    count: int = 0
    success_count: int = 0
    failure_rate: float = 0.0
    quality_score: float = 0.0
    quality_variance: float = 0.0  # 评分方差（跨样本）
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0


class Anomaly(BaseModel):
    """异常标记：评分方差过大、失败率过高、成本超预算等。"""

    type: str
    model_id: str | None = None
    detail: str = ""


class Metrics(BaseModel):
    """一次 run 的聚合指标。"""

    run_id: str
    models: list[ModelMetrics] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    budget_limit_usd: float = 0.0
    budget_exceeded: bool = False
    sample_count: int = 0
    anomalies: list[Anomaly] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 错误与状态
# --------------------------------------------------------------------------- #
class ErrorRecord(BaseModel):
    """节点执行中的错误记录。"""

    node: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


class RunStatus(StrEnum):
    planning = "planning"
    awaiting_approval = "awaiting_approval"
    executing = "executing"
    scoring = "scoring"
    aggregating = "aggregating"
    analyzing = "analyzing"
    completed = "completed"
    failed = "failed"
    interrupted = "interrupted"
