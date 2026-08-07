"""数据契约单测：LLMEndpointConfig 继承、新字段默认值、向后兼容。"""

from app.domain.schemas import (
    EvaluationPlan,
    JudgeConfig,
    JudgeResult,
    ModelConfig,
    PlannerConfig,
    RunConfig,
)


def test_model_config_endpoint_fields():
    m = ModelConfig(
        provider="provider_a", model="m", base_url="https://x",
        pricing={"input": 1, "output": 2},
    )
    assert m.temperature == 0.0
    assert m.timeout == 60.0
    assert m.api_key_env is None
    assert m.concurrency == 3


def test_judge_config_defaults_backcompat():
    j = JudgeConfig()
    assert j.provider == "fake"
    assert j.model == "judge"
    assert j.repeats == 2
    assert j.base_url is None


def test_planner_config():
    p = PlannerConfig(provider="provider_a", model="m", base_url="https://x")
    assert p.provider == "provider_a"
    assert p.max_tokens is None


def test_run_config_optional_fields():
    # Planner 场景：models/dataset 可空
    rc = RunConfig(run_name="r", budget_limit_usd=1.0)
    assert rc.models == []
    assert rc.dataset == ""
    assert rc.planner is None
    assert rc.judge.provider == "fake"


def test_run_config_with_planner():
    rc = RunConfig(
        run_name="r", budget_limit_usd=1.0,
        planner=PlannerConfig(provider="provider_a", model="m", api_key_env="PLANNER_API_KEY"),
    )
    assert rc.planner is not None
    assert rc.planner.api_key_env == "PLANNER_API_KEY"


def test_eval_plan_empty_models_allowed():
    # requires_clarification 时允许空 models（非空校验由 validate_plan 完成）
    p = EvaluationPlan(
        run_name="r", judge=JudgeConfig(), budget_limit_usd=1.0,
        requires_clarification=True, missing_fields=["models"],
    )
    assert p.models == []


def test_judge_result_cost_fields_default():
    jr = JudgeResult(anonymous_id="a", request_key="rk", scores={"correctness": 8.0})
    assert jr.prompt_tokens == 0
    assert jr.completion_tokens == 0
    assert jr.cost_usd == 0.0
