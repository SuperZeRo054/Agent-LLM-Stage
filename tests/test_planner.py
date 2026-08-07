"""Planner 单测：mock adapter，验证自然语言->EvaluationPlan、澄清、降级、未配 planner。"""

import pytest

from app.adapters.base import RawModelOutput
from app.domain.schemas import PlannerConfig, RunConfig
from app.graph.nodes import _parse_plan_json, _plan_with_llm, plan_evaluation


def _cfg() -> RunConfig:
    return RunConfig(
        run_name="r",
        budget_limit_usd=1.0,
        planner=PlannerConfig(provider="provider_a", model="m", api_key_env="PLANNER_API_KEY"),
    )


def test_parse_plan_json_valid():
    raw = (
        '{"run_name":"r","dataset":"datasets/dataset_v1.jsonl",'
        '"models":[{"provider":"fake","model":"alpha"}],'
        '"judge":{"provider":"fake","model":"judge","repeats":2},'
        '"budget_limit_usd":1.0,"requires_clarification":false}'
    )
    plan = _parse_plan_json(raw, _cfg())
    assert plan.requires_clarification is False
    assert plan.models[0].model == "alpha"
    assert plan.dataset == "datasets/dataset_v1.jsonl"


def test_parse_plan_json_clarification():
    raw = '{"requires_clarification":true,"missing_fields":["models"],"run_name":"r"}'
    plan = _parse_plan_json(raw, _cfg())
    assert plan.requires_clarification is True
    assert "models" in plan.missing_fields


def test_parse_plan_json_code_fence():
    raw = (
        '```json\n{"run_name":"r","models":[{"provider":"fake","model":"alpha"}],'
        '"judge":{"provider":"fake","model":"judge"},"budget_limit_usd":1.0}\n```'
    )
    plan = _parse_plan_json(raw, _cfg())
    assert plan.models[0].model == "alpha"


def test_parse_plan_json_no_models_raises():
    raw = '{"run_name":"r","models":[],"judge":{"provider":"fake","model":"judge"},"budget_limit_usd":1.0}'
    with pytest.raises(ValueError, match="未产出 models"):
        _parse_plan_json(raw, _cfg())


class _FakeAdapter:
    def __init__(self, answer, error=False):
        self._answer = answer
        self._error = error

    async def invoke(self, prompt, context, *, json_mode=False):
        if self._error:
            raise RuntimeError("boom")
        return RawModelOutput(answer=self._answer, prompt_tokens=5, completion_tokens=10)


@pytest.mark.asyncio
async def test_plan_with_llm_valid(monkeypatch):
    plan_json = (
        '{"run_name":"r","dataset":"datasets/dataset_v1.jsonl",'
        '"models":[{"provider":"fake","model":"alpha"}],'
        '"judge":{"provider":"fake","model":"judge","repeats":2},'
        '"budget_limit_usd":1.0,"requires_clarification":false}'
    )
    monkeypatch.setattr("app.graph.nodes.get_adapter", lambda cfg: _FakeAdapter(plan_json))
    plan = await _plan_with_llm("比较模型", _cfg())
    assert plan.requires_clarification is False
    assert plan.models[0].model == "alpha"


@pytest.mark.asyncio
async def test_plan_with_llm_clarification(monkeypatch):
    plan_json = '{"requires_clarification":true,"missing_fields":["budget"],"run_name":"r"}'
    monkeypatch.setattr("app.graph.nodes.get_adapter", lambda cfg: _FakeAdapter(plan_json))
    plan = await _plan_with_llm("比较模型", _cfg())
    assert plan.requires_clarification is True
    assert "budget" in plan.missing_fields


@pytest.mark.asyncio
async def test_plan_with_llm_degrades_on_failure(monkeypatch):
    # adapter 持续抛错 -> 降级为 requires_clarification
    monkeypatch.setattr("app.graph.nodes.get_adapter", lambda cfg: _FakeAdapter("", error=True))
    plan = await _plan_with_llm("比较模型", _cfg())
    assert plan.requires_clarification is True
    assert any("planner_failed" in f for f in plan.missing_fields)


@pytest.mark.asyncio
async def test_plan_evaluation_nl_calls_planner(monkeypatch):
    plan_json = (
        '{"run_name":"r","dataset":"datasets/dataset_v1.jsonl",'
        '"models":[{"provider":"fake","model":"alpha"}],'
        '"judge":{"provider":"fake","model":"judge","repeats":2},'
        '"budget_limit_usd":1.0,"requires_clarification":false}'
    )
    monkeypatch.setattr("app.graph.nodes.get_adapter", lambda cfg: _FakeAdapter(plan_json))
    state = {"config": _cfg().model_dump(mode="json"), "request": "比较两个模型"}
    result = await plan_evaluation(state)
    assert "plan" in result
    assert result["status"] != "failed"


@pytest.mark.asyncio
async def test_plan_evaluation_nl_without_planner():
    cfg = RunConfig(run_name="r", budget_limit_usd=1.0)  # 无 planner
    state = {"config": cfg.model_dump(mode="json"), "request": "比较模型"}
    result = await plan_evaluation(state)
    assert result["status"] == "failed"
    assert result["errors"]
