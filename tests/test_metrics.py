"""指标聚合单测（文档 8.2：纯函数、可单测）。"""

from app.domain.schemas import (
    ModelResult,
    ScoreResult,
    TaskStatus,
)
from app.services.metrics import calculate_metrics, percentile


def test_percentile_basic():
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0
    assert percentile([1, 2, 3, 4, 5], 0) == 1.0
    assert percentile([1, 2, 3, 4, 5], 100) == 5.0


def test_percentile_empty():
    assert percentile([], 50) == 0.0
    assert percentile([42], 95) == 42.0


def _result(case_id, model_id, status, latency, cost=0.001, tokens=10):
    return ModelResult(
        request_key=f"r:{case_id}:{model_id}",
        run_id="run-1",
        case_id=case_id,
        model_id=model_id,
        provider="fake",
        latency_ms=latency,
        total_tokens=tokens,
        cost_usd=cost,
        status=status,
        answer="ok" if status == TaskStatus.success else "",
    )


def _score(request_key, model_id, quality):
    return ScoreResult(
        request_key=request_key,
        run_id="run-1",
        case_id="c",
        model_id=model_id,
        anonymous_id="anon",
        quality_score=quality,
    )


def test_calculate_metrics_failure_rate_and_latency():
    results = [
        _result("c1", "alpha", TaskStatus.success, 100),
        _result("c2", "alpha", TaskStatus.success, 200),
        _result("c3", "alpha", TaskStatus.success, 300),
        _result("c4", "alpha", TaskStatus.success, 400),
        _result("c5", "alpha", TaskStatus.failed, 0),
    ]
    scores = [_score(r.request_key, "alpha", q) for r, q in zip(results, [9, 8, 7, 6, 0], strict=True)]
    m = calculate_metrics("run-1", results, scores, budget_limit_usd=10.0)
    alpha = m.models[0]
    assert alpha.count == 5
    assert alpha.success_count == 4
    assert alpha.failure_rate == 0.2
    # P50 of [100,200,300,400] = 250, P95 ~ near 400
    assert alpha.p50_latency_ms == 250.0
    assert alpha.p95_latency_ms >= 300
    assert alpha.quality_score == 6.0  # mean of [9,8,7,6,0]


def test_calculate_metrics_anomaly_high_failure():
    results = [_result(f"c{i}", "beta", TaskStatus.failed if i < 2 else TaskStatus.success, 100) for i in range(5)]
    scores = [_score(r.request_key, "beta", 5.0) for r in results]
    m = calculate_metrics("run-1", results, scores, budget_limit_usd=10.0)
    types = [a.type for a in m.anomalies]
    assert "high_failure_rate" in types  # 2/5 = 0.4 > 0.15


def test_calculate_metrics_budget_exceeded():
    results = [_result("c1", "alpha", TaskStatus.success, 100, cost=5.0)]
    scores = [_score(results[0].request_key, "alpha", 8.0)]
    m = calculate_metrics("run-1", results, scores, budget_limit_usd=1.0)
    assert m.budget_exceeded is True
    assert any(a.type == "budget_exceeded" for a in m.anomalies)
