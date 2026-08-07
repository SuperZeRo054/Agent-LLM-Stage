"""指标聚合（文档 8.2、第 11 节）。

纯函数、可单测：统计质量分、P50/P95 延迟、token 成本、失败率、评分方差与异常标记。
"""

from __future__ import annotations

from app.domain.schemas import (
    Anomaly,
    Metrics,
    ModelMetrics,
    ModelResult,
    ScoreResult,
    TaskStatus,
)

# 异常阈值
FAILURE_RATE_THRESHOLD = 0.15
SCORE_VARIANCE_THRESHOLD = 4.0  # 跨样本质量分方差


def percentile(values: list[float], p: float) -> float:
    """线性插值百分位。空列表返回 0。"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


def calculate_metrics(
    run_id: str,
    results: list[ModelResult],
    scores: list[ScoreResult],
    budget_limit_usd: float,
) -> Metrics:
    """聚合一次 run 的指标。"""
    scores_by_key = {s.request_key: s for s in scores}
    models: list[ModelMetrics] = []
    anomalies: list[Anomaly] = []

    # 按 model_id 分组
    model_ids: list[str] = []
    seen: set[str] = set()
    for r in results:
        key = f"{r.provider}:{r.model_id}"
        if key not in seen:
            seen.add(key)
            model_ids.append(r.model_id)

    total_cost = 0.0
    for mid in model_ids:
        rs = [r for r in results if r.model_id == mid]
        provider = rs[0].provider if rs else ""
        count = len(rs)
        success = sum(1 for r in rs if r.status == TaskStatus.success)
        failure_rate = round(1 - (success / count) if count else 0.0, 4)

        latencies = [float(r.latency_ms) for r in rs if r.status == TaskStatus.success]
        tokens = sum(r.total_tokens for r in rs)
        cost = round(sum(r.cost_usd for r in rs), 6)
        total_cost += cost

        # 质量分跨样本均值与方差
        qs = [scores_by_key[r.request_key].quality_score for r in rs if r.request_key in scores_by_key]
        quality = round(sum(qs) / len(qs), 4) if qs else 0.0
        qvar = round(sum((q - quality) ** 2 for q in qs) / len(qs), 4) if qs else 0.0

        models.append(
            ModelMetrics(
                model_id=mid,
                provider=provider,
                count=count,
                success_count=success,
                failure_rate=failure_rate,
                quality_score=quality,
                quality_variance=qvar,
                p50_latency_ms=round(percentile(latencies, 50), 2),
                p95_latency_ms=round(percentile(latencies, 95), 2),
                total_tokens=tokens,
                total_cost_usd=cost,
            )
        )

        # 异常标记
        if failure_rate > FAILURE_RATE_THRESHOLD:
            anomalies.append(
                Anomaly(type="high_failure_rate", model_id=mid, detail=f"失败率 {failure_rate:.0%}")
            )
        if qvar > SCORE_VARIANCE_THRESHOLD:
            anomalies.append(
                Anomaly(type="high_score_variance", model_id=mid, detail=f"质量分方差 {qvar:.2f}")
            )

    # Judge 调用成本（阶段二真实 Judge；fake judge 为 0）
    judge_cost = round(sum(jr.cost_usd for s in scores for jr in s.judge_scores), 6)
    total_cost += judge_cost

    budget_exceeded = total_cost > budget_limit_usd if budget_limit_usd > 0 else False
    if budget_exceeded:
        anomalies.append(
            Anomaly(type="budget_exceeded", detail=f"总成本 ${total_cost:.4f} > 预算 ${budget_limit_usd:.2f}")
        )

    sample_count = len({r.case_id for r in results})

    return Metrics(
        run_id=run_id,
        models=sorted(models, key=lambda m: m.model_id),
        total_cost_usd=round(total_cost, 6),
        budget_limit_usd=budget_limit_usd,
        budget_exceeded=budget_exceeded,
        sample_count=sample_count,
        anomalies=anomalies,
    )
