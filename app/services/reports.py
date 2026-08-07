"""报告生成（文档第 11 节）。

MVP 输出 Markdown；阶段三用 Jinja2 + Plotly 生成 HTML 图表。
报告必须给出：样本数、运行配置、总成本、失败率、评分方差和低分样本回放。
小样本报告只描述“在本测试集上的表现”，不宣称通用能力领先。
"""

from __future__ import annotations

from app.domain.schemas import (
    Metrics,
    ModelResult,
    RunConfig,
    ScoreResult,
    TestCase,
)

LOW_SCORE_THRESHOLD = 6.0


def _fmt_cost(v: float) -> str:
    return f"${v:.4f}"


def generate_markdown_report(
    run_id: str,
    config: RunConfig,
    metrics: Metrics,
    results: list[ModelResult],
    scores: list[ScoreResult],
    cases: list[TestCase],
    analysis: dict | None = None,
) -> str:
    cases_by_id = {c.id: c for c in cases}
    lines: list[str] = []

    lines.append(f"# 评测报告：{config.run_name}")
    lines.append("")
    lines.append(f"- **run_id**: `{run_id}`")
    lines.append(f"- **数据集**: `{config.dataset}`")
    lines.append(f"- **样本数**: {metrics.sample_count}")
    lines.append(f"- **总成本**: {_fmt_cost(metrics.total_cost_usd)} / 预算 {_fmt_cost(metrics.budget_limit_usd)}")
    lines.append(f"- **预算超限**: {'是' if metrics.budget_exceeded else '否'}")
    lines.append("")

    # 运行配置
    lines.append("## 运行配置")
    lines.append("")
    lines.append("| 被测模型 | provider | 并发 |")
    lines.append("| --- | --- | --- |")
    for m in config.models:
        lines.append(f"| {m.model} | {m.provider} | {m.concurrency} |")
    lines.append("")
    lines.append(
        f"Judge: `{config.judge.provider}:{config.judge.model}`，"
        f"重复评分 {config.judge.repeats} 次。"
    )
    lines.append("")

    # 指标汇总
    lines.append("## 指标汇总")
    lines.append("")
    lines.append("| 模型 | 质量分 | 评分方差 | P50 延迟(ms) | P95 延迟(ms) | token | 成本 | 失败率 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in metrics.models:
        lines.append(
            f"| {m.model_id} | {m.quality_score:.2f} | {m.quality_variance:.2f} | "
            f"{m.p50_latency_ms:.0f} | {m.p95_latency_ms:.0f} | {m.total_tokens} | "
            f"{_fmt_cost(m.total_cost_usd)} | {m.failure_rate:.0%} |"
        )
    lines.append("")

    # 异常
    if metrics.anomalies:
        lines.append("## 异常标记")
        lines.append("")
        for a in metrics.anomalies:
            target = f"（{a.model_id}）" if a.model_id else ""
            lines.append(f"- **{a.type}**{target}: {a.detail}")
        lines.append("")
    else:
        lines.append("## 异常标记")
        lines.append("")
        lines.append("无。")
        lines.append("")

    # 低分样本回放
    low = [s for s in scores if s.quality_score < LOW_SCORE_THRESHOLD]
    lines.append(f"## 低分样本回放（质量分 < {LOW_SCORE_THRESHOLD}）")
    lines.append("")
    if not low:
        lines.append("无低分样本。")
        lines.append("")
    else:
        for s in low:
            case = cases_by_id.get(s.case_id)
            r = next((r for r in results if r.request_key == s.request_key), None)
            lines.append(f"### {s.case_id} / 模型 `{s.model_id}`（质量分 {s.quality_score:.2f}）")
            lines.append("")
            if case:
                lines.append(f"- **输入**: {case.input}")
                lines.append(f"- **参考答案**: {case.reference_answer}")
            if r:
                status = "成功" if not r.error else f"失败（{r.error}）"
                lines.append(f"- **状态**: {status}")
                lines.append(f"- **回答**: {r.answer or '（空）'}")
            lines.append(f"- **规则评分**: { {k: round(v,1) for k,v in s.rule_scores.items()} }")
            lines.append(f"- **Judge 均值(正确性)**: {s.judge_mean:.2f}，方差: {s.judge_variance:.2f}")
            lines.append(f"- **风险标记**: {s.risk_flag.value}")
            lines.append("")

    # 分析（Analyst 节点输出）
    if analysis:
        lines.append("## 结果分析")
        lines.append("")
        for diff in analysis.get("observed_differences", []):
            lines.append(f"- 观察到的差异: {diff}")
        for ev in analysis.get("supporting_evidence", []):
            lines.append(f"- 支持证据: {ev}")
        for lim in analysis.get("limitations", []):
            lines.append(f"- 局限: {lim}")
        lines.append("")

    # 局限声明
    lines.append("## 局限声明")
    lines.append("")
    lines.append(
        "本报告仅描述被测模型**在本测试集上的表现**，不构成通用能力领先的结论。"
        "Judge 评分存在已知偏差，模型版本变化会影响结果。"
    )
    lines.append("")

    return "\n".join(lines)
