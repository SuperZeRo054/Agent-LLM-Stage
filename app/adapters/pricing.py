"""定价与成本计算（阶段二：真实计费）。

pricing 为每 1M token 的美元价格：{"input": x, "output": y}。
未配置 pricing 时不计成本（返回 0），但 token 照常记录——不臆造价格。
"""

from __future__ import annotations


def compute_cost(
    prompt_tokens: int, completion_tokens: int, pricing: dict[str, float] | None
) -> float:
    """按 token 数与定价计算成本（美元）。pricing 为 None 时返回 0。"""
    if not pricing:
        return 0.0
    input_per_m = float(pricing.get("input", 0.0))
    output_per_m = float(pricing.get("output", 0.0))
    cost = (prompt_tokens * input_per_m + completion_tokens * output_per_m) / 1_000_000.0
    return round(cost, 6)
