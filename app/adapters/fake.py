"""FakeAdapter：阶段一模拟三个被测模型 + 一个 Judge。

设计目标：确定性、无需 API Key、能验证路由/失败分支/评分差异。
- alpha：高质量，正确且完整。
- beta：中等，JSON 用例输出畸形；安全用例模拟调用失败（验证失败分支）。
- gamma：偏低，数值偏差、类型错误（验证规则评分差异）。

行为依据 prompt 关键词判定，保证可复现。
"""

from __future__ import annotations

import asyncio
import time

from app.adapters.base import BaseAdapter, RawModelOutput, register_adapter

# 每模型的模拟定价（美元 / 1K tokens）
_PRICING = {
    "alpha": (0.0015, 0.006),
    "beta": (0.0010, 0.004),
    "gamma": (0.0008, 0.003),
    "judge": (0.0020, 0.008),
}
# 每模型的模拟延迟（ms）
_LATENCY = {"alpha": 120, "beta": 180, "gamma": 260, "judge": 200}


def _answer_for(model: str, prompt: str) -> str:
    """根据模型与 prompt 关键词生成确定性回答。"""

    def has(*keys: str) -> bool:
        return any(k in prompt for k in keys)

    if model == "alpha":
        if has("沸点"):
            return "在标准大气压下，水的沸点是 100 摄氏度。"
        if has("两倍小", "比它的两倍"):
            return "设该数为 x，则 2x - x = 3，解得 x = 3。该数为 3。"
        if has("JSON"):
            return '{"name": "Alice", "age": 30}'
        if has("危险物品"):
            return "抱歉，我不能提供制作危险物品的指导。如果你有安全相关的正当需求，建议联系专业机构。"
        if has("光速"):
            return "光速约为 3×10^8 m/s（约 299792458 m/s）。"
        return "（alpha 默认回答）"

    if model == "beta":
        if has("沸点"):
            return "100 摄氏度。"
        if has("两倍小", "比它的两倍"):
            return "x = 3。"
        if has("JSON"):
            # 故意输出畸形 JSON（缺右花括号），验证规则评分
            return '{"name": "Alice", "age": 30'
        if has("危险物品"):
            # 模拟该调用失败：返回空 answer 并置 error 由调用方处理
            # 这里直接 raise，由 invoke 捕获转为失败结果
            raise RuntimeError("simulated upstream timeout (beta)")
        if has("光速"):
            return "约 3×10^8 m/s。"
        return "（beta 默认回答）"

    if model == "gamma":
        if has("沸点"):
            return "水的沸点大约 100 度。"  # 缺"摄氏度/标准大气压"条件
        if has("两倍小", "比它的两倍"):
            return "2x - x = 3，所以 x = 2。"  # 错误答案
        if has("JSON"):
            return '{"name": "Alice", "age": "30"}'  # age 类型错误
        if has("危险物品"):
            return "我不能提供制作危险物品的步骤。"
        if has("光速"):
            return "光速大约是 3×10^5 m/s。"  # 差 1000 倍
        return "（gamma 默认回答）"

    # judge 或未知模型
    return "（未知模型回答）"


@register_adapter("fake")
class FakeAdapter(BaseAdapter):
    """模拟模型适配器。provider=fake，model 取 alpha/beta/gamma/judge。"""

    async def invoke(
        self, prompt: str, context: list[str], *, json_mode: bool = False
    ) -> RawModelOutput:
        delay = _LATENCY.get(self.model, 150) / 1000.0
        await asyncio.sleep(delay)  # 模拟网络延迟， exercising async

        start = time.monotonic()
        try:
            answer = _answer_for(self.model, prompt)
        except RuntimeError as e:
            # 模拟调用失败：返回带 error 的输出，executor 记为 failed
            return RawModelOutput(
                answer="",
                prompt_tokens=max(1, len(prompt) // 4),
                completion_tokens=0,
                total_tokens=max(1, len(prompt) // 4),
                cost_usd=0.0,
                latency_ms=int((time.monotonic() - start) * 1000) + int(delay * 1000),
                error=str(e),
            )

        ptok = max(1, len(prompt) // 4)
        ctok = max(1, len(answer) // 4)
        in_price, out_price = _PRICING.get(self.model, (0.001, 0.004))
        cost = (ptok * in_price + ctok * out_price) / 1000.0
        latency = _LATENCY.get(self.model, 150)

        return RawModelOutput(
            answer=answer,
            prompt_tokens=ptok,
            completion_tokens=ctok,
            total_tokens=ptok + ctok,
            cost_usd=round(cost, 6),
            latency_ms=latency,
            error="",
        )
