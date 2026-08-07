"""Provider A Adapter：OpenAI 兼容接口（langchain-openai SDK）。

阶段二真实接入。依赖 langchain-openai（pyproject 的 [phase2] extra）。
缺 API Key 或依赖时抛清晰错误。支持 json_mode（response_format=json_object），
供 Judge/Planner 结构化输出。
"""

from __future__ import annotations

import time

from app.adapters.base import BaseAdapter, RawModelOutput, register_adapter
from app.adapters.pricing import compute_cost


@register_adapter("provider_a")
class ProviderAAdapter(BaseAdapter):
    """OpenAI 兼容的 Provider A（langchain-openai SDK）。"""

    async def invoke(
        self, prompt: str, context: list[str], *, json_mode: bool = False
    ) -> RawModelOutput:
        api_key = self.resolve_api_key()
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "未安装 phase2 依赖（langchain-openai）；pip install -e .[phase2]"
            ) from e

        kwargs: dict = {
            "model": self.model,
            "api_key": api_key,
            "temperature": self.config.temperature,
            "timeout": self.config.timeout,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        if json_mode:
            # OpenAI json_object 模式要求 prompt 含 "json" 字样（由调用方保证）
            kwargs["response_format"] = {"type": "json_object"}

        llm = ChatOpenAI(**kwargs)
        messages: list[dict] = []
        if context:
            messages.append({"role": "system", "content": "\n\n".join(context)})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        try:
            resp = await llm.ainvoke(messages)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"provider_a 调用失败: {type(e).__name__}: {e}") from e
        latency = int((time.monotonic() - start) * 1000)

        usage = getattr(resp, "usage_metadata", None) or {}
        ptok = int(usage.get("input_tokens", 0))
        ctok = int(usage.get("output_tokens", 0))
        content = resp.content
        answer = content if isinstance(content, str) else str(content)

        return RawModelOutput(
            answer=answer,
            prompt_tokens=ptok,
            completion_tokens=ctok,
            total_tokens=ptok + ctok,
            cost_usd=compute_cost(ptok, ctok, self.config.pricing),
            latency_ms=latency,
        )
