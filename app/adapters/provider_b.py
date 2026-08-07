"""Provider B Adapter：OpenAI 兼容接口（httpx 裸调）。

阶段二真实接入，展示不依赖 SDK 的轻量直连方式（体现 ModelAdapter 接口通用性）。
依赖 httpx（[phase2] extra）。POST {base_url}/chat/completions，解析标准 OpenAI 响应。
"""

from __future__ import annotations

import time

from app.adapters.base import BaseAdapter, RawModelOutput, register_adapter
from app.adapters.pricing import compute_cost


@register_adapter("provider_b")
class ProviderBAdapter(BaseAdapter):
    """Provider B（httpx 直连 OpenAI 兼容 /chat/completions）。"""

    async def invoke(
        self, prompt: str, context: list[str], *, json_mode: bool = False
    ) -> RawModelOutput:
        api_key = self.resolve_api_key()
        try:
            import httpx  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "未安装 phase2 依赖（httpx）；pip install -e .[phase2]"
            ) from e

        base = (self.config.base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base}/chat/completions"

        messages: list[dict] = []
        if context:
            messages.append({"role": "system", "content": "\n\n".join(context)})
        messages.append({"role": "user", "content": prompt})

        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                resp = await client.post(url, json=body, headers=headers)
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"provider_b HTTP {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"provider_b 调用失败: {type(e).__name__}: {e}") from e
        latency = int((time.monotonic() - start) * 1000)

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {}) or {}
        ptok = int(usage.get("prompt_tokens", 0))
        ctok = int(usage.get("completion_tokens", 0))

        return RawModelOutput(
            answer=content if isinstance(content, str) else str(content),
            prompt_tokens=ptok,
            completion_tokens=ctok,
            total_tokens=ptok + ctok,
            cost_usd=compute_cost(ptok, ctok, self.config.pricing),
            latency_ms=latency,
        )
