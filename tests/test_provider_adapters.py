"""真实 Adapter 单测：mock SDK/HTTP，验证 token 解析、成本、json_mode、异常处理。"""

import pytest

from app.adapters.pricing import compute_cost
from app.domain.schemas import ModelConfig


# --------------------------------------------------------------------------- #
# 定价
# --------------------------------------------------------------------------- #
def test_compute_cost_with_pricing():
    assert compute_cost(1000, 500, {"input": 0.15, "output": 0.6}) == round(
        (1000 * 0.15 + 500 * 0.6) / 1_000_000, 6
    )


def test_compute_cost_no_pricing():
    assert compute_cost(1000, 500, None) == 0.0


# --------------------------------------------------------------------------- #
# Provider A（langchain-openai SDK）
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, content, ptok, ctok):
        self.content = content
        self.usage_metadata = {"input_tokens": ptok, "output_tokens": ctok, "total_tokens": ptok + ctok}


class _FakeChat:
    """替身 langchain_openai.ChatOpenAI。"""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeChat.last_kwargs = kwargs

    async def ainvoke(self, messages):
        return _FakeResp("hello world", 12, 34)


@pytest.mark.asyncio
async def test_provider_a_basic(monkeypatch):
    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _FakeChat)
    monkeypatch.setenv("PROVIDER_A_API_KEY", "sk-test")

    from app.adapters.provider_a import ProviderAAdapter

    cfg = ModelConfig(
        provider="provider_a", model="gpt-4o-mini",
        base_url="https://api.openai.com/v1", api_key_env="PROVIDER_A_API_KEY",
        pricing={"input": 0.15, "output": 0.6},
    )
    out = await ProviderAAdapter(cfg).invoke("ping", [], json_mode=True)
    assert out.answer == "hello world"
    assert out.prompt_tokens == 12 and out.completion_tokens == 34
    assert out.total_tokens == 46
    assert out.cost_usd == compute_cost(12, 34, cfg.pricing)
    assert _FakeChat.last_kwargs.get("response_format") == {"type": "json_object"}
    assert _FakeChat.last_kwargs.get("base_url") == "https://api.openai.com/v1"


@pytest.mark.asyncio
async def test_provider_a_error_wrapped(monkeypatch):
    class _BoomChat(_FakeChat):
        async def ainvoke(self, messages):
            raise RuntimeError("upstream down")

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _BoomChat)
    monkeypatch.setenv("PROVIDER_A_API_KEY", "sk-test")
    from app.adapters.provider_a import ProviderAAdapter

    cfg = ModelConfig(provider="provider_a", model="m", api_key_env="PROVIDER_A_API_KEY")
    with pytest.raises(RuntimeError, match="provider_a 调用失败"):
        await ProviderAAdapter(cfg).invoke("p", [])


@pytest.mark.asyncio
async def test_provider_a_missing_key(monkeypatch):
    monkeypatch.delenv("PROVIDER_A_API_KEY", raising=False)
    from app.adapters.provider_a import ProviderAAdapter

    cfg = ModelConfig(provider="provider_a", model="m", api_key_env="PROVIDER_A_API_KEY")
    with pytest.raises(RuntimeError, match="未配置环境变量"):
        await ProviderAAdapter(cfg).invoke("p", [])


# --------------------------------------------------------------------------- #
# Provider B（httpx 裸调）
# --------------------------------------------------------------------------- #
class _FakeHttpResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _FakeHttpClient:
    last: dict = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        _FakeHttpClient.last = {"url": url, "json": json, "headers": headers}
        return _FakeHttpResp(
            {"choices": [{"message": {"content": "ans"}}], "usage": {"prompt_tokens": 8, "completion_tokens": 9}}
        )


@pytest.mark.asyncio
async def test_provider_b_basic(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpClient)
    monkeypatch.setenv("PROVIDER_B_API_KEY", "sk-test")
    from app.adapters.provider_b import ProviderBAdapter

    cfg = ModelConfig(
        provider="provider_b", model="deepseek-chat",
        base_url="https://api.deepseek.com/v1", api_key_env="PROVIDER_B_API_KEY",
        pricing={"input": 0.27, "output": 1.1},
    )
    out = await ProviderBAdapter(cfg).invoke("q", ["ctx"], json_mode=True)
    assert out.answer == "ans"
    assert out.prompt_tokens == 8 and out.completion_tokens == 9
    assert out.cost_usd == compute_cost(8, 9, cfg.pricing)
    assert _FakeHttpClient.last["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert _FakeHttpClient.last["json"]["response_format"] == {"type": "json_object"}
    # context 进 system message
    assert _FakeHttpClient.last["json"]["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_provider_b_http_error(monkeypatch):
    import httpx

    class _ErrClient(_FakeHttpClient):
        async def post(self, url, json=None, headers=None):
            raise httpx.ConnectError("conn refused")

    monkeypatch.setattr(httpx, "AsyncClient", _ErrClient)
    monkeypatch.setenv("PROVIDER_B_API_KEY", "sk-test")
    from app.adapters.provider_b import ProviderBAdapter

    cfg = ModelConfig(provider="provider_b", model="m", base_url="https://x", api_key_env="PROVIDER_B_API_KEY")
    with pytest.raises(RuntimeError, match="provider_b 调用失败"):
        await ProviderBAdapter(cfg).invoke("p", [])
