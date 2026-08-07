"""ModelAdapter 协议与工厂。

被测模型与 Judge 都通过该协议调用。executor 只依赖协议，不感知具体供应商。
阶段二：invoke 增加 json_mode（Judge/Planner 结构化输出用）；get_adapter 接受
LLMEndpointConfig（被测模型 / Judge / Planner 通用）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.domain.schemas import LLMEndpointConfig

logger = logging.getLogger(__name__)


@dataclass
class RawModelOutput:
    """单次模型调用的原始输出（供 executor 包装为 ModelResult）。"""

    answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: str = ""


@runtime_checkable
class ModelAdapter(Protocol):
    """统一的模型调用协议。"""

    provider: str
    model: str

    async def invoke(
        self, prompt: str, context: list[str], *, json_mode: bool = False
    ) -> RawModelOutput:
        """调用模型，返回原始输出。

        json_mode=True 时，实现方应让模型返回严格 JSON（如 OpenAI response_format），
        供 Judge/Planner 解析。实现方负责重试与超时（executor 再包一层限流）。
        """
        ...


# --------------------------------------------------------------------------- #
# 工厂注册表
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, type[BaseAdapter]] = {}

# provider -> 默认密钥环境变量名（api_key_env 未显式指定时使用）
_DEFAULT_KEY_ENV = {
    "provider_a": "PROVIDER_A_API_KEY",
    "provider_b": "PROVIDER_B_API_KEY",
}


def register_adapter(provider: str):
    """装饰器：注册一个 Adapter 实现类。"""

    def _wrap(cls: type[BaseAdapter]) -> type[BaseAdapter]:
        _REGISTRY[provider] = cls
        return cls

    return _wrap


def get_adapter(config: LLMEndpointConfig) -> BaseAdapter:
    """根据 LLMEndpointConfig.provider 取得 Adapter 实例。未注册则报错。"""
    cls = _REGISTRY.get(config.provider)
    if cls is None:
        raise ValueError(
            f"未注册的 provider: {config.provider}（已注册: {list(_REGISTRY)}）"
        )
    return cls(config)


def registered_providers() -> list[str]:
    """返回已注册的 provider 名称（供 validate_plan 校验白名单）。"""
    return list(_REGISTRY)


class BaseAdapter:
    """Adapter 基类：持有 LLMEndpointConfig，供子类实现 invoke。"""

    def __init__(self, config: LLMEndpointConfig) -> None:
        self.config = config
        self.provider = config.provider
        self.model = config.model

    def resolve_api_key(self) -> str:
        """解析 API Key：优先 config.api_key_env，否则按 provider 推导。缺则抛错。"""
        env = self.config.api_key_env or _DEFAULT_KEY_ENV.get(self.provider)
        if not env:
            raise RuntimeError(
                f"{self.provider} 未配置 api_key_env，且 provider 无默认密钥推导"
            )
        key = os.environ.get(env, "")
        if not key:
            raise RuntimeError(f"{self.provider} 未配置环境变量 {env}")
        return key

    async def invoke(
        self, prompt: str, context: list[str], *, json_mode: bool = False
    ) -> RawModelOutput:  # pragma: no cover
        raise NotImplementedError


# 触发各 provider 模块注册（import 即注册）
def _load_builtin_adapters() -> None:
    # 延迟导入，避免未安装 phase2 依赖时 import 失败
    from app.adapters import fake  # noqa: F401

    try:
        from app.adapters import provider_a, provider_b  # noqa: F401
    except Exception as e:  # phase2 依赖未安装时跳过，但在非测试环境给出 warning
        if os.environ.get("PYTEST_CURRENT_TEST"):
            logger.debug("phase2 adapters 未加载: %s", e)
        else:
            logger.warning("phase2 adapters 加载失败（provider_a/provider_b 不可用）: %s", e)


_load_builtin_adapters()
