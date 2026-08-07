"""模型并发执行器（文档 9.3）。

- 生成 case × model 任务矩阵后，asyncio 并发执行。
- 每个 ModelConfig 独立 Semaphore（限流），指数退避重试。
- 单任务失败写入结果，不中断整个 run。
- request_key = run_id + case_id + model_id，配合 RunRepository 实现幂等：断点续跑时
  已完成任务从仓库加载、跳过调用，不重复扣费。
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.adapters.base import RawModelOutput, get_adapter
from app.domain.schemas import (
    ModelConfig,
    ModelResult,
    TaskItem,
    TaskStatus,
    TestCase,
)
from app.repositories.runs import RunRepository

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 0.2  # 秒


def build_prompt(case: TestCase) -> tuple[str, list[str]]:
    """从 TestCase 组装 (prompt, context)。"""
    parts: list[str] = []
    if case.context:
        parts.extend(case.context)
    parts.append(case.input)
    return "\n\n".join(parts), list(case.context)


async def _call_with_retry(
    adapter, prompt: str, context: list[str], request_key: str
) -> RawModelOutput:
    """带指数退避的重试。最终失败返回带 error 的输出。"""
    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            out = await adapter.invoke(prompt, context)
            if out.error:
                # adapter 内部已判定失败（如 fake 的 beta 安全用例）
                last_error = out.error
                # 仍可重试，但确定性失败会耗尽次数
            else:
                return out
        except Exception as e:  # noqa: BLE001
            last_error = f"{type(e).__name__}: {e}"
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(BACKOFF_BASE * (2**attempt))
    return RawModelOutput(answer="", error=last_error or "unknown error")


async def execute_batch(
    tasks: list[TaskItem],
    cases: list[TestCase],
    model_configs: list[ModelConfig],
    repo: RunRepository,
) -> list[ModelResult]:
    """并发执行全部任务，返回完整 ModelResult 列表（含历史与新增）。"""
    cases_by_id = {c.id: c for c in cases}
    cfg_by_model = {f"{m.provider}:{m.model}": m for m in model_configs}

    # 每个 ModelConfig 一个 Semaphore（限流）
    sems = {key: asyncio.Semaphore(cfg.concurrency) for key, cfg in cfg_by_model.items()}
    # 每个 ModelConfig 一个 adapter 实例（缓存）
    adapters = {key: get_adapter(cfg) for key, cfg in cfg_by_model.items()}

    async def run_one(task: TaskItem) -> ModelResult:
        # 幂等：已完成则直接加载，跳过调用
        if repo.has_result(task.request_key):
            results = repo.load_model_results()
            for r in results:
                if r.request_key == task.request_key:
                    return r

        cfg = cfg_by_model[f"{task.provider}:{task.model_id}"]
        adapter = adapters[f"{task.provider}:{task.model_id}"]
        sem = sems[f"{task.provider}:{task.model_id}"]
        case = cases_by_id[task.case_id]
        prompt, context = build_prompt(case)

        started = time.time()
        async with sem:
            out = await _call_with_retry(adapter, prompt, context, task.request_key)
        ended = time.time()

        status = TaskStatus.success if not out.error else TaskStatus.failed
        result = ModelResult(
            request_key=task.request_key,
            run_id=task.run_id,
            case_id=task.case_id,
            model_id=task.model_id,
            provider=task.provider,
            config_snapshot={"provider": cfg.provider, "model": cfg.model, "concurrency": cfg.concurrency},
            prompt_version="v1",
            answer=out.answer,
            started_at=started,
            ended_at=ended,
            latency_ms=out.latency_ms,
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            total_tokens=out.total_tokens,
            cost_usd=out.cost_usd,
            retries=0,
            status=status,
            error=out.error,
        )
        # 立即持久化：中断后可从仓库恢复，不重复扣费
        repo.upsert_model_result(result)
        return result

    coros = [run_one(t) for t in tasks]
    results = await asyncio.gather(*coros)
    # 保持稳定顺序：按 case_id, model_id
    results.sort(key=lambda r: (r.case_id, r.model_id))
    return list(results)
