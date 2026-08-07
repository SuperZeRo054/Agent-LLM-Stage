"""执行器单测：并发执行与 request_key 幂等去重（文档 9.3）。"""

import pytest

from app.adapters.base import BaseAdapter, RawModelOutput, register_adapter
from app.domain.schemas import ModelConfig, TaskItem, TaskStatus, TestCase
from app.repositories.runs import RunRepository
from app.services.executor import execute_batch

_INVOCATIONS = 0


@register_adapter("fake_count")
class CountingAdapter(BaseAdapter):
    """计数 invoke 次数，验证幂等去重。"""

    async def invoke(self, prompt: str, context: list[str]) -> RawModelOutput:
        global _INVOCATIONS
        _INVOCATIONS += 1
        return RawModelOutput(answer="counted-answer", prompt_tokens=4, completion_tokens=4, total_tokens=8)


def _make_tasks(cases, models, run_id="run-exec"):
    tasks = []
    for c in cases:
        for m in models:
            tasks.append(
                TaskItem(
                    request_key=f"{run_id}:{c.id}:{m.provider}:{m.model}",
                    run_id=run_id,
                    case_id=c.id,
                    model_id=m.model,
                    provider=m.provider,
                )
            )
    return tasks


@pytest.mark.asyncio
async def test_execute_batch_runs_all(tmp_path):
    global _INVOCATIONS
    _INVOCATIONS = 0
    cases = [TestCase(id=f"c{i}", input=f"q{i}") for i in range(3)]
    models = [ModelConfig(provider="fake_count", model="m1", concurrency=2)]
    tasks = _make_tasks(cases, models)
    repo = RunRepository("run-exec", base_dir=tmp_path)

    results = await execute_batch(tasks, cases, models, repo)
    assert len(results) == 3
    assert all(r.status == TaskStatus.success for r in results)
    assert _INVOCATIONS == 3  # 每个任务调用一次


@pytest.mark.asyncio
async def test_execute_batch_idempotent_no_duplicate_calls(tmp_path):
    """第二次执行同一批任务，已完成的不重复调用（不重复扣费）。"""
    global _INVOCATIONS
    _INVOCATIONS = 0
    cases = [TestCase(id=f"c{i}", input=f"q{i}") for i in range(3)]
    models = [ModelConfig(provider="fake_count", model="m1", concurrency=2)]
    tasks = _make_tasks(cases, models)
    repo = RunRepository("run-exec", base_dir=tmp_path)

    await execute_batch(tasks, cases, models, repo)
    first = _INVOCATIONS
    assert first == 3

    # 第二次：repo 已有全部结果，应跳过调用
    results2 = await execute_batch(tasks, cases, models, repo)
    assert _INVOCATIONS == first  # 未新增调用
    assert len(results2) == 3


@pytest.mark.asyncio
async def test_execute_batch_persists_results_for_resume(tmp_path):
    """结果立即落盘，支持断点续跑（即使内存状态丢失）。"""
    cases = [TestCase(id="c1", input="q1")]
    models = [ModelConfig(provider="fake_count", model="m1", concurrency=1)]
    tasks = _make_tasks(cases, models)
    repo = RunRepository("run-exec", base_dir=tmp_path)
    await execute_batch(tasks, cases, models, repo)
    # 模拟新进程：新建 repo 读取已落盘结果
    repo2 = RunRepository("run-exec", base_dir=tmp_path)
    loaded = repo2.load_model_results()
    assert len(loaded) == 1
    assert loaded[0].answer == "counted-answer"
