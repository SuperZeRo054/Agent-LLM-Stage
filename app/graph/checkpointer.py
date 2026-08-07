"""Checkpoint 后端工厂（文档第 2、6 节）。

- memory：内存，用于单元测试与单进程内联恢复。
- sqlite：Phase 1 持久化，支持 CLI `resume` 跨进程恢复（仅 Agent 状态，非业务数据）。
- postgres：Phase 3 LangGraph PostgreSQL Checkpointer。

业务数据从不使用 SQLite（文档 6.1）；此处仅指 Agent 状态 checkpoint。
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver


@asynccontextmanager
async def memory_checkpointer() -> AsyncIterator[MemorySaver]:
    yield MemorySaver()


@asynccontextmanager
async def sqlite_checkpointer(path: str | None = None) -> AsyncIterator:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = path or os.environ.get("CHECKPOINTER_SQLITE_PATH", "./runs/checkpoints.sqlite")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await saver.setup()
        yield saver


@asynccontextmanager
async def postgres_checkpointer(uri: str | None = None) -> AsyncIterator:
    """Phase 3：LangGraph PostgreSQL Checkpointer。"""
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as e:
        raise RuntimeError(
            "未安装 phase3 依赖（langgraph-checkpoint-postgres）；pip install -e .[phase3]"
        ) from e
    uri = uri or os.environ.get("LANGGRAPH_PG_URI")
    if not uri:
        raise RuntimeError("未配置 LANGGRAPH_PG_URI")
    async with AsyncPostgresSaver.from_conn_string(uri) as saver:
        await saver.setup()
        yield saver


@asynccontextmanager
async def get_checkpointer(backend: str | None = None) -> AsyncIterator:
    """按 backend（或环境变量 CHECKPOINTER_BACKEND）取得已初始化的 checkpointer。"""
    backend = backend or os.environ.get("CHECKPOINTER_BACKEND", "memory")
    if backend == "memory":
        async with memory_checkpointer() as c:
            yield c
    elif backend == "sqlite":
        async with sqlite_checkpointer() as c:
            yield c
    elif backend == "postgres":
        async with postgres_checkpointer() as c:
            yield c
    else:
        raise ValueError(f"未知 checkpointer backend: {backend}")
