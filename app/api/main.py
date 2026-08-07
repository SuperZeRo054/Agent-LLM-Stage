"""FastAPI 服务骨架（阶段三交付）。

阶段一仅提供最小骨架与 run 状态查询；完整 API 在阶段三配合 Supabase Auth/RLS 实现。
运行：uvicorn app.api.main:app（需 pip install -e .[phase3]）
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph.checkpointer import get_checkpointer
from app.graph.graph import build_graph
from app.repositories.runs import RunRepository

app = FastAPI(title="Agent-LLM-Stage API", version="0.1.0")


class RunRequest(BaseModel):
    config: dict[str, Any]
    auto_approve: bool = False
    checkpointer: str = "sqlite"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/runs")
async def create_run(req: RunRequest) -> dict:
    """创建并启动一次评测（阶段三增加认证与 RLS）。"""
    from app.utils import new_run_id

    cfg = dict(req.config)
    if req.auto_approve:
        cfg["auto_approve"] = True
    run_id = new_run_id(cfg["run_name"])
    thread_cfg = {"configurable": {"thread_id": run_id}}
    async with get_checkpointer(req.checkpointer) as saver:
        graph = build_graph(saver)
        state = await graph.ainvoke(
            {"run_id": run_id, "config": cfg, "request": None}, config=thread_cfg
        )
        if "__interrupt__" in state and req.auto_approve:
            from langgraph.types import Command

            state = await graph.ainvoke(Command(resume="approved"), config=thread_cfg)
    return {"run_id": run_id, "status": state.get("status"), "report_path": state.get("report_path")}


@app.get("/runs/{run_id}/status")
async def run_status(run_id: str, checkpointer: str = "sqlite") -> dict:
    async with get_checkpointer(checkpointer) as saver:
        graph = build_graph(saver)
        snap = await graph.aget_state({"configurable": {"thread_id": run_id}})
        if not snap or not snap.values:
            raise HTTPException(status_code=404, detail="run not found")
        return {
            "run_id": run_id,
            "current_node": snap.values.get("current_node"),
            "status": snap.values.get("status"),
            "next": list(snap.next) if snap.next else [],
        }


@app.get("/runs/{run_id}/report")
async def run_report(run_id: str) -> dict:
    repo = RunRepository(run_id)
    if not repo.report_path.exists():
        raise HTTPException(status_code=404, detail="report not found")
    return {"run_id": run_id, "report": repo.report_path.read_text("utf-8")}
