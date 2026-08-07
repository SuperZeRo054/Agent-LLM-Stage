"""Typer CLI：run / status / resume / report（文档第 12 节阶段一交付）。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
import yaml
from langgraph.types import Command
from rich.console import Console
from rich.table import Table

from app.graph.checkpointer import get_checkpointer
from app.graph.graph import build_graph
from app.repositories.runs import RunRepository
from app.utils import new_run_id

try:  # 阶段二：加载 .env 中的 API Key（未装 python-dotenv 时跳过）
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

app = typer.Typer(help="Agent-LLM-Stage：基于 LangGraph 的 LLM 评测 Agent", no_args_is_help=True)
console = Console()


def _load_config(config_path: str, auto_approve: bool) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise typer.BadParameter(f"配置文件不存在: {config_path}")
    cfg = yaml.safe_load(path.read_text("utf-8"))
    if auto_approve:
        cfg["auto_approve"] = True
    return cfg


async def _run(config: dict, request: str | None, checkpointer_backend: str) -> dict:
    run_id = new_run_id(config["run_name"])
    thread_cfg = {"configurable": {"thread_id": run_id}}
    init_state = {"run_id": run_id, "config": config, "request": request}

    async with get_checkpointer(checkpointer_backend) as saver:
        graph = build_graph(saver)
        state = await graph.ainvoke(init_state, config=thread_cfg)

        # 若在 human_approval 暂停（interrupt），按 auto_approve 决定是否自动恢复
        if "__interrupt__" in state:
            if config.get("auto_approve"):
                console.print("[yellow]检测到审批节点，auto_approve=True，自动放行。[/yellow]")
                state = await graph.ainvoke(Command(resume="approved"), config=thread_cfg)
            else:
                intr = state["__interrupt__"][0].value
                console.print(f"[cyan]已暂停等待审批：[/cyan]{intr.get('message')}")
                console.print(f"  run_id = {run_id}")
                decision = typer.prompt("输入审批决定（approved / rejected）", default="approved")
                state = await graph.ainvoke(Command(resume=decision), config=thread_cfg)

    state["run_id"] = run_id
    return state


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="YAML 配置路径"),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="跳过人工审批"),
    request: str = typer.Option(None, "--request", help="自然语言请求（阶段二）"),
    checkpointer: str = typer.Option("sqlite", "--checkpointer", help="memory / sqlite / postgres"),
):
    """启动一次评测。"""
    cfg = _load_config(config, auto_approve)
    state = asyncio.run(_run(cfg, request, checkpointer))
    _print_summary(state)


@app.command()
def status(
    run_id: str = typer.Option(..., "--run-id", help="运行 ID"),
    checkpointer: str = typer.Option("sqlite", "--checkpointer"),
):
    """查看 run 的当前节点与状态。"""
    snapshot = asyncio.run(_get_state(run_id, checkpointer))
    if snapshot is None:
        console.print(f"[red]未找到 run_id={run_id} 的状态（checkpointer 中无记录）。[/red]")
        raise typer.Exit(1)

    values = snapshot.values
    table = Table(title=f"Run 状态：{run_id}")
    table.add_column("字段", style="cyan")
    table.add_column("值")
    table.add_row("current_node", str(values.get("current_node")))
    table.add_row("status", str(values.get("status")))
    table.add_row("next", ", ".join(snapshot.next) if snapshot.next else "(结束)")
    table.add_row("cases", str(len(values.get("cases", []))))
    table.add_row("tasks", str(len(values.get("tasks", []))))
    table.add_row("model_results", str(len(values.get("model_results", []))))
    table.add_row("score_results", str(len(values.get("score_results", []))))
    table.add_row("errors", str(len(values.get("errors", []))))
    table.add_row("report_path", str(values.get("report_path")))
    console.print(table)


@app.command()
def resume(
    run_id: str = typer.Option(..., "--run-id"),
    decision: str = typer.Option("approved", "--decision", help="approved / rejected"),
    checkpointer: str = typer.Option("sqlite", "--checkpointer"),
):
    """从 checkpoint 恢复（含中断后的人工审批恢复）。"""
    thread_cfg = {"configurable": {"thread_id": run_id}}

    async def _do() -> dict:
        async with get_checkpointer(checkpointer) as saver:
            graph = build_graph(saver)
            snap = await graph.aget_state(thread_cfg)
            if not snap or not snap.next:
                return {"status": "已完成或无待恢复节点", "run_id": run_id}
            state = await graph.ainvoke(Command(resume=decision), config=thread_cfg)
            return state

    state = asyncio.run(_do())
    _print_summary(state)


@app.command()
def report(
    run_id: str = typer.Option(..., "--run-id"),
):
    """查看 / 重新生成 Markdown 报告。"""
    repo = RunRepository(run_id)
    if not repo.report_path.exists():
        console.print(f"[red]报告不存在: {repo.report_path}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]报告路径: {repo.report_path}[/green]\n")
    console.print(repo.report_path.read_text("utf-8"))


# --------------------------------------------------------------------------- #
async def _get_state(run_id: str, checkpointer: str):
    async with get_checkpointer(checkpointer) as saver:
        graph = build_graph(saver)
        return await graph.aget_state({"configurable": {"thread_id": run_id}})


def _print_summary(state: dict) -> None:
    console.print("\n[bold green]== 评测完成 ==[/bold green]")
    table = Table(title=f"Run：{state.get('run_id')}")
    table.add_column("字段", style="cyan")
    table.add_column("值")
    table.add_row("status", str(state.get("status")))
    table.add_row("current_node", str(state.get("current_node")))
    table.add_row("model_results", str(len(state.get("model_results", []))))
    table.add_row("score_results", str(len(state.get("score_results", []))))
    metrics = state.get("metrics") or {}
    table.add_row("total_cost_usd", str(metrics.get("total_cost_usd")))
    table.add_row("anomalies", str(len(metrics.get("anomalies", []))))
    table.add_row("report_path", str(state.get("report_path")))
    console.print(table)


if __name__ == "__main__":
    app()
