"""提示词加载：读取 prompts/*.txt 文本，供 Judge / Planner / Analyst 复用。"""

from __future__ import annotations

from functools import cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@cache
def load_prompt(name: str) -> str:
    """读取 prompts/<name>.txt 全文。name 为不含扩展名的文件名，禁止目录穿越。"""
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"非法提示词名: {name!r}")
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"提示词文件不存在: {path}")
    return path.read_text("utf-8")
