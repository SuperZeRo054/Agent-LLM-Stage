"""通用工具：数据集加载（路径白名单 + Schema 校验）、run_id 生成、JSON 容错。"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from app.domain.schemas import TestCase


def new_run_id(run_name: str) -> str:
    """生成不可变的 run_id：run_name（清洗）+ 短 uuid。也用作 LangGraph thread_id。"""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_name).strip("-") or "run"
    return f"{slug}-{uuid.uuid4().hex[:8]}"


# ```json\n...\n``` 或 ```\n...\n```（可选语言标记）
_FENCE_FULL_RE = re.compile(
    r"^```[a-zA-Z0-9_+-]*[ \t]*\n?(.*?)[ \t]*\n?```[ \t]*$", re.DOTALL
)
# 仅开头围栏，无结尾
_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9_+-]*[ \t]*\n?(.*)$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """剥离模型可能包裹的 ```json ... ``` 代码块，返回纯文本。

    兼容完整围栏、仅开头围栏、无语言标记等情况。
    """
    t = text.strip()
    m = _FENCE_FULL_RE.match(t)
    if m:
        return m.group(1).strip()
    if t.startswith("```"):
        m = _FENCE_OPEN_RE.match(t)
        if m:
            return m.group(1).strip().rstrip("`").strip()
    return t


def parse_json_lenient(text: str) -> dict:
    """容错 JSON 解析：剥离代码块 -> 严格解析 -> json_repair 修复。

    用于解析 LLM 返回的 JSON。当模型在字符串字段中混入未转义引号
    （例如把候选回答原样塞进 evidence）导致严格解析失败时，json_repair
    可挽救大部分结构。失败抛 ValueError。
    """
    t = strip_code_fence(text)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    try:
        import json_repair

        obj = json_repair.repair_json(t, return_objects=True)
        if isinstance(obj, dict):
            return obj
    except Exception:  # noqa: BLE001 -- json_repair 失败则整体失败
        pass
    raise ValueError(f"JSON 解析失败（严格与 json_repair 均失败）: {t[:120]!r}")


def load_dataset(path: str, base_dir: str | Path | None = None) -> list[TestCase]:
    """读取 JSONL 测试集并校验（文档 8.2 load_dataset 工具契约）。

    路径白名单：解析后必须位于项目根目录之下，防止路径穿越。
    """
    base = Path(base_dir or Path.cwd()).resolve()
    target = (base / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        target.relative_to(base)
    except ValueError as e:
        raise ValueError(f"数据集路径不在允许的白名单目录内: {target}") from e
    if not target.exists():
        raise FileNotFoundError(f"数据集不存在: {target}")

    cases: list[TestCase] = []
    for lineno, line in enumerate(target.read_text("utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{target}:{lineno} JSON 解析失败: {e}") from e
        cases.append(TestCase(**obj))
    if not cases:
        raise ValueError(f"数据集为空: {target}")
    return cases
