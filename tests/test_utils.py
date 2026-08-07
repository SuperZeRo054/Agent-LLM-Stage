"""数据集加载单测（文档 8.2 load_dataset：路径白名单 + Schema 校验）。"""

import json
from pathlib import Path

import pytest

from app.utils import load_dataset, new_run_id


def _write(path: Path, lines):
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in lines), encoding="utf-8")
    return str(path)


def test_load_dataset_ok(tmp_path):
    f = _write(
        tmp_path / "ds.jsonl",
        [
            {"id": "c1", "input": "q1", "reference_answer": "a1"},
            {"id": "c2", "input": "q2", "reference_answer": "a2", "rubric": {"correctness": "x"}},
        ],
    )
    cases = load_dataset(f, base_dir=tmp_path)
    assert len(cases) == 2
    assert cases[0].id == "c1"
    assert cases[1].rubric.correctness == "x"


def test_load_dataset_empty_file_raises(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="为空"):
        load_dataset(str(f), base_dir=tmp_path)


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(str(tmp_path / "nope.jsonl"), base_dir=tmp_path)


def test_load_dataset_rejects_path_traversal(tmp_path):
    # 解析后超出白名单基目录
    outside = tmp_path.parent / "outside_ds.jsonl"
    with pytest.raises(ValueError, match="白名单"):
        load_dataset(str(outside), base_dir=tmp_path)


def test_new_run_id_unique_and_stable_prefix():
    a = new_run_id("My Run 1")
    b = new_run_id("My Run 1")
    assert a.startswith("My-Run-1-")
    assert a != b  # uuid 后缀保证唯一
