"""运行记录与原始结果存储（文档 6.1 业务数据域）。

阶段一：JSON 文件落盘（每个 run 一个目录），按 request_key 幂等 upsert。
阶段三：切换为 Supabase PostgreSQL + SQLAlchemy，接口保持不变。

原始模型响应等大对象存文件/Storage，checkpoint 只存 ID 回链，避免过大。
单线程 asyncio 中同步 read-modify-write 天然原子，无需加锁。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.domain.schemas import ModelResult, ScoreResult


class RunRepository:
    """单次 run 的持久化仓库。"""

    def __init__(self, run_id: str, base_dir: str | Path | None = None) -> None:
        self.run_id = run_id
        base = Path(base_dir or os.environ.get("RUNS_DIR", "./runs"))
        self.run_dir = base / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ----- 路径 -----
    @property
    def results_path(self) -> Path:
        return self.run_dir / "model_results.json"

    @property
    def scores_path(self) -> Path:
        return self.run_dir / "score_results.json"

    @property
    def meta_path(self) -> Path:
        return self.run_dir / "meta.json"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.md"

    # ----- 模型结果（幂等 upsert by request_key）-----
    def _load_results(self) -> dict[str, dict]:
        if self.results_path.exists():
            return json.loads(self.results_path.read_text("utf-8"))
        return {}

    def upsert_model_result(self, result: ModelResult) -> None:
        data = self._load_results()
        data[result.request_key] = result.model_dump(mode="json")
        self.results_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

    def load_model_results(self) -> list[ModelResult]:
        data = self._load_results()
        return [ModelResult(**v) for v in data.values()]

    def has_result(self, request_key: str) -> bool:
        return request_key in self._load_results()

    # ----- 评分结果 -----
    def _load_scores(self) -> dict[str, dict]:
        if self.scores_path.exists():
            return json.loads(self.scores_path.read_text("utf-8"))
        return {}

    def upsert_score_result(self, score: ScoreResult) -> None:
        data = self._load_scores()
        data[score.request_key] = score.model_dump(mode="json")
        self.scores_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

    def load_score_results(self) -> list[ScoreResult]:
        data = self._load_scores()
        return [ScoreResult(**v) for v in data.values()]

    # ----- 元数据与报告 -----
    def save_meta(self, meta: dict) -> None:
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")

    def load_meta(self) -> dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text("utf-8"))
        return {}

    def save_report(self, content: str) -> Path:
        self.report_path.write_text(content, "utf-8")
        return self.report_path
