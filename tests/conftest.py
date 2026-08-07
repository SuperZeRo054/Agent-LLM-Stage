"""pytest 公共配置：把运行产物导向临时目录，避免污染仓库。"""

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.gettempdir()) / "agent_llm_stage_test_runs"
_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("RUNS_DIR", str(_TMP))
