# 阶段二实施计划：真实模型接入 + LLM Judge + LLM Planner

> 决策：**联网端到端验证**（用户提供真实 key/端点）+ **完整阶段二**（含 Planner）。
> 原则：保持 `ModelAdapter` 抽象干净，新增 provider 只写一个类 + `@register_adapter`，核心流程不动。

## 1. 目标与范围

把阶段一预留的脚手架/`NotImplementedError` 全部填成真实实现，且 fake 路径完整保留（36 个旧测试不回归）：

1. `provider_a`（langchain-openai SDK，OpenAI 兼容）+ `provider_b`（httpx 裸调，OpenAI 兼容）真实接入，记录真实 token/成本。
2. `judge_candidate` 真实分支：Judge 走 adapter + 严格 JSON Schema 校验 + 有限重试。
3. `plan_evaluation` / `intake_request` 真实 Planner 分支：自然语言请求 -> 结构化 `EvaluationPlan`，缺信息 `requires_clarification`。
4. 限流（已有 Semaphore）、超时（adapter 内）、真实计费（pricing）对接。

「留出接口」的体现：抽 `LLMEndpointConfig` 基类 + `get_adapter` 工厂 + `@register_adapter`；provider_a（SDK）与 provider_b（裸 HTTP）双示例展示同一协议两种接入方式，加新 provider 零侵入核心。

## 2. 关键设计

### 2.1 数据契约扩展（`app/domain/schemas.py`，向后兼容）

抽连接配置基类，`ModelConfig`/`JudgeConfig` 继承，新增 `PlannerConfig`：

```python
class LLMEndpointConfig(BaseModel):
    provider: str
    model: str
    base_url: str | None = None        # OpenAI 兼容端点；None 时用 SDK 默认
    api_key_env: str | None = None     # 环境变量名；None 时按 provider 推导（PROVIDER_A_API_KEY 等）
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int | None = None
    timeout: float = Field(default=60.0, gt=0)
    pricing: dict[str, float] | None = None  # {"input": x, "output": y} 每 1M token USD

class ModelConfig(LLMEndpointConfig):
    concurrency: int = Field(default=3, ge=1, le=64)

class JudgeConfig(LLMEndpointConfig):
    repeats: int = Field(default=2, ge=1, le=10)

class PlannerConfig(LLMEndpointConfig):
    pass
```

- 旧 YAML（`provider/model/concurrency` 或 `provider/model/repeats`）默认值不变，**向后兼容**。
- `RunConfig` 增 `planner: PlannerConfig | None = None`。
- `JudgeResult` 增可选 `prompt_tokens/completion_tokens/cost_usd`（默认 0），让 Judge 成本可计入总成本。

### 2.2 Adapter 层（`app/adapters/`）

- `base.py`：`invoke` 增加 keyword-only `json_mode: bool = False`（默认 False，向后兼容；runtime_checkable 不检查签名，fake 不破坏）。`get_adapter` 接受 `LLMEndpointConfig`（ModelConfig/JudgeConfig/PlannerConfig 通用）。
- `pricing.py`（新）：`compute_cost(prompt_tokens, completion_tokens, pricing) -> float`；`pricing` 为 None 时 cost=0（不臆造价格，token 照记）。
- `provider_a.py`：`ChatOpenAI(model, api_key, base_url, temperature, max_tokens, timeout)`；`json_mode` 用 `response_format={"type":"json_object"}`；`messages = context + [user prompt]`；从 `usage_metadata` 取 token；异常 `raise RuntimeError`（executor 重试）。
- `provider_b.py`：`httpx.AsyncClient(timeout)` POST `{base_url}/chat/completions`；解析 `choices[0].message.content` 与 `usage.prompt_tokens/completion_tokens`；`json_mode` 设 `response_format={"type":"json_object"}`；异常 raise。
- `fake.py`：`invoke` 加 `json_mode` 参数；fake judge 在 `json_mode=True` 时返回合法 JSON 文本（与真实 judge 路径统一可测）。

密钥读取：`api_key_env` 指定变量名，否则按 provider 推导（`provider_a` -> `PROVIDER_A_API_KEY`，`provider_b` -> `PROVIDER_B_API_KEY`，judge/planner 用各自 `api_key_env` 或 `JUDGE_API_KEY`/`PLANNER_API_KEY`）。缺 key 抛清晰错误。

### 2.3 真实 LLM Judge（`app/services/scoring.py`）

`judge_candidate` 真实分支（`judge.provider != "fake"`）：
1. `JudgeConfig` -> `get_adapter` 拿 judge adapter。
2. `build_judge_prompt`（改用 `prompts/judge_v1.txt` 作系统指令 + case 信息拼接）。
3. `adapter.invoke(prompt, [], json_mode=True)`。
4. 解析 JSON -> 严格校验（`scores` 三维度 0-10、`risk_flag` 合法、必要字段）。
5. 解析/校验失败：有限重试（`JUDGE_PARSE_RETRIES=2`），仍失败则降级为 `risk_flag=other` 的 JudgeResult 并记 warning（不中断 run）。
6. 记 token/cost 到 `JudgeResult`。
fake 分支（`_judge_heuristic`）保留不变。

### 2.4 真实 LLM Planner（`app/graph/nodes.py`）

- `intake_request`：`request`（自然语言）存在时不再报错，生成 run_id、status=planning，保留 request 传入下游。
- `plan_evaluation`：
  - `request` 存在且 `config.planner` 已配 -> 调 Planner adapter（`json_mode=True`，用 `prompts/planner_v1.txt` + 用户 request）-> 解析 `EvaluationPlan`；缺关键信息 -> `requires_clarification=True` + `missing_fields`。
  - `config` 存在（YAML）-> 现有确定性路径不变。
  - `request` 存在但未配 planner -> 记 error「自然语言请求需在配置中提供 planner」。
- 新增 `app/prompts.py: load_prompt(name) -> str` 读 `prompts/*.txt`（Judge/Planner 复用）。

### 2.5 计费与指标（`app/services/metrics.py`）

`calculate_metrics` 汇总时把 `ScoreResult.judge_scores` 的 token/cost 计入 `total_cost_usd`（被测模型成本 + Judge 成本）。anomaly 的 `budget_exceeded` 因此含全部真实成本。

### 2.6 .env 与依赖

- `pyproject.toml`：`python-dotenv>=1.0` 从 phase3 提到 phase2。
- `app/cli.py`：启动 `load_dotenv()`（找不到 .env 不报错）。
- `.env.example`：补 `PROVIDER_A_BASE_URL`、`PROVIDER_B_BASE_URL`、`PLANNER_API_KEY` 等示例与注释。
- 安装：`.venv/Scripts/python.exe -m pip install -e ".[dev,phase2]"`（清华镜像或 7897 代理）。

## 3. 配置示例

新增 `configs/run.real.yaml`（保留 `run.example.yaml` fake 不动），示例真实 provider：

```yaml
run_name: model-comparison-real
dataset: datasets/dataset_v1.jsonl
models:
  - provider: provider_a
    model: <被测模型1>
    base_url: <OpenAI兼容端点>
    api_key_env: PROVIDER_A_API_KEY
    concurrency: 2
    pricing: {input: 0.15, output: 0.6}   # 每 1M token USD
  - provider: provider_b
    model: <被测模型2>
    base_url: <OpenAI兼容端点>
    api_key_env: PROVIDER_B_API_KEY
    pricing: {input: 0.27, output: 1.1}
judge:
  provider: provider_a
  model: <judge模型>
  base_url: <OpenAI兼容端点>
  api_key_env: JUDGE_API_KEY
  repeats: 2
  pricing: {input: 0.15, output: 0.6}
budget_limit_usd: 1.0
auto_approve: false
```

## 4. 测试策略（mock，不联网；自动化测试不产生费用）

- `tests/test_provider_adapters.py`（新）：mock `ChatOpenAI`/`httpx`，验证 provider_a/b 的 token 解析、cost 计算、`json_mode` 透传、超时/异常 -> RuntimeError。
- `tests/test_scoring.py` 扩展：mock judge adapter 返回合法/非法/缺字段 JSON，验证校验、重试、降级；fake 路径不回归。
- `tests/test_planner.py`（新）：mock planner adapter，验证自然语言 -> `EvaluationPlan`、`requires_clarification`、未配 planner 报错。
- `tests/test_schemas.py`（新）：`LLMEndpointConfig` 继承、新字段默认值、旧 YAML 兼容解析。
- 现有 36 个 fake 测试全绿；`ruff check` 通过。

## 5. 联网端到端验证（需你提供）

mock 全绿后，实跑一次真实评测。**需要你提供**（退出计划后）：
- OpenAI 兼容 `base_url`（如 OpenAI `https://api.openai.com/v1`、智谱 `https://open.bigmodel.cn/api/paas/v4`、DeepSeek `https://api.deepseek.com/v1` 等）。
- API key（写入 `.env`，不提交；你也可用 `! ` 前缀在会话里 export）。
- 1-2 个被测模型名 + 1 个 Judge 模型名。
- 定价（每 1M token USD，可选；不填则只记 token 不计成本）。

外网 API 走你的 `7897` 代理（adapter 经 httpx/openai 自动尊重 `HTTPS_PROXY`）。验证点：真实 token/cost 入库、真实 Judge 评分与方差、报告含真实数据、断点续跑不重复扣费。

## 6. 文件改动清单

| 文件 | 改动 |
| --- | --- |
| `app/domain/schemas.py` | `LLMEndpointConfig` 基类；ModelConfig/JudgeConfig 继承；`PlannerConfig`；RunConfig+.planner；JudgeResult+token/cost |
| `app/adapters/base.py` | invoke+`json_mode`；get_adapter 接受 LLMEndpointConfig |
| `app/adapters/pricing.py`（新） | cost 计算 |
| `app/adapters/provider_a.py` | 真实 langchain-openai |
| `app/adapters/provider_b.py` | 真实 httpx |
| `app/adapters/fake.py` | invoke+json_mode |
| `app/prompts.py`（新） | load_prompt |
| `app/services/scoring.py` | 真实 Judge 分支 + 重试 + 降级 |
| `app/graph/nodes.py` | 真实 Planner 分支 |
| `app/services/metrics.py` | 汇总 Judge 成本 |
| `app/cli.py` | load_dotenv |
| `pyproject.toml` | dotenv 提到 phase2 |
| `.env.example` / `configs/run.real.yaml`（新） | 配置示例 |
| `tests/` | 4 个测试文件新增/扩展 |
| `框架搭建报告.md` | 追加阶段二完成章节 |

## 7. 执行顺序

1. 装依赖（phase2 + dotenv）。
2. schemas 扩展（基础，其他依赖它）。
3. adapter 层（base/pricing/provider_a/provider_b/fake）。
4. prompts.py + 真实 Judge + 真实 Planner。
5. metrics 计费汇总。
6. cli .env + 配置示例。
7. mock 测试全绿 + ruff。
8. 联网端到端实跑（你提供 key）。
9. 更新报告。

## 8. 风险与回退

- 真实 API 不可用/超时：adapter 抛错 -> executor 重试 3 次 -> 记 failed，run 不中断（阶段一已验证失败分支）。联网验证失败不影响 mock 测试交付。
- JSON 解析失败：Judge 降级、Planner 记 error 转 failed_report，不崩。
- 向后兼容：所有新字段有默认值，旧 fake 配置与旧测试零改动。
