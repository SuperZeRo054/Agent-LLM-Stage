# Agent-LLM-Stage

基于 **LangGraph** 的 LLM 评测 Agent。用户提交“用哪些模型、哪份测试集、按什么规则评测”的任务后，Agent 自动规划并执行一次完整评测，最后给出可追溯的对比报告。

```text
理解评测任务 -> 生成执行计划 -> 并发调用多个 LLM
-> 规则校验与 LLM Judge 盲评 -> 汇总指标 -> 解释结果并生成报告
```

## 核心设计

- **一个 Agent、一个 StateGraph**：11 个节点，职责单一，可测试。
- **LLM 智能受边界约束**：LLM 只负责规划、语义评分、解释报告；模型调用、指标计算、落库、重试保持确定性。
- **可恢复执行**：每个节点完成后 checkpoint；中断后从最后成功节点恢复，且不重复扣费（幂等 request key）。
- **盲评 Judge**：被测模型身份对 Judge 隐藏，候选顺序随机化，每条评分两次取均值与方差。

## 目录结构

```text
app/
  graph/        # StateGraph、状态、节点、路由
  domain/       # Pydantic Schema
  adapters/     # ModelAdapter 协议 + Fake/真实 Adapter
  services/     # executor / scoring / metrics / reports
  repositories/ # 运行记录与原始结果
  api/          # FastAPI 骨架
  cli.py        # Typer CLI：run / status / resume / report
datasets/       # JSONL 测试集
configs/        # 运行配置示例
prompts/        # planner / judge / analyst 提示词
tests/          # 单元测试
docker-compose.yml
pyproject.toml
```

## 快速开始

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
# Windows (Git Bash)
source .venv/Scripts/activate
# 或 PowerShell: .venv\Scripts\Activate.ps1

# 2. 安装依赖（阶段一 + 开发工具）
pip install -e ".[dev]"

# 3. 运行一次模拟评测（Fake Adapter，无需 API Key）
agent-llm-stage run --config configs/run.example.yaml

# 4. 查看状态与报告
agent-llm-stage status --run-id <run_id>
agent-llm-stage report --run-id <run_id>
```

## CLI

| 命令 | 说明 |
| --- | --- |
| `run` | 读取 YAML 配置，启动一次评测（可 `--auto-approve` 跳过人工审批） |
| `status` | 查看 run 的当前节点与状态 |
| `resume` | 从 checkpoint 恢复（含中断后的人工审批恢复） |
| `report` | 重新生成 Markdown 报告 |

## 开发阶段

- **阶段一（当前）**：Graph 骨架 + Fake Adapter + 内存 checkpoint + CLI + 单测。
- **阶段二**：接入真实模型、限流重试、规则评分 + Judge 双次盲评。
- **阶段三**：Supabase PostgreSQL 业务表 + RLS、PostgreSQL Checkpointer、HTML 报告、Docker Compose。

详见 `Agent-LLM-Stage-开发框架与实现方案.md` 与 `框架搭建报告.md`。

## 安全约束

- 不反序列化不受信任的配置或 checkpoint。
- 服务端密钥只放环境变量，绝不返回给浏览器。
- Judge 不接收被测模型身份；`interrupt()` 前的外部副作用可安全重放。
