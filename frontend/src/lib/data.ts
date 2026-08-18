export type RunStatus = "completed" | "running" | "failed"

export interface EvalRun {
  id: string
  models: string[]
  dataset: string
  cases: number
  status: RunStatus
  score: number | null
  cost: number | null
  created: string
}

export interface EvalEvent {
  ts: string
  msg: string
  type: "ok" | "warn" | "err"
}

export interface ModelStat {
  model: string
  avg: number
  runs: number
}

export const SAMPLE_RUNS: EvalRun[] = [
  {
    id: "run-20260807-a1b2",
    models: ["deepseek-v4-flash", "kimi-k3"],
    dataset: "MMLU-Pro",
    cases: 120,
    status: "completed",
    score: 87.3,
    cost: 0.015,
    created: "2026-08-07 14:22",
  },
  {
    id: "run-20260807-c3d4",
    models: ["glm-5.2", "gpt-4o"],
    dataset: "HumanEval",
    cases: 164,
    status: "completed",
    score: 92.1,
    cost: 0.042,
    created: "2026-08-07 11:05",
  },
  {
    id: "run-20260806-e5f6",
    models: ["deepseek-v4-flash"],
    dataset: "C-Eval",
    cases: 200,
    status: "completed",
    score: 79.8,
    cost: 0.008,
    created: "2026-08-06 16:40",
  },
  {
    id: "run-20260806-b7a8",
    models: ["kimi-k3", "glm-5.2"],
    dataset: "GSM8K",
    cases: 85,
    status: "completed",
    score: 94.5,
    cost: 0.031,
    created: "2026-08-06 09:15",
  },
  {
    id: "run-20260805-d9c0",
    models: ["gpt-4o", "deepseek-v4-flash", "kimi-k3"],
    dataset: "MMLU-Pro",
    cases: 120,
    status: "completed",
    score: 88.9,
    cost: 0.067,
    created: "2026-08-05 20:30",
  },
  {
    id: "run-20260805-e1f2",
    models: ["glm-5.2"],
    dataset: "HumanEval",
    cases: 164,
    status: "failed",
    score: null,
    cost: null,
    created: "2026-08-05 15:12",
  },
  {
    id: "run-20260804-a3b4",
    models: ["kimi-k3"],
    dataset: "C-Eval",
    cases: 200,
    status: "completed",
    score: 82.4,
    cost: 0.019,
    created: "2026-08-04 10:45",
  },
  {
    id: "run-20260804-c5d6",
    models: ["deepseek-v4-flash", "glm-5.2"],
    dataset: "GSM8K",
    cases: 85,
    status: "running",
    score: null,
    cost: null,
    created: "2026-08-04 08:00",
  },
]

export const SAMPLE_EVENTS: EvalEvent[] = [
  {
    ts: "2026-08-07 14:22",
    msg: "Run run-20260807-a1b2 completed — 87.3% avg score across 2 models",
    type: "ok",
  },
  {
    ts: "2026-08-07 11:05",
    msg: "Run run-20260807-c3d4 completed — 92.1% on HumanEval (glm-5.2 + gpt-4o)",
    type: "ok",
  },
  {
    ts: "2026-08-07 09:30",
    msg: "New config pushed: DeepSeek v4 endpoint updated",
    type: "warn",
  },
  {
    ts: "2026-08-06 16:40",
    msg: "Run run-20260806-e5f6 completed — 79.8% on C-Eval (deepseek-v4-flash)",
    type: "ok",
  },
  {
    ts: "2026-08-06 09:15",
    msg: "Run run-20260806-b7a8 completed — 94.5% on GSM8K (kimi-k3 + glm-5.2)",
    type: "ok",
  },
  {
    ts: "2026-08-05 20:30",
    msg: "Run run-20260805-d9c0 completed — 88.9% (3-model comparison on MMLU-Pro)",
    type: "ok",
  },
  {
    ts: "2026-08-05 15:12",
    msg: "Run run-20260805-e1f2 FAILED — Judge API timeout after 3 retries",
    type: "err",
  },
  {
    ts: "2026-08-05 10:00",
    msg: "Kimi API key rotated — previous key approaching expiration",
    type: "warn",
  },
]

export function getModelStats(runs: EvalRun[]): ModelStat[] {
  const map = new Map<string, { total: number; count: number }>()
  for (const run of runs) {
    if (run.status !== "completed" || run.score === null) continue
    for (const model of run.models) {
      const entry = map.get(model) ?? { total: 0, count: 0 }
      entry.total += run.score
      entry.count += 1
      map.set(model, entry)
    }
  }
  return [...map.entries()]
    .map(([model, v]) => ({
      model,
      avg: Number((v.total / v.count).toFixed(1)),
      runs: v.count,
    }))
    .sort((a, b) => b.avg - a.avg)
}

export function getKpis(runs: EvalRun[]) {
  const completed = runs.filter((r) => r.status === "completed")
  const active = runs.filter((r) => r.status === "running")
  const avgScore =
    completed.length > 0
      ? completed.reduce((s, r) => s + (r.score ?? 0), 0) / completed.length
      : null
  const totalCost = completed.reduce((s, r) => s + (r.cost ?? 0), 0)
  return {
    totalRuns: runs.length,
    activeNow: active.length,
    avgScore,
    totalCost,
    completedCount: completed.length,
  }
}

export const SCORE_TREND = [
  { label: "08-01", value: 84.2 },
  { label: "08-02", value: 86.1 },
  { label: "08-03", value: 83.7 },
  { label: "08-04", value: 82.4 },
  { label: "08-05", value: 88.9 },
  { label: "08-06", value: 87.2 },
  { label: "08-07", value: 89.7 },
]

export const DAILY_ACTIVITY = [
  { label: "08-04", value: 2 },
  { label: "08-05", value: 5 },
  { label: "08-06", value: 4 },
  { label: "08-07", value: 6 },
  { label: "08-08", value: 3 },
  { label: "08-09", value: 7 },
  { label: "08-10", value: 5 },
  { label: "08-11", value: 8 },
  { label: "08-12", value: 6 },
  { label: "08-13", value: 9 },
  { label: "08-14", value: 7 },
  { label: "08-15", value: 10 },
  { label: "08-16", value: 8 },
  { label: "08-17", value: 12 },
]
