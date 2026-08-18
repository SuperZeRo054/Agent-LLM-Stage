import { useState } from "react"
import {
  Activity,
  Gauge,
  Moon,
  RefreshCw,
  Sun,
  TrendingUp,
  Trophy,
  Wallet,
} from "lucide-react"

import { BarChart, Sparkline } from "@/components/charts"
import { EventLog } from "@/components/event-log"
import { RunsTable } from "@/components/runs-table"
import { StatCard } from "@/components/stat-card"
import { useTheme } from "@/components/theme-provider"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  DAILY_ACTIVITY,
  SCORE_TREND,
  SAMPLE_EVENTS,
  SAMPLE_RUNS,
  getKpis,
  getModelStats,
} from "@/lib/data"

export function App() {
  const [refreshing, setRefreshing] = useState(false)
  const { theme, setTheme } = useTheme()
  const kpis = getKpis(SAMPLE_RUNS)
  const modelStats = getModelStats(SAMPLE_RUNS)

  const handleRefresh = () => {
    setRefreshing(true)
    setTimeout(() => setRefreshing(false), 800)
  }

  return (
    <div className="min-h-svh bg-background text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6">
          <div className="flex items-center gap-2.5">
            <div className="grid size-7 place-items-center rounded-md bg-primary text-sm font-black text-primary-foreground">
              LS
            </div>
            <h1 className="text-sm font-bold">
              Agent-LLM-Stage
              <span className="ml-1.5 font-normal text-muted-foreground">
                · Evaluation Dashboard
              </span>
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              aria-label={
                theme === "dark" ? "Switch to light mode" : "Switch to dark mode"
              }
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? <Sun /> : <Moon />}
            </Button>
            <Button
              variant="outline"
              onClick={handleRefresh}
              disabled={refreshing}
            >
              <RefreshCw className={refreshing ? "animate-spin" : ""} />
              Refresh
            </Button>
            <Button>+ New Run</Button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        {/* KPI cards */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <StatCard
            label="Total Runs"
            value={String(kpis.totalRuns)}
            delta="All time"
            icon={<Gauge />}
          />
          <StatCard
            label="Active Now"
            value={String(kpis.activeNow)}
            delta={kpis.activeNow > 0 ? "In progress" : "Idle"}
            accent
            icon={<Activity />}
          />
          <StatCard
            label="Avg Score"
            value={
              kpis.avgScore !== null ? `${kpis.avgScore.toFixed(1)}%` : "--"
            }
            delta={`${kpis.completedCount} completed`}
            icon={<TrendingUp />}
          />
          <StatCard
            label="Total Cost"
            value={`$${kpis.totalCost.toFixed(3)}`}
            delta="USD across all runs"
            icon={<Wallet />}
          />
        </div>

        {/* Tabs */}
        <Tabs defaultValue="runs" className="mt-6 gap-4">
          <TabsList>
            <TabsTrigger value="runs">Runs</TabsTrigger>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="activity">Activity Log</TabsTrigger>
          </TabsList>

          {/* Runs tab */}
          <TabsContent value="runs" className="mt-4">
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
              <Card className="xl:col-span-2">
                <CardHeader>
                  <CardTitle className="text-xs font-semibold tracking-[1.4px] uppercase">
                    Recent Evaluation Runs
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <RunsTable runs={SAMPLE_RUNS} />
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-xs font-semibold tracking-[1.4px] uppercase">
                    Score Trend (7 days)
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Sparkline data={SCORE_TREND} />
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Metrics tab */}
          <TabsContent value="metrics" className="mt-4">
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
              <Card className="xl:col-span-2">
                <CardHeader>
                  <CardTitle className="text-xs font-semibold tracking-[1.4px] uppercase">
                    Model Score Comparison
                  </CardTitle>
                </CardHeader>
                <CardContent className="h-64">
                  <BarChart
                    data={modelStats.map((m) => ({
                      label: m.model,
                      value: m.avg,
                    }))}
                  />
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-1.5 text-xs font-semibold tracking-[1.4px] uppercase">
                    <Trophy className="size-3.5 text-primary" />
                    Top Models
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col gap-2">
                  {modelStats.map((m, i) => (
                    <div
                      key={m.model}
                      className="flex items-center justify-between rounded-md border border-border p-2.5"
                    >
                      <div className="flex items-center gap-2.5">
                        <span className="grid size-6 place-items-center rounded bg-primary/15 font-mono text-[10px] font-bold text-primary">
                          {i + 1}
                        </span>
                        <div>
                          <p className="font-mono text-xs">{m.model}</p>
                          <p className="text-[11px] text-muted-foreground">
                            {m.runs} run{m.runs > 1 ? "s" : ""}
                          </p>
                        </div>
                      </div>
                      <span className="text-base font-bold tabular-nums">
                        {m.avg.toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Activity tab */}
          <TabsContent value="activity" className="mt-4">
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
              <Card className="xl:col-span-2">
                <CardHeader>
                  <CardTitle className="text-xs font-semibold tracking-[1.4px] uppercase">
                    Evaluation Activity (14 days)
                  </CardTitle>
                </CardHeader>
                <CardContent className="h-64">
                  <BarChart
                    data={DAILY_ACTIVITY}
                    unit=""
                    colorVar="var(--chart-2)"
                  />
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-xs font-semibold tracking-[1.4px] uppercase">
                    Recent Events
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <EventLog events={SAMPLE_EVENTS} />
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <footer className="mt-8 flex flex-wrap items-center justify-between gap-2 border-t border-border pt-4 text-[11px] text-muted-foreground">
          <p>
            Agent-LLM-Stage — LangGraph-based LLM evaluation agent.
          </p>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="font-mono">
              phase 3 preview
            </Badge>
            <Badge variant="ghost" className="font-mono">
              70 tests passing
            </Badge>
          </div>
        </footer>
      </main>
    </div>
  )
}

export default App
