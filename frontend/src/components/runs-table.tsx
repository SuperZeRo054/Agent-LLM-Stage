import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { EvalRun } from "@/lib/data"
import { cn } from "@/lib/utils"

const STATUS_BADGE: Record<
  EvalRun["status"],
  { label: string; className: string }
> = {
  completed: {
    label: "completed",
    className:
      "border-transparent bg-secondary/20 text-secondary-foreground dark:bg-secondary/40",
  },
  running: {
    label: "running",
    className:
      "border-transparent bg-primary/15 text-primary dark:text-primary",
  },
  failed: {
    label: "failed",
    className: "border-transparent bg-destructive/15 text-destructive",
  },
}

interface RunsTableProps {
  runs: EvalRun[]
  className?: string
}

export function RunsTable({ runs, className }: RunsTableProps) {
  return (
    <Table className={className}>
      <TableHeader>
        <TableRow>
          <TableHead className="font-mono">Run ID</TableHead>
          <TableHead>Models</TableHead>
          <TableHead>Dataset</TableHead>
          <TableHead className="text-right">Cases</TableHead>
          <TableHead>Status</TableHead>
          <TableHead className="text-right">Score</TableHead>
          <TableHead className="text-right">Cost</TableHead>
          <TableHead>Created</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {runs.map((run) => {
          const badge = STATUS_BADGE[run.status]
          return (
            <TableRow key={run.id}>
              <TableCell className="font-mono text-xs">{run.id}</TableCell>
              <TableCell>
                <div className="flex flex-wrap gap-1">
                  {run.models.map((m) => (
                    <Badge key={m} variant="outline" className="font-mono text-[11px]">
                      {m}
                    </Badge>
                  ))}
                </div>
              </TableCell>
              <TableCell>{run.dataset}</TableCell>
              <TableCell className="text-right tabular-nums">
                {run.cases}
              </TableCell>
              <TableCell>
                <Badge className={cn("font-mono", badge.className)}>
                  {badge.label}
                </Badge>
              </TableCell>
              <TableCell
                className={cn(
                  "text-right font-mono tabular-nums",
                  run.score === null && "text-muted-foreground"
                )}
              >
                {run.score !== null ? `${run.score.toFixed(1)}%` : "--"}
              </TableCell>
              <TableCell
                className={cn(
                  "text-right font-mono tabular-nums",
                  run.cost === null && "text-muted-foreground"
                )}
              >
                {run.cost !== null ? `$${run.cost.toFixed(3)}` : "--"}
              </TableCell>
              <TableCell className="text-muted-foreground whitespace-nowrap">
                {run.created}
              </TableCell>
            </TableRow>
          )
        })}
      </TableBody>
    </Table>
  )
}
