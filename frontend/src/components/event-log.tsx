import type { EvalEvent } from "@/lib/data"
import { cn } from "@/lib/utils"

const DOT_CLASS: Record<EvalEvent["type"], string> = {
  ok: "bg-secondary",
  warn: "bg-primary",
  err: "bg-destructive",
}

interface EventLogProps {
  events: EvalEvent[]
  className?: string
}

export function EventLog({ events, className }: EventLogProps) {
  return (
    <div
      className={cn("flex max-h-[280px] flex-col gap-2 overflow-y-auto", className)}
      role="log"
      aria-label="Recent evaluation events"
    >
      {events.map((event, i) => (
        <div
          key={i}
          className="flex items-start gap-2.5 rounded-md border border-border p-2.5"
        >
          <span
            className={cn(
              "mt-1.5 size-2 shrink-0 rounded-full",
              DOT_CLASS[event.type]
            )}
            aria-hidden
          />
          <div className="min-w-0">
            <p className="text-xs leading-relaxed text-foreground">
              {event.msg}
            </p>
            <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">
              {event.ts}
            </p>
          </div>
        </div>
      ))}
    </div>
  )
}
