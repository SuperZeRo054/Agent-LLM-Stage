import type { ReactNode } from "react"

import { TiltCard } from "@/components/motion/tilt-card"
import { cn } from "@/lib/utils"

interface StatCardProps {
  label: string
  value: string
  delta: string
  accent?: boolean
  icon?: ReactNode
  className?: string
}

export function StatCard({
  label,
  value,
  delta,
  accent = false,
  icon,
  className,
}: StatCardProps) {
  return (
    <TiltCard
      max={8}
      glare
      className={cn(
        "rounded-lg border border-border bg-card text-card-foreground",
        className
      )}
    >
      <div className="p-5">
        <div className="flex items-center justify-between">
          <p className="text-[11px] font-semibold tracking-[1.4px] text-muted-foreground uppercase">
            {label}
          </p>
          {icon ? (
            <span className="text-muted-foreground [&_svg]:size-4">{icon}</span>
          ) : null}
        </div>
        <p
          className={cn(
            "mt-2 text-4xl font-black tracking-tight tabular-nums",
            accent ? "text-primary" : "text-foreground"
          )}
        >
          {value}
        </p>
        <p className="mt-1 text-xs text-muted-foreground">{delta}</p>
      </div>
    </TiltCard>
  )
}
