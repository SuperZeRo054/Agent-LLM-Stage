import { motion, useReducedMotion } from "motion/react"

import { EASE_OUT } from "@/lib/ease"
import { cn } from "@/lib/utils"

interface BarDatum {
  label: string
  value: number
}

interface BarChartProps {
  data: BarDatum[]
  unit?: string
  colorVar?: string
  className?: string
  labelClassName?: string
}

export function BarChart({
  data,
  unit = "%",
  colorVar = "var(--chart-1)",
  className,
  labelClassName,
}: BarChartProps) {
  const reduce = useReducedMotion()
  const max = Math.max(...data.map((d) => d.value), 1)

  return (
    <div
      className={cn("flex h-full w-full items-end gap-1.5", className)}
      role="img"
      aria-label={`Bar chart: ${data.map((d) => `${d.label} ${d.value}${unit}`).join(", ")}`}
    >
      {data.map((d, i) => {
        const heightPct = (d.value / max) * 100
        return (
          <div
            key={`${d.label}-${i}`}
            className="group flex h-full min-w-0 flex-1 flex-col items-center justify-end gap-1.5"
          >
            <span className="text-[10px] font-semibold text-foreground opacity-0 transition-opacity group-hover:opacity-100">
              {d.value}
              {unit}
            </span>
            <motion.div
              initial={reduce ? false : { height: 0 }}
              whileInView={{ height: `${heightPct}%` }}
              viewport={{ once: true }}
              transition={{ duration: 0.7, delay: i * 0.05, ease: EASE_OUT }}
              className="w-full min-w-2 rounded-t-[4px]"
              style={{ backgroundColor: colorVar }}
            />
            <span
              className={cn(
                "w-full truncate text-center text-[10px] text-muted-foreground",
                labelClassName
              )}
            >
              {d.label}
            </span>
          </div>
        )
      })}
    </div>
  )
}

interface SparklineProps {
  data: BarDatum[]
  unit?: string
  className?: string
}

export function Sparkline({ data, unit = "%", className }: SparklineProps) {
  const reduce = useReducedMotion()
  const w = 360
  const h = 120
  const pad = 16
  const min = Math.min(...data.map((d) => d.value))
  const max = Math.max(...data.map((d) => d.value))
  const range = Math.max(max - min, 1)

  const points = data.map((d, i) => {
    const x = pad + (i / Math.max(data.length - 1, 1)) * (w - pad * 2)
    const y = h - pad - ((d.value - min) / range) * (h - pad * 2)
    return { x, y, ...d }
  })

  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
    .join(" ")
  const areaPath = `${path} L ${points.at(-1)?.x.toFixed(1)} ${h - pad} L ${points[0].x.toFixed(1)} ${h - pad} Z`

  return (
    <div className={className}>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="h-full w-full"
        role="img"
        aria-label={`Trend chart: ${data.map((d) => `${d.label} ${d.value}${unit}`).join(", ")}`}
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id="spark-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--chart-1)" stopOpacity="0.25" />
            <stop offset="100%" stopColor="var(--chart-1)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <motion.path
          d={areaPath}
          fill="url(#spark-fill)"
          initial={reduce ? false : { opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
        />
        <motion.path
          d={path}
          fill="none"
          stroke="var(--chart-1)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={reduce ? false : { pathLength: 0 }}
          whileInView={{ pathLength: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1, ease: EASE_OUT }}
        />
        {points.map((p) => (
          <circle
            key={p.label}
            cx={p.x}
            cy={p.y}
            r="3"
            fill="var(--chart-1)"
          >
            <title>{`${p.label}: ${p.value}${unit}`}</title>
          </circle>
        ))}
      </svg>
      <div className="mt-1 flex justify-between">
        {points.map((p) => (
          <span key={p.label} className="text-[10px] text-muted-foreground">
            {p.label}
          </span>
        ))}
      </div>
    </div>
  )
}
