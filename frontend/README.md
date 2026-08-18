# Agent-LLM-Stage Frontend

LLM evaluation dashboard for [Agent-LLM-Stage](../README.md) — a
LangGraph-based LLM evaluation agent.

Built with React 19 + TypeScript + Vite + Tailwind CSS v4 + shadcn/ui
(Base UI) + [beUI](https://beui.dev) motion components. Themed after the
ClickHouse design system: dark neon cockpit, oversized stats, terminal-grade
contrast.

## Stack

- **shadcn/ui** (Base UI) — tabs, table, card, badge, button
- **beUI** — `tilt-card` (3D perspective tilt + cursor-tracked glare)
- **Motion** — chart animations (bar growth, sparkline draw-in)
- **Theme** — dark default with light mode toggle (`d` key or header button)

## Develop

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # production bundle in dist/
npm run lint
npm run typecheck
```

## Add components

```bash
npx shadcn@latest add button
npx shadcn@latest add @beui/tilt-card   # beUI registry items
```

## Structure

```
src/
├── components/
│   ├── motion/tilt-card.tsx   # beUI tilt card
│   ├── ui/                    # shadcn/ui primitives
│   ├── charts.tsx             # SVG bar chart + sparkline
│   ├── runs-table.tsx         # evaluation runs table
│   ├── event-log.tsx          # activity event log
│   └── stat-card.tsx          # tilt KPI card
├── lib/
│   ├── data.ts                # sample data + aggregations
│   ├── ease.ts                # beUI motion tokens
│   └── hooks/use-hover-capable.ts
├── App.tsx                    # dashboard layout
└── main.tsx
```

## Design

OpenDesign binding (`.open-design.json` at repo root):
`clickhouse` design system + `flowai-live-dashboard-template` workflow.
Dark canvas (`oklch(0.13 …)`), neon yellow-green primary, charcoal borders,
Inter/Geist heavy weights.
