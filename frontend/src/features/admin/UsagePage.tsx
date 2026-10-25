import { useQuery } from '@tanstack/react-query'

import { errorMessage, getJson } from '@/api/client'
import type { MetricsSummary, UsageResponse } from '@/api/types'
import { Card, ErrorState, SkeletonRows } from '@/components/ui'
import { formatINR, formatPercent, humanize } from '@/lib/format'

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Card className="p-4">
      <p className="text-xs text-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums">{value}</p>
      {hint && <p className="mt-0.5 text-xs text-muted">{hint}</p>}
    </Card>
  )
}

function Bars({ rows, format }: { rows: Array<{ label: string; value: number }>; format: (n: number) => string }) {
  const max = Math.max(1, ...rows.map((r) => r.value))
  return (
    <ul className="space-y-2">
      {rows.map((row) => (
        <li key={row.label} className="grid grid-cols-[8rem_1fr_5rem] items-center gap-3">
          <span className="truncate text-muted">{row.label}</span>
          <div className="h-2.5 rounded-full bg-surface-2">
            <div className="h-full rounded-full bg-accent" style={{ width: `${(row.value / max) * 100}%` }} />
          </div>
          <span className="text-right tabular-nums">{format(row.value)}</span>
        </li>
      ))}
    </ul>
  )
}

export default function UsagePage() {
  const summary = useQuery({
    queryKey: ['admin-metrics'],
    queryFn: () => getJson<MetricsSummary>('/admin/metrics/summary'),
    refetchInterval: 30_000,
  })
  const usage = useQuery({ queryKey: ['admin-usage'], queryFn: () => getJson<UsageResponse>('/admin/usage?days=7') })

  if (summary.isLoading || usage.isLoading) return <SkeletonRows />
  if (summary.isError || usage.isError) {
    return (
      <ErrorState
        message={errorMessage(summary.error ?? usage.error)}
        onRetry={() => {
          void summary.refetch()
          void usage.refetch()
        }}
      />
    )
  }
  const s = summary.data!
  const u = usage.data!
  const providers = Object.entries(s.calls_by_provider)

  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat
          label="Spend today"
          value={formatINR(s.spend_inr)}
          hint={s.budget_inr > 0 ? `${s.budget_used_percent ?? 0}% of ${formatINR(s.budget_inr)} budget` : 'No daily budget set'}
        />
        <Stat label="LLM calls today" value={String(s.llm_calls)} />
        <Stat
          label="Fallback model share"
          value={formatPercent(s.fallback_rate)}
          hint={`${s.fallback_calls} call${s.fallback_calls === 1 ? '' : 's'} answered by the fallback`}
        />
        <Stat label="Agent runs today" value={String(Object.values(s.runs_today).reduce((a, b) => a + b, 0))} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card className="p-4">
          <h2 className="mb-3 font-semibold">Spend, last 7 days</h2>
          <Bars
            rows={[...u.history].reverse().map((d) => ({ label: d.day, value: d.inr }))}
            format={(n) => formatINR(n)}
          />
          <p className="mt-3 text-xs text-muted">{u.note}</p>
        </Card>
        <Card className="p-4">
          <h2 className="mb-3 font-semibold">Claims by status</h2>
          <Bars
            rows={Object.entries(s.claims_by_status).map(([k, v]) => ({ label: humanize(k), value: v }))}
            format={(n) => String(n)}
          />
        </Card>
        <Card className="p-4">
          <h2 className="mb-3 font-semibold">Calls by provider today</h2>
          {providers.length === 0 ? (
            <p className="text-muted">No LLM calls yet today.</p>
          ) : (
            <Bars rows={providers.map(([k, v]) => ({ label: k, value: v }))} format={(n) => String(n)} />
          )}
        </Card>
        <Card className="p-4">
          <h2 className="mb-3 font-semibold">Agent runs today</h2>
          {Object.keys(s.runs_today).length === 0 ? (
            <p className="text-muted">No runs yet today.</p>
          ) : (
            <Bars rows={Object.entries(s.runs_today).map(([k, v]) => ({ label: humanize(k), value: v }))} format={(n) => String(n)} />
          )}
        </Card>
      </div>
    </div>
  )
}
