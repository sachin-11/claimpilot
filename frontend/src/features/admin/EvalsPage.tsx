import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import Markdown from 'react-markdown'

import { errorMessage, getJson } from '@/api/client'
import type { EvalRunDetail, EvalRunSummary } from '@/api/types'
import { Badge, Card, EmptyState, ErrorState, SkeletonRows } from '@/components/ui'
import { cn } from '@/lib/cn'
import { formatDateTime, formatPercent, humanize } from '@/lib/format'

const RATIO_KEYS = new Set(['decision_accuracy', 'amount_within_tolerance', 'adversarial_pass_rate', 'error_rate', 'hit_at_3', 'mrr', 'faithfulness', 'citation_ok'])

function headline(key: string, value: number | null): string {
  if (value === null) return '—'
  if (RATIO_KEYS.has(key)) return formatPercent(value)
  if (key === 'cost_per_claim_usd') return `$${value.toFixed(4)}`
  if (key === 'latency_p95_s') return `${value.toFixed(1)} s`
  return String(value)
}

function Report({ id }: { id: string }) {
  const query = useQuery({
    queryKey: ['eval', id],
    queryFn: () => getJson<EvalRunDetail>(`/admin/evals/${encodeURIComponent(id)}`),
  })
  if (query.isLoading) return <SkeletonRows rows={8} />
  if (query.isError) return <ErrorState message={errorMessage(query.error)} onRetry={() => query.refetch()} />
  const run = query.data!
  if (!run.report_md) return <EmptyState title="This run has no report.md" />
  return (
    <article className="prose-report max-w-none leading-relaxed">
      <Markdown>{run.report_md}</Markdown>
    </article>
  )
}

export default function EvalsPage() {
  const list = useQuery({ queryKey: ['evals'], queryFn: () => getJson<EvalRunSummary[]>('/admin/evals') })
  const [selected, setSelected] = useState<string | null>(null)

  if (list.isLoading) return <SkeletonRows />
  if (list.isError) return <ErrorState message={errorMessage(list.error)} onRetry={() => list.refetch()} />
  const runs = list.data ?? []
  if (runs.length === 0) {
    return (
      <Card>
        <EmptyState
          title="No evaluation results yet"
          hint="Run `make eval-e2e` or `make eval-run`; the reports appear here."
        />
      </Card>
    )
  }
  const current = selected ?? runs.find((r) => r.has_report)?.id ?? null

  return (
    <div className="grid gap-4 lg:grid-cols-[22rem_1fr]">
      <Card className="h-fit max-h-[70vh] overflow-y-auto">
        <ul aria-label="Evaluation runs" className="divide-y divide-line">
          {runs.map((run) => (
            <li key={run.id}>
              <button
                type="button"
                onClick={() => setSelected(run.id)}
                aria-current={run.id === current}
                className={cn('block w-full px-4 py-3 text-left hover:bg-surface-2', run.id === current && 'bg-accent-soft')}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-medium">{run.name}</span>
                  <Badge tone={run.kind === 'e2e' ? 'accent' : 'info'}>{run.kind === 'e2e' ? 'End to end' : 'RAG'}</Badge>
                </div>
                <p className="text-xs text-muted">{formatDateTime(run.created_at)}</p>
                <p className="mt-1 flex flex-wrap gap-x-3 text-xs text-muted">
                  {Object.entries(run.headline)
                    .filter(([, v]) => v !== null && v !== undefined)
                    .slice(0, 4)
                    .map(([k, v]) => (
                      <span key={k}>
                        {humanize(k)} <strong className="text-fg">{headline(k, v)}</strong>
                      </span>
                    ))}
                </p>
              </button>
            </li>
          ))}
        </ul>
      </Card>
      <Card className="min-w-0 p-5">{current ? <Report id={current} /> : <EmptyState title="Pick a run" />}</Card>
    </div>
  )
}
