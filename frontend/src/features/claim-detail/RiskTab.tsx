import type { RiskAssessment } from '@/api/types'
import { SeverityBadge } from '@/components/status'
import { Badge, EmptyState } from '@/components/ui'
import { humanize } from '@/lib/format'

/** Review threshold of the backend (RISK_REVIEW_THRESHOLD); only used to colour the gauge. */
const REVIEW_THRESHOLD = 50

export function RiskGauge({ score }: { score: number }) {
  const clamped = Math.max(0, Math.min(100, score))
  const tone = clamped >= REVIEW_THRESHOLD ? 'bg-bad' : clamped >= 25 ? 'bg-warn' : 'bg-ok'
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-3xl font-semibold tabular-nums">{clamped}</span>
        <span className="text-xs text-muted">of 100 · review from {REVIEW_THRESHOLD}</span>
      </div>
      <div
        role="meter"
        aria-label="Risk score"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={clamped}
        className="relative h-2.5 overflow-hidden rounded-full bg-surface-2"
      >
        <div className={`h-full ${tone} transition-[width] duration-500`} style={{ width: `${clamped}%` }} />
        <span className="absolute inset-y-0 w-px bg-fg/40" style={{ left: `${REVIEW_THRESHOLD}%` }} aria-hidden />
      </div>
    </div>
  )
}

export function RiskTab({ risk }: { risk: RiskAssessment | null | undefined }) {
  if (!risk) {
    return <EmptyState title="No risk assessment yet" hint="It appears after the risk agent has run." />
  }
  return (
    <div className="space-y-5">
      <RiskGauge score={risk.score} />
      {risk.summary && <p className="leading-relaxed">{risk.summary}</p>}
      {risk.flags.length === 0 ? (
        <p className="rounded-md bg-ok-soft px-3 py-2 text-ok">No risk flags were raised.</p>
      ) : (
        <ul className="space-y-2" aria-label="Risk flags">
          {risk.flags.map((flag) => (
            <li key={flag.code} className="flex items-start gap-3 rounded-md border border-line px-3 py-2.5">
              <SeverityBadge severity={flag.severity} />
              <div className="min-w-0">
                <p className="font-medium">
                  {humanize(flag.code.replace(/^llm_/, ''))}
                  {flag.source === 'llm' && (
                    <Badge className="ml-2" tone="info">
                      model-suggested
                    </Badge>
                  )}
                </p>
                <p className="text-muted">{flag.detail}</p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
