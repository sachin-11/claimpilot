import type { DecisionDraft } from '@/api/types'
import { DecisionBadge } from '@/components/status'
import { EmptyState } from '@/components/ui'
import { CitationChip } from '@/features/citations/Citations'
import { formatINR, humanize } from '@/lib/format'

export function DecisionTab({ decision }: { decision: DecisionDraft | null | undefined }) {
  if (!decision) {
    return <EmptyState title="No decision yet" hint="The draft decision appears after coverage and risk are done." />
  }
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="mb-1 text-xs text-muted">Draft decision</p>
          <DecisionBadge decision={decision.decision} />
        </div>
        <dl className="flex gap-6 text-right">
          <div>
            <dt className="text-xs text-muted">Claimed</dt>
            <dd className="text-lg font-semibold tabular-nums">{formatINR(decision.claimed_amount)}</dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Payable</dt>
            <dd className="text-lg font-semibold tabular-nums text-accent" data-testid="payable">
              {formatINR(decision.payable_amount)}
            </dd>
          </div>
        </dl>
      </div>

      {decision.deductions.length > 0 && (
        <section>
          <h3 className="mb-2 font-semibold">Deductions</h3>
          <div className="overflow-x-auto rounded-md border border-line">
            <table className="w-full min-w-[24rem] text-left">
              <thead className="bg-surface-2 text-xs uppercase tracking-wide text-muted">
                <tr>
                  <th className="px-3 py-2 font-medium">Rule</th>
                  <th className="px-3 py-2 font-medium">What changed</th>
                  <th className="px-3 py-2 text-right font-medium">Amount</th>
                </tr>
              </thead>
              <tbody>
                {decision.deductions.map((d, i) => (
                  <tr key={i} className="border-t border-line">
                    <td className="whitespace-nowrap px-3 py-2 font-medium">{humanize(d.rule)}</td>
                    <td className="px-3 py-2">{d.description}</td>
                    <td className="whitespace-nowrap px-3 py-2 text-right tabular-nums">−{formatINR(d.amount)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {decision.reasons.length > 0 && (
        <section>
          <h3 className="mb-2 font-semibold">Reasons</h3>
          <ul className="list-disc space-y-1.5 pl-5 leading-relaxed">
            {decision.reasons.map((reason, i) => (
              <li key={i}>{reason}</li>
            ))}
          </ul>
        </section>
      )}

      {decision.cited_clauses.length > 0 && (
        <section>
          <h3 className="mb-2 font-semibold">Policy clauses behind this decision</h3>
          <div className="flex flex-wrap gap-1.5">
            {decision.cited_clauses.map((c) => (
              <CitationChip key={c.chunk_id} chunkId={c.chunk_id} label={c.section_path} pages={c.pages} />
            ))}
          </div>
        </section>
      )}

      {decision.risk_flags.length > 0 && (
        <p className="text-muted">Risk flags considered: {decision.risk_flags.map(humanize).join(', ')}.</p>
      )}
    </div>
  )
}
