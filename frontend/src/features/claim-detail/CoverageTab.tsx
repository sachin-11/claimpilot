import type { CoverageCheck } from '@/api/types'
import { ResultBadge } from '@/components/status'
import { EmptyState } from '@/components/ui'
import { CitationChip } from '@/features/citations/Citations'
import { humanize } from '@/lib/format'

export const CHECK_LABEL: Record<string, string> = {
  initial_waiting: 'Initial waiting period',
  specific_disease_waiting: 'Specific disease waiting period',
  pre_existing: 'Pre-existing condition',
  exclusion: 'Exclusions',
  room_rent: 'Room rent limit',
  icu_limit: 'ICU limit',
  sub_limit: 'Procedure sub-limit',
  co_pay: 'Co-payment',
  other: 'Other',
}

const RESULT_BAR: Record<string, string> = {
  pass: 'border-l-ok',
  fail: 'border-l-bad',
  partial: 'border-l-warn',
  unclear: 'border-l-line',
}

export function CoverageTab({ coverage }: { coverage: CoverageCheck[] | null | undefined }) {
  if (!coverage || coverage.length === 0) {
    return (
      <EmptyState
        title="No coverage checks yet"
        hint="They appear after the coverage agent has compared the claim with the policy wording."
      />
    )
  }
  const applicable = coverage.filter((c) => c.applies)
  const skipped = coverage.filter((c) => !c.applies)
  return (
    <div className="space-y-4">
      <ul className="space-y-3">
        {applicable.map((check) => (
          <li
            key={check.check}
            data-testid="coverage-row"
            className={`rounded-lg border border-line border-l-4 bg-surface p-4 ${RESULT_BAR[check.result] ?? 'border-l-line'}`}
          >
            <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-semibold">{CHECK_LABEL[check.check] ?? humanize(check.check)}</h3>
              <ResultBadge result={check.result} />
            </div>
            <p className="leading-relaxed text-fg/90">{check.explanation}</p>
            {check.cited_clauses.length > 0 ? (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {check.cited_clauses.map((clause) => (
                  <CitationChip
                    key={clause.chunk_id}
                    chunkId={clause.chunk_id}
                    label={clause.section_path}
                    pages={clause.pages}
                  />
                ))}
              </div>
            ) : (
              <p className="mt-2 text-xs text-muted">No policy clause was cited for this check.</p>
            )}
          </li>
        ))}
      </ul>
      {skipped.length > 0 && (
        <p className="text-muted">
          Not relevant for this claim: {skipped.map((c) => CHECK_LABEL[c.check] ?? humanize(c.check)).join(', ')}.
        </p>
      )}
    </div>
  )
}
