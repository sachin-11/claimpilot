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
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full min-w-[34rem] text-left">
          <thead className="bg-surface-2 text-xs uppercase tracking-wide text-muted">
            <tr>
              <th className="px-3 py-2 font-medium">Check</th>
              <th className="px-3 py-2 font-medium">Result</th>
              <th className="px-3 py-2 font-medium">Why, and where in the policy</th>
            </tr>
          </thead>
          <tbody>
            {applicable.map((check) => (
              <tr key={check.check} className="border-t border-line align-top" data-testid="coverage-row">
                <td className="whitespace-nowrap px-3 py-3 font-medium">{CHECK_LABEL[check.check] ?? humanize(check.check)}</td>
                <td className="px-3 py-3">
                  <ResultBadge result={check.result} />
                </td>
                <td className="space-y-2 px-3 py-3">
                  <p>{check.explanation}</p>
                  {check.cited_clauses.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
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
                    <p className="text-xs text-muted">No policy clause was cited for this check.</p>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {skipped.length > 0 && (
        <p className="text-muted">
          Not relevant for this claim: {skipped.map((c) => CHECK_LABEL[c.check] ?? humanize(c.check)).join(', ')}.
        </p>
      )}
    </div>
  )
}
