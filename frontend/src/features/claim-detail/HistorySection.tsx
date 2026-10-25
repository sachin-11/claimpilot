import type { HistoryItem } from '@/api/types'
import { DecisionBadge } from '@/components/status'
import { Badge, Card } from '@/components/ui'
import { formatDateTime, formatINR, humanize } from '@/lib/format'

const ACTION_LABEL: Record<string, string> = { approved: 'Approved', edited: 'Edited', rejected: 'Rejected' }

export function HistorySection({ history }: { history: HistoryItem[] }) {
  return (
    <section className="mt-6" aria-labelledby="history-title">
      <h2 id="history-title" className="mb-2 text-base font-semibold">
        Decision history
      </h2>
      <Card>
        {history.length === 0 ? (
          <p className="px-4 py-6 text-center text-muted">No decisions have been recorded for this claim yet.</p>
        ) : (
          <ul className="divide-y divide-line">
            {history.map((item) => (
              <li key={item.decision_id} className="flex flex-wrap items-start gap-x-6 gap-y-2 px-4 py-3">
                <div className="min-w-[10rem]">
                  <DecisionBadge decision={item.decision} />
                  <p className="mt-1 text-xs text-muted">{formatDateTime(item.created_at)}</p>
                  {item.model_used && <p className="text-xs text-muted">{item.model_used}</p>}
                </div>
                <div>
                  <p className="text-xs text-muted">Draft payable</p>
                  <p className="tabular-nums">{formatINR(item.payable_amount)}</p>
                </div>
                <div className="min-w-0 flex-1">
                  {item.approval ? (
                    <>
                      <p>
                        <Badge tone={item.approval.action === 'rejected' ? 'bad' : 'ok'}>
                          {ACTION_LABEL[item.approval.action] ?? humanize(item.approval.action)}
                        </Badge>{' '}
                        by <strong>{item.approval.reviewer_name}</strong> on {formatDateTime(item.approval.created_at)}
                        {item.approval.final_amount && <> · {formatINR(item.approval.final_amount)}</>}
                      </p>
                      {item.approval.comment && <p className="mt-1 text-muted">“{item.approval.comment}”</p>}
                    </>
                  ) : (
                    <p className="text-muted">No human decision yet.</p>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </section>
  )
}
