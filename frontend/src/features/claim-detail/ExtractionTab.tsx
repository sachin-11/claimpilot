import { Crosshair } from 'lucide-react'

import { errorMessage } from '@/api/client'
import type { ExtractionData, ExtractionResponse } from '@/api/types'
import { SeverityBadge } from '@/components/status'
import { Badge, EmptyState, ErrorState, SkeletonRows } from '@/components/ui'
import { cn } from '@/lib/cn'
import { formatDate, formatINR, formatPercent, humanize } from '@/lib/format'

import type { DocumentTarget } from './DocumentViewer'

interface FieldDef {
  key: keyof ExtractionData
  label: string
  format?: (value: unknown) => string
}

const money = (v: unknown) => formatINR(v as string | number | null)
const date = (v: unknown) => formatDate(v as string | null)

const FIELDS: FieldDef[] = [
  { key: 'patient_name', label: 'Patient' },
  { key: 'patient_age', label: 'Age' },
  { key: 'gender', label: 'Gender' },
  { key: 'policy_number', label: 'Policy number' },
  { key: 'hospital_name', label: 'Hospital' },
  { key: 'admission_date', label: 'Admitted', format: date },
  { key: 'discharge_date', label: 'Discharged', format: date },
  { key: 'length_of_stay_days', label: 'Length of stay (nights)' },
  { key: 'diagnosis_text', label: 'Diagnosis' },
  { key: 'icd10_code', label: 'ICD-10' },
  { key: 'procedure', label: 'Procedure' },
  { key: 'room_type', label: 'Room type' },
  { key: 'room_rent_per_day', label: 'Room rent / day', format: money },
  { key: 'icu_days', label: 'ICU days' },
  { key: 'bill_subtotal', label: 'Bill subtotal', format: money },
  { key: 'taxes', label: 'Taxes', format: money },
  { key: 'bill_total', label: 'Bill total', format: money },
]

const confidenceTone = { high: 'bg-ok', medium: 'bg-warn', low: 'bg-bad' } as const

export function ExtractionTab({
  extraction,
  loading,
  error,
  onRetry,
  onJump,
}: {
  extraction: ExtractionResponse | null | undefined
  loading: boolean
  error: unknown
  onRetry: () => void
  onJump: (target: Omit<DocumentTarget, 'nonce'>) => void
}) {
  if (loading) return <SkeletonRows />
  if (error) return <ErrorState message={errorMessage(error)} onRetry={onRetry} />
  if (!extraction) {
    return (
      <EmptyState
        title="Not extracted yet"
        hint="The fields appear here once the documents have been read. Run the agents to extract and decide."
      />
    )
  }
  const data = extraction.extraction as ExtractionData
  const sources = data.field_sources ?? {}
  const confidence = data.field_confidence ?? {}
  const items = data.line_items ?? []
  const issues = extraction.consistency_issues
  const problems = [
    ...extraction.missing_documents.map((m) => `Missing document: ${humanize(m)}`),
    ...extraction.validation_errors,
  ]

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2 text-muted">
        <span>Version {extraction.version}</span>
        <span>·</span>
        <span>Confidence {formatPercent(extraction.confidence, 0)}</span>
        {extraction.model_used && (
          <>
            <span>·</span>
            <Badge>{extraction.model_used}</Badge>
          </>
        )}
      </div>

      {(problems.length > 0 || issues.length > 0) && (
        <section aria-label="Issues" className="space-y-2">
          {problems.map((p) => (
            <p key={p} className="rounded-md bg-warn-soft px-3 py-2 text-warn">
              {p}
            </p>
          ))}
          {issues.map((issue, i) => (
            <div key={i} className="flex items-start gap-2 rounded-md border border-line px-3 py-2">
              <SeverityBadge severity={issue.severity} />
              <p>
                <span className="font-medium">{humanize(issue.type)}.</span> {issue.detail}
              </p>
            </div>
          ))}
        </section>
      )}

      <dl className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
        {FIELDS.map(({ key, label, format }) => {
          const raw = data[key]
          if (raw === null || raw === undefined || raw === '') return null
          const level = confidence[key as string]
          const source = sources[key as string]
          return (
            <div key={key} className="min-w-0">
              <dt className="flex items-center gap-1.5 text-xs text-muted">
                {level && (
                  <span
                    className={cn('size-2 rounded-full', confidenceTone[level])}
                    title={`${humanize(level)} confidence`}
                    aria-label={`${level} confidence`}
                    role="img"
                  />
                )}
                {label}
              </dt>
              <dd className="flex items-baseline gap-2">
                <span className="min-w-0 break-words font-medium">{format ? format(raw) : String(raw)}</span>
                {source?.page ? (
                  <button
                    type="button"
                    className="inline-flex shrink-0 items-center gap-1 text-xs text-accent hover:underline"
                    onClick={() => onJump({ docType: source.doc_type, page: source.page })}
                    aria-label={`Show ${label} in ${humanize(source.doc_type)}, page ${source.page}`}
                  >
                    <Crosshair className="size-3" aria-hidden />
                    {humanize(source.doc_type)} p. {source.page}
                  </button>
                ) : null}
              </dd>
            </div>
          )
        })}
      </dl>

      {items.length > 0 && (
        <section>
          <h3 className="mb-2 font-semibold">Bill line items</h3>
          <div className="overflow-x-auto rounded-md border border-line">
            <table className="w-full min-w-[28rem] text-left">
              <thead className="bg-surface-2 text-xs uppercase tracking-wide text-muted">
                <tr>
                  <th className="px-3 py-2 font-medium">Item</th>
                  <th className="px-3 py-2 font-medium">Category</th>
                  <th className="px-3 py-2 text-right font-medium">Qty</th>
                  <th className="px-3 py-2 text-right font-medium">Amount</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item, i) => (
                  <tr key={i} className="border-t border-line">
                    <td className="px-3 py-2">{item.description}</td>
                    <td className="px-3 py-2 text-muted">{humanize(item.category)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{item.quantity ?? '—'}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{formatINR(item.amount)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
