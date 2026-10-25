import { useQuery } from '@tanstack/react-query'
import { ChevronLeft, ChevronRight, Plus, Search } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'

import { api, errorMessage, unwrap } from '@/api/client'
import type { ClaimStatus } from '@/api/types'
import { StatusChip } from '@/components/status'
import { Button, Card, EmptyState, ErrorState, Input, PageHeader, Select, SkeletonRows } from '@/components/ui'
import { formatDate, formatINR } from '@/lib/format'

const STATUSES: ClaimStatus[] = [
  'submitted',
  'processing',
  'awaiting_approval',
  'approved',
  'partially_approved',
  'rejected',
  'documents_requested',
  'failed',
]
const PAGE_SIZE = 20

export function ClaimsListPage() {
  const [params, setParams] = useSearchParams()
  const status = (params.get('status') ?? '') as ClaimStatus | ''
  const page = Math.max(1, Number(params.get('page') ?? 1) || 1)
  const q = params.get('q') ?? ''
  const [search, setSearch] = useState(q)

  // search as you type, but only hit the API once typing pauses
  useEffect(() => {
    const id = setTimeout(() => {
      if (search === q) return
      setParams((p) => {
        const next = new URLSearchParams(p)
        if (search) next.set('q', search)
        else next.delete('q')
        next.delete('page')
        return next
      })
    }, 300)
    return () => clearTimeout(id)
  }, [search, q, setParams])

  const query = useQuery({
    queryKey: ['claims', { status, page, q }],
    queryFn: () =>
      unwrap(
        api.GET('/claims', {
          params: { query: { status: status ? [status] : undefined, q: q || undefined, page, size: PAGE_SIZE } },
        }),
      ),
    placeholderData: (previous) => previous,
  })

  const setFilter = (key: string, value: string) =>
    setParams((p) => {
      const next = new URLSearchParams(p)
      if (value) next.set(key, value)
      else next.delete(key)
      next.delete('page')
      return next
    })
  const goToPage = (n: number) =>
    setParams((p) => {
      const next = new URLSearchParams(p)
      next.set('page', String(n))
      return next
    })

  const data = query.data
  const pages = data ? Math.max(1, Math.ceil(data.total / data.size)) : 1

  return (
    <>
      <PageHeader
        title="Claims"
        subtitle={data ? `${data.total} claim${data.total === 1 ? '' : 's'}` : undefined}
        actions={
          <Link to="/claims/new">
            <Button variant="primary">
              <Plus className="size-4" aria-hidden /> New claim
            </Button>
          </Link>
        }
      />
      <Card>
        <div className="flex flex-wrap items-center gap-3 border-b border-line p-3">
          <div className="relative w-full sm:w-64">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted" aria-hidden />
            <Input
              aria-label="Search by claim number"
              placeholder="Search claim number"
              className="pl-9"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <Select
            aria-label="Filter by status"
            className="w-full sm:w-52"
            value={status}
            onChange={(e) => setFilter('status', e.target.value)}
          >
            <option value="">All statuses</option>
            {STATUSES.map((s) => (
              <option key={s} value={s}>
                {s.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())}
              </option>
            ))}
          </Select>
        </div>

        {query.isLoading ? (
          <SkeletonRows className="p-3" />
        ) : query.isError ? (
          <ErrorState message={errorMessage(query.error)} onRetry={() => query.refetch()} />
        ) : data && data.items.length === 0 ? (
          <EmptyState
            title={q || status ? 'No claims match these filters' : 'No claims yet'}
            hint={q || status ? 'Try a different search or clear the filter.' : 'Upload the first claim to get started.'}
            action={
              !q && !status ? (
                <Link to="/claims/new">
                  <Button variant="primary">New claim</Button>
                </Link>
              ) : undefined
            }
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[40rem] text-left">
              <thead className="text-[12px] uppercase tracking-wide text-muted">
                <tr className="border-b border-line">
                  <th className="px-4 py-2.5 font-medium">Claim</th>
                  <th className="px-4 py-2.5 font-medium">Customer</th>
                  <th className="hidden px-4 py-2.5 font-medium md:table-cell">Hospital</th>
                  <th className="px-4 py-2.5 text-right font-medium">Claimed</th>
                  <th className="px-4 py-2.5 font-medium">Status</th>
                  <th className="hidden px-4 py-2.5 font-medium sm:table-cell">Created</th>
                </tr>
              </thead>
              <tbody>
                {data?.items.map((claim) => (
                  <tr key={claim.id} className="border-b border-line last:border-0 hover:bg-surface-2">
                    <td className="px-4 py-3">
                      <Link
                        to={`/claims/${claim.claim_number}`}
                        className="font-medium text-accent hover:underline"
                      >
                        {claim.claim_number}
                      </Link>
                      {claim.diagnosis_text && (
                        <p className="max-w-xs truncate text-xs text-muted">{claim.diagnosis_text}</p>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      {claim.customer_name ?? '—'}
                      {claim.customer_phone_masked && (
                        <p className="text-xs text-muted">{claim.customer_phone_masked}</p>
                      )}
                    </td>
                    <td className="hidden max-w-[14rem] truncate px-4 py-3 md:table-cell">{claim.hospital_name ?? '—'}</td>
                    <td className="px-4 py-3 text-right tabular-nums">{formatINR(claim.claimed_amount)}</td>
                    <td className="px-4 py-3">
                      <StatusChip status={claim.status} />
                    </td>
                    <td className="hidden whitespace-nowrap px-4 py-3 text-muted sm:table-cell">
                      {formatDate(claim.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {data && data.total > data.size && (
          <div className="flex items-center justify-between border-t border-line px-4 py-2.5">
            <span className="text-muted">
              Page {page} of {pages}
            </span>
            <div className="flex gap-2">
              <Button size="sm" disabled={page <= 1} onClick={() => goToPage(page - 1)} aria-label="Previous page">
                <ChevronLeft className="size-4" aria-hidden /> Prev
              </Button>
              <Button size="sm" disabled={page >= pages} onClick={() => goToPage(page + 1)} aria-label="Next page">
                Next <ChevronRight className="size-4" aria-hidden />
              </Button>
            </div>
          </div>
        )}
      </Card>
    </>
  )
}
