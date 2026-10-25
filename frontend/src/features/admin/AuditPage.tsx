import { useQuery } from '@tanstack/react-query'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useState } from 'react'

import { errorMessage, getJson } from '@/api/client'
import type { AuditLogPage } from '@/api/types'
import { Badge, Button, Card, EmptyState, ErrorState, Input, Select, SkeletonRows } from '@/components/ui'
import { formatDateTime } from '@/lib/format'

const PAGE = 25

export default function AuditPage() {
  const [entityType, setEntityType] = useState('')
  const [entityId, setEntityId] = useState('')
  const [page, setPage] = useState(0)
  const [expanded, setExpanded] = useState<string | null>(null)

  const query = useQuery({
    queryKey: ['audit', entityType, entityId, page],
    queryFn: () => {
      const params = new URLSearchParams({ limit: String(PAGE), offset: String(page * PAGE) })
      if (entityType) params.set('entity_type', entityType)
      if (entityId.trim()) params.set('entity_id', entityId.trim())
      return getJson<AuditLogPage>(`/admin/audit-logs?${params}`)
    },
    placeholderData: (previous) => previous,
  })
  const total = query.data?.total ?? 0
  const pages = Math.max(1, Math.ceil(total / PAGE))

  return (
    <Card>
      <div className="flex flex-wrap items-center gap-3 border-b border-line p-3">
        <Select
          aria-label="Entity type"
          className="w-full sm:w-44"
          value={entityType}
          onChange={(e) => {
            setEntityType(e.target.value)
            setPage(0)
          }}
        >
          <option value="">All entities</option>
          <option value="claim">Claim</option>
          <option value="user">User</option>
          <option value="policy">Policy</option>
        </Select>
        <Input
          aria-label="Entity id"
          className="w-full sm:w-72"
          placeholder="Entity id (claim UUID)"
          value={entityId}
          onChange={(e) => {
            setEntityId(e.target.value)
            setPage(0)
          }}
        />
        <span className="ml-auto text-muted">{total} entries</span>
      </div>
      {query.isLoading ? (
        <SkeletonRows className="p-3" />
      ) : query.isError ? (
        <ErrorState message={errorMessage(query.error)} onRetry={() => query.refetch()} />
      ) : query.data && query.data.items.length === 0 ? (
        <EmptyState title="No audit entries" hint="Nothing matches these filters." />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[44rem] text-left">
            <thead className="text-[12px] uppercase tracking-wide text-muted">
              <tr className="border-b border-line">
                <th className="px-4 py-2.5 font-medium">When</th>
                <th className="px-4 py-2.5 font-medium">Who</th>
                <th className="px-4 py-2.5 font-medium">Action</th>
                <th className="px-4 py-2.5 font-medium">Entity</th>
                <th className="px-4 py-2.5" />
              </tr>
            </thead>
            <tbody>
              {query.data?.items.map((entry) => (
                <>
                  <tr key={entry.id} className="border-b border-line align-top" data-testid="audit-row">
                    <td className="whitespace-nowrap px-4 py-2.5 text-muted">{formatDateTime(entry.created_at)}</td>
                    <td className="px-4 py-2.5">
                      <Badge>{entry.actor_type}</Badge>
                      {entry.actor_id && <span className="ml-1.5 text-xs text-muted">{entry.actor_id.slice(0, 18)}</span>}
                    </td>
                    <td className="px-4 py-2.5 font-medium">{entry.action}</td>
                    <td className="px-4 py-2.5">
                      {entry.entity_type}
                      <span className="ml-1.5 text-xs text-muted">{entry.entity_id.slice(0, 8)}</span>
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <button
                        type="button"
                        className="text-xs text-accent hover:underline"
                        aria-expanded={expanded === entry.id}
                        onClick={() => setExpanded(expanded === entry.id ? null : entry.id)}
                      >
                        {expanded === entry.id ? 'Hide' : 'Details'}
                      </button>
                    </td>
                  </tr>
                  {expanded === entry.id && (
                    <tr key={`${entry.id}-d`} className="border-b border-line bg-surface-2">
                      <td colSpan={5} className="px-4 py-3">
                        <pre className="overflow-x-auto whitespace-pre-wrap text-xs">{JSON.stringify(entry.details, null, 2)}</pre>
                        {entry.request_id && <p className="mt-1 text-xs text-muted">request {entry.request_id}</p>}
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {total > PAGE && (
        <div className="flex items-center justify-between border-t border-line px-4 py-2.5">
          <span className="text-muted">
            Page {page + 1} of {pages}
          </span>
          <div className="flex gap-2">
            <Button size="sm" disabled={page === 0} onClick={() => setPage(page - 1)} aria-label="Previous page">
              <ChevronLeft className="size-4" aria-hidden /> Prev
            </Button>
            <Button size="sm" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)} aria-label="Next page">
              Next <ChevronRight className="size-4" aria-hidden />
            </Button>
          </div>
        </div>
      )}
    </Card>
  )
}
