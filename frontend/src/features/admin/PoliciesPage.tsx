import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { RefreshCw } from 'lucide-react'

import { api, errorMessage, unwrap } from '@/api/client'
import { toast } from '@/components/toast'
import { Badge, Button, Card, EmptyState, ErrorState, SkeletonRows } from '@/components/ui'
import { formatDateTime } from '@/lib/format'

const statusTone = { success: 'ok', failed: 'bad', running: 'info' } as const

export default function PoliciesPage() {
  const queryClient = useQueryClient()
  const query = useQuery({
    queryKey: ['policies'],
    queryFn: () => unwrap(api.GET('/policies')),
    refetchInterval: (q) => (q.state.data?.some((p) => p.last_run?.status === 'running') ? 3000 : false),
  })
  const ingest = useMutation({
    mutationFn: (code: string) =>
      unwrap(api.POST('/policies/ingest', { body: { policy_code: code, all: false, force: false } })),
    onSuccess: (_, code) => {
      toast.info(`Ingestion of ${code} started`)
      void queryClient.invalidateQueries({ queryKey: ['policies'] })
    },
    onError: (error) => toast.error(errorMessage(error)),
  })

  if (query.isLoading) return <SkeletonRows />
  if (query.isError) return <ErrorState message={errorMessage(query.error)} onRetry={() => query.refetch()} />
  const policies = query.data ?? []
  if (policies.length === 0) return <Card><EmptyState title="No policies" hint="Run the seed to load the policy catalogue." /></Card>

  return (
    <Card className="overflow-x-auto">
      <table className="w-full min-w-[44rem] text-left">
        <thead className="text-[12px] uppercase tracking-wide text-muted">
          <tr className="border-b border-line">
            <th className="px-4 py-2.5 font-medium">Policy</th>
            <th className="px-4 py-2.5 font-medium">Wording</th>
            <th className="px-4 py-2.5 text-right font-medium">Chunks</th>
            <th className="px-4 py-2.5 font-medium">Last ingestion</th>
            <th className="px-4 py-2.5" />
          </tr>
        </thead>
        <tbody>
          {policies.map((p) => (
            <tr key={p.policy_code} className="border-b border-line last:border-0">
              <td className="px-4 py-3">
                <p className="font-medium">{p.name}</p>
                <p className="text-xs text-muted">
                  {p.policy_code} · v{p.version}
                  {!p.is_active && ' · inactive'}
                </p>
              </td>
              <td className="px-4 py-3 text-muted">{p.source_pdf ?? <Badge tone="warn">No PDF ingested</Badge>}</td>
              <td className="px-4 py-3 text-right tabular-nums">{p.chunk_count}</td>
              <td className="px-4 py-3">
                {p.last_run ? (
                  <div>
                    <Badge tone={statusTone[p.last_run.status]}>{p.last_run.status}</Badge>
                    <p className="mt-0.5 text-xs text-muted">
                      {formatDateTime(p.last_run.finished_at ?? p.last_run.started_at)} · {p.last_run.chunks_total} chunks
                    </p>
                    {p.last_run.error && <p className="text-xs text-bad">{p.last_run.error}</p>}
                  </div>
                ) : (
                  <span className="text-muted">Never</span>
                )}
              </td>
              <td className="px-4 py-3 text-right">
                <Button
                  size="sm"
                  loading={ingest.isPending && ingest.variables === p.policy_code}
                  disabled={p.last_run?.status === 'running'}
                  onClick={() => ingest.mutate(p.policy_code)}
                >
                  <RefreshCw className="size-3.5" aria-hidden /> Ingest
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}
