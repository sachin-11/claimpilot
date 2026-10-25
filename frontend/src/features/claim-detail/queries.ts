import { useQuery } from '@tanstack/react-query'

import { api, ApiError, getJson, unwrap } from '@/api/client'
import type { RunSnapshot } from '@/api/types'

export const claimKey = (claimNumber: string) => ['claim', claimNumber] as const

export function useClaim(claimNumber: string) {
  return useQuery({
    queryKey: claimKey(claimNumber),
    queryFn: () =>
      unwrap(api.GET('/claims/{claim_number}', { params: { path: { claim_number: claimNumber } } })),
  })
}

/** 404 = "no extraction yet", which is a normal state, not an error. */
export function useExtraction(claimId: string | undefined, refetchMs: number | false = false) {
  return useQuery({
    queryKey: ['extraction', claimId],
    enabled: !!claimId,
    refetchInterval: refetchMs,
    queryFn: async () => {
      try {
        return await unwrap(
          api.GET('/claims/{claim_id}/extraction', { params: { path: { claim_id: claimId! } } }),
        )
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null
        throw error
      }
    },
  })
}

export const runKey = (claimId: string | undefined, runId: string | null | undefined) =>
  ['run', claimId, runId] as const

export function useRunSnapshot(claimId: string | undefined, runId: string | null | undefined) {
  return useQuery({
    queryKey: runKey(claimId, runId),
    enabled: !!claimId && !!runId,
    queryFn: () => getJson<RunSnapshot>(`/claims/${claimId}/runs/${runId}`),
    // while the graph is still working the snapshot is refreshed by the stream's onFinished
    refetchInterval: (query) => (query.state.data?.status === 'running' ? 4000 : false),
  })
}

export function useClause(chunkId: string | null) {
  return useQuery({
    queryKey: ['clause', chunkId],
    enabled: !!chunkId,
    staleTime: Infinity, // a chunk is immutable
    queryFn: () =>
      unwrap(api.GET('/policies/chunks/{chunk_id}', { params: { path: { chunk_id: chunkId! } } })),
  })
}
