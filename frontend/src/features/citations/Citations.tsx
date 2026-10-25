import { BookOpen } from 'lucide-react'
import { createContext, type ReactNode, useCallback, useContext, useMemo, useState } from 'react'

import { Drawer } from '@/components/overlays'
import { Badge, ErrorState, SkeletonRows } from '@/components/ui'
import { useClause } from '@/features/claim-detail/queries'
import { errorMessage } from '@/api/client'
import { formatPages } from '@/lib/format'

interface CitationContext {
  open: (chunkId: string) => void
}
const Ctx = createContext<CitationContext | null>(null)

/** Provides one clause drawer for everything below it; any CitationChip opens it. */
export function CitationProvider({ children }: { children: ReactNode }) {
  const [chunkId, setChunkId] = useState<string | null>(null)
  const open = useCallback((id: string) => setChunkId(id), [])
  const value = useMemo(() => ({ open }), [open])
  return (
    <Ctx.Provider value={value}>
      {children}
      <ClauseDrawer chunkId={chunkId} onClose={() => setChunkId(null)} />
    </Ctx.Provider>
  )
}

export function useCitations(): CitationContext {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useCitations needs a CitationProvider')
  return ctx
}

export function CitationChip({
  chunkId,
  label,
  pages,
}: {
  chunkId: string
  label: string | null | undefined
  pages?: string | null
}) {
  const { open } = useCitations()
  const text = label?.trim() || 'Policy clause'
  return (
    <button
      type="button"
      onClick={() => open(chunkId)}
      title="Open the full clause"
      className="inline-flex max-w-full items-center gap-1.5 rounded-md border border-line bg-surface-2 px-2 py-1 text-left text-xs hover:border-accent hover:text-accent"
    >
      <BookOpen className="size-3.5 shrink-0" aria-hidden />
      <span className="truncate">{text}</span>
      {pages && <span className="shrink-0 text-muted">p. {pages}</span>}
    </button>
  )
}

function ClauseDrawer({ chunkId, onClose }: { chunkId: string | null; onClose: () => void }) {
  const query = useClause(chunkId)
  const clause = query.data
  return (
    <Drawer
      open={chunkId !== null}
      onClose={onClose}
      title={clause?.section_path ?? 'Policy clause'}
      subtitle={clause ? `${clause.policy_name} · ${formatPages(clause.page_start, clause.page_end)}` : undefined}
    >
      {query.isLoading ? (
        <SkeletonRows rows={5} />
      ) : query.isError ? (
        <ErrorState message={errorMessage(query.error)} onRetry={() => query.refetch()} />
      ) : clause ? (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-1.5">
            <Badge tone="accent">{clause.policy_code}</Badge>
            {clause.clauses.map((c) => (
              <Badge key={c}>Clause {c}</Badge>
            ))}
          </div>
          <p className="whitespace-pre-wrap leading-relaxed" data-testid="clause-text">
            {clause.content}
          </p>
        </div>
      ) : null}
    </Drawer>
  )
}
