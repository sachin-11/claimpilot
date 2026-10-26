import { ChevronLeft, ChevronRight, ScanLine } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { Suspense, useEffect, useMemo, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'

import { API_BASE, authFetch, errorMessage } from '@/api/client'
import type { DocumentOut } from '@/api/types'
import { Badge, Button, Card, EmptyState, ErrorState, Skeleton } from '@/components/ui'
import { cn } from '@/lib/cn'
import { humanize } from '@/lib/format'

pdfjs.GlobalWorkerOptions.workerSrc = new URL('pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url).toString()

/** Uploaded files are stored under their hash: show what the document is instead of 64 hex characters. */
export function documentLabel(doc: DocumentOut): string {
  if (!/^[0-9a-f]{32,}(\.\w+)?$/i.test(doc.file_name)) return doc.file_name
  const pages = doc.page_count ? ` · ${doc.page_count} page${doc.page_count === 1 ? '' : 's'}` : ''
  return `${humanize(doc.doc_type)}${pages}`
}

export interface DocumentTarget {
  docType: string
  page: number | null
  /** changes on every jump request so jumping to the same page twice still works */
  nonce: number
}

/** The file is fetched with the bearer token (an <img>/<iframe> src cannot send it), then shown from a blob URL. */
function useDocumentBlob(claimId: string, doc: DocumentOut | undefined) {
  const query = useQuery({
    queryKey: ['document-file', claimId, doc?.id],
    enabled: !!doc,
    staleTime: Infinity,
    gcTime: 5 * 60_000,
    queryFn: async () => {
      const response = await authFetch(new Request(`${API_BASE}/claims/${claimId}/documents/${doc!.id}/file`))
      if (!response.ok) throw new Error(`The file could not be loaded (${response.status})`)
      return response.blob()
    },
  })
  const url = useMemo(() => (query.data ? URL.createObjectURL(query.data) : null), [query.data])
  useEffect(
    () => () => {
      if (url) URL.revokeObjectURL(url)
    },
    [url],
  )
  return {
    url,
    loading: query.isLoading,
    error: query.isError ? errorMessage(query.error) : null,
  }
}

/** Left column of the claim screen: one tab per uploaded document, page navigation, jump targets. */
export function DocumentViewer({
  claimId,
  documents,
  target,
}: {
  claimId: string
  documents: DocumentOut[]
  target: DocumentTarget | null
}) {
  const [index, setIndex] = useState(0)
  const [page, setPage] = useState(1)
  const [pages, setPages] = useState<number | null>(null)
  const doc = documents[index]
  const blob = useDocumentBlob(claimId, doc)
  const [width, setWidth] = useState(520)
  const [container, setContainer] = useState<HTMLDivElement | null>(null)

  // follow "show me where this came from" requests from the extraction tab (adjusting state while
  // rendering, guarded by the request's nonce, instead of an effect that sets state)
  const [handled, setHandled] = useState(0)
  if (target && target.nonce !== handled) {
    setHandled(target.nonce)
    const found = documents.findIndex((d) => d.doc_type === target.docType)
    if (found >= 0) {
      setIndex(found)
      setPage(target.page ?? 1)
      setPages(null)
    }
  }

  useEffect(() => {
    if (!container) return
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.max(280, Math.floor(entry.contentRect.width) - 2))
    })
    observer.observe(container)
    return () => observer.disconnect()
  }, [container])

  const isPdf = doc?.mime_type === 'application/pdf'
  const total = pages ?? doc?.page_count ?? 1
  const options = useMemo(() => ({}), [])

  if (documents.length === 0) {
    return (
      <Card>
        <EmptyState title="No documents" hint="This claim has no uploaded documents." />
      </Card>
    )
  }

  return (
    <Card className="flex h-full min-h-[28rem] flex-col overflow-hidden">
      <div role="tablist" aria-label="Documents" className="flex gap-1 overflow-x-auto border-b border-line px-2 pt-2">
        {documents.map((d, i) => (
          <button
            key={d.id}
            role="tab"
            type="button"
            aria-selected={i === index}
            onClick={() => {
              setIndex(i)
              setPage(1)
              setPages(null)
            }}
            className={cn(
              'flex items-center gap-1.5 whitespace-nowrap rounded-t-md border-b-2 px-3 py-2 font-medium',
              i === index ? 'border-accent text-fg' : 'border-transparent text-muted hover:text-fg',
            )}
          >
            {humanize(d.doc_type)}
            {d.is_scanned && (
              <Badge tone="warn" title="Scanned: read with OCR">
                <ScanLine className="size-3" aria-hidden /> Scanned
              </Badge>
            )}
          </button>
        ))}
      </div>

      <div ref={setContainer} className="min-h-0 flex-1 overflow-auto bg-surface-2 p-2" data-testid="document-pane">
        {blob.loading && <Skeleton className="h-[32rem] w-full" />}
        {blob.error && <ErrorState message={blob.error} />}
        {/* react-pdf suspends while pdf.js starts: keep that local, or the whole page is hidden */}
        {blob.url && isPdf && (
          <Suspense fallback={<Skeleton className="h-[32rem] w-full" />}>
            <Document
              file={blob.url}
              options={options}
              loading={<Skeleton className="h-[32rem] w-full" />}
              error={<ErrorState message="This PDF could not be displayed." />}
              onLoadSuccess={({ numPages }) => {
                setPages(numPages)
                setPage((p) => Math.min(p, numPages))
              }}
            >
              <Page pageNumber={page} width={width} className="mx-auto shadow" />
            </Document>
          </Suspense>
        )}
        {blob.url && !isPdf && (
          <img src={blob.url} alt={doc?.file_name} className="mx-auto max-w-full rounded shadow" />
        )}
      </div>

      <div className="flex items-center justify-between gap-2 border-t border-line px-3 py-2">
        <span className="min-w-0 truncate text-muted" title={doc?.file_name}>
          {doc ? documentLabel(doc) : ''}
        </span>
        {isPdf && (
          <div className="flex items-center gap-1">
            <Button size="sm" variant="ghost" disabled={page <= 1} onClick={() => setPage(page - 1)} aria-label="Previous page">
              <ChevronLeft className="size-4" aria-hidden />
            </Button>
            <span className="whitespace-nowrap tabular-nums" aria-live="polite">
              Page {page} / {total}
            </span>
            <Button size="sm" variant="ghost" disabled={page >= total} onClick={() => setPage(page + 1)} aria-label="Next page">
              <ChevronRight className="size-4" aria-hidden />
            </Button>
          </div>
        )}
      </div>
    </Card>
  )
}
