import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Play } from 'lucide-react'
import { useCallback, useId, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { api, errorMessage, unwrap } from '@/api/client'
import type { ClaimDetail, RunSnapshot } from '@/api/types'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { StatusChip } from '@/components/status'
import { Tabs, tabPanelProps } from '@/components/tabs'
import { toast } from '@/components/toast'
import { DecisionBadge } from '@/components/status'
import { Button, Card, EmptyState, ErrorState, Skeleton } from '@/components/ui'
import { CitationProvider } from '@/features/citations/Citations'
import { buildTimeline, timelineFromSnapshot, withModels } from '@/features/run/timeline'
import { useRunStream } from '@/features/run/useRunStream'
import { formatDate, formatINR } from '@/lib/format'

import { AgentRunTab } from './AgentRunTab'
import { ApprovalPanel } from './ApprovalPanel'
import { CoverageTab } from './CoverageTab'
import { DecisionTab } from './DecisionTab'
import { DocumentViewer, type DocumentTarget } from './DocumentViewer'
import { ExtractionTab } from './ExtractionTab'
import { HistorySection } from './HistorySection'
import { claimKey, runKey, useClaim, useExtraction, useRunSnapshot } from './queries'
import { RiskTab } from './RiskTab'

type TabId = 'extraction' | 'run' | 'coverage' | 'risk' | 'decision'

function ClaimHeader({ claim }: { claim: ClaimDetail }) {
  const facts: Array<[string, string]> = [
    ['Customer', claim.customer?.name ?? '—'],
    ['Policy', claim.policy_name ? `${claim.policy_name} (${claim.policy_code})` : (claim.policy_code ?? '—')],
    ['Hospital', claim.hospital_name ?? '—'],
    ['Stay', claim.admission_date ? `${formatDate(claim.admission_date)} → ${formatDate(claim.discharge_date)}` : '—'],
    ['Diagnosis', claim.diagnosis_text ?? '—'],
  ]
  return (
    <>
      <div className="sticky top-[3.3rem] z-20 -mx-4 mb-4 border-b border-line bg-bg/95 px-4 py-2 backdrop-blur sm:-mx-6 sm:px-6">
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-1">
          <div className="flex min-w-0 items-center gap-3">
            <Link to="/claims" className="rounded p-1 text-muted hover:bg-surface-2 hover:text-fg" aria-label="All claims">
              <ArrowLeft className="size-4" aria-hidden />
            </Link>
            <h1 className="text-lg font-semibold tracking-tight">{claim.claim_number}</h1>
            <StatusChip status={claim.status} />
            <span className="hidden text-muted md:inline">{claim.customer?.name ?? ''}</span>
          </div>
          <dl className="flex items-baseline gap-5">
            <div className="flex items-baseline gap-1.5">
              <dt className="text-xs text-muted">Claimed</dt>
              <dd className="font-semibold tabular-nums">{formatINR(claim.claimed_amount)}</dd>
            </div>
            <div className="flex items-baseline gap-1.5">
              <dt className="text-xs text-muted">Approved</dt>
              <dd className="font-semibold tabular-nums" data-testid="approved-amount">
                {formatINR(claim.approved_amount)}
              </dd>
            </div>
          </dl>
        </div>
      </div>
      <Card className="mb-4 p-4">
        <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2 xl:grid-cols-5">
          {facts.map(([label, value]) => (
            <div key={label} className="min-w-0">
              <dt className="text-xs text-muted">{label}</dt>
              <dd className="truncate" title={value}>
                {value}
              </dd>
            </div>
          ))}
        </dl>
        <p className="mt-2 text-xs text-muted">Created {formatDate(claim.created_at)}</p>
      </Card>
    </>
  )
}

function SummaryStrip({ run, onOpen }: { run: RunSnapshot; onOpen: (tab: TabId) => void }) {
  const decision = run.decision
  if (!decision) return null
  const applicable = (run.coverage ?? []).filter((c) => c.applies)
  const passed = applicable.filter((c) => c.result === 'pass').length
  const score = run.risk?.score ?? null
  const claimed = Number(decision.claimed_amount)
  const payable = Number(decision.payable_amount)
  const cut = claimed > 0 ? Math.round(((claimed - payable) / claimed) * 100) : 0
  const riskTone = score === null ? '' : score >= 50 ? 'text-bad' : score >= 25 ? 'text-warn' : 'text-ok'
  const tile = 'rounded-lg border border-line bg-surface p-3 text-left transition-colors hover:bg-surface-2'
  return (
    <div className="mb-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4" data-testid="summary-strip">
      <button type="button" className={tile} onClick={() => onOpen('decision')}>
        <p className="text-xs text-muted">Draft decision</p>
        <div className="mt-1.5">
          <DecisionBadge decision={decision.decision} />
        </div>
      </button>
      <button type="button" className={tile} onClick={() => onOpen('decision')}>
        <p className="text-xs text-muted">Payable</p>
        <p className="text-xl font-semibold tabular-nums text-accent">{formatINR(payable)}</p>
        <p className="text-xs text-muted">{cut > 0 ? `${cut}% below the ${formatINR(claimed)} claimed` : 'Full amount claimed'}</p>
      </button>
      <button type="button" className={tile} onClick={() => onOpen('coverage')}>
        <p className="text-xs text-muted">Coverage checks</p>
        <p className="text-xl font-semibold tabular-nums">
          {passed}
          <span className="text-base font-normal text-muted"> of {applicable.length} passed</span>
        </p>
      </button>
      <button type="button" className={tile} onClick={() => onOpen('risk')}>
        <p className="text-xs text-muted">Risk score</p>
        <p className={`text-xl font-semibold tabular-nums ${riskTone}`}>
          {score ?? '—'}
          <span className="text-base font-normal text-muted"> / 100</span>
        </p>
      </button>
    </div>
  )
}

function ClaimScreen({ claim }: { claim: ClaimDetail }) {
  const queryClient = useQueryClient()
  const tabsId = useId()
  const [tab, setTab] = useState<TabId>('extraction')
  const [startedRun, setStartedRun] = useState<string | null>(null)
  const [target, setTarget] = useState<DocumentTarget | null>(null)
  const runId = startedRun ?? claim.latest_run_id

  const refreshAll = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: claimKey(claim.claim_number) })
    void queryClient.invalidateQueries({ queryKey: runKey(claim.id, runId) })
    void queryClient.invalidateQueries({ queryKey: ['extraction', claim.id] })
    void queryClient.invalidateQueries({ queryKey: ['claims'] })
  }, [queryClient, claim.claim_number, claim.id, runId])

  const stream = useRunStream(claim.id, runId, refreshAll)
  const snapshot = useRunSnapshot(claim.id, runId)
  const extraction = useExtraction(claim.id, claim.status === 'processing' ? 4000 : false)

  const timeline = useMemo(() => {
    if (stream.timeline.steps.length > 0) return withModels(stream.timeline, snapshot.data)
    if (snapshot.data) return timelineFromSnapshot(snapshot.data)
    return buildTimeline([])
  }, [stream.timeline, snapshot.data])

  const start = useMutation({
    mutationFn: () => unwrap(api.POST('/claims/{claim_id}/process', { params: { path: { claim_id: claim.id } } })),
    onSuccess: (result) => {
      setStartedRun(result.run_id)
      setTab('run')
      void queryClient.invalidateQueries({ queryKey: claimKey(claim.claim_number) })
    },
    onError: (error) => toast.error(errorMessage(error)),
  })

  const run = snapshot.data
  const awaiting = claim.status === 'awaiting_approval' && run?.status === 'awaiting_approval' && run.decision
  const canStart =
    ['submitted', 'failed', 'documents_requested'].includes(claim.status) &&
    !(runId && timeline.state === 'running' && stream.connection !== 'failed' && claim.status === 'processing')

  const tabs = [
    { id: 'extraction' as const, label: 'Extraction' },
    { id: 'run' as const, label: 'Agent run' },
    { id: 'coverage' as const, label: 'Coverage' },
    { id: 'risk' as const, label: 'Risk' },
    { id: 'decision' as const, label: 'Decision' },
  ]

  return (
    <CitationProvider>
      <ClaimHeader claim={claim} />
      {run && <SummaryStrip run={run} onOpen={setTab} />}

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="min-w-0 xl:sticky xl:top-[6.5rem] xl:h-[calc(100vh-7.5rem)] xl:self-start">
          <ErrorBoundary what="the document viewer">
            <DocumentViewer claimId={claim.id} documents={claim.documents} target={target} />
          </ErrorBoundary>
        </div>

        <div className="min-w-0 space-y-4">
          {awaiting && run && runId && (
            <ApprovalPanel
              claimId={claim.id}
              runId={runId}
              decision={run.decision!}
              claimedAmount={Number(claim.claimed_amount)}
              uploadedByMe={claim.uploaded_by_me}
              onDone={refreshAll}
            />
          )}
          <Card>
            <div className="sticky top-[6.5rem] z-10 flex flex-wrap items-center justify-between gap-2 rounded-t-lg bg-surface pr-3">
              <Tabs tabs={tabs} value={tab} onChange={setTab} label="Claim details" className="flex-1 border-b-0" />
              {canStart && (
                <Button variant="primary" size="sm" loading={start.isPending} onClick={() => start.mutate()}>
                  <Play className="size-3.5" aria-hidden /> {runId ? 'Run agents again' : 'Run agents'}
                </Button>
              )}
            </div>
            <div {...tabPanelProps(tabsId)} className="border-t border-line p-4">
              {tab === 'extraction' && (
                <ExtractionTab
                  extraction={extraction.data}
                  loading={extraction.isLoading}
                  error={extraction.error}
                  onRetry={() => extraction.refetch()}
                  onJump={(t) => setTarget({ ...t, nonce: Date.now() })}
                />
              )}
              {tab === 'run' && <AgentRunTab timeline={timeline} connection={stream.connection} hasRun={!!runId} />}
              {tab === 'coverage' && <CoverageTab coverage={run?.coverage} />}
              {tab === 'risk' && <RiskTab risk={run?.risk} />}
              {tab === 'decision' && <DecisionTab decision={run?.decision} />}
            </div>
          </Card>

        </div>
      </div>

      <HistorySection history={claim.history} />
    </CitationProvider>
  )
}

export default function ClaimDetailPage() {
  const { claimNumber = '' } = useParams()
  const query = useClaim(claimNumber)

  if (query.isLoading) {
    return (
      <div className="space-y-4" role="status" aria-label="Loading claim">
        <Skeleton className="h-28 w-full" />
        <div className="grid gap-4 xl:grid-cols-2">
          <Skeleton className="h-[30rem]" />
          <Skeleton className="h-[30rem]" />
        </div>
      </div>
    )
  }
  if (query.isError) {
    const notFound = (query.error as { status?: number }).status === 404
    return notFound ? (
      <EmptyState
        title="Claim not found"
        hint={`There is no claim ${claimNumber}.`}
        action={
          <Link to="/claims">
            <Button>Back to claims</Button>
          </Link>
        }
      />
    ) : (
      <ErrorState message={errorMessage(query.error)} onRetry={() => query.refetch()} />
    )
  }
  return <ClaimScreen claim={query.data!} />
}
