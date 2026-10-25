import { AlertTriangle, CheckCircle2, Circle, Loader2, Wifi, WifiOff, Wrench } from 'lucide-react'

import { Badge, EmptyState } from '@/components/ui'
import type { ConnectionState } from '@/features/run/useRunStream'
import { durationMs, type Timeline, type TimelineStep } from '@/features/run/timeline'
import { cn } from '@/lib/cn'
import { formatDuration, formatTime, humanize } from '@/lib/format'

const NODE_LABEL: Record<string, string> = {
  intake: 'Intake · read the documents',
  supervisor: 'Supervisor · choose the next step',
  coverage: 'Coverage · check the policy wording',
  risk: 'Risk · look for red flags',
  decision: 'Decision · draft the outcome',
  approval: 'Approval · waiting for a person',
  finalize: 'Finalize · notify the customer',
  error: 'Error handling',
}

function StepIcon({ step }: { step: TimelineStep }) {
  if (step.status === 'running') return <Loader2 className="size-5 animate-spin text-info" aria-label="Running" />
  if (step.status === 'error') return <AlertTriangle className="size-5 text-bad" aria-label="Failed" />
  return <CheckCircle2 className="size-5 text-ok" aria-label="Done" />
}

export function ConnectionBanner({ connection }: { connection: ConnectionState }) {
  if (connection === 'reconnecting') {
    return (
      <p role="status" className="flex items-center gap-2 rounded-md bg-warn-soft px-3 py-2 text-warn">
        <WifiOff className="size-4" aria-hidden /> Connection lost. Reconnecting…
      </p>
    )
  }
  if (connection === 'failed') {
    return (
      <p role="alert" className="flex items-center gap-2 rounded-md bg-bad-soft px-3 py-2 text-bad">
        <WifiOff className="size-4" aria-hidden /> Live updates stopped. Reload the page to continue watching.
      </p>
    )
  }
  if (connection === 'live') {
    return (
      <p className="flex items-center gap-2 text-xs text-muted">
        <Wifi className="size-3.5 text-ok" aria-hidden /> Live
      </p>
    )
  }
  return null
}

function Step({ step, last }: { step: TimelineStep; last: boolean }) {
  const duration = durationMs(step)
  return (
    <li className="relative flex gap-3 pb-5" data-testid="timeline-step" data-node={step.node} data-status={step.status}>
      {!last && <span className="absolute left-[9px] top-6 h-[calc(100%-1.5rem)] w-px bg-line" aria-hidden />}
      <span className="mt-0.5 shrink-0 bg-surface">
        <StepIcon step={step} />
      </span>
      <div className="min-w-0 flex-1 animate-fade-up">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-medium">{NODE_LABEL[step.node] ?? humanize(step.node)}</span>
          {step.model && (
            <Badge tone={step.fallback ? 'warn' : 'neutral'} title={step.fallback ? 'The fallback model answered' : undefined}>
              {step.model}
              {step.fallback ? ' · fallback' : ''}
            </Badge>
          )}
          <span className="ml-auto flex items-center gap-2 text-xs tabular-nums text-muted">
            {step.startedAt > 0 && <time>{formatTime(new Date(step.startedAt).toISOString())}</time>}
            {duration !== null && <span>{formatDuration(duration)}</span>}
          </span>
        </div>
        {step.summary && <p className="mt-0.5 text-muted">{step.summary}</p>}
        {step.routing && (
          <p className="mt-0.5 text-muted">
            Chose{' '}
            <strong className="text-fg">
              {Array.isArray(step.routing.chosen) ? step.routing.chosen.join(' + ') : (step.routing.chosen ?? 'nothing')}
            </strong>{' '}
            from {step.routing.allowed.join(', ') || 'no options'} ({step.routing.source === 'llm' ? 'model' : 'rules'})
          </p>
        )}
        {step.tools.length > 0 && (
          <ul className="mt-1.5 space-y-1">
            {step.tools.map((tool, i) => (
              <li key={i} className="flex items-start gap-1.5 text-xs text-muted">
                <Wrench className="mt-0.5 size-3 shrink-0" aria-hidden />
                <span className="min-w-0 break-words">
                  <code className="rounded bg-surface-2 px-1">{tool.tool}</code>
                  {tool.args && <span className="ml-1.5">{tool.args}</span>}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </li>
  )
}

export function AgentRunTab({
  timeline,
  connection,
  hasRun,
}: {
  timeline: Timeline
  connection: ConnectionState
  hasRun: boolean
}) {
  if (!hasRun) {
    return (
      <EmptyState
        icon={<Circle className="size-8" aria-hidden />}
        title="The agents have not run yet"
        hint="Use “Run agents” above to read the documents, check coverage and risk, and draft a decision."
      />
    )
  }
  return (
    <div className="space-y-4">
      <ConnectionBanner connection={connection} />
      {timeline.error && (
        <p role="alert" className="rounded-md bg-bad-soft px-3 py-2 text-bad">
          The run failed: {timeline.error}
        </p>
      )}
      {timeline.steps.length === 0 ? (
        <p className="text-muted" role="status">
          Waiting for the first step…
        </p>
      ) : (
        <ol aria-label="Agent run timeline" aria-live="polite">
          {timeline.steps.map((step, i) => (
            <Step key={step.key} step={step} last={i === timeline.steps.length - 1} />
          ))}
        </ol>
      )}
      {timeline.state === 'awaiting_approval' && (
        <p className={cn('rounded-md bg-warn-soft px-3 py-2 text-warn')} role="status">
          The agents are done. A reviewer has to approve, edit or reject the draft decision.
        </p>
      )}
    </div>
  )
}
