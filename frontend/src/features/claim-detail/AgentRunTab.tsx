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

const TOOL_LABEL: Record<string, string> = {
  get_claim: 'Read the claim record',
  get_claim_extraction: 'Read the data extracted from the documents',
  get_customer_history: "Looked up the customer's earlier claims",
  list_claims: 'Listed related claims',
  list_policies: 'Listed the available policies',
  get_policy_rules: 'Loaded the policy rules',
  search_policy: 'Searched the policy wording',
  update_claim_status: 'Updated the claim status',
  save_decision_draft: 'Saved the draft decision',
  send_status_update: 'Notified the customer',
}

const RESULT_TONE: Record<string, 'ok' | 'bad' | 'warn' | 'neutral'> = {
  pass: 'ok',
  fail: 'bad',
  partial: 'warn',
  unclear: 'neutral',
}

/** "a=pass, b=fail" -> [["a","pass"],["b","fail"]]; null when the text is not in that shape. */
function parseResults(summary: string): Array<[string, string]> | null {
  const parts = summary.split(', ')
  const pairs = parts.map((p) => p.split('='))
  if (pairs.length === 0 || pairs.some((p) => p.length !== 2 || !(p[1] ?? '') || !(p[0] ?? ''))) return null
  return pairs.map(([k, v]) => [k as string, v as string])
}

interface ToolGroup {
  tool: string
  args: string
  count: number
}

/** Consecutive identical calls collapse into one row ("x2"). */
function groupTools(tools: TimelineStep['tools']): ToolGroup[] {
  const groups: ToolGroup[] = []
  for (const t of tools) {
    const last = groups.at(-1)
    if (last && last.tool === t.tool && last.args === t.args) last.count += 1
    else groups.push({ tool: t.tool, args: t.args, count: 1 })
  }
  return groups
}

function ToolRow({ group }: { group: ToolGroup }) {
  const query = /query=([^,]*)/.exec(group.args)?.[1]?.trim()
  const long = group.args.length > 90
  return (
    <li className="flex items-start gap-2 text-[13px] text-muted">
      <Wrench className="mt-1 size-3 shrink-0" aria-hidden />
      <div className="min-w-0">
        <span className="text-fg/80">{TOOL_LABEL[group.tool] ?? humanize(group.tool)}</span>
        {group.count > 1 && <span className="ml-1 text-xs">×{group.count}</span>}
        {query && <span className="ml-1.5 italic">“{query}”</span>}
        {(long || (group.args && !query)) && (
          <details className="mt-0.5 text-xs">
            <summary className="cursor-pointer select-none text-muted hover:text-fg">
              <code className="rounded bg-surface-2 px-1">{group.tool}</code> details
            </summary>
            <p className="mt-1 break-words rounded bg-surface-2 p-2 font-mono text-[11.5px] leading-relaxed">{group.args}</p>
          </details>
        )}
      </div>
    </li>
  )
}

function StepSummary({ summary }: { summary: string }) {
  const results = parseResults(summary)
  if (!results) return <p className="mt-1 text-fg/80">{summary}</p>
  return (
    <div className="mt-1.5 flex flex-wrap gap-1.5">
      {results.map(([name, value]) => (
        <Badge key={name} tone={RESULT_TONE[value] ?? 'neutral'}>
          {humanize(name)}: {humanize(value)}
        </Badge>
      ))}
    </div>
  )
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
        {step.summary && <StepSummary summary={step.summary} />}
        {step.routing && (
          <p className="mt-1 text-fg/80">
            Next:{' '}
            <strong>
              {(Array.isArray(step.routing.chosen) ? step.routing.chosen : [step.routing.chosen ?? 'nothing'])
                .map(humanize)
                .join(' + ')}
            </strong>
            <span className="text-muted">
              {' '}
              ·{' '}
              {step.routing.allowed.length <= 1
                ? 'the only step allowed'
                : step.routing.source === 'llm'
                  ? 'chosen by the model'
                  : 'chosen by the rules'}
            </span>
          </p>
        )}
        {step.tools.length > 0 && (
          <ul className="mt-2 space-y-1 border-l-2 border-line pl-3">
            {groupTools(step.tools).map((group, i) => (
              <ToolRow key={i} group={group} />
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
