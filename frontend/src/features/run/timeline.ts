import type { RunEvent, RunSnapshot } from '@/api/types'

export type StepStatus = 'running' | 'done' | 'error'

export interface ToolCall {
  tool: string
  args: string
  at: number
}

export interface TimelineStep {
  /** unique per execution (a node can run more than once, e.g. the supervisor) */
  key: string
  node: string
  status: StepStatus
  startedAt: number
  endedAt: number | null
  summary: string
  tools: ToolCall[]
  /** supervisor routing: what was allowed / chosen and whether the LLM or the rules decided */
  routing: { allowed: string[]; chosen: string[] | string | null; source: string } | null
  model: string | null
  fallback: boolean
}

export interface Timeline {
  steps: TimelineStep[]
  /** the run has paused for a human, finished, or failed */
  state: 'running' | 'awaiting_approval' | 'completed' | 'failed' | 'idle'
  error: string | null
}

const when = (event: RunEvent): number => {
  const ts = event.data.ts
  if (typeof ts === 'string') {
    const parsed = Date.parse(ts)
    if (!Number.isNaN(parsed)) return parsed
  }
  return event.receivedAt
}

/** `{claim_number: "CLM-014"}` -> `claim_number=CLM-014` (the API sends the arguments as an object). */
export function formatArgs(args: unknown): string {
  if (args === null || args === undefined) return ''
  if (typeof args !== 'object') return String(args)
  return Object.entries(args as Record<string, unknown>)
    .map(([key, value]) => `${key}=${typeof value === 'object' ? JSON.stringify(value) : String(value)}`)
    .join(', ')
}

function asList(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : []
}

/**
 * Folds the SSE events into rows, in the order the nodes started. Pure, so the same function renders
 * a live run and a replay. Coverage and risk run in parallel: `node_completed` closes the open row
 * of that node name, not "the last row".
 */
export function buildTimeline(events: RunEvent[]): Timeline {
  const steps: TimelineStep[] = []
  let state: Timeline['state'] = events.length ? 'running' : 'idle'
  let error: string | null = null

  const openRow = (node: string): TimelineStep | undefined =>
    [...steps].reverse().find((s) => s.node === node && s.status === 'running')

  const open = (node: string, at: number): TimelineStep => {
    const step: TimelineStep = {
      key: `${node}-${steps.length}`,
      node,
      status: 'running',
      startedAt: at,
      endedAt: null,
      summary: '',
      tools: [],
      routing: null,
      model: null,
      fallback: false,
    }
    steps.push(step)
    return step
  }

  for (const event of events) {
    const at = when(event)
    const data = event.data
    switch (event.event) {
      case 'node_started':
        open(String(data.node), at)
        break
      case 'tool_called': {
        const agent = String(data.agent ?? '')
        const row = openRow(agent) ?? [...steps].reverse().find((s) => s.status === 'running')
        row?.tools.push({ tool: String(data.tool), args: formatArgs(data.args), at })
        break
      }
      case 'routing': {
        const row = openRow('supervisor') ?? open('supervisor', at)
        const chosen = data.chosen
        row.routing = {
          allowed: asList(data.allowed),
          chosen: Array.isArray(chosen) ? asList(chosen) : chosen == null ? null : String(chosen),
          source: String(data.source ?? ''),
        }
        break
      }
      case 'node_completed': {
        const node = String(data.node)
        const row = openRow(node) ?? open(node, at)
        row.status = 'done'
        row.endedAt = at
        row.summary = String(data.summary ?? '')
        break
      }
      case 'awaiting_approval':
        state = 'awaiting_approval'
        break
      case 'completed':
        state = 'completed'
        break
      case 'error': {
        state = 'failed'
        const errors = Array.isArray(data.errors) ? (data.errors as Array<Record<string, string>>) : []
        error = errors.map((e) => e.message ?? e.error ?? JSON.stringify(e)).join('; ') || 'The run failed'
        for (const step of steps) if (step.status === 'running') step.status = 'error'
        break
      }
    }
  }
  // the approval node stays open while the run waits for a human
  if (state === 'awaiting_approval') {
    for (const step of steps) if (step.status === 'running' && step.node === 'approval') step.status = 'done'
  }
  return { steps, state, error }
}

export const durationMs = (step: TimelineStep): number | null =>
  step.endedAt === null ? null : Math.max(0, step.endedAt - step.startedAt)

/** Adds the model that answered per node (from the final run snapshot) and marks the fallback. */
export function withModels(timeline: Timeline, snapshot: RunSnapshot | null | undefined): Timeline {
  if (!snapshot) return timeline
  const fallback = snapshot.fallback_model
  return {
    ...timeline,
    steps: timeline.steps.map((step) => {
      const model = snapshot.models_used[step.node] ?? null
      return { ...step, model, fallback: !!model && !!fallback && model === fallback }
    }),
  }
}

/** A finished run opened from scratch (no events kept): one row per completed step. */
export function timelineFromSnapshot(snapshot: RunSnapshot): Timeline {
  const steps: TimelineStep[] = snapshot.completed_steps.map((node, i) => ({
    key: `${node}-${i}`,
    node,
    status: 'done',
    startedAt: 0,
    endedAt: null,
    summary: '',
    tools: [],
    routing: null,
    model: null,
    fallback: false,
  }))
  const state: Timeline['state'] =
    snapshot.status === 'awaiting_approval'
      ? 'awaiting_approval'
      : snapshot.status === 'completed'
        ? 'completed'
        : snapshot.status === 'failed'
          ? 'failed'
          : 'running'
  return withModels(
    {
      steps,
      state,
      error: snapshot.errors.map((e) => e.message ?? e.error ?? JSON.stringify(e)).join('; ') || null,
    },
    snapshot,
  )
}
