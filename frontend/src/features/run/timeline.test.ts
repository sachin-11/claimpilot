import { describe, expect, it } from 'vitest'

import type { RunEvent, RunEventName, RunSnapshot } from '@/api/types'

import { buildTimeline, durationMs, formatArgs, timelineFromSnapshot, withModels } from './timeline'

let clock = Date.parse('2026-10-25T10:00:00Z')
const ev = (event: RunEventName, data: Record<string, unknown> = {}, secondsLater = 1): RunEvent => {
  clock += secondsLater * 1000
  return { event, data: { ...data, ts: new Date(clock).toISOString() }, receivedAt: clock }
}

describe('buildTimeline', () => {
  it('lists steps in the order they started, with durations', () => {
    const t = buildTimeline([
      ev('node_started', { node: 'intake' }),
      ev('tool_called', { agent: 'intake', tool: 'get_claim', args: 'claim_number=CLM-014' }),
      ev('node_completed', { node: 'intake', summary: 'policy POL-AS; 0 missing document(s)' }, 3),
      ev('node_started', { node: 'supervisor' }),
      ev('routing', { allowed: ['coverage', 'risk'], chosen: ['coverage', 'risk'], source: 'rules' }),
      ev('node_completed', { node: 'supervisor', summary: 'next: coverage, risk' }),
    ])
    expect(t.steps.map((s) => s.node)).toEqual(['intake', 'supervisor'])
    expect(t.steps[0]).toMatchObject({ status: 'done', summary: 'policy POL-AS; 0 missing document(s)' })
    expect(durationMs(t.steps[0]!)).toBe(4000)
    expect(t.steps[0]?.tools).toEqual([expect.objectContaining({ tool: 'get_claim' })])
    expect(t.steps[1]?.routing).toEqual({ allowed: ['coverage', 'risk'], chosen: ['coverage', 'risk'], source: 'rules' })
    expect(t.state).toBe('running')
  })

  it('closes the right row when coverage and risk run in parallel', () => {
    const t = buildTimeline([
      ev('node_started', { node: 'coverage' }),
      ev('node_started', { node: 'risk' }),
      ev('tool_called', { agent: 'coverage', tool: 'search_policy', args: 'q' }),
      ev('node_completed', { node: 'risk', summary: 'score 10' }),
      ev('node_completed', { node: 'coverage', summary: 'room_rent=partial' }),
    ])
    expect(t.steps.map((s) => [s.node, s.status])).toEqual([
      ['coverage', 'done'],
      ['risk', 'done'],
    ])
    expect(t.steps[0]?.tools).toHaveLength(1)
    expect(t.steps[1]?.summary).toBe('score 10')
  })

  it('keeps every visit of a node that runs more than once', () => {
    const t = buildTimeline([
      ev('node_started', { node: 'supervisor' }),
      ev('node_completed', { node: 'supervisor' }),
      ev('node_started', { node: 'supervisor' }),
    ])
    expect(t.steps.map((s) => s.status)).toEqual(['done', 'running'])
    expect(new Set(t.steps.map((s) => s.key)).size).toBe(2)
  })

  it('knows when the run is waiting for a person, finished, or failed', () => {
    const base = [ev('node_started', { node: 'approval' })]
    expect(buildTimeline([...base, ev('awaiting_approval', {})]).state).toBe('awaiting_approval')
    expect(buildTimeline([...base, ev('awaiting_approval', {})]).steps[0]?.status).toBe('done')
    expect(buildTimeline([ev('completed', {})]).state).toBe('completed')
    const failed = buildTimeline([ev('node_started', { node: 'coverage' }), ev('error', { errors: [{ node: 'coverage', message: 'LLM down' }] })])
    expect(failed.state).toBe('failed')
    expect(failed.error).toBe('LLM down')
    expect(failed.steps[0]?.status).toBe('error')
  })

  it('is idle without events', () => {
    expect(buildTimeline([])).toEqual({ steps: [], state: 'idle', error: null })
  })

  it('falls back to the arrival time when an event has no server timestamp', () => {
    const t = buildTimeline([
      { event: 'node_started', data: { node: 'intake' }, receivedAt: 1000 },
      { event: 'node_completed', data: { node: 'intake' }, receivedAt: 2500 },
    ])
    expect(durationMs(t.steps[0]!)).toBe(1500)
  })
})

const snapshot = (over: Partial<RunSnapshot> = {}): RunSnapshot => ({
  run_id: 'r1',
  claim_number: 'CLM-014',
  status: 'awaiting_approval',
  completed_steps: ['intake', 'supervisor', 'coverage', 'risk', 'decision'],
  next: ['approval'],
  awaiting_approval: null,
  decision: null,
  coverage: null,
  risk: null,
  approval: null,
  notification: null,
  errors: [],
  models_used: { coverage: 'gpt-4o-mini', decision: 'llama-3.3-70b-versatile' },
  fallback_model: 'llama-3.3-70b-versatile',
  ...over,
})

describe('models and snapshots', () => {
  it('marks the node the fallback model answered', () => {
    const t = withModels(buildTimeline([ev('node_started', { node: 'coverage' }), ev('node_started', { node: 'decision' })]), snapshot())
    expect(t.steps.find((s) => s.node === 'coverage')).toMatchObject({ model: 'gpt-4o-mini', fallback: false })
    expect(t.steps.find((s) => s.node === 'decision')).toMatchObject({ model: 'llama-3.3-70b-versatile', fallback: true })
  })

  it('renders a finished run from the snapshot alone', () => {
    const t = timelineFromSnapshot(snapshot())
    expect(t.steps.map((s) => s.node)).toEqual(['intake', 'supervisor', 'coverage', 'risk', 'decision'])
    expect(t.steps.every((s) => s.status === 'done')).toBe(true)
    expect(t.state).toBe('awaiting_approval')
    expect(timelineFromSnapshot(snapshot({ status: 'failed', errors: [{ message: 'boom' }] })).error).toBe('boom')
  })
})

describe('tool arguments', () => {
  it('prints the argument object as key=value pairs, never [object Object]', () => {
    expect(formatArgs({ claim_number: 'CLM-014' })).toBe('claim_number=CLM-014')
    expect(formatArgs({ query: 'room rent', policy_codes: ['POL-AS'], top_n: 5 })).toBe(
      'query=room rent, policy_codes=["POL-AS"], top_n=5',
    )
    expect(formatArgs('plain')).toBe('plain')
    expect(formatArgs(undefined)).toBe('')
    const t = buildTimeline([
      ev('node_started', { node: 'intake' }),
      ev('tool_called', { agent: 'intake', tool: 'get_claim', args: { claim_number: 'CLM-014' } }),
    ])
    expect(t.steps[0]?.tools[0]?.args).toBe('claim_number=CLM-014')
  })
})
