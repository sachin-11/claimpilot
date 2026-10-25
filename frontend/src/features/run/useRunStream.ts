import { useEffect, useMemo, useRef, useState } from 'react'

import { streamEvents } from '@/api/sse'
import type { RunEvent, RunEventName } from '@/api/types'

import { buildTimeline, type Timeline } from './timeline'

export type ConnectionState = 'idle' | 'connecting' | 'live' | 'reconnecting' | 'ended' | 'failed'

const KNOWN: ReadonlySet<string> = new Set<RunEventName>([
  'node_started',
  'tool_called',
  'node_completed',
  'routing',
  'awaiting_approval',
  'completed',
  'error',
])

interface Result {
  events: RunEvent[]
  timeline: Timeline
  connection: ConnectionState
  /** the review payload carried by the `awaiting_approval` event, if the run paused */
  review: Record<string, unknown> | null
}

interface StreamState {
  key: string | null
  events: RunEvent[]
  connection: ConnectionState
}

function parseData(data: string): Record<string, unknown> | null {
  try {
    const value: unknown = JSON.parse(data)
    return value && typeof value === 'object' ? (value as Record<string, unknown>) : {}
  } catch {
    return null
  }
}

/**
 * Subscribes to a run's SSE stream. Each (re)connection replays the run from its first event, so the
 * list is reset when a connection opens instead of appending duplicates. Stops when the run pauses,
 * completes or fails (the server closes the stream), and on unmount.
 */
export function useRunStream(
  claimId: string | undefined,
  runId: string | null | undefined,
  onFinished?: () => void,
): Result {
  const key = claimId && runId ? `${claimId}/${runId}` : null
  const [state, setState] = useState<StreamState>({ key: null, events: [], connection: 'idle' })
  const onFinishedRef = useRef(onFinished)
  useEffect(() => {
    onFinishedRef.current = onFinished
  })

  useEffect(() => {
    if (!claimId || !runId || !key) return
    const controller = new AbortController()
    let finished = false
    let buffer: RunEvent[] = []
    const publish = (connection: ConnectionState) => setState({ key, events: buffer, connection })

    streamEvents(`/claims/${claimId}/runs/${runId}/stream`, {
      signal: controller.signal,
      onOpen: () => {
        buffer = []
        publish('live')
      },
      onMessage: ({ event, data }) => {
        if (!KNOWN.has(event)) return
        const parsed = parseData(data)
        if (parsed === null) return
        if (event === 'awaiting_approval' || event === 'completed' || event === 'error') finished = true
        buffer = [...buffer, { event: event as RunEventName, data: parsed, receivedAt: Date.now() }]
        publish('live')
      },
      shouldReconnect: () => !finished,
      onRetry: () => {
        publish('reconnecting')
      },
    })
      .then(() => {
        if (controller.signal.aborted) return
        publish('ended')
        onFinishedRef.current?.()
      })
      .catch(() => {
        if (!controller.signal.aborted) publish('failed')
      })

    return () => controller.abort()
  }, [claimId, runId, key])

  // state from a previous run (or none yet) must not leak into the current one
  const current: StreamState =
    state.key === key ? state : { key, events: [], connection: key ? 'connecting' : 'idle' }
  const timeline = useMemo(() => buildTimeline(current.events), [current.events])
  const review = useMemo(() => {
    const last = [...current.events].reverse().find((e) => e.event === 'awaiting_approval')
    return last ? last.data : null
  }, [current.events])
  return { events: current.events, timeline, connection: current.connection, review }
}
