import { EventStreamContentType, fetchEventSource } from '@microsoft/fetch-event-source'

import { API_BASE, ApiError, authFetch, toApiError } from './client'

export interface SseMessage {
  event: string
  data: string
}

interface StreamOptions {
  signal: AbortSignal
  method?: 'GET' | 'POST'
  body?: unknown
  onOpen?: () => void
  onMessage: (message: SseMessage) => void
  /** called on every dropped connection before the next attempt; return false to stop retrying */
  onRetry?: (error: unknown, attempt: number) => boolean | void
  /** the server closes the stream when a run pauses/ends; return true to reconnect anyway */
  shouldReconnect?: () => boolean
}

class Fatal extends Error {}

/**
 * Authenticated SSE (EventSource cannot send an Authorization header). Goes through `authFetch`, so
 * a 401 refreshes the token once; network drops reconnect with a growing delay.
 */
export async function streamEvents(path: string, options: StreamOptions): Promise<void> {
  let attempt = 0
  try {
    await fetchEventSource(`${API_BASE}${path}`, {
      method: options.method ?? 'GET',
      headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: options.signal,
      openWhenHidden: true,
      fetch: (input, init) => authFetch(new Request(input as RequestInfo, init)),
      async onopen(response) {
        if (response.ok && response.headers.get('content-type')?.startsWith(EventStreamContentType)) {
          attempt = 0
          options.onOpen?.()
          return
        }
        const body: unknown = await response.json().catch(() => null)
        const error = toApiError(response.status, body)
        // 4xx (except a rate limit) will not get better by retrying
        if (response.status >= 400 && response.status < 500 && response.status !== 429) {
          throw new Fatal(error.message, { cause: error })
        }
        throw error
      },
      onmessage(message) {
        options.onMessage({ event: message.event, data: message.data })
      },
      onclose() {
        if (options.shouldReconnect?.()) throw new ApiError(0, 'closed', 'stream closed')
        throw new Fatal('done')
      },
      onerror(error) {
        if (error instanceof Fatal) throw error
        attempt += 1
        if (options.onRetry?.(error, attempt) === false) throw new Fatal('stopped')
        return Math.min(1000 * 2 ** (attempt - 1), 15_000)
      },
    })
  } catch (error) {
    if (error instanceof Fatal) {
      if (error.message === 'done' || error.message === 'stopped') return
      throw error.cause ?? error
    }
    if (options.signal.aborted) return
    throw error
  }
}
