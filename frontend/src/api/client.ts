import createClient from 'openapi-fetch'

import { type AuthUser, useAuth } from '@/stores/auth'

import type { paths } from './schema'

/** Same-origin: nginx (container) / the Vite proxy (dev) forwards `/api/*` to the backend. */
export const API_BASE = '/api'

const AJAX_HEADER = { 'X-Requested-With': 'claimpilot' }

export class ApiError extends Error {
  status: number
  code: string
  requestId: string | null

  constructor(status: number, code: string, message: string, requestId: string | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.requestId = requestId
  }
}

/** Turns the backend's `{error: {code, message, request_id}}` (or FastAPI's `detail`) into an ApiError. */
export function toApiError(status: number, body: unknown): ApiError {
  const record = (body && typeof body === 'object' ? body : {}) as Record<string, unknown>
  const envelope = record.error as Record<string, unknown> | undefined
  if (envelope && typeof envelope.message === 'string') {
    return new ApiError(
      status,
      String(envelope.code ?? 'error'),
      envelope.message,
      typeof envelope.request_id === 'string' ? envelope.request_id : null,
    )
  }
  const detail = record.detail
  if (typeof detail === 'string') return new ApiError(status, 'error', detail)
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0] as { msg?: string }
    return new ApiError(status, 'validation_error', first?.msg ?? 'Invalid request')
  }
  return new ApiError(status, 'error', `Request failed (${status})`)
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message
  if (error instanceof Error) return error.message
  return 'Something went wrong'
}

// --------------------------------------------------------------------------- silent refresh

let refreshing: Promise<boolean> | null = null

async function doRefresh(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: AJAX_HEADER,
    })
    if (!res.ok) {
      useAuth.getState().clear()
      return false
    }
    const body = (await res.json()) as {
      access_token: string
      expires_at: string
      user: AuthUser
    }
    useAuth.getState().setSession(body)
    return true
  } catch {
    // network error: we do not know the session is dead, so do not log the user out
    return false
  }
}

/** One refresh at a time: parallel 401s share the same request (the refresh token rotates). */
export function refreshSession(): Promise<boolean> {
  refreshing ??= doRefresh().finally(() => {
    refreshing = null
  })
  return refreshing
}

const isAuthPath = (url: string): boolean => /\/auth\/(login|refresh|logout)$/.test(new URL(url, 'http://x').pathname)

/**
 * fetch with the bearer token. On a 401 it refreshes the session ONCE and retries the request;
 * when the refresh fails the session is cleared (the route guard then sends the user to /login).
 */
export async function authFetch(input: Request): Promise<Response> {
  const retry = input.clone()
  const token = useAuth.getState().token
  if (token) input.headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(input)
  if (response.status !== 401 || isAuthPath(input.url)) return response
  if (!(await refreshSession())) {
    if (useAuth.getState().status !== 'anonymous') return response
    useAuth.getState().clear('Your session expired. Please sign in again.')
    return response
  }
  retry.headers.set('Authorization', `Bearer ${useAuth.getState().token}`)
  return fetch(retry)
}

export const api = createClient<paths>({ baseUrl: API_BASE, fetch: authFetch })

/** Resolves with the data or throws ApiError (so react-query shows the same messages everywhere). */
export async function unwrap<T>(
  call: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  const { data, error, response } = await call
  if (error !== undefined || !response.ok) throw toApiError(response.status, error)
  return data as T
}

/** For endpoints whose response model the OpenAPI schema leaves open (typed by hand in types.ts). */
export async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(new Request(`${API_BASE}${path}`, init))
  const body: unknown = response.status === 204 ? null : await response.json().catch(() => null)
  if (!response.ok) throw toApiError(response.status, body)
  return body as T
}

export async function login(email: string, password: string): Promise<void> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) throw toApiError(response.status, body)
  useAuth.getState().setSession(
    body as { access_token: string; expires_at: string; user: AuthUser },
  )
}

export async function logout(): Promise<void> {
  try {
    await fetch(`${API_BASE}/auth/logout`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { ...AJAX_HEADER, 'Content-Type': 'application/json' },
      body: JSON.stringify({ everywhere: false }),
    })
  } finally {
    useAuth.getState().clear()
  }
}
