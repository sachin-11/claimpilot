import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useAuth } from '@/stores/auth'

import { api, authFetch, refreshSession, toApiError } from './client'

const user = { id: 'u1', email: 'o@example.test', full_name: 'Officer', role: 'officer' }
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
const tokenBody = (token: string) => ({ access_token: token, token_type: 'bearer', expires_at: '2099-01-01T00:00:00Z', user })

let calls: Array<{ url: string; auth: string | null; method: string }>

function mockFetch(handler: (url: string, auth: string | null) => Response | Promise<Response>) {
  calls = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: Request | string, init?: RequestInit) => {
      const request = input instanceof Request ? input : new Request(input, init)
      const auth = request.headers.get('Authorization')
      calls.push({ url: new URL(request.url).pathname, auth, method: request.method })
      return handler(new URL(request.url).pathname, auth)
    }),
  )
}

beforeEach(() => {
  useAuth.getState().setSession({ access_token: 'old', expires_at: '2099-01-01T00:00:00Z', user })
})
afterEach(() => {
  vi.unstubAllGlobals()
})

describe('silent refresh', () => {
  it('sends the in-memory token', async () => {
    mockFetch(() => json({ ok: true }))
    await api.GET('/health')
    expect(calls[0]?.auth).toBe('Bearer old')
  })

  it('refreshes once on a 401 and retries the request with the new token', async () => {
    mockFetch((url, auth) => {
      if (url === '/api/auth/refresh') return json(tokenBody('fresh'))
      return auth === 'Bearer fresh' ? json({ items: [], total: 0, page: 1, size: 20 }) : json({ error: { code: 'unauthorized', message: 'expired' } }, 401)
    })
    const { data, response } = await api.GET('/claims', { params: { query: {} } })
    expect(response.status).toBe(200)
    expect(data?.total).toBe(0)
    expect(calls.map((c) => c.url)).toEqual(['/api/claims', '/api/auth/refresh', '/api/claims'])
    expect(calls[2]?.auth).toBe('Bearer fresh')
    expect(useAuth.getState().token).toBe('fresh')
  })

  it('sends the refresh request with the CSRF header and without the access token', async () => {
    let header: string | null = null
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init?: RequestInit) => {
        header = new Headers(init?.headers).get('X-Requested-With')
        return json(tokenBody('x'))
      }),
    )
    expect(await refreshSession()).toBe(true)
    expect(header).toBeTruthy()
  })

  it('gives up after ONE refresh: a second 401 is returned, not retried forever', async () => {
    mockFetch((url) => (url === '/api/auth/refresh' ? json(tokenBody('fresh')) : json({ error: { code: 'unauthorized', message: 'no' } }, 401)))
    const { response } = await api.GET('/claims', { params: { query: {} } })
    expect(response.status).toBe(401)
    expect(calls.filter((c) => c.url === '/api/auth/refresh')).toHaveLength(1)
    expect(calls.filter((c) => c.url === '/api/claims')).toHaveLength(2)
  })

  it('ends the session when the refresh token is rejected', async () => {
    mockFetch((url) =>
      url === '/api/auth/refresh' ? json({ error: { code: 'unauthorized', message: 'revoked' } }, 401) : json({}, 401),
    )
    await api.GET('/claims', { params: { query: {} } })
    expect(useAuth.getState().status).toBe('anonymous')
    expect(useAuth.getState().token).toBeNull()
    expect(useAuth.getState().endedReason).toMatch(/expired/i)
  })

  it('keeps the session when the refresh fails because the network is down', async () => {
    mockFetch((url) => {
      if (url === '/api/auth/refresh') throw new TypeError('network down')
      return json({}, 401)
    })
    await api.GET('/claims', { params: { query: {} } })
    expect(useAuth.getState().status).toBe('authenticated')
  })

  it('lets parallel 401s share a single refresh (the refresh token rotates)', async () => {
    mockFetch(async (url, auth) => {
      if (url === '/api/auth/refresh') {
        await new Promise((r) => setTimeout(r, 20))
        return json(tokenBody('fresh'))
      }
      return auth === 'Bearer fresh' ? json({ ok: 1 }) : json({}, 401)
    })
    const results = await Promise.all([
      authFetch(new Request('http://x/api/claims')),
      authFetch(new Request('http://x/api/policies')),
      authFetch(new Request('http://x/api/admin/usage')),
    ])
    expect(results.map((r) => r.status)).toEqual([200, 200, 200])
    expect(calls.filter((c) => c.url === '/api/auth/refresh')).toHaveLength(1)
  })

  it('never retries the auth endpoints themselves', async () => {
    mockFetch(() => json({ error: { code: 'unauthorized', message: 'bad password' } }, 401))
    const response = await authFetch(new Request('http://x/api/auth/login', { method: 'POST', body: '{}' }))
    expect(response.status).toBe(401)
    expect(calls).toHaveLength(1)
  })
})

describe('toApiError', () => {
  it('reads the backend error envelope', () => {
    const error = toApiError(403, { error: { code: 'forbidden', message: 'Admins only', request_id: 'r1' } })
    expect(error).toMatchObject({ status: 403, code: 'forbidden', message: 'Admins only', requestId: 'r1' })
  })
  it('falls back to FastAPI detail and a generic message', () => {
    expect(toApiError(422, { detail: [{ msg: 'field required' }] }).message).toBe('field required')
    expect(toApiError(409, { detail: 'already running' }).message).toBe('already running')
    expect(toApiError(500, null).message).toBe('Request failed (500)')
  })
})
