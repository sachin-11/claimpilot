import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter } from 'react-router-dom'

import { type AuthUser, useAuth } from '@/stores/auth'

export const officer: AuthUser = { id: 'u-off', email: 'officer@example.test', full_name: 'Olivia Officer', role: 'officer' }
export const admin: AuthUser = { id: 'u-adm', email: 'admin@example.test', full_name: 'Adam Admin', role: 'admin' }

export function signIn(user: AuthUser) {
  useAuth.getState().setSession({ access_token: 'test-token', expires_at: '2099-01-01T00:00:00Z', user })
}

export function renderWithProviders(ui: ReactElement, { route = '/' }: { route?: string } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  )
}

export const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
