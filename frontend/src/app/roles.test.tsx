import { screen } from '@testing-library/react'
import { Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ApprovalPanel } from '@/features/claim-detail/ApprovalPanel'
import { useAuth } from '@/stores/auth'
import { admin, officer, renderWithProviders, signIn } from '@/test/render'

import { AppShell } from './AppShell'
import { RequireAdmin, RequireAuth } from './guards'

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 200 })))
})
afterEach(() => {
  vi.unstubAllGlobals()
  useAuth.getState().clear()
})

function Shell() {
  return (
    <Routes>
      <Route element={<RequireAuth />}>
        <Route element={<AppShell />}>
          <Route path="/claims" element={<p>claims page</p>} />
          <Route path="/admin" element={<RequireAdmin><p>admin page</p></RequireAdmin>} />
        </Route>
      </Route>
      <Route path="/login" element={<p>login page</p>} />
    </Routes>
  )
}

describe('role visibility', () => {
  it('shows the Admin link to admins only', () => {
    signIn(admin)
    const { unmount } = renderWithProviders(<Shell />, { route: '/claims' })
    expect(screen.getAllByRole('link', { name: /Admin/ }).length).toBeGreaterThan(0)
    unmount()

    signIn(officer)
    renderWithProviders(<Shell />, { route: '/claims' })
    expect(screen.queryByRole('link', { name: /Admin/ })).not.toBeInTheDocument()
  })

  it('does not render the admin page for an officer who types the URL', () => {
    signIn(officer)
    renderWithProviders(<Shell />, { route: '/admin' })
    expect(screen.getByText('Admins only')).toBeInTheDocument()
    expect(screen.queryByText('admin page')).not.toBeInTheDocument()
  })

  it('sends a signed-out visitor to the login page', () => {
    useAuth.getState().clear()
    renderWithProviders(<Shell />, { route: '/claims' })
    expect(screen.getByText('login page')).toBeInTheDocument()
  })
})

const decision = {
  decision: 'partial_approve' as const,
  payable_amount: '53666.86',
  claimed_amount: '70000',
  deductions: [],
  reasons: [],
  cited_clauses: [],
  risk_flags: [],
  decision_id: 'd1',
}

describe('approval panel permissions', () => {
  const panel = (uploadedByMe: boolean) => (
    <ApprovalPanel claimId="c1" runId="r1" decision={decision} claimedAmount={70000} uploadedByMe={uploadedByMe} onDone={() => {}} />
  )

  it('explains why an officer cannot decide the claim they uploaded, and offers no buttons', () => {
    signIn(officer)
    renderWithProviders(panel(true))
    expect(screen.getByText(/You uploaded this claim, so another officer/)).toBeInTheDocument()
    expect(screen.queryByRole('radio')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Approve|Reject/ })).not.toBeInTheDocument()
  })

  it('lets another officer decide', () => {
    signIn(officer)
    renderWithProviders(panel(false))
    expect(screen.getByRole('radio', { name: 'Approve' })).toBeEnabled()
    expect(screen.getByRole('radio', { name: 'Reject' })).toBeInTheDocument()
  })

  it('lets an admin override their own upload, with a notice that it is audited', () => {
    signIn(admin)
    renderWithProviders(panel(true))
    expect(screen.getByText(/This is audited/)).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Approve' })).toBeEnabled()
  })
})
