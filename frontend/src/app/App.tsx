import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'

import { Toaster } from '@/components/toast'
import { EmptyState, Skeleton } from '@/components/ui'
import { LoginPage } from '@/features/auth/LoginPage'
import { ClaimsListPage } from '@/features/claims/ClaimsListPage'
import { NewClaimPage } from '@/features/claims/NewClaimPage'

import { AppShell } from './AppShell'
import { RequireAdmin, RequireAuth, useBootstrapSession } from './guards'

// the heavy screens (PDF viewer, markdown, charts) load on demand
const ClaimDetailPage = lazy(() => import('@/features/claim-detail/ClaimDetailPage'))
const PolicyChatPage = lazy(() => import('@/features/chat/PolicyChatPage'))
const AdminLayout = lazy(() => import('@/features/admin/AdminLayout'))
const AdminPolicies = lazy(() => import('@/features/admin/PoliciesPage'))
const AdminAudit = lazy(() => import('@/features/admin/AuditPage'))
const AdminUsage = lazy(() => import('@/features/admin/UsagePage'))
const AdminEvals = lazy(() => import('@/features/admin/EvalsPage'))

const Loading = () => (
  <div className="space-y-3" role="status" aria-label="Loading">
    <Skeleton className="h-8 w-56" />
    <Skeleton className="h-64 w-full" />
  </div>
)

export function App() {
  useBootstrapSession()
  return (
    <>
      <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route element={<RequireAuth />}>
            <Route element={<AppShell />}>
              <Route index element={<Navigate to="/claims" replace />} />
              <Route path="/claims" element={<ClaimsListPage />} />
              <Route path="/claims/new" element={<NewClaimPage />} />
              <Route path="/claims/:claimNumber" element={<ClaimDetailPage />} />
              <Route path="/policy-chat" element={<PolicyChatPage />} />
              <Route
                path="/admin"
                element={
                  <RequireAdmin>
                    <AdminLayout />
                  </RequireAdmin>
                }
              >
                <Route index element={<Navigate to="policies" replace />} />
                <Route path="policies" element={<AdminPolicies />} />
                <Route path="audit" element={<AdminAudit />} />
                <Route path="usage" element={<AdminUsage />} />
                <Route path="evals" element={<AdminEvals />} />
              </Route>
              <Route path="*" element={<EmptyState title="Page not found" />} />
            </Route>
          </Route>
        </Routes>
      </Suspense>
      <Toaster />
    </>
  )
}
