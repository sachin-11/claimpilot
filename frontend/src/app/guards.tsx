import { type ReactNode, useEffect } from 'react'
import { Navigate, Outlet, useLocation } from 'react-router-dom'

import { refreshSession } from '@/api/client'
import { EmptyState, Skeleton } from '@/components/ui'
import { isAdmin, useAuth } from '@/stores/auth'

/** First load: the access token is memory-only, so ask the refresh cookie for one before routing. */
export function useBootstrapSession(): void {
  const status = useAuth((s) => s.status)
  useEffect(() => {
    if (status !== 'unknown') return
    void refreshSession().then((ok) => {
      if (!ok && useAuth.getState().status === 'unknown') useAuth.getState().clear()
    })
  }, [status])
}

export function RequireAuth() {
  const status = useAuth((s) => s.status)
  const location = useLocation()
  if (status === 'unknown') {
    return (
      <div className="grid min-h-screen place-items-center" role="status" aria-label="Loading">
        <Skeleton className="h-8 w-40" />
      </div>
    )
  }
  if (status === 'anonymous') {
    const next = location.pathname + location.search
    return <Navigate to="/login" replace state={{ next }} />
  }
  return <Outlet />
}

export function RequireAdmin({ children }: { children: ReactNode }) {
  const user = useAuth((s) => s.user)
  if (!isAdmin(user)) {
    return (
      <EmptyState
        title="Admins only"
        hint="You are signed in as an officer. Ask an admin if you need access to this page."
      />
    )
  }
  return <>{children}</>
}
