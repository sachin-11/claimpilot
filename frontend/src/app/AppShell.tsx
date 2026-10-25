import { useQueryClient } from '@tanstack/react-query'
import { FileText, LogOut, Menu, MessageSquare, Moon, Shield, Sun, X } from 'lucide-react'
import { useState } from 'react'
import { Link, NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'

import { logout } from '@/api/client'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { Badge } from '@/components/ui'
import { cn } from '@/lib/cn'
import { useTheme } from '@/lib/theme'
import { isAdmin, useAuth } from '@/stores/auth'

const linkClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    'flex items-center gap-2 rounded-md px-3 py-2 font-medium transition-colors',
    isActive ? 'bg-accent-soft text-accent' : 'text-muted hover:bg-surface-2 hover:text-fg',
  )

export function AppShell() {
  const user = useAuth((s) => s.user)
  const { theme, toggle } = useTheme()
  const navigate = useNavigate()
  const location = useLocation()
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)

  const signOut = async () => {
    await logout()
    queryClient.clear()
    navigate('/login', { replace: true })
  }

  const nav = (
    <nav aria-label="Main" className="flex flex-col gap-1" onClick={() => setOpen(false)}>
      <NavLink to="/claims" className={linkClass}>
        <FileText className="size-4" aria-hidden /> Claims
      </NavLink>
      <NavLink to="/policy-chat" className={linkClass}>
        <MessageSquare className="size-4" aria-hidden /> Policy chat
      </NavLink>
      {isAdmin(user) && (
        <NavLink to="/admin" className={linkClass}>
          <Shield className="size-4" aria-hidden /> Admin
        </NavLink>
      )}
    </nav>
  )

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[14rem_1fr]">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-surface focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <aside className="hidden border-r border-line bg-surface lg:flex lg:flex-col">
        <Link to="/claims" className="flex items-center gap-2 px-5 py-4 text-base font-semibold">
          <span className="grid size-7 place-items-center rounded-md bg-accent text-accent-fg">✓</span>
          ClaimPilot
        </Link>
        <div className="flex-1 px-3">{nav}</div>
      </aside>

      <div className="flex min-w-0 flex-col">
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-line bg-surface/95 px-4 py-2.5 backdrop-blur">
          <button
            type="button"
            className="rounded p-1.5 text-muted hover:bg-surface-2 lg:hidden"
            aria-label={open ? 'Close menu' : 'Open menu'}
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
          <span className="font-semibold lg:hidden">ClaimPilot</span>
          <div className="ml-auto flex items-center gap-3">
            {user && (
              <div className="hidden items-center gap-2 sm:flex">
                <span className="text-muted">{user.full_name}</span>
                <Badge tone={isAdmin(user) ? 'accent' : 'neutral'}>{user.role}</Badge>
              </div>
            )}
            <button
              type="button"
              onClick={toggle}
              className="rounded p-1.5 text-muted hover:bg-surface-2 hover:text-fg"
              aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            >
              {theme === 'dark' ? <Sun className="size-5" /> : <Moon className="size-5" />}
            </button>
            <button
              type="button"
              onClick={signOut}
              className="flex items-center gap-1.5 rounded px-2 py-1.5 text-muted hover:bg-surface-2 hover:text-fg"
            >
              <LogOut className="size-4" aria-hidden /> Sign out
            </button>
          </div>
        </header>
        {open && <div className="border-b border-line bg-surface p-3 lg:hidden">{nav}</div>}
        <main id="main" className="mx-auto w-full max-w-[96rem] flex-1 px-4 py-5 sm:px-6">
          <ErrorBoundary resetKey={location.pathname}>
            <Outlet />
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
