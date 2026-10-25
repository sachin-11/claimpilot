import { create } from 'zustand'

export interface AuthUser {
  id: string
  email: string
  full_name: string
  role: string
}

/** `unknown` until the first silent refresh finished; the access token only ever lives here (memory). */
export type AuthStatus = 'unknown' | 'authenticated' | 'anonymous'

interface AuthState {
  status: AuthStatus
  token: string | null
  expiresAt: string | null
  user: AuthUser | null
  /** why the session ended, shown once on the login page */
  endedReason: string | null
  setSession: (session: { access_token: string; expires_at: string; user: AuthUser }) => void
  clear: (reason?: string | null) => void
}

export const useAuth = create<AuthState>((set) => ({
  status: 'unknown',
  token: null,
  expiresAt: null,
  user: null,
  endedReason: null,
  setSession: ({ access_token, expires_at, user }) =>
    set({
      status: 'authenticated',
      token: access_token,
      expiresAt: expires_at,
      user,
      endedReason: null,
    }),
  clear: (reason = null) =>
    set({ status: 'anonymous', token: null, expiresAt: null, user: null, endedReason: reason }),
}))

export const isAdmin = (user: AuthUser | null): boolean => user?.role === 'admin'
