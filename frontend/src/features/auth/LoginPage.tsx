import { zodResolver } from '@hookform/resolvers/zod'
import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { z } from 'zod'

import { errorMessage, login } from '@/api/client'
import { Button, Card, Field, Input } from '@/components/ui'
import { useAuth } from '@/stores/auth'

const schema = z.object({
  email: z.string().min(1, 'Enter your email').regex(/^[^@\s]+@[^@\s]+$/, 'Enter a valid email'),
  password: z.string().min(1, 'Enter your password'),
})
type Values = z.infer<typeof schema>

export function LoginPage() {
  const status = useAuth((s) => s.status)
  const endedReason = useAuth((s) => s.endedReason)
  const navigate = useNavigate()
  const location = useLocation()
  const next = (location.state as { next?: string } | null)?.next ?? '/claims'
  const [error, setError] = useState<string | null>(null)
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<Values>({ resolver: zodResolver(schema) })

  if (status === 'authenticated') return <Navigate to={next} replace />

  const onSubmit = handleSubmit(async (values) => {
    setError(null)
    try {
      await login(values.email, values.password)
      navigate(next, { replace: true })
    } catch (e) {
      setError(errorMessage(e))
    }
  })

  return (
    <div className="grid min-h-screen lg:grid-cols-[1.1fr_1fr]">
      <aside className="relative hidden overflow-hidden bg-accent p-12 text-accent-fg lg:flex lg:flex-col lg:justify-between">
        <div
          aria-hidden
          className="pointer-events-none absolute -right-24 -top-24 size-96 rounded-full bg-white/10 blur-2xl"
        />
        <div
          aria-hidden
          className="pointer-events-none absolute -bottom-32 -left-16 size-[28rem] rounded-full bg-black/10 blur-3xl"
        />
        <div className="relative flex items-center gap-3">
          <span className="grid size-10 place-items-center rounded-lg bg-white/20 text-lg">✓</span>
          <span className="text-xl font-semibold tracking-tight">ClaimPilot</span>
        </div>
        <div className="relative max-w-md">
          <h2 className="text-3xl font-semibold leading-tight tracking-tight">
            Health insurance claims, reviewed with evidence.
          </h2>
          <p className="mt-4 text-base/relaxed opacity-90">
            AI agents read the bill and discharge summary, check coverage against the policy wording and
            cite the clause. A human officer always makes the final call.
          </p>
          <ul className="mt-8 space-y-3 text-sm">
            {[
              'Every decision cites the policy clause it relies on',
              'Live agent timeline: intake, coverage, risk, decision',
              'Human approval before any claim is settled',
            ].map((t) => (
              <li key={t} className="flex items-start gap-3">
                <span className="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full bg-white/20 text-xs">
                  ✓
                </span>
                <span>{t}</span>
              </li>
            ))}
          </ul>
        </div>
        <p className="relative text-xs opacity-70">Portfolio project · demo data only</p>
      </aside>

      <main className="grid place-items-center px-4 py-12">
        <div className="w-full max-w-sm">
          <div className="mb-8 flex items-center gap-2 lg:hidden">
            <span className="grid size-9 place-items-center rounded-lg bg-accent text-accent-fg">✓</span>
            <span className="text-lg font-semibold">ClaimPilot</span>
          </div>
          <h1 className="text-2xl font-semibold tracking-tight">Welcome back</h1>
          <p className="mb-6 mt-1 text-muted">Sign in to review claims</p>
          {endedReason && !error && (
            <p role="status" className="mb-4 rounded-md bg-warn-soft px-3 py-2 text-warn">
              {endedReason}
            </p>
          )}
          {error && (
            <p role="alert" className="mb-4 rounded-md bg-bad-soft px-3 py-2 text-bad">
              {error}
            </p>
          )}
          <Card className="p-6 shadow-sm">
            <form onSubmit={onSubmit} className="space-y-4" noValidate>
              <Field label="Email" htmlFor="email" error={errors.email?.message}>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@company.com"
                  autoComplete="username"
                  autoFocus
                  {...register('email')}
                />
              </Field>
              <Field label="Password" htmlFor="password" error={errors.password?.message}>
                <Input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  autoComplete="current-password"
                  {...register('password')}
                />
              </Field>
              <Button type="submit" variant="primary" className="w-full" loading={isSubmitting}>
                Sign in
              </Button>
            </form>
          </Card>
          <p className="mt-6 text-center text-xs text-muted">
            Access is limited to claims officers and admins.
          </p>
        </div>
      </main>
    </div>
  )
}
