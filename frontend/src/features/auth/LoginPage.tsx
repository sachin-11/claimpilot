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
    <div className="grid min-h-screen place-items-center px-4">
      <Card className="w-full max-w-sm p-6">
        <div className="mb-5 flex items-center gap-2">
          <span className="grid size-8 place-items-center rounded-md bg-accent text-accent-fg">✓</span>
          <div>
            <h1 className="text-lg font-semibold leading-tight">ClaimPilot</h1>
            <p className="text-muted">Sign in to review claims</p>
          </div>
        </div>
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
        <form onSubmit={onSubmit} className="space-y-4" noValidate>
          <Field label="Email" htmlFor="email" error={errors.email?.message}>
            <Input id="email" type="email" autoComplete="username" autoFocus {...register('email')} />
          </Field>
          <Field label="Password" htmlFor="password" error={errors.password?.message}>
            <Input id="password" type="password" autoComplete="current-password" {...register('password')} />
          </Field>
          <Button type="submit" variant="primary" className="w-full" loading={isSubmitting}>
            Sign in
          </Button>
        </form>
      </Card>
    </div>
  )
}
