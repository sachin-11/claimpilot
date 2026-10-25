import { CheckCircle2, Info, XCircle } from 'lucide-react'
import { create } from 'zustand'

import { cn } from '@/lib/cn'

type ToastKind = 'success' | 'error' | 'info'
interface ToastItem {
  id: number
  kind: ToastKind
  message: string
}

interface ToastStore {
  items: ToastItem[]
  push: (kind: ToastKind, message: string) => void
  dismiss: (id: number) => void
}

let nextId = 1

export const useToasts = create<ToastStore>((set) => ({
  items: [],
  push: (kind, message) => {
    const id = nextId++
    set((s) => ({ items: [...s.items, { id, kind, message }] }))
    setTimeout(() => set((s) => ({ items: s.items.filter((t) => t.id !== id) })), kind === 'error' ? 8000 : 4500)
  },
  dismiss: (id) => set((s) => ({ items: s.items.filter((t) => t.id !== id) })),
}))

export const toast = {
  success: (message: string) => useToasts.getState().push('success', message),
  error: (message: string) => useToasts.getState().push('error', message),
  info: (message: string) => useToasts.getState().push('info', message),
}

const icons = { success: CheckCircle2, error: XCircle, info: Info }
const colors: Record<ToastKind, string> = {
  success: 'text-ok',
  error: 'text-bad',
  info: 'text-info',
}

export function Toaster() {
  const items = useToasts((s) => s.items)
  const dismiss = useToasts((s) => s.dismiss)
  return (
    <div
      className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-[min(24rem,calc(100vw-2rem))] flex-col gap-2"
      aria-live="polite"
    >
      {items.map((item) => {
        const Icon = icons[item.kind]
        return (
          <div
            key={item.id}
            role={item.kind === 'error' ? 'alert' : 'status'}
            className="animate-fade-up pointer-events-auto flex items-start gap-3 rounded-lg border border-line bg-surface p-3 shadow-lg"
          >
            <Icon className={cn('mt-0.5 size-5 shrink-0', colors[item.kind])} aria-hidden />
            <p className="flex-1">{item.message}</p>
            <button
              type="button"
              onClick={() => dismiss(item.id)}
              className="text-muted hover:text-fg"
              aria-label="Dismiss"
            >
              ×
            </button>
          </div>
        )
      })}
    </div>
  )
}
