import { type KeyboardEvent, type ReactNode, useId } from 'react'

import { cn } from '@/lib/cn'

export interface TabDef<T extends string> {
  id: T
  label: ReactNode
}

/** Accessible tab strip (roving arrows); the panel is rendered by the caller with `tabPanelProps`. */
export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
  label,
  className,
}: {
  tabs: TabDef<T>[]
  value: T
  onChange: (id: T) => void
  label: string
  className?: string
}) {
  const base = useId()
  const move = (event: KeyboardEvent, index: number) => {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return
    event.preventDefault()
    const step = event.key === 'ArrowRight' ? 1 : -1
    const next = tabs[(index + step + tabs.length) % tabs.length]
    if (next) {
      onChange(next.id)
      document.getElementById(`${base}-${next.id}`)?.focus()
    }
  }
  return (
    <div role="tablist" aria-label={label} className={cn('flex gap-1 overflow-x-auto border-b border-line', className)}>
      {tabs.map((tab, index) => {
        const selected = tab.id === value
        return (
          <button
            key={tab.id}
            id={`${base}-${tab.id}`}
            role="tab"
            type="button"
            aria-selected={selected}
            aria-controls={`${base}-panel`}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(tab.id)}
            onKeyDown={(event) => move(event, index)}
            className={cn(
              '-mb-px whitespace-nowrap border-b-2 px-3 py-2 font-medium transition-colors',
              selected ? 'border-accent text-fg' : 'border-transparent text-muted hover:text-fg',
            )}
          >
            {tab.label}
          </button>
        )
      })}
    </div>
  )
}

export function tabPanelProps(baseId: string) {
  return { role: 'tabpanel', id: `${baseId}-panel` } as const
}
