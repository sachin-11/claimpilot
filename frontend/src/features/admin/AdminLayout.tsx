import { NavLink, Outlet } from 'react-router-dom'

import { PageHeader } from '@/components/ui'
import { cn } from '@/lib/cn'

const links = [
  { to: 'policies', label: 'Policies' },
  { to: 'audit', label: 'Audit log' },
  { to: 'usage', label: 'Usage' },
  { to: 'evals', label: 'Evaluations' },
]

export default function AdminLayout() {
  return (
    <>
      <PageHeader title="Admin" subtitle="Policies, audit trail, LLM spend and evaluation results." />
      <nav aria-label="Admin sections" className="mb-5 flex gap-1 overflow-x-auto border-b border-line">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            className={({ isActive }) =>
              cn(
                '-mb-px whitespace-nowrap border-b-2 px-3 py-2 font-medium',
                isActive ? 'border-accent text-fg' : 'border-transparent text-muted hover:text-fg',
              )
            }
          >
            {link.label}
          </NavLink>
        ))}
      </nav>
      <Outlet />
    </>
  )
}
