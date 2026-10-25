import { Badge } from '@/components/ui'
import { humanize } from '@/lib/format'

type Tone = 'neutral' | 'ok' | 'warn' | 'bad' | 'info' | 'accent'

const claimTone: Record<string, Tone> = {
  submitted: 'neutral',
  processing: 'info',
  awaiting_approval: 'warn',
  approved: 'ok',
  partially_approved: 'ok',
  rejected: 'bad',
  documents_requested: 'warn',
  failed: 'bad',
}

const claimLabel: Record<string, string> = {
  awaiting_approval: 'Awaiting approval',
  partially_approved: 'Partially approved',
  documents_requested: 'Documents requested',
}

export function StatusChip({ status }: { status: string }) {
  return <Badge tone={claimTone[status] ?? 'neutral'}>{claimLabel[status] ?? humanize(status)}</Badge>
}

const decisionTone: Record<string, Tone> = {
  approve: 'ok',
  partial_approve: 'ok',
  reject: 'bad',
  flag_for_review: 'warn',
  request_documents: 'info',
}
const decisionLabel: Record<string, string> = {
  approve: 'Approve',
  partial_approve: 'Partial approval',
  reject: 'Reject',
  flag_for_review: 'Flag for review',
  request_documents: 'Request documents',
}

export function DecisionBadge({ decision }: { decision: string }) {
  return <Badge tone={decisionTone[decision] ?? 'neutral'}>{decisionLabel[decision] ?? humanize(decision)}</Badge>
}

const resultTone: Record<string, Tone> = { pass: 'ok', fail: 'bad', partial: 'warn', unclear: 'neutral' }
const resultLabel: Record<string, string> = { pass: 'Pass', fail: 'Fail', partial: 'Partial', unclear: 'Unclear' }

export function ResultBadge({ result }: { result: string }) {
  return <Badge tone={resultTone[result] ?? 'neutral'}>{resultLabel[result] ?? humanize(result)}</Badge>
}

const severityTone: Record<string, Tone> = { low: 'neutral', medium: 'warn', high: 'bad', info: 'info', warning: 'warn', critical: 'bad' }

export function SeverityBadge({ severity }: { severity: string }) {
  return <Badge tone={severityTone[severity] ?? 'neutral'}>{humanize(severity)}</Badge>
}
