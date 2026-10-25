import { zodResolver } from '@hookform/resolvers/zod'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Lock } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useForm } from 'react-hook-form'

import { api, errorMessage, unwrap } from '@/api/client'
import type { DecisionDraft } from '@/api/types'
import { Dialog } from '@/components/overlays'
import { toast } from '@/components/toast'
import { Button, Card, Field, Input, Textarea } from '@/components/ui'
import { formatINR } from '@/lib/format'
import { isAdmin, useAuth } from '@/stores/auth'

import {
  type ApprovalAction,
  type ApprovalValues,
  approvalBody,
  approvalSchema,
  canAccept,
  lockReason,
} from './approvalRules'

const ACTION_LABEL: Record<ApprovalAction, string> = {
  approved: 'Approve',
  edited: 'Edit amount',
  rejected: 'Reject',
}

export function ApprovalPanel({
  claimId,
  runId,
  decision,
  claimedAmount,
  uploadedByMe,
  onDone,
}: {
  claimId: string
  runId: string
  decision: DecisionDraft
  claimedAmount: number
  uploadedByMe: boolean
  onDone: () => void
}) {
  const admin = isAdmin(useAuth((s) => s.user))
  const queryClient = useQueryClient()
  const [action, setAction] = useState<ApprovalAction>(canAccept(decision.decision) ? 'approved' : 'edited')
  const [confirming, setConfirming] = useState<ApprovalValues | null>(null)
  const locked = lockReason({ uploadedByMe, isAdmin: admin })

  const schema = useMemo(
    () => approvalSchema(action, { decision: decision.decision, claimedAmount, uploadedByMe, isAdmin: admin }),
    [action, decision.decision, claimedAmount, uploadedByMe, admin],
  )
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<ApprovalValues>({
    resolver: zodResolver(schema),
    defaultValues: { amount: decision.payable_amount ?? '', comment: '' },
  })

  const mutation = useMutation({
    mutationFn: (values: ApprovalValues) =>
      unwrap(
        api.POST('/claims/{claim_id}/runs/{run_id}/approve', {
          params: { path: { claim_id: claimId, run_id: runId } },
          body: approvalBody(action, values),
        }),
      ),
    onSuccess: (result) => {
      toast.success(`Recorded. The claim is now ${result.final_status.replace(/_/g, ' ')}.`)
      setConfirming(null)
      reset()
      void queryClient.invalidateQueries({ queryKey: ['claim'] })
      void queryClient.invalidateQueries({ queryKey: ['claims'] })
      onDone()
    },
    onError: (error) => {
      setConfirming(null)
      toast.error(errorMessage(error))
    },
  })

  const accept = canAccept(decision.decision)
  const summary =
    action === 'approved'
      ? `Accept the draft: ${decision.decision.replace(/_/g, ' ')}${decision.payable_amount ? `, pay ${formatINR(decision.payable_amount)}` : ''}.`
      : action === 'edited'
        ? 'Pay a different amount than the draft.'
        : 'Reject this claim.'

  return (
    <Card className="p-4" aria-label="Approval">
      <h2 className="mb-1 font-semibold">Your decision</h2>
      {locked ? (
        <p role="status" className="flex items-start gap-2 rounded-md bg-warn-soft px-3 py-2 text-warn">
          <Lock className="mt-0.5 size-4 shrink-0" aria-hidden />
          {locked}
        </p>
      ) : (
        <>
          {uploadedByMe && admin && (
            <p className="mb-2 rounded-md bg-info-soft px-3 py-2 text-info">
              You uploaded this claim. As an admin you can still decide it, with a comment that explains why. This is audited.
            </p>
          )}
          <div role="radiogroup" aria-label="Decision" className="mb-3 grid grid-cols-3 gap-2">
            {(['approved', 'edited', 'rejected'] as const).map((value) => {
              const disabled = value === 'approved' && !accept
              return (
                <button
                  key={value}
                  type="button"
                  role="radio"
                  aria-checked={action === value}
                  disabled={disabled}
                  title={disabled ? 'This draft was flagged for review: there is no outcome to accept. Edit or reject.' : undefined}
                  onClick={() => setAction(value)}
                  className={`rounded-md border px-2 py-2 font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                    action === value ? 'border-accent bg-accent-soft text-accent' : 'border-line hover:bg-surface-2'
                  }`}
                >
                  {ACTION_LABEL[value]}
                </button>
              )
            })}
          </div>
          <p className="mb-3 text-muted">{summary}</p>

          <form
            noValidate
            onSubmit={handleSubmit((values) => setConfirming(values))}
            className="space-y-3"
            key={action /* fresh validation state per action */}
          >
            {action === 'edited' && (
              <Field
                label="Amount to pay (₹)"
                htmlFor="amount"
                error={errors.amount?.message}
                hint={`Between ₹1 and ${formatINR(claimedAmount)} (the claimed amount).`}
              >
                <Input id="amount" inputMode="decimal" autoComplete="off" {...register('amount')} />
              </Field>
            )}
            <Field
              label={action === 'approved' ? 'Comment (optional)' : 'Comment'}
              htmlFor="comment"
              error={errors.comment?.message}
              hint="Internal note. It is never sent to the customer."
            >
              <Textarea id="comment" rows={3} {...register('comment')} />
            </Field>
            <Button type="submit" variant={action === 'rejected' ? 'danger' : 'primary'} className="w-full">
              {ACTION_LABEL[action]}…
            </Button>
          </form>
        </>
      )}

      <Dialog
        open={confirming !== null}
        onClose={() => setConfirming(null)}
        title={`${ACTION_LABEL[action]} this claim?`}
        footer={
          <>
            <Button onClick={() => setConfirming(null)} disabled={mutation.isPending}>
              Cancel
            </Button>
            <Button
              variant={action === 'rejected' ? 'danger' : 'primary'}
              loading={mutation.isPending}
              onClick={() => confirming && mutation.mutate(confirming)}
            >
              Confirm
            </Button>
          </>
        }
      >
        <p>{summary}</p>
        {action === 'edited' && confirming && (
          <p className="mt-2 font-medium">New amount: {formatINR(confirming.amount)}</p>
        )}
        <p className="mt-2 text-muted">The customer is notified, and this cannot be undone.</p>
      </Dialog>
    </Card>
  )
}
