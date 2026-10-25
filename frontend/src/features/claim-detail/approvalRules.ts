import { z } from 'zod'

import type { DecisionName } from '@/api/types'

export type ApprovalAction = 'approved' | 'edited' | 'rejected'

export const MIN_OVERRIDE_COMMENT = 10

export interface ApprovalContext {
  decision: DecisionName
  claimedAmount: number
  /** the signed-in user uploaded this claim */
  uploadedByMe: boolean
  isAdmin: boolean
}

/** Why the panel is locked for this user (null = they may decide). Mirrors the API's separation of duties. */
export function lockReason(ctx: Pick<ApprovalContext, 'uploadedByMe' | 'isAdmin'>): string | null {
  if (ctx.uploadedByMe && !ctx.isAdmin) {
    return 'You uploaded this claim, so another officer has to decide it (separation of duties).'
  }
  return null
}

/** `approved` accepts the draft; a flag_for_review draft has no outcome to accept. */
export function canAccept(decision: DecisionName): boolean {
  return decision !== 'flag_for_review'
}

export function approvalSchema(action: ApprovalAction, ctx: ApprovalContext) {
  return z
    .object({ amount: z.string().trim(), comment: z.string().trim() })
    .superRefine((value, issue) => {
      const needsOverrideNote = ctx.uploadedByMe && ctx.isAdmin
      if (action === 'edited') {
        const amount = Number(value.amount)
        if (value.amount === '' || !/^\d+(\.\d{1,2})?$/.test(value.amount) || !Number.isFinite(amount)) {
          issue.addIssue({ code: 'custom', path: ['amount'], message: 'Enter the amount to pay, e.g. 45000' })
        } else if (amount <= 0) {
          issue.addIssue({ code: 'custom', path: ['amount'], message: 'The amount must be more than zero. Reject the claim instead.' })
        } else if (amount > ctx.claimedAmount) {
          issue.addIssue({ code: 'custom', path: ['amount'], message: 'The amount cannot be more than the claimed amount' })
        }
        if (!value.comment) {
          issue.addIssue({ code: 'custom', path: ['comment'], message: 'Explain why you changed the amount' })
        }
      }
      if (action === 'rejected' && !value.comment) {
        issue.addIssue({ code: 'custom', path: ['comment'], message: 'Give a reason for rejecting' })
      }
      if (needsOverrideNote && value.comment.length < MIN_OVERRIDE_COMMENT) {
        issue.addIssue({
          code: 'custom',
          path: ['comment'],
          message: `You uploaded this claim: an admin override needs a comment of at least ${MIN_OVERRIDE_COMMENT} characters`,
        })
      }
    })
}

export type ApprovalValues = z.infer<ReturnType<typeof approvalSchema>>

/** The body for POST .../approve. */
export function approvalBody(action: ApprovalAction, values: ApprovalValues) {
  return {
    action,
    final_amount: action === 'edited' ? values.amount : undefined,
    comment: values.comment || undefined,
  }
}
