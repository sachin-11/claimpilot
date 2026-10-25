import { describe, expect, it } from 'vitest'

import { approvalBody, approvalSchema, canAccept, lockReason, type ApprovalContext } from './approvalRules'

const ctx: ApprovalContext = { decision: 'partial_approve', claimedAmount: 100000, uploadedByMe: false, isAdmin: false }

const errorsOf = (action: 'approved' | 'edited' | 'rejected', values: { amount?: string; comment?: string }, c = ctx) => {
  const result = approvalSchema(action, c).safeParse({ amount: values.amount ?? '', comment: values.comment ?? '' })
  return result.success ? {} : Object.fromEntries(result.error.issues.map((i) => [String(i.path[0]), i.message]))
}

describe('approval validation', () => {
  it('approving needs nothing', () => {
    expect(errorsOf('approved', {})).toEqual({})
  })

  it('rejecting needs a reason', () => {
    expect(errorsOf('rejected', {}).comment).toMatch(/reason/i)
    expect(errorsOf('rejected', { comment: '   ' }).comment).toBeDefined()
    expect(errorsOf('rejected', { comment: 'Cosmetic procedure' })).toEqual({})
  })

  it('editing needs an amount and a comment', () => {
    const empty = errorsOf('edited', {})
    expect(empty.amount).toBeDefined()
    expect(empty.comment).toBeDefined()
    expect(errorsOf('edited', { amount: '45000', comment: 'Room rent capped' })).toEqual({})
  })

  it('the edited amount must be above zero and not above the claimed amount', () => {
    expect(errorsOf('edited', { amount: '0', comment: 'x' }).amount).toMatch(/more than zero/)
    expect(errorsOf('edited', { amount: '100000.01', comment: 'x' }).amount).toMatch(/more than the claimed/)
    expect(errorsOf('edited', { amount: '100000', comment: 'x' })).toEqual({})
    expect(errorsOf('edited', { amount: '12abc', comment: 'x' }).amount).toBeDefined()
    expect(errorsOf('edited', { amount: '1.234', comment: 'x' }).amount).toBeDefined()
  })

  it('an admin deciding their own upload needs a real explanation', () => {
    const own: ApprovalContext = { ...ctx, uploadedByMe: true, isAdmin: true }
    expect(errorsOf('approved', {}, own).comment).toMatch(/at least 10/)
    expect(errorsOf('approved', { comment: 'short' }, own).comment).toBeDefined()
    expect(errorsOf('approved', { comment: 'Second officer unavailable today' }, own)).toEqual({})
  })
})

describe('separation of duties', () => {
  it('locks the panel for the officer who uploaded the claim, with a reason', () => {
    expect(lockReason({ uploadedByMe: true, isAdmin: false })).toMatch(/another officer/)
  })
  it('leaves it open for other officers and for admins', () => {
    expect(lockReason({ uploadedByMe: false, isAdmin: false })).toBeNull()
    expect(lockReason({ uploadedByMe: true, isAdmin: true })).toBeNull()
  })
})

describe('what can be accepted', () => {
  it('a flagged draft has no outcome to accept', () => {
    expect(canAccept('flag_for_review')).toBe(false)
    expect(canAccept('partial_approve')).toBe(true)
    expect(canAccept('request_documents')).toBe(true)
  })
})

describe('request body', () => {
  it('sends the amount only for an edit and drops an empty comment', () => {
    expect(approvalBody('approved', { amount: '5', comment: '' })).toEqual({ action: 'approved', final_amount: undefined, comment: undefined })
    expect(approvalBody('edited', { amount: '4500', comment: 'ok' })).toEqual({ action: 'edited', final_amount: '4500', comment: 'ok' })
  })
})
