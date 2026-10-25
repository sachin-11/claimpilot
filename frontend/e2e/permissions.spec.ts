import { expect, test } from '@playwright/test'

import { ADMIN, login, logout, OFFICER, shot, uniqueClaimNumber, uploadAndProcess } from './helpers'

test('an officer cannot approve their own upload, and only an admin sees the audit log', async ({ page }) => {
  const claimNumber = uniqueClaimNumber('PERM')

  await login(page, OFFICER)
  // officers have no Admin area, even by URL
  await expect(page.getByRole('link', { name: 'Admin' })).toHaveCount(0)
  await page.goto('/admin/audit')
  await expect(page.getByText('Admins only')).toBeVisible()

  await page.goto('/claims')
  await uploadAndProcess(page, claimNumber)
  await expect(page.getByText(/You uploaded this claim, so another officer/)).toBeVisible({ timeout: 60_000 })
  await expect(page.getByRole('button', { name: /Approve|Reject/ })).toHaveCount(0)
  await shot(page, '11-separation-of-duties')

  // an admin can read the audit trail, including what happened to this claim
  await logout(page)
  await login(page, ADMIN)
  await expect(page.getByRole('link', { name: 'Admin' }).first()).toBeVisible()
  await page.getByRole('link', { name: 'Admin' }).first().click()
  await page.getByRole('link', { name: 'Audit log' }).click()
  await expect(page.getByRole('heading', { name: 'Admin', exact: true })).toBeVisible()
  await expect(page.getByTestId('audit-row').first()).toBeVisible()
  await shot(page, '12-admin-audit')

  await page.getByRole('link', { name: 'Usage' }).click()
  await expect(page.getByText('Spend today')).toBeVisible()
  await shot(page, '13-admin-usage')

  await page.getByRole('link', { name: 'Policies' }).click()
  await expect(page.getByRole('cell', { name: /Arogya Sanjeevani/ })).toBeVisible()
})
