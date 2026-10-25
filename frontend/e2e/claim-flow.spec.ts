import { expect, test } from '@playwright/test'

import { login, logout, OFFICER, OFFICER_2, shot, uniqueClaimNumber, uploadAndProcess } from './helpers'

test('officer uploads a claim, watches the agents, a second officer checks a citation and approves', async ({ page }) => {
  const claimNumber = uniqueClaimNumber('E2E')

  // --- officer 1 uploads and starts the run
  await login(page, OFFICER)
  await shot(page, '01-claims-list')
  await uploadAndProcess(page, claimNumber)

  // the live timeline fills in and ends at the human approval step
  await page.getByRole('tab', { name: 'Agent run' }).click()
  const timeline = page.getByRole('list', { name: 'Agent run timeline' })
  await expect(timeline.getByTestId('timeline-step').first()).toBeVisible()
  await expect(page.getByText(/The agents are done/)).toBeVisible({ timeout: 60_000 })
  const nodes = await timeline.getByTestId('timeline-step').evaluateAll((els) => els.map((e) => e.getAttribute('data-node')))
  expect(nodes.slice(0, 2)).toEqual(['intake', 'supervisor'])
  expect(nodes).toEqual(expect.arrayContaining(['coverage', 'risk', 'decision']))
  expect(nodes.indexOf('decision')).toBeGreaterThan(nodes.indexOf('coverage'))
  await expect(page.getByTestId('timeline-step').filter({ hasText: 'Decision' }).first()).toHaveAttribute('data-status', 'done')
  await shot(page, '04-agent-timeline')

  // the uploader cannot decide their own claim
  await expect(page.getByText(/You uploaded this claim, so another officer/)).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Approve' })).toHaveCount(0)

  // --- officer 2 reviews
  await logout(page)
  await login(page, OFFICER_2)
  await page.goto(`/claims/${claimNumber}`)
  await expect(page.getByRole('heading', { name: claimNumber })).toBeVisible()
  await expect(page.getByText('Awaiting approval').first()).toBeVisible()

  await page.getByRole('tab', { name: 'Extraction' }).click()
  await expect(page.getByText('Vedanta Health Institute').first()).toBeVisible()
  await shot(page, '03-claim-detail')

  // a citation chip opens the full clause in the drawer
  await page.getByRole('tab', { name: 'Coverage' }).click()
  const rows = page.getByTestId('coverage-row')
  await expect(rows.first()).toBeVisible()
  await shot(page, '05-coverage')
  const chip = rows.first().getByRole('button').first()
  const chipText = (await chip.innerText()).replace(/\s+p\.\s*[\d-]+$/, '').trim()
  await chip.click()
  const drawer = page.getByRole('dialog')
  await expect(drawer).toBeVisible()
  await expect(drawer.getByTestId('clause-text')).not.toBeEmpty()
  await expect(drawer.getByRole('heading').first()).toContainText(chipText.split('>').pop()!.trim().slice(0, 12))
  await shot(page, '06-citation-drawer')
  await page.keyboard.press('Escape')
  await expect(drawer).toBeHidden()

  await page.getByRole('tab', { name: 'Risk' }).click()
  await expect(page.getByRole('meter', { name: 'Risk score' })).toBeVisible()
  await shot(page, '07-risk')
  await page.getByRole('tab', { name: 'Decision' }).click()
  await expect(page.getByTestId('payable')).not.toHaveText('—')
  await shot(page, '08-decision')

  // approve the draft, with a confirmation step
  await page.getByRole('radio', { name: 'Approve' }).click()
  await shot(page, '09-approval-panel')
  await page.getByRole('button', { name: /^Approve…/ }).click()
  const confirm = page.getByRole('dialog')
  await expect(confirm.getByText('Approve this claim?')).toBeVisible()
  await confirm.getByRole('button', { name: 'Confirm' }).click()

  await expect(page.getByText(/Approved|Partially approved/).first()).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('approved-amount')).not.toHaveText('—')
  await expect(page.getByRole('radiogroup', { name: 'Decision' })).toHaveCount(0)
  await expect(page.getByRole('heading', { name: 'Decision history' })).toBeVisible()
  await expect(page.getByText(/by Second Officer/)).toBeVisible()
  await shot(page, '10-approved')
})
