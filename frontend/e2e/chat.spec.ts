import { expect, test } from '@playwright/test'

import { login, OFFICER, shot } from './helpers'

test('policy chat streams a cited answer, opens the clause, and says so when the policy is silent', async ({ page }) => {
  await login(page, OFFICER)
  await page.getByRole('link', { name: 'Policy chat' }).first().click()
  await expect(page.getByRole('heading', { name: 'Policy chat' })).toBeVisible()

  await page.getByLabel('Policy', { exact: true }).selectOption('POL-AS')
  await page.getByLabel('Your question').fill('What is the initial waiting period of thirty days?')
  await page.getByRole('button', { name: 'Ask' }).click()

  const answer = page.getByTestId('assistant-message').first()
  await expect(answer).toContainText(/thirty days|30 days/i, { timeout: 30_000 })
  const source = answer.getByRole('group', { name: 'Sources' }).or(answer.getByLabel('Sources'))
  await expect(source.getByRole('button').first()).toBeVisible()
  await shot(page, '14-policy-chat')

  await source.getByRole('button').first().click()
  await expect(page.getByRole('dialog').getByTestId('clause-text')).toContainText(/waiting/i)
  await page.keyboard.press('Escape')

  // a question the wording does not answer
  await page.getByLabel('Your question').fill('Are unicorn transplants reimbursable?')
  await page.getByRole('button', { name: 'Ask' }).click()
  await expect(page.getByText('Not covered by the policy text')).toBeVisible({ timeout: 30_000 })

  // dark theme
  await page.getByRole('button', { name: /Switch to dark theme/ }).click()
  await shot(page, '15-dark-theme')
  await page.getByRole('button', { name: /Switch to light theme/ }).click()
})
