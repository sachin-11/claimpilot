import { expect, type Page } from '@playwright/test'
import { existsSync, mkdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

export const PASSWORD = process.env.DEMO_PASSWORD ?? 'demo-pass-12345'
export const OFFICER = 'demo-officer@example.test'
export const OFFICER_2 = 'demo-officer2@example.test'
export const ADMIN = 'demo-admin@example.test'

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
/** claim CLM-014 of the generated data set: appendicitis, room rent above the limit -> partial approval */
const CLAIM_DIR = path.join(REPO, 'data', 'claims', 'CLM-014')
export const CLAIM_FILES = [path.join(CLAIM_DIR, 'hospital_bill.pdf'), path.join(CLAIM_DIR, 'discharge_summary.pdf')]
export const POLICY_NUMBER = 'POL-AS/2024/7084652'
export const CLAIMED_AMOUNT = '71415'

const SHOTS = path.join(REPO, 'docs', 'screenshots')

/** Screenshots for the README / posts: only when SHOTS=1, so a normal run does not rewrite tracked files. */
export async function shot(page: Page, name: string): Promise<void> {
  if (!process.env.SHOTS) return
  if (!existsSync(SHOTS)) mkdirSync(SHOTS, { recursive: true })
  await page.waitForTimeout(400) // let animations settle
  await page.screenshot({ path: path.join(SHOTS, `${name}.png`) })
}

export async function login(page: Page, email: string): Promise<void> {
  await page.goto('/login')
  await page.getByLabel('Email').fill(email)
  await page.getByLabel('Password').fill(PASSWORD)
  await page.getByRole('button', { name: 'Sign in' }).click()
  await expect(page).toHaveURL(/\/claims/)
}

export async function logout(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Sign out' }).click()
  await expect(page).toHaveURL(/\/login/)
}

/** Uploads the two CLM-014 documents under a fresh claim number and starts the agents. */
export async function uploadAndProcess(page: Page, claimNumber: string): Promise<void> {
  await page.getByRole('link', { name: 'New claim' }).first().click()
  await expect(page.getByRole('heading', { name: 'New claim' })).toBeVisible()
  await page.getByTestId('file-input').setInputFiles(CLAIM_FILES)
  await expect(page.getByText('hospital_bill.pdf')).toBeVisible()
  await page.getByLabel('Policy number').fill(POLICY_NUMBER)
  await page.getByLabel('Claimed amount (₹)').fill(CLAIMED_AMOUNT)
  await page.getByLabel('Claim number (optional)').fill(claimNumber)
  await shot(page, '02-new-claim')
  await page.getByRole('button', { name: 'Run full processing' }).click()
  await expect(page).toHaveURL(new RegExp(`/claims/${claimNumber}$`))
}

export const uniqueClaimNumber = (prefix: string): string => `${prefix}-${Date.now().toString(36).toUpperCase()}`
