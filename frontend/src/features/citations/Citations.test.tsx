import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CoverageCheck } from '@/api/types'
import { CoverageTab } from '@/features/claim-detail/CoverageTab'
import { jsonResponse, officer, renderWithProviders, signIn } from '@/test/render'

import { CitationProvider } from './Citations'

const CLAUSE_A = '11111111-1111-4111-8111-111111111111'
const CLAUSE_B = '22222222-2222-4222-8222-222222222222'

const clause = (id: string, section: string, content: string) => ({
  id,
  policy_code: 'POL-AS',
  policy_name: 'Arogya Sanjeevani',
  chunk_index: 3,
  section_path: section,
  page_start: 6,
  page_end: 7,
  clauses: ['6.2'],
  content,
})

const coverage: CoverageCheck[] = [
  {
    check: 'initial_waiting',
    applies: true,
    result: 'pass',
    explanation: 'Admitted after the first 30 days.',
    cited_clauses: [{ chunk_id: CLAUSE_A, section_path: '6 Waiting Period > 6.2', pages: '6' }],
  },
  {
    check: 'room_rent',
    applies: true,
    result: 'partial',
    explanation: 'Room rent is above the limit.',
    cited_clauses: [{ chunk_id: CLAUSE_B, section_path: '4. COVERAGE', pages: '6-7' }],
  },
  { check: 'co_pay', applies: false, result: 'unclear', explanation: '', cited_clauses: [] },
]

const fetched: string[] = []

beforeEach(() => {
  signIn(officer)
  fetched.length = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: Request) => {
      const path = new URL(input.url).pathname
      fetched.push(path)
      if (path.endsWith(CLAUSE_A)) return jsonResponse(clause(CLAUSE_A, '6 Waiting Period > 6.2', 'Any illness is not covered during the first thirty days.'))
      if (path.endsWith(CLAUSE_B)) return jsonResponse(clause(CLAUSE_B, '4. COVERAGE', 'Room rent is limited to one percent of the sum insured per day.'))
      return jsonResponse({ error: { code: 'not_found', message: 'nope' } }, 404)
    }),
  )
})
afterEach(() => vi.unstubAllGlobals())

describe('citation chip -> clause drawer', () => {
  it('opens the clause that the clicked chip points to', async () => {
    const user = userEvent.setup()
    renderWithProviders(
      <CitationProvider>
        <CoverageTab coverage={coverage} />
      </CitationProvider>,
    )
    expect(screen.getAllByTestId('coverage-row')).toHaveLength(2) // the co-pay check does not apply
    expect(screen.getByText(/Not relevant for this claim: Co-payment/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /4\. COVERAGE/ }))
    const drawer = await screen.findByRole('dialog')
    expect(await within(drawer).findByTestId('clause-text')).toHaveTextContent('Room rent is limited to one percent')
    expect(within(drawer).getByText('4. COVERAGE', { selector: 'h2' })).toBeInTheDocument()
    expect(within(drawer).getByText(/Arogya Sanjeevani/)).toBeInTheDocument()
    expect(fetched).toEqual([`/api/policies/chunks/${CLAUSE_B}`]) // only that clause was requested

    await user.keyboard('{Escape}')
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  })

  it('shows the other clause for the other chip', async () => {
    const user = userEvent.setup()
    renderWithProviders(
      <CitationProvider>
        <CoverageTab coverage={coverage} />
      </CitationProvider>,
    )
    await user.click(screen.getByRole('button', { name: /6 Waiting Period/ }))
    expect(await screen.findByTestId('clause-text')).toHaveTextContent('first thirty days')
  })

  it('shows an error with a retry instead of an empty drawer when the clause cannot be loaded', async () => {
    const user = userEvent.setup()
    const missing = { ...coverage[0]!, cited_clauses: [{ chunk_id: '33333333-3333-4333-8333-333333333333', section_path: 'Gone', pages: null }] }
    renderWithProviders(
      <CitationProvider>
        <CoverageTab coverage={[missing]} />
      </CitationProvider>,
    )
    await user.click(screen.getByRole('button', { name: /Gone/ }))
    expect(await screen.findByText('Could not load this')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument()
  })

  it('says so when a check has no cited clause', () => {
    renderWithProviders(
      <CitationProvider>
        <CoverageTab coverage={[{ ...coverage[0]!, cited_clauses: [] }]} />
      </CitationProvider>,
    )
    expect(screen.getByText(/No policy clause was cited/)).toBeInTheDocument()
  })
})
