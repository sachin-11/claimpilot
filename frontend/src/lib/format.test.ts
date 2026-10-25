import { describe, expect, it } from 'vitest'

import { formatDate, formatDuration, formatINR, formatPages, formatPercent, humanize } from './format'

describe('formatINR', () => {
  it('groups digits the Indian way', () => {
    expect(formatINR(123456)).toBe('₹1,23,456')
    expect(formatINR('1234567')).toBe('₹12,34,567')
    expect(formatINR(999)).toBe('₹999')
    expect(formatINR(10000000)).toBe('₹1,00,00,000')
  })
  it('keeps decimals only when there are some', () => {
    expect(formatINR('53666.86')).toBe('₹53,666.86')
    expect(formatINR('5000.00')).toBe('₹5,000')
    expect(formatINR(1234.5)).toBe('₹1,234.5')
  })
  it('handles negatives, zero and missing values', () => {
    expect(formatINR(-2500)).toBe('-₹2,500')
    expect(formatINR(0)).toBe('₹0')
    expect(formatINR(null)).toBe('—')
    expect(formatINR(undefined)).toBe('—')
    expect(formatINR('')).toBe('—')
    expect(formatINR('abc')).toBe('—')
  })
})

describe('formatDate', () => {
  it('prints DD MMM YYYY', () => {
    expect(formatDate('2026-10-25')).toBe('25 Oct 2026')
    expect(formatDate('2026-01-05')).toBe('5 Jan 2026')
  })
  it('does not shift a plain date by timezone', () => {
    expect(formatDate('2026-03-01')).toBe('1 Mar 2026')
  })
  it('accepts timestamps and tolerates junk', () => {
    expect(formatDate('2026-10-25T10:30:00')).toBe('25 Oct 2026')
    expect(formatDate(null)).toBe('—')
    expect(formatDate('not a date')).toBe('—')
  })
})

describe('formatDuration', () => {
  it('uses ms below a second, seconds below a minute', () => {
    expect(formatDuration(850)).toBe('850 ms')
    expect(formatDuration(4200)).toBe('4.2 s')
    expect(formatDuration(65_000)).toBe('1 m 05 s')
    expect(formatDuration(null)).toBe('—')
  })
})

describe('small helpers', () => {
  it('formats percentages, pages and labels', () => {
    expect(formatPercent(0.275)).toBe('27.5%')
    expect(formatPercent(null)).toBe('—')
    expect(formatPages(8, 8)).toBe('p. 8')
    expect(formatPages(8, 9)).toBe('pp. 8–9')
    expect(formatPages(null)).toBe('')
    expect(humanize('initial_waiting')).toBe('Initial waiting')
  })
})
