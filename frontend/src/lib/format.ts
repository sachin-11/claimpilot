/** Display formatting. Money is Indian-grouped (₹1,23,456), dates are DD MMM YYYY, durations s/ms. */

const inr = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 2, minimumFractionDigits: 0 })
const inrWhole = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 })

/** `₹1,23,456` (decimals only when there are some). Accepts the API's decimal strings. */
export function formatINR(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === '') return '—'
  const n = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(n)) return '—'
  const sign = n < 0 ? '-' : ''
  const abs = Math.abs(n)
  const formatted = Number.isInteger(abs) ? inrWhole.format(abs) : inr.format(abs)
  return `${sign}₹${formatted}`
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/** `25 Oct 2026` from an ISO date or timestamp (a date-only string is not shifted by timezone). */
export function formatDate(value: string | null | undefined): string {
  if (!value) return '—'
  const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (dateOnly) {
    const [, y, m, d] = dateOnly
    return `${Number(d)} ${MONTHS[Number(m) - 1]} ${y}`
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return `${date.getDate()} ${MONTHS[date.getMonth()]} ${date.getFullYear()}`
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  const hh = String(date.getHours()).padStart(2, '0')
  const mm = String(date.getMinutes()).padStart(2, '0')
  return `${formatDate(value)}, ${hh}:${mm}`
}

export function formatTime(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
}

/** `850 ms`, `4.2 s`, `1 m 05 s`. */
export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '—'
  if (ms < 1000) return `${Math.round(ms)} ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`
  const minutes = Math.floor(ms / 60_000)
  const seconds = Math.round((ms % 60_000) / 1000)
  return `${minutes} m ${String(seconds).padStart(2, '0')} s`
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

/** `"8"` or `"8-9"` -> `p. 8` / `pp. 8-9`. */
export function formatPages(start: number | null | undefined, end?: number | null): string {
  if (!start) return ''
  return end && end !== start ? `pp. ${start}–${end}` : `p. ${start}`
}

/** `initial_waiting` -> `Initial waiting`. */
export function humanize(value: string | null | undefined): string {
  if (!value) return '—'
  const text = value.replace(/_/g, ' ').trim()
  return text.charAt(0).toUpperCase() + text.slice(1)
}
