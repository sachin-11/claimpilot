import { describe, expect, it } from 'vitest'

import { checkFiles, formatBytes, MAX_FILES, MAX_UPLOAD_MB } from './uploadRules'

const file = (name: string, size = 1000) => new File([new Uint8Array(size)], name)

describe('checkFiles', () => {
  it('accepts PDF, PNG and JPG', () => {
    const { accepted, problems } = checkFiles([], [file('bill.pdf'), file('scan.PNG'), file('photo.jpeg'), file('x.jpg')])
    expect(accepted).toHaveLength(4)
    expect(problems).toEqual([])
  })
  it('names every rejected file and why', () => {
    const { accepted, problems } = checkFiles(
      [],
      [file('notes.docx'), file('empty.pdf', 0), file('huge.pdf', MAX_UPLOAD_MB * 1024 * 1024 + 1), file('ok.pdf')],
    )
    expect(accepted.map((f) => f.name)).toEqual(['ok.pdf'])
    expect(problems.map((p) => p.name)).toEqual(['notes.docx', 'empty.pdf', 'huge.pdf'])
    expect(problems[0]?.message).toMatch(/PDF, PNG or JPG/)
    expect(problems[2]?.message).toMatch(/10 MB/)
  })
  it('skips duplicates and caps the number of files', () => {
    const first = file('a.pdf')
    expect(checkFiles([first], [file('a.pdf')]).problems[0]?.message).toMatch(/Already/)
    const many = Array.from({ length: MAX_FILES + 2 }, (_, i) => file(`f${i}.pdf`, 100 + i))
    const result = checkFiles([], many)
    expect(result.accepted).toHaveLength(MAX_FILES)
    expect(result.problems).toHaveLength(2)
  })
  it('formats sizes', () => {
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(2048)).toBe('2 KB')
    expect(formatBytes(3.5 * 1024 * 1024)).toBe('3.5 MB')
  })
})
