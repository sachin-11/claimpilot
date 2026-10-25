/** Client-side checks, the same limits the API enforces (it re-checks the file's real type). */
export const MAX_UPLOAD_MB = 10
export const MAX_FILES = 10
export const ACCEPTED_EXTENSIONS = ['pdf', 'png', 'jpg', 'jpeg'] as const

export interface FileProblem {
  name: string
  message: string
}

const extensionOf = (name: string): string => name.split('.').pop()?.toLowerCase() ?? ''

export function checkFiles(existing: File[], incoming: File[]): { accepted: File[]; problems: FileProblem[] } {
  const accepted: File[] = []
  const problems: FileProblem[] = []
  const known = new Set(existing.map((f) => `${f.name}:${f.size}`))
  for (const file of incoming) {
    if (!ACCEPTED_EXTENSIONS.includes(extensionOf(file.name) as (typeof ACCEPTED_EXTENSIONS)[number])) {
      problems.push({ name: file.name, message: 'Only PDF, PNG or JPG files are accepted' })
    } else if (file.size === 0) {
      problems.push({ name: file.name, message: 'The file is empty' })
    } else if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      problems.push({ name: file.name, message: `Larger than ${MAX_UPLOAD_MB} MB` })
    } else if (known.has(`${file.name}:${file.size}`)) {
      problems.push({ name: file.name, message: 'Already added' })
    } else if (existing.length + accepted.length >= MAX_FILES) {
      problems.push({ name: file.name, message: `At most ${MAX_FILES} files per claim` })
    } else {
      accepted.push(file)
      known.add(`${file.name}:${file.size}`)
    }
  }
  return { accepted, problems }
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
