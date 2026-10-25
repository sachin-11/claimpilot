import { zodResolver } from '@hookform/resolvers/zod'
import { useQueryClient } from '@tanstack/react-query'
import { FileText, Image as ImageIcon, Upload, X } from 'lucide-react'
import { useRef, useState } from 'react'
import { useForm } from 'react-hook-form'
import { useNavigate } from 'react-router-dom'
import { z } from 'zod'

import { api, errorMessage, unwrap } from '@/api/client'
import { uploadClaim } from '@/api/upload'
import { toast } from '@/components/toast'
import { Button, Card, Field, Input, PageHeader } from '@/components/ui'
import { cn } from '@/lib/cn'

import { checkFiles, type FileProblem, formatBytes, MAX_FILES, MAX_UPLOAD_MB } from './uploadRules'

const schema = z.object({
  policyNumber: z.string().trim().min(3, 'Enter the policy number (at least 3 characters)').max(100),
  claimedAmount: z
    .string()
    .trim()
    .min(1, 'Enter the claimed amount')
    .refine((v) => /^\d+(\.\d{1,2})?$/.test(v), 'Use digits only, e.g. 125000 or 125000.50')
    .refine((v) => Number(v) > 0, 'The amount must be more than zero'),
  claimNumber: z
    .string()
    .trim()
    .regex(/^([A-Za-z0-9][A-Za-z0-9_-]{0,49})?$/, 'Letters, digits, - and _ only')
    .optional(),
})
type Values = z.infer<typeof schema>

type Mode = 'extract' | 'process'

export function NewClaimPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const input = useRef<HTMLInputElement>(null)
  const [files, setFiles] = useState<File[]>([])
  const [problems, setProblems] = useState<FileProblem[]>([])
  const [dragging, setDragging] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)
  const [stage, setStage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<Values>({ resolver: zodResolver(schema) })

  const addFiles = (list: FileList | File[]) => {
    const { accepted, problems: found } = checkFiles(files, [...list])
    setFiles((current) => [...current, ...accepted])
    setProblems(found)
  }

  const submit = (mode: Mode) =>
    handleSubmit(async (values) => {
      setError(null)
      if (files.length === 0) {
        setProblems([{ name: 'Documents', message: 'Add at least one document' }])
        return
      }
      try {
        setStage('Uploading')
        setProgress(0)
        const created = await uploadClaim({
          policyNumber: values.policyNumber,
          claimedAmount: values.claimedAmount,
          claimNumber: values.claimNumber || undefined,
          files,
          onProgress: setProgress,
        })
        setProgress(1)
        setStage(mode === 'process' ? 'Starting the agents' : 'Extracting')
        if (mode === 'process') {
          await unwrap(api.POST('/claims/{claim_id}/process', { params: { path: { claim_id: created.id } } }))
        } else {
          await unwrap(api.POST('/claims/{claim_id}/extract', { params: { path: { claim_id: created.id } } }))
        }
        await queryClient.invalidateQueries({ queryKey: ['claims'] })
        toast.success(`Claim ${created.claim_number} created`)
        navigate(`/claims/${created.claim_number}`)
      } catch (e) {
        setError(errorMessage(e))
        setStage(null)
        setProgress(null)
      }
    })

  const busy = stage !== null

  return (
    <>
      <PageHeader title="New claim" subtitle="Upload the hospital bill and discharge summary." />
      <form className="grid gap-5 lg:grid-cols-[1fr_22rem]" onSubmit={(e) => e.preventDefault()} noValidate>
        <Card className="p-5">
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              addFiles(e.dataTransfer.files)
            }}
            className={cn(
              'flex flex-col items-center gap-2 rounded-lg border-2 border-dashed px-4 py-10 text-center transition-colors',
              dragging ? 'border-accent bg-accent-soft' : 'border-line',
            )}
          >
            <Upload className="size-8 text-muted" aria-hidden />
            <p className="font-medium">Drag and drop documents here</p>
            <p className="text-muted">
              PDF, PNG or JPG, up to {MAX_UPLOAD_MB} MB each, at most {MAX_FILES} files
            </p>
            <Button onClick={() => input.current?.click()} disabled={busy}>
              Choose files
            </Button>
            <input
              ref={input}
              type="file"
              multiple
              hidden
              accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg"
              aria-label="Documents"
              data-testid="file-input"
              onChange={(e) => {
                if (e.target.files) addFiles(e.target.files)
                e.target.value = ''
              }}
            />
          </div>

          {problems.length > 0 && (
            <ul role="alert" className="mt-3 space-y-1 rounded-md bg-bad-soft px-3 py-2 text-bad">
              {problems.map((p, i) => (
                <li key={i}>
                  <strong>{p.name}</strong>: {p.message}
                </li>
              ))}
            </ul>
          )}

          {files.length > 0 && (
            <ul className="mt-4 divide-y divide-line rounded-md border border-line">
              {files.map((file, index) => (
                <li key={`${file.name}-${file.size}`} className="flex items-center gap-3 px-3 py-2">
                  {file.type.startsWith('image/') ? (
                    <ImageIcon className="size-4 text-muted" aria-hidden />
                  ) : (
                    <FileText className="size-4 text-muted" aria-hidden />
                  )}
                  <span className="min-w-0 flex-1 truncate">{file.name}</span>
                  <span className="text-muted">{formatBytes(file.size)}</span>
                  <button
                    type="button"
                    className="rounded p-1 text-muted hover:bg-surface-2 hover:text-fg disabled:opacity-40"
                    aria-label={`Remove ${file.name}`}
                    disabled={busy}
                    onClick={() => setFiles((current) => current.filter((_, i) => i !== index))}
                  >
                    <X className="size-4" aria-hidden />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card className="h-fit space-y-4 p-5">
          <Field label="Policy number" htmlFor="policyNumber" error={errors.policyNumber?.message}>
            <Input id="policyNumber" autoComplete="off" disabled={busy} {...register('policyNumber')} />
          </Field>
          <Field label="Claimed amount (₹)" htmlFor="claimedAmount" error={errors.claimedAmount?.message}>
            <Input id="claimedAmount" inputMode="decimal" autoComplete="off" disabled={busy} {...register('claimedAmount')} />
          </Field>
          <Field
            label="Claim number (optional)"
            htmlFor="claimNumber"
            hint="Leave empty to generate one."
            error={errors.claimNumber?.message}
          >
            <Input id="claimNumber" autoComplete="off" disabled={busy} {...register('claimNumber')} />
          </Field>

          {error && (
            <p role="alert" className="rounded-md bg-bad-soft px-3 py-2 text-bad">
              {error}
            </p>
          )}

          {busy && (
            <div role="status" aria-live="polite">
              <div className="mb-1 flex justify-between text-xs text-muted">
                <span>{stage}…</span>
                {progress !== null && stage === 'Uploading' && <span>{Math.round(progress * 100)}%</span>}
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-surface-2">
                <div
                  className="h-full bg-accent transition-[width]"
                  style={{ width: `${Math.round((progress ?? 0) * 100)}%` }}
                />
              </div>
            </div>
          )}

          <div className="flex flex-col gap-2">
            <Button variant="primary" disabled={busy} onClick={submit('process')}>
              Run full processing
            </Button>
            <Button disabled={busy} onClick={submit('extract')}>
              Extract only
            </Button>
          </div>
        </Card>
      </form>
    </>
  )
}
