import { API_BASE, refreshSession, toApiError } from './client'
import type { components } from './schema'
import { useAuth } from '@/stores/auth'

export type ClaimUpload = components['schemas']['ClaimUploadResponse']

interface UploadInput {
  policyNumber: string
  claimedAmount: string
  claimNumber?: string
  files: File[]
  onProgress: (fraction: number) => void
  signal?: AbortSignal
}

function send(input: UploadInput, token: string | null): Promise<{ status: number; body: unknown }> {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.set('policy_number', input.policyNumber)
    form.set('claimed_amount', input.claimedAmount)
    if (input.claimNumber) form.set('claim_number', input.claimNumber)
    for (const file of input.files) form.append('files', file)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/claims`)
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`)
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) input.onProgress(event.loaded / event.total)
    }
    xhr.onload = () => {
      let body: unknown = null
      try {
        body = JSON.parse(xhr.responseText)
      } catch {
        /* non-JSON error page */
      }
      resolve({ status: xhr.status, body })
    }
    xhr.onerror = () => reject(new Error('Network error while uploading'))
    xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'))
    input.signal?.addEventListener('abort', () => xhr.abort())
    xhr.send(form)
  })
}

/** multipart upload with real progress (fetch cannot report it); refreshes an expired token once. */
export async function uploadClaim(input: UploadInput): Promise<ClaimUpload> {
  let result = await send(input, useAuth.getState().token)
  if (result.status === 401 && (await refreshSession())) {
    input.onProgress(0)
    result = await send(input, useAuth.getState().token)
  }
  if (result.status < 200 || result.status >= 300) throw toApiError(result.status, result.body)
  return result.body as ClaimUpload
}
