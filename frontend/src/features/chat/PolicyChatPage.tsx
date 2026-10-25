import { useQuery } from '@tanstack/react-query'
import { Check, Copy, Send, Square } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { api, errorMessage, unwrap } from '@/api/client'
import { streamEvents } from '@/api/sse'
import type { PolicyAnswer } from '@/api/types'
import { toast } from '@/components/toast'
import { Badge, Button, Card, EmptyState, PageHeader, Select, Textarea } from '@/components/ui'
import { CitationChip, CitationProvider } from '@/features/citations/Citations'
import { cn } from '@/lib/cn'

interface Message {
  id: number
  role: 'user' | 'assistant'
  text: string
  streaming?: boolean
  error?: string
  result?: PolicyAnswer
}

const HISTORY_TURNS = 6
const EXAMPLES = [
  'What is the initial waiting period?',
  'Is cataract surgery covered, and from when?',
  'Are pre-existing diseases covered?',
]

/** Strips the `[1]`-style markers the model puts after a sentence (the chips below carry the sources). */
export const stripMarkers = (text: string): string => text.replace(/\s*\[\d+(?:\s*,\s*\d+)*\]/g, '').trim()

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      type="button"
      className="inline-flex items-center gap-1 text-xs text-muted hover:text-fg"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text)
          setCopied(true)
          setTimeout(() => setCopied(false), 1500)
        } catch {
          toast.error('Could not copy to the clipboard')
        }
      }}
    >
      {copied ? <Check className="size-3.5" aria-hidden /> : <Copy className="size-3.5" aria-hidden />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

function AssistantMessage({ message }: { message: Message }) {
  const result = message.result
  const notCovered = result?.insufficient_context
  return (
    <div className="max-w-3xl space-y-2" data-testid="assistant-message">
      {notCovered ? (
        <div className="rounded-md border border-warn/40 bg-warn-soft px-3 py-2.5 text-warn" role="status">
          <p className="font-medium">Not covered by the policy text</p>
          <p className="mt-0.5">{stripMarkers(message.text)}</p>
        </div>
      ) : (
        <p className="whitespace-pre-wrap leading-relaxed">
          {stripMarkers(message.text)}
          {message.streaming && <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-fg align-middle" aria-hidden />}
        </p>
      )}
      {message.error && (
        <p role="alert" className="rounded-md bg-bad-soft px-3 py-2 text-bad">
          {message.error}
        </p>
      )}
      {result && !notCovered && result.citations.length > 0 && (
        <div className="flex flex-wrap gap-1.5" aria-label="Sources">
          {result.citations.map((c) => (
            <CitationChip
              key={c.chunk_id}
              chunkId={c.chunk_id}
              label={c.section_path}
              pages={c.page_start ? (c.page_end && c.page_end !== c.page_start ? `${c.page_start}-${c.page_end}` : String(c.page_start)) : null}
            />
          ))}
        </div>
      )}
      {result && (
        <div className="flex items-center gap-3 text-xs text-muted">
          <Badge tone={result.confidence === 'high' ? 'ok' : result.confidence === 'medium' ? 'warn' : 'neutral'}>
            {result.confidence} confidence
          </Badge>
          {result.cache_hit && <span>cached</span>}
          {result.warnings && result.warnings.length > 0 && <span>{result.warnings.join(' · ')}</span>}
          <CopyButton text={stripMarkers(message.text)} />
        </div>
      )}
    </div>
  )
}

export default function PolicyChatPage() {
  const policies = useQuery({ queryKey: ['policies'], queryFn: () => unwrap(api.GET('/policies')) })
  const [policy, setPolicy] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const controller = useRef<AbortController | null>(null)
  const nextId = useRef(1)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' })
  }, [messages])
  useEffect(() => () => controller.current?.abort(), [])

  const update = (id: number, patch: (m: Message) => Message) =>
    setMessages((current) => current.map((m) => (m.id === id ? patch(m) : m)))

  const send = async (text: string) => {
    const query = text.trim()
    if (!query || busy) return
    const history = messages
      .filter((m) => !m.error && m.text)
      .slice(-HISTORY_TURNS)
      .map((m) => ({ role: m.role, content: m.role === 'assistant' ? stripMarkers(m.text) : m.text }))
    const assistantId = nextId.current + 1
    setMessages((current) => [
      ...current,
      { id: nextId.current, role: 'user', text: query },
      { id: assistantId, role: 'assistant', text: '', streaming: true },
    ])
    nextId.current += 2
    setDraft('')
    setBusy(true)
    const ctl = new AbortController()
    controller.current = ctl
    try {
      await streamEvents('/policies/query/stream', {
        method: 'POST',
        signal: ctl.signal,
        body: {
          query,
          policy_codes: policy ? [policy] : undefined,
          chat_history: history.length ? history : undefined,
          use_cache: true,
        },
        onRetry: () => false, // a question is not safe to replay silently
        onMessage: ({ event, data }) => {
          const value: unknown = JSON.parse(data)
          if (event === 'token' && typeof value === 'string') {
            update(assistantId, (m) => ({ ...m, text: m.text + value }))
          } else if (event === 'done') {
            const result = value as PolicyAnswer
            update(assistantId, (m) => ({ ...m, text: result.answer, result, streaming: false }))
          } else if (event === 'error') {
            const message = (value as { message?: string }).message ?? 'The answer could not be generated'
            update(assistantId, (m) => ({ ...m, streaming: false, error: message }))
          }
        },
      })
    } catch (error) {
      update(assistantId, (m) => ({ ...m, streaming: false, error: errorMessage(error) }))
    } finally {
      update(assistantId, (m) => ({ ...m, streaming: false }))
      setBusy(false)
      controller.current = null
    }
  }

  return (
    <CitationProvider>
      <PageHeader
        title="Policy chat"
        subtitle="Ask about a policy. Answers come only from the policy wording and cite the clauses."
        actions={
          <Select
            aria-label="Policy"
            className="w-64"
            value={policy}
            onChange={(e) => setPolicy(e.target.value)}
            disabled={busy}
          >
            <option value="">All policies</option>
            {policies.data?.map((p) => (
              <option key={p.policy_code} value={p.policy_code}>
                {p.name} ({p.policy_code})
              </option>
            ))}
          </Select>
        }
      />
      <Card className="flex h-[calc(100vh-14rem)] min-h-[26rem] flex-col">
        <div className="flex-1 space-y-5 overflow-y-auto p-4" aria-live="polite" aria-label="Conversation">
          {messages.length === 0 ? (
            <EmptyState
              title="Ask a question about the policy"
              hint="For example:"
              action={
                <div className="flex flex-wrap justify-center gap-2">
                  {EXAMPLES.map((example) => (
                    <Button key={example} size="sm" onClick={() => void send(example)}>
                      {example}
                    </Button>
                  ))}
                </div>
              }
            />
          ) : (
            messages.map((m) =>
              m.role === 'user' ? (
                <div key={m.id} className="flex justify-end">
                  <p className="max-w-xl whitespace-pre-wrap rounded-lg bg-accent px-3.5 py-2 text-accent-fg">{m.text}</p>
                </div>
              ) : (
                <AssistantMessage key={m.id} message={m} />
              ),
            )
          )}
          <div ref={bottom} />
        </div>
        <form
          className="flex items-end gap-2 border-t border-line p-3"
          onSubmit={(e) => {
            e.preventDefault()
            void send(draft)
          }}
        >
          <Textarea
            aria-label="Your question"
            rows={1}
            value={draft}
            placeholder="Ask about waiting periods, exclusions, limits…"
            className={cn('max-h-32 min-h-9! flex-1 resize-none py-2')}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void send(draft)
              }
            }}
          />
          {busy ? (
            <Button onClick={() => controller.current?.abort()} aria-label="Stop">
              <Square className="size-4" aria-hidden /> Stop
            </Button>
          ) : (
            <Button type="submit" variant="primary" disabled={!draft.trim()}>
              <Send className="size-4" aria-hidden /> Ask
            </Button>
          )}
        </form>
      </Card>
    </CitationProvider>
  )
}
