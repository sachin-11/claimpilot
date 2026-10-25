import type { components } from './schema'

type S = components['schemas']

export type ClaimListItem = S['ClaimListItem']
export type ClaimDetail = S['ClaimDetail']
export type DocumentOut = S['DocumentOut']
export type HistoryItem = S['HistoryItem']
export type ApprovalOut = S['ApprovalOut']
export type ClaimStatus = S['ClaimStatus']
export type ExtractionResponse = S['ExtractionResponse']
export type ConsistencyIssue = S['ConsistencyIssue']
export type PolicyClause = S['PolicyClauseRead']
export type PolicySummary = S['PolicySummary']
export type PolicyAnswer = S['PolicyAnswerResult']
export type Citation = S['Citation']
export type ApproveRequest = S['ApproveRequest']
export type ApproveResponse = S['ApproveResponse']

// ---- shapes the OpenAPI schema leaves open (the agent state is plain JSON); see backend/app/agents/state.py

export type CheckName =
  | 'initial_waiting'
  | 'specific_disease_waiting'
  | 'pre_existing'
  | 'exclusion'
  | 'room_rent'
  | 'icu_limit'
  | 'sub_limit'
  | 'co_pay'
  | 'other'
export type CheckResult = 'pass' | 'fail' | 'partial' | 'unclear'
export type DecisionName =
  | 'approve'
  | 'partial_approve'
  | 'reject'
  | 'flag_for_review'
  | 'request_documents'

export interface CitedClause {
  chunk_id: string
  section_path: string | null
  pages: string | null
}

export interface CoverageCheck {
  check: CheckName
  applies: boolean
  result: CheckResult
  explanation: string
  cited_clauses: CitedClause[]
}

export interface RiskFlag {
  code: string
  severity: 'low' | 'medium' | 'high'
  detail: string
  source: 'rule' | 'llm'
}

export interface RiskAssessment {
  flags: RiskFlag[]
  score: number
  summary: string
}

export interface DeductionLine {
  rule: string
  description: string
  amount: string | number
}

export interface DecisionDraft {
  decision: DecisionName
  payable_amount: string | null
  claimed_amount: string | null
  deductions: DeductionLine[]
  reasons: string[]
  cited_clauses: CitedClause[]
  risk_flags: string[]
  decision_id: string | null
}

export interface HumanApproval {
  action: 'approved' | 'rejected' | 'edited'
  final_amount: string | null
  comment: string | null
  reviewer: string
}

export type RunStatus = 'running' | 'awaiting_approval' | 'completed' | 'failed' | 'stalled' | string

export interface ReviewPayload {
  claim_number: string
  claim_id: string
  run_id: string
  decision: DecisionDraft | null
  coverage: CoverageCheck[]
  risk: RiskAssessment | null
  extraction_highlights: Record<string, unknown>
  missing_documents: string[]
}

export interface RunSnapshot {
  run_id: string
  claim_number: string
  status: RunStatus
  completed_steps: string[]
  next: string[]
  awaiting_approval: ReviewPayload | null
  decision: DecisionDraft | null
  coverage: CoverageCheck[] | null
  risk: RiskAssessment | null
  approval: HumanApproval | null
  notification: { status?: string; [key: string]: unknown } | null
  errors: Array<Record<string, string>>
  models_used: Record<string, string>
  fallback_model?: string
}

export interface LineItem {
  description: string
  category: string
  quantity: string | number | null
  unit_price: string | number | null
  amount: string | number
}

export interface FieldSource {
  doc_type: string
  page: number | null
  method: string
}

export interface ExtractionData {
  patient_name?: string | null
  patient_age?: number | null
  gender?: string | null
  policy_number?: string | null
  hospital_name?: string | null
  admission_date?: string | null
  discharge_date?: string | null
  length_of_stay_days?: number | null
  diagnosis_text?: string | null
  icd10_code?: string | null
  procedure?: string | null
  room_type?: string | null
  room_rent_per_day?: string | number | null
  icu_days?: number | null
  line_items?: LineItem[]
  bill_subtotal?: string | number | null
  taxes?: string | number | null
  bill_total?: string | number | null
  claimed_amount?: string | number | null
  field_sources?: Record<string, FieldSource>
  field_confidence?: Record<string, 'low' | 'medium' | 'high'>
}

// ---- admin

export interface AuditLogEntry {
  id: string
  created_at: string
  actor_type: string
  actor_id: string | null
  action: string
  entity_type: string
  entity_id: string
  details: Record<string, unknown>
  request_id: string | null
}

export interface AuditLogPage {
  total: number
  items: AuditLogEntry[]
}

export interface UsageDay {
  day: string
  inr: number
  calls: number
  tokens_in: number
  tokens_out: number
}

export interface UsageResponse {
  currency: string
  usd_inr_rate: number
  today: UsageDay & {
    budget_inr: number
    remaining_inr: number | null
    budget_used_percent: number | null
  }
  history: UsageDay[]
  note: string
}

export interface MetricsSummary {
  day: string
  spend_inr: number
  budget_inr: number
  budget_used_percent: number | null
  llm_calls: number
  calls_by_provider: Record<string, number>
  fallback_calls: number
  fallback_rate: number | null
  runs_today: Record<string, number>
  claims_by_status: Record<string, number>
}

export interface EvalRunSummary {
  id: string
  kind: 'rag' | 'e2e'
  name: string
  created_at: string | null
  has_report: boolean
  headline: Record<string, number | null>
}

export interface EvalRunDetail extends EvalRunSummary {
  report_md: string
  config?: unknown
}

// ---- streaming events

export type RunEventName =
  | 'node_started'
  | 'tool_called'
  | 'node_completed'
  | 'routing'
  | 'awaiting_approval'
  | 'completed'
  | 'error'

export interface RunEvent {
  event: RunEventName
  data: Record<string, unknown>
  receivedAt: number
}
