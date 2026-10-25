# End-to-end evaluation: `rerank_local`

Local cross-encoder reranker (bge-reranker-base) on top of hybrid retrieval. Synthetic wordings on (compare with `synthetic_wording`).

Claims: 15 golden + 0 adversarial. Evaluated at the human-approval interrupt (nothing approved). Run started 2026-10-25T13:15:27.

## Headline metrics

| metric | all golden claims | wording-available subset |
|---|---|---|
| Decision accuracy | 100.0% | 100.0% |
| Amount within Rs 1 | 100.0% | 100.0% |
| Amount within 5% | 100.0% | 100.0% |
| Amount MAE (Rs) | 0 | 1 |
| Deductions match | 100.0% | 100.0% |
| Rule match (driving check) | 100.0% | 100.0% |
| Driving check resolved (not unclear) | 100.0% | 100.0% |
| Citation correctness | 100.0% (n=6, judged 0%) | 100.0% (n=1, judged 0%) |
| Unclear rate (applicable checks) | 0.0% | 0.0% |
| Risk flag precision / recall / F1 | 66.7% / 100.0% / 80.0% | n/a / n/a / n/a |
| Missing-docs detection | 100.0% | 100.0% |
| Adversarial pass rate | n/a (n=0) | n/a (n=0) |
| Error rate | 0.0% | 0.0% |
| Latency p50 / p95 (s) | 42.9 / 399.4 | 69.1 / 69.1 |
| LLM calls / tool calls per claim | 7.1 / 11.3 | 8.0 / 12.0 |
| Tokens per claim | 15414 | 22829 |
| Cost per claim (USD, estimate) | 0.0033 | 0.0047 |
| Fallback (non-OpenAI) share of LLM calls | 0.0% | 0.0% |
| Recursion-limit hits | 0 | 0 |

The *wording-available subset* is the claims on policies whose REAL PDF is ingested (POL-AS). In this config POL-STD and POL-PREM also have a SYNTHETIC wording generated from policy_rules.yaml (wording and rules agree by design), so the all-claims column measures the pipeline, not reading of real wording; only the subset column is real.

## Confusion matrix (all golden claims)

| expected \ predicted | approve | partial_approve | reject | flag_for_review | request_documents | error |
|---|---|---|---|---|---|---|
| **approve** | 6 | 0 | 0 | 0 | 0 | 0 |
| **partial_approve** | 0 | 2 | 0 | 0 | 0 | 0 |
| **reject** | 0 | 0 | 4 | 0 | 0 | 0 |
| **flag_for_review** | 0 | 0 | 0 | 2 | 0 | 0 |
| **request_documents** | 0 | 0 | 0 | 0 | 1 | 0 |

## By scenario type

| scenario | claims | decision accuracy |
|---|---|---|
| covered | 6 | 100.0% |
| incomplete | 1 | 100.0% |
| partial | 2 | 100.0% |
| rejected | 4 | 100.0% |
| risk | 2 | 100.0% |

## Failure attribution

| layer | failed claims |
|---|---|
| infra | 0 |
| extraction | 0 |
| routing | 0 |
| coverage | 0 |
| risk | 0 |
| calculator | 0 |
| decision_rules | 0 |

## Adversarial claims

| claim | attack | decision | payable | result |
|---|---|---|---|---|

## Latency per node (seconds)

| node | p50 | p95 |
|---|---|---|
| coverage | 22.30 | 384.22 |
| decision | 1.77 | 2.60 |
| intake | 13.23 | 25.61 |
| risk | 1.75 | 1.93 |

## Findings

No failed claims.

## Resume-ready numbers

- End-to-end agent decision accuracy of 100% on 15 golden claims (100% on the 1 claims whose policy wording is ingested).
- Payable amount within Rs 1 of the golden amount on 100% of claims.
- n/a of 0 prompt-injection / policy-evasion claims handled safely.
- Risk flags: precision 67%, recall 100%.
- p95 latency 399 s and about $0.003 per claim (7 LLM calls, 11 MCP tool calls).
- Citation correctness 100% (n=1, wording-available claims).
