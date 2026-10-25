# Configuration comparison

| metric | default | synthetic_wording |
|---|---|---|
| Decision accuracy | 27.5% | 87.5% |
| Amount within Rs 1 | 83.3% | 97.2% |
| Amount within 5% | 83.3% | 97.2% |
| Amount MAE (Rs) | 19427 | 3696 |
| Deductions match | 66.7% | 94.4% |
| Rule match (driving check) | 60.0% | 93.3% |
| Driving check resolved (not unclear) | 13.3% | 73.3% |
| Citation correctness | 100.0% (n=6, judged 0%) | 100.0% (n=15, judged 0%) |
| Unclear rate (applicable checks) | 83.5% | 8.8% |
| Risk flag precision / recall / F1 | 85.7% / 100.0% / 92.3% | 85.7% / 100.0% / 92.3% |
| Missing-docs detection | 100.0% | 100.0% |
| Adversarial pass rate | 100.0% (n=5) | 100.0% (n=5) |
| Error rate | 0.0% | 0.0% |
| Latency p50 / p95 (s) | 27.7 / 38.7 | 23.0 / 34.6 |
| LLM calls / tool calls per claim | 8.6 / 14.4 | 7.5 / 12.1 |
| Tokens per claim | 42811 | 35923 |
| Cost per claim (USD, estimate) | 0.0060 | 0.0051 |
| Fallback (Groq / other) share of LLM calls | 0.0% | 0.0% |
| Recursion-limit hits | 0 | 0 |

Descriptions:
- **default**: Current settings (supervisor LLM on, Claude Haiku 5.5 via ANTHROPIC_API_KEY, no reranker). Numbers before 2026-10-25 were gpt-4o-mini.
- **synthetic_wording**: Default settings, plus generated wordings for POL-STD and POL-PREM (built from policy_rules.yaml, so wording and rules agree by design). Measures the pipeline when every policy has a wording; the citation numbers on those two policies are NOT evidence of reading real wording.
