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
| Latency p50 / p95 (s) | 27.5 / 41.1 | 21.8 / 30.3 |
| LLM calls / tool calls per claim | 8.7 / 14.6 | 7.3 / 11.8 |
| Tokens per claim | 43421 | 33969 |
| Cost per claim (USD, estimate) | 0.0061 | 0.0049 |
| Fallback (non-OpenAI) share of LLM calls | 100.0% | 100.0% |
| Recursion-limit hits | 0 | 0 |

Descriptions:
- **default**: Current settings (supervisor LLM on, Claude Haiku 5.5 via ANTHROPIC_API_KEY, no reranker). Numbers before 2026-10-25 were gpt-4o-mini.
- **synthetic_wording**: Default settings, plus generated wordings for POL-STD and POL-PREM (built from policy_rules.yaml, so wording and rules agree by design). Measures the pipeline when every policy has a wording; the citation numbers on those two policies are NOT evidence of reading real wording.
