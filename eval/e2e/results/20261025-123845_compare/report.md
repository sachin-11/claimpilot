# Configuration comparison

| metric | default | synthetic_wording |
|---|---|---|
| Decision accuracy | 37.5% | 95.0% |
| Amount within Rs 1 | 83.3% | 97.2% |
| Amount within 5% | 83.3% | 97.2% |
| Amount MAE (Rs) | 19427 | 3696 |
| Deductions match | 66.7% | 94.4% |
| Rule match (driving check) | 60.0% | 93.3% |
| Driving check resolved (not unclear) | 33.3% | 93.3% |
| Citation correctness | 100.0% (n=6, judged 0%) | 100.0% (n=15, judged 0%) |
| Unclear rate (applicable checks) | 75.8% | 3.3% |
| Risk flag precision / recall / F1 | 85.7% / 100.0% / 92.3% | 85.7% / 100.0% / 92.3% |
| Missing-docs detection | 100.0% | 100.0% |
| Adversarial pass rate | 100.0% (n=5) | 100.0% (n=5) |
| Error rate | 0.0% | 0.0% |
| Latency p50 / p95 (s) | 26.5 / 34.5 | 24.2 / 34.6 |
| LLM calls / tool calls per claim | 8.2 / 13.5 | 7.0 / 11.1 |
| Tokens per claim | 15226 | 15376 |
| Cost per claim (USD, estimate) | 0.0032 | 0.0033 |
| Fallback (non-OpenAI) share of LLM calls | 0.0% | 0.0% |
| Recursion-limit hits | 0 | 0 |

Descriptions:
- **default**: Current settings (supervisor LLM on, OpenAI primary with Groq fallback, no reranker).
- **synthetic_wording**: Default settings, plus generated wordings for POL-STD and POL-PREM (built from policy_rules.yaml, so wording and rules agree by design). Measures the pipeline when every policy has a wording; the citation numbers on those two policies are NOT evidence of reading real wording.
