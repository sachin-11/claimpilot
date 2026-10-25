# Configuration comparison

| metric | synthetic_wording | deterministic_routing | fallback_only |
|---|---|---|---|
| Decision accuracy | 92.5% | 90.0% | 0.0% |
| Amount within Rs 1 | 97.2% | 97.2% | 0.0% |
| Amount within 5% | 97.2% | 97.2% | 0.0% |
| Amount MAE (Rs) | 3696 | 3696 | 110875 |
| Deductions match | 94.4% | 94.4% | 0.0% |
| Rule match (driving check) | 93.3% | 93.3% | 0.0% |
| Driving check resolved (not unclear) | 93.3% | 93.3% | 0.0% |
| Citation correctness | 100.0% (n=15, judged 0%) | 100.0% (n=15, judged 0%) | 0.0% (n=15, judged 0%) |
| Unclear rate (applicable checks) | 3.3% | 5.4% | n/a |
| Risk flag precision / recall / F1 | 75.0% / 100.0% / 85.7% | 75.0% / 100.0% / 85.7% | n/a / 0.0% / n/a |
| Missing-docs detection | 97.5% | 97.5% | 90.0% |
| Adversarial pass rate | 100.0% (n=5) | 100.0% (n=5) | 0.0% (n=5) |
| Error rate | 0.0% | 0.0% | 100.0% |
| Latency p50 / p95 (s) | 24.3 / 38.5 | 9.0 / 11.6 | n/a / n/a |
| LLM calls / tool calls per claim | 7.1 / 11.2 | 4.7 / 10.2 | 0.0 / 3.0 |
| Tokens per claim | 15487 | 10591 | 0 |
| Cost per claim (USD, estimate) | 0.0033 | 0.0019 | 0.0000 |
| Fallback (non-OpenAI) share of LLM calls | 0.0% | 0.0% | n/a |
| Recursion-limit hits | 0 | 0 | 0 |

Descriptions:
- **synthetic_wording**: Default settings, plus generated wordings for POL-STD and POL-PREM (built from policy_rules.yaml, so wording and rules agree by design). Measures the pipeline when every policy has a wording; the citation numbers on those two policies are NOT evidence of reading real wording.
- **deterministic_routing**: Supervisor LLM disabled; routing uses the guards and the default order only. Synthetic wordings on (compare with `synthetic_wording`).
- **fallback_only**: OpenAI primary disabled; every chat call (extraction included) goes to the Groq model. Synthetic wordings on (compare with `synthetic_wording`).
