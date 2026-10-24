# data_gen: synthetic claims generator (Module 1)

Generates fictional Indian health-insurance claims (PDFs + JSON) and a golden dataset of
expected outcomes. No real patient data. Used later to test extraction, RAG coverage checks,
agents and evaluation.

> **Verify before final use:** the values in `policy_rules.yaml` are placeholders. Check them
> against the real policy wordings in `data/policies/`. Expected outcomes are computed *only*
> from that YAML, so after editing it, re-run the generator and the golden data follows.

## Run

From the repo root:

```bash
pip install -r data_gen/requirements.txt
python -m data_gen.generate --count 40 --seed 42 --out data
python -m pytest data_gen/tests
```

| Flag | Meaning |
|---|---|
| `--count` | Total claims, scaled from the default mix 15/8/7/6/4 |
| `--mix covered=15,partial=8,...` | Explicit scenario counts (overrides `--count`) |
| `--seed` | Same seed gives the same claims and golden data |
| `--no-llm` | Force template narratives |
| `--scan-count` | How many claims also get `*_scanned.png` (default 10) |
| `--out` | Output root; writes `claims/` and `golden/` |

Re-running deletes and recreates `data/claims/CLM-*`. `data/policies/` is never touched.

## Output

```
data/claims/CLM-001/
  claim_form.json
  hospital_bill.pdf
  discharge_summary.pdf          (absent for missing_discharge_summary)
  *_scanned.png                  (~10 claims; page 1 only, rotated/blurred/noisy/low contrast)
data/golden/expected.json        one record per claim, with deduction breakdown
data/golden/summary.csv          same, one row per claim
```

Narrative text in the discharge summary uses OpenAI (`GEN_MODEL`, default `gpt-4o-mini`,
temperature 0.7) when `OPENAI_API_KEY` is set. Responses are cached in `data/.llm_cache/`
so reruns are free. With no key, a bad key, or any API error it falls back to templates.
The key is read from the environment or the repo-root `.env`.

## Scenarios

| Scenario | Decision | What is built |
|---|---|---|
| `covered` | approve | Within every limit. Uses policies with 0% co-pay, so payable = claimed |
| `partial` | partial_approve | `room_rent` (proportionate deduction), `copay`, `sub_limit` (e.g. cataract per eye) |
| `rejected` | reject | `initial_waiting`, `specific_waiting:<disease>`, `pre_existing`, `exclusion:<tag>` |
| `risk` | flag_for_review | `high_bill`, `early_admission` (accident claim days after inception), `date_mismatch` (summary dates differ from bill), `duplicate_items`, `discharge_before_admission`. The inconsistency is really present in the documents |
| `incomplete` | request_documents | `missing_discharge_summary`, `blank_field` (admission date blank in summary), `missing_policy_number`, `unreadable_bill_total` |

Every claim is re-rolled until `calculator.compute_payout` confirms it lands in the requested
scenario (a "rejected / cataract waiting" claim really is rejected by that rule).

## Golden record notes

- `expected_payable_amount`: `final` for covered/partial/rejected (0 when rejected).
  For `risk` it is `provisional` (payable if the flags are cleared; duplicates excluded).
  For `incomplete` it is `null` (`not_determinable`).
- `expected_policy_rule`: the YAML rule that drove the outcome, e.g.
  `specific_disease_waiting.cataract`, `room_rent_limit`, `exclusions.cosmetic_surgery`. For
  partial claims it is the largest deduction. `base_coverage`, `risk.<flag>` and
  `document_completeness` are used for the other scenarios.
- Deduction order: room-rent proportionate -> ICU cap -> sub-limit -> sum insured -> co-pay.
  Invariant: `claimed - sum(deductions) == payable`.

## Modules

`config` (YAML + models) · `models` · `calculator` (pure payout logic) · `planner` (scenarios,
bill building) · `pdf_renderer` · `scan_simulator` · `llm_text` · `writer` · `generate` (CLI).
Tests: `tests/test_calculator.py`.
