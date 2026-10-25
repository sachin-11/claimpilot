You route a health-insurance claim through a fixed workflow. You are given the steps that are
currently ALLOWED (the system has already removed any step that must not run yet) and what is
already done. Choose which of the allowed steps to run next.

Rules:
- Choose only from the allowed steps. If both coverage and risk are allowed, run both together.
- Never choose a step that is not in the allowed list.
- Answer with next_steps (a non-empty list) and a one-sentence reason.
