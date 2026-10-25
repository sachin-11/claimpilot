"""Synthetic policy wordings for the policies that have no real PDF (POL-STD, POL-PREM).

The text is generated from the same `policy_rules.yaml` the golden answers are computed from, so
the wording and the rules agree BY DESIGN. That makes a run with these wordings a test of retrieval,
citation and decision plumbing, not of how well the agent reads real, messy wording. The report
labels it as synthetic; only the real POL-AS wording is used for the headline citation numbers.
"""

import html
from pathlib import Path
from typing import Any

import pymupdf

EXCLUSION_TEXT = {
    "cosmetic_surgery": (
        "7.1. Cosmetic or Plastic Surgery",
        "Expenses for cosmetic, aesthetic or plastic surgery, including rhinoplasty, are excluded "
        "unless required to reconstruct after an accident, burns or cancer.",
    ),
    "dental_unless_accident": (
        "7.2. Dental Treatment",
        "Dental treatment and dental surgery of any kind, including implants, are excluded unless "
        "necessitated by an accident and requiring hospitalisation.",
    ),
    "obesity_treatment": (
        "7.3. Obesity and Weight Control",
        "Expenses for obesity or weight-control treatment, including bariatric surgery, are "
        "excluded.",
    ),
}


def _rs(amount: float | int) -> str:
    return f"Rs {amount:,.0f}"


def _words(key: str) -> str:
    return key.replace("_", " ")


def _room_clause(rule: dict[str, Any], kind: str) -> str:
    pct, cap = rule.get("percent_of_si_per_day"), rule.get("max_per_day")
    if pct is None and cap is None:
        return f"There is no limit on {kind} per day; any room category may be chosen."
    return (
        f"The {kind} admissible per day is {pct:g}% of the sum insured, subject to a maximum of "
        f"{_rs(cap)} per day."
    )


def clauses(policy: dict[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Wording as (section heading, [(clause heading, text)]), generated from the policy rules."""
    name = policy["name"]
    ped, initial = policy["pre_existing_waiting_months"], policy["initial_waiting_days"]
    sections: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "1. PREAMBLE",
            [
                (
                    "1.1. Scope",
                    f"This is the policy wording of {name}. It is a synthetic wording generated "
                    "for evaluation from the policy rules and is not a real insurance product.",
                )
            ],
        ),
        (
            "3. DEFINITIONS",
            [
                (
                    "3.1. Pre-Existing Disease",
                    "Any condition, ailment or injury diagnosed or treated before the policy start "
                    f"date is a pre-existing disease and is subject to the waiting period of {ped} "
                    "months in clause 6.1.",
                ),
                (
                    "3.2. Room Rent",
                    "Room rent means the daily charge of the room category occupied, including "
                    "nursing charges and service charges.",
                ),
            ],
        ),
    ]
    cover: list[tuple[str, str]] = [
        (
            "4.1. In-patient Hospitalisation",
            "Medically necessary in-patient treatment is covered up to the sum insured, after the "
            "waiting periods in section 6 and subject to the exclusions in section 7.",
        ),
        ("4.2. Room Rent", _room_clause(policy["room_rent_limit"], "room rent")),
        ("4.3. ICU Charges", _room_clause(policy["icu_limit"], "ICU charges")),
    ]
    if policy["room_rent_limit"].get("max_per_day") is not None:
        cats = ", ".join(_words(c) for c in policy["proportionate_categories"])
        cover.append(
            (
                "4.4. Proportionate Deduction",
                "If the room rent chosen is above the limit in clause 4.2, the associated "
                f"charges ({cats}) are reduced in the same proportion as the room rent limit bears "
                "to the room rent actually charged.",
            )
        )
    for key, sub in (policy.get("sub_limits") or {}).items():
        unit = "per eye" if sub["unit"] == "per_eye" else "per admission"
        cover.append(
            (
                f"4.{len(cover) + 1}. Sub-limit for {_words(key).title()}",
                f"The maximum payable for {_words(key)} is {_rs(sub['amount'])} {unit}.",
            )
        )
    if policy["maternity"]["covered"]:
        cover.append(
            (
                f"4.{len(cover) + 1}. Maternity",
                f"Maternity expenses are covered after {policy['maternity']['waiting_months']} "
                "months from the policy start date, within the maternity sub-limit.",
            )
        )
    co_pay = policy["co_pay_percent"]
    cover.append(
        (
            f"4.{len(cover) + 1}. Co-pay",
            f"The insured bears {co_pay:g}% of every admissible claim as co-payment."
            if co_pay
            else "No co-payment applies; the insured bears no share of an admissible claim.",
        )
    )
    sections.append(("4. COVERAGE", cover))

    waits = policy["specific_disease_waiting"]
    spec = "; ".join(f"{_words(k)}: {m} months" for k, m in waits.items())
    sections.append(
        (
            "6. WAITING PERIOD",
            [
                (
                    "6.1. Pre-Existing Diseases",
                    f"Pre-existing diseases are covered only after {ped} months of continuous "
                    "coverage from the policy start date.",
                ),
                (
                    "6.2. First Thirty Days Waiting Period",
                    f"Any illness is not covered during the first {initial} days from the policy "
                    "start date. This does not apply to injury caused by an accident.",
                ),
                (
                    "6.3. Specific Waiting Period",
                    f"The following are covered only after the stated period from the policy start "
                    f"date: {spec}.",
                ),
            ],
        )
    )
    excl: list[tuple[str, str]] = [
        EXCLUSION_TEXT[e] for e in policy["exclusions"] if e in EXCLUSION_TEXT
    ]
    maternity = policy["maternity"]
    if not maternity["covered"]:
        excl.append(
            (
                f"7.{len(excl) + 1}. Maternity",
                "Expenses for maternity, delivery and childbirth are excluded.",
            )
        )
    sections.append(("7. EXCLUSIONS", excl))
    return sections


def to_html(policy: dict[str, Any]) -> str:
    parts = [f"<h1>{html.escape(policy['name'])}</h1>", "<p>Policy Wording</p>"]
    for heading, items in clauses(policy):
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        for title, text in items:
            parts.append(f"<p>{html.escape(title)} {html.escape(text)}</p>")
    return "\n".join(parts)


def write_pdf(policy: dict[str, Any], path: Path) -> Path:
    css = (
        "h1 { font-size: 18pt; } h2 { font-size: 13pt; font-weight: bold; margin-top: 14pt; }"
        "p { font-size: 10.5pt; margin: 4pt 0; }"
    )
    story = pymupdf.Story(html=to_html(policy), user_css=css)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = pymupdf.DocumentWriter(str(path))
    page_rect = pymupdf.paper_rect("a4")
    where = page_rect + (54, 54, -54, -54)
    more = 1
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    return path


def write_wordings(policies: list[dict[str, Any]], policies_dir: Path) -> dict[str, Path]:
    """One PDF per policy under the name the policy row expects (`source_pdf`)."""
    return {p["policy_id"]: write_pdf(p, policies_dir / p["source_pdf"]) for p in policies}
