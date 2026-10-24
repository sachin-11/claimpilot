"""Discharge-summary narrative: OpenAI with on-disk cache, template fallback.

Standalone on purpose: the data generator is a dev tool and must not import the backend app.
The fallback guarantees the script runs with no API key, no network, or any LLM failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path

from data_gen.models import ClaimSpec, Narrative

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7
ROOT_ENV = Path(__file__).parent.parent / ".env"

# Flipped on the first LLM failure so a bad key does not cost one failed call per claim.
_llm_failed = False


def load_env_file(path: Path = ROOT_ENV) -> None:
    """Populate os.environ from a .env file without overriding existing variables."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _context(spec: ClaimSpec) -> dict[str, object]:
    return {
        "age": spec.patient.age,
        "gender": spec.patient.gender,
        "diagnosis": spec.diagnosis_name,
        "icd10": spec.icd10,
        "procedure": spec.procedure,
        "stay_days": spec.stay_days,
        "icu_days": spec.icu_days,
        "accident": spec.is_accident,
        "pre_existing": spec.ped_conditions,
    }


def _prompt(spec: ClaimSpec) -> str:
    return (
        "Write the clinical narrative for an Indian hospital discharge summary. This is fictional "
        "synthetic data. Do not include any patient name, phone number or address.\n"
        f"Case: {json.dumps(_context(spec))}\n"
        "Return a JSON object with keys: "
        '"history" (2-3 sentences: presenting complaints and relevant history), '
        '"course_in_hospital" (3-4 sentences: investigations, treatment, procedure, recovery), '
        '"discharge_advice" (list of 4-5 short instructions including medicines and follow-up).'
    )


def _template(spec: ClaimSpec, rng: random.Random) -> Narrative:
    ctx = spec
    gender = "male" if spec.patient.gender == "Male" else "female"
    pronoun = "He" if gender == "male" else "She"
    ped = f" Known case of {spec.ped_conditions[0].split(' (')[0]}." if spec.ped_conditions else ""

    if spec.is_accident:
        history = (
            f"{spec.patient.age}-year-old {gender} brought to the emergency department after a road "
            f"traffic accident with pain and inability to bear weight. Diagnosed as {spec.diagnosis_name}."
        )
    else:
        complaint = rng.choice(
            ["pain and discomfort for the past several days", "fever and weakness for 3-4 days",
             "progressive symptoms over the last two weeks"]
        )
        history = (
            f"{spec.patient.age}-year-old {gender} presented with {complaint}. On evaluation, "
            f"diagnosed with {spec.diagnosis_name} ({spec.icd10}).{ped}"
        )
    if spec.procedure:
        course = (
            f"Pre-operative work-up was satisfactory. {pronoun} underwent {spec.procedure} under "
            f"{rng.choice(['general', 'spinal', 'regional'])} anaesthesia without intra-operative "
            f"complications. Post-operative recovery was uneventful with adequate pain control."
        )
    else:
        course = (
            f"{pronoun} was admitted, investigated and started on IV fluids, antibiotics and "
            f"supportive care. Clinical parameters improved steadily over {spec.stay_days} days."
        )
    if spec.icu_days:
        course += f" Close monitoring was done in the ICU for {spec.icu_days} day(s)."
    advice = [
        "Take the prescribed medicines regularly as per schedule",
        "Maintain adequate hydration and a balanced diet",
        "Avoid strenuous activity for two weeks",
        f"Follow-up with {spec.doctor} after 7 days or earlier if symptoms worsen",
    ]
    if spec.procedure:
        advice.insert(1, "Keep the surgical wound clean and dry")
    del ctx
    return Narrative(history=history, course_in_hospital=course, discharge_advice=advice)


def _call_openai(prompt: str, model: str) -> Narrative:
    from openai import OpenAI

    client = OpenAI(timeout=60)
    response = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return Narrative.model_validate(json.loads(response.choices[0].message.content or "{}"))


def get_narrative(spec: ClaimSpec, *, use_llm: bool, model: str, cache_dir: Path, seed: int) -> Narrative:
    global _llm_failed
    rng = random.Random(f"{seed}:{spec.claim_id}:narrative")
    if not use_llm or _llm_failed or not os.getenv("OPENAI_API_KEY"):
        return _template(spec, rng)

    prompt = _prompt(spec)
    key = hashlib.sha256(f"{model}|{TEMPERATURE}|{prompt}".encode()).hexdigest()[:24]
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        return Narrative.model_validate_json(cache_file.read_text(encoding="utf-8"))
    try:
        narrative = _call_openai(prompt, model)
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the template
        _llm_failed = True
        logger.warning("LLM call failed at %s (%s); using templates for the rest of this run", spec.claim_id, exc)
        return _template(spec, rng)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(narrative.model_dump_json(indent=2), encoding="utf-8")
    return narrative
