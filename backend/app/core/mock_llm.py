"""Deterministic stand-ins for the chat model and the embeddings (`MOCK_LLM=true`).

For UI end-to-end tests and demos without API keys or cost. They implement exactly the part of the
LangChain interface the app uses (`with_structured_output`, `bind_tools`, `ainvoke`, `astream`) and
answer from the PROMPT CONTENT, so a run is repeatable and reads sensibly:

- extraction: parses the text layer of the generated bills and discharge summaries;
- supervisor / coverage / risk / decision: follow the facts and the clauses the prompt contains
  (coverage searches the policy through the real tool and cites what it finds);
- policy answers: pick the clause that shares the most words with the question.

Never for production: `Settings` refuses `MOCK_LLM=true` when APP_ENV=production, and
`get_chat_model` / `get_embeddings` check it again.
"""

import hashlib
import json
import math
import re
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage

MODEL_NAME = "mock-llm"
EMBEDDING_DIMENSIONS = 1536
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it me my of on or the to "
    "what when which who why will with would you your this that there their than then into "
    "about under over after before during any all not no yes".split()
)


def _text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else " ".join(str(part) for part in content)


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1]


def _json_after(text: str, marker: str) -> Any:
    """The JSON value that follows `marker` in `text`."""
    start = text.index(marker) + len(marker)
    value, _ = json.JSONDecoder().raw_decode(text[start:].lstrip())
    return value


def _meta(text_in: int = 0, text_out: int = 0) -> dict[str, Any]:
    return {
        "response_metadata": {"model_name": MODEL_NAME},
        "usage_metadata": {
            "input_tokens": text_in,
            "output_tokens": text_out,
            "total_tokens": text_in + text_out,
        },
    }


def _size(messages: Sequence[BaseMessage]) -> int:
    return max(sum(len(_text(m)) for m in messages) // 4, 1)


# ------------------------------------------------------------------------------- embeddings


def embed_text(text: str) -> list[float]:
    """Bag-of-words hashing into 1536 dimensions, L2 normalised: texts that share words are close
    (cosine), so the real hybrid search and ranking work on top of it."""
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for word in _words(text) or ["empty"]:
        digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSIONS
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class MockEmbeddings:
    async def aembed_query(self, text: str) -> list[float]:
        return embed_text(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embed_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embed_text(t) for t in texts]


# ------------------------------------------------------------------------------- extraction

LABELS = {
    "patient name": "patient_name",
    "age / sex": "age_sex",
    "date of admission": "admission",
    "date of discharge": "discharge",
    "diagnosis": "diagnosis",
    "procedure performed": "procedure",
}
CATEGORY_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("icu", ("icu", "intensive care")),
    ("room", ("room rent", "room charge", "bed charge", "ward")),
    ("ot", ("operation theatre", "theatre charge", "ot charge")),
    (
        "surgeon",
        ("surgeon", "anaesthetist", "anesthetist", "physician", "consultant", "medical officer"),
    ),
    ("consumables", ("consumable", "disposable", "cannula", "syringe", "infusion", "dressing")),
    (
        "diagnostics",
        (
            "usg",
            "blood",
            "x-ray",
            "xray",
            "radiology",
            "imaging",
            "culture",
            "serology",
            "check-up",
            "antigen",
            "platelet",
            "gas",
        ),
    ),
    ("medicines", ("tablet", "injection", "iv fluid", "oral", "supplement", "syrup", "medicine")),
)


NUMERIC = re.compile(r"[\d,]+(?:\.\d+)?")


def _money(raw: str) -> float | None:
    cleaned = re.sub(r"[^\d.]", "", raw)
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _iso(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), "%d-%m-%Y").date().isoformat()
    except ValueError:
        return None


def _category(description: str) -> str:
    lowered = description.lower()
    for category, words in CATEGORY_WORDS:
        if any(word in lowered for word in words):
            return category
    return "other"


# lines that are labels, never values: a blank field is followed by the next label
LABEL_LINES = frozenset(
    {
        *LABELS,
        "date of",
        "uhid",
        "phone",
        "patient phone",
        "bill no.",
        "bill date",
        "treating doctor",
        "length of stay",
        "#",
    }
)


def _label_values(lines: list[str]) -> dict[str, str]:
    """`Label` on one to three lines followed by its value on the next line."""
    found: dict[str, str] = {}
    i = 0
    while i < len(lines):
        for span in (3, 2, 1):
            candidate = " ".join(lines[i : i + span]).strip().lower()
            if candidate in LABELS and i + span < len(lines):
                value = lines[i + span].strip()
                blank = value.lower() in LABEL_LINES or not value.strip("_- ")  # "______" = blank
                if not blank:
                    found.setdefault(LABELS[candidate], value)
                i += span - 1
                break
        i += 1
    return found


def _bill_table(lines: list[str]) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    """The charge table (# / description / qty / rate / amount, a value per line) and the totals."""
    items: list[dict[str, Any]] = []
    totals: dict[str, float | None] = {"subtotal": None, "taxes": None, "total": None}
    start = next((i for i, ln in enumerate(lines) if ln.lower().startswith("amount (rs")), None)
    if start is None:
        return items, totals
    i = start + 1
    expected = 1
    while i < len(lines) and lines[i] == str(expected):
        j = i + 1
        description: list[str] = []
        while j < len(lines) and not NUMERIC.fullmatch(lines[j]):  # a long name wraps to more lines
            description.append(lines[j])
            j += 1
        if j + 2 >= len(lines) + 0 and j + 2 > len(lines) - 1:
            break
        qty, rate, amount = _money(lines[j]), _money(lines[j + 1]), _money(lines[j + 2])
        text = " ".join(description)
        if amount is not None and text:
            items.append(
                {
                    "description": text,
                    "category": _category(text),
                    "quantity": qty,
                    "unit_price": rate,
                    "amount": amount,
                }
            )
        i = j + 3
        expected += 1
    for j in range(i, len(lines) - 1):
        head = lines[j].lower()
        if head.startswith("sub total"):
            totals["subtotal"] = _money(lines[j + 1])
        elif head.startswith("gst") or head.startswith("tax"):
            totals["taxes"] = _money(lines[j + 1])
        elif head == "total":
            totals["total"] = _money(lines[j + 1])
    return items, totals


def mock_extraction(messages: Sequence[BaseMessage]) -> Any:
    from app.intake.schemas import LLMExtraction, LLMFieldSource, LLMLineItem, LLMObservation

    text = _text(messages[-1]) if messages else ""
    blocks = re.findall(r'<document type="([a-z_]+)" page="(\d+)">\n(.*?)\n</document>', text, re.S)
    bill = next(((int(p), b) for t, p, b in blocks if t == "hospital_bill"), None)
    summary = next(((int(p), b) for t, p, b in blocks if t == "discharge_summary"), None)
    bill_lines = [ln.strip() for ln in bill[1].splitlines() if ln.strip()] if bill else []
    summary_lines = [ln.strip() for ln in summary[1].splitlines() if ln.strip()] if summary else []
    bill_fields, summary_fields = _label_values(bill_lines), _label_values(summary_lines)
    items, totals = _bill_table(bill_lines)

    def pick(name: str) -> tuple[str | None, str | None]:
        if summary_fields.get(name):
            return summary_fields[name], "discharge_summary"
        if bill_fields.get(name):
            return bill_fields[name], "hospital_bill"
        return None, None

    patient, patient_doc = pick("patient_name")
    admission, admission_doc = pick("admission")
    discharge, discharge_doc = pick("discharge")
    diagnosis_raw, diagnosis_doc = pick("diagnosis")
    procedure, procedure_doc = pick("procedure")
    icd = None
    diagnosis = diagnosis_raw
    if diagnosis_raw:
        match = re.search(r"\(ICD-10:\s*([A-Za-z]\d{2}(?:\.\w{1,4})?)\)", diagnosis_raw)
        if match:
            icd = match.group(1).upper()
            diagnosis = diagnosis_raw[: match.start()].strip()
    age = gender = None
    age_sex = summary_fields.get("age_sex") or bill_fields.get("age_sex")
    if age_sex and "/" in age_sex:
        left, right = (part.strip() for part in age_sex.split("/", 1))
        age, gender = (int(left) if left.isdigit() else None), (right or None)
    hospital_doc = "hospital_bill" if bill else "discharge_summary"
    hospital = (bill_lines or summary_lines or [None])[0]
    room = next((i for i in items if i["category"] == "room"), None)
    icu = next((i for i in items if i["category"] == "icu"), None)
    room_type = room["description"].split(" - ", 1)[-1] if room else None

    sources: list[LLMFieldSource] = []

    def source(field: str, value: Any, doc: str | None, page: int = 1) -> None:
        if value not in (None, [], "") and doc:
            sources.append(LLMFieldSource(field=field, doc_type=doc, page=page, confidence="high"))

    bill_page = bill[0] if bill else 1
    summary_page = summary[0] if summary else 1
    for name, value, doc in (
        ("patient_name", patient, patient_doc),
        ("admission_date", admission, admission_doc),
        ("discharge_date", discharge, discharge_doc),
        ("diagnosis_text", diagnosis, diagnosis_doc),
        ("icd10_code", icd, "discharge_summary"),
        ("procedure", procedure, procedure_doc),
        ("patient_age", age, "discharge_summary" if summary else "hospital_bill"),
        ("gender", gender, "discharge_summary" if summary else "hospital_bill"),
    ):
        source(name, value, doc, summary_page if doc == "discharge_summary" else bill_page)
    for name, value in (
        ("hospital_name", hospital),
        ("room_type", room_type),
        ("room_rent_per_day", room["unit_price"] if room else None),
        ("icu_days", int(icu["quantity"]) if icu and icu["quantity"] else None),
        ("line_items", items),
        ("bill_subtotal", totals["subtotal"]),
        ("taxes", totals["taxes"]),
        ("bill_total", totals["total"]),
    ):
        source(name, value, hospital_doc, bill_page)

    observations = []
    for doc_type, lines_, fields_ in (
        ("hospital_bill", bill_lines, bill_fields),
        ("discharge_summary", summary_lines, summary_fields),
    ):
        if lines_:
            observations.append(
                LLMObservation(
                    doc_type=doc_type,
                    patient_name=fields_.get("patient_name"),
                    admission_date=_iso(fields_.get("admission")),
                    discharge_date=_iso(fields_.get("discharge")),
                    total_amount=totals["total"] if doc_type == "hospital_bill" else None,
                )
            )

    return LLMExtraction(
        patient_name=patient,
        patient_age=age,
        gender=gender,
        hospital_name=hospital,
        admission_date=_iso(admission),
        discharge_date=_iso(discharge),
        diagnosis_text=diagnosis,
        icd10_code=icd,
        procedure=procedure,
        room_type=room_type,
        room_rent_per_day=room["unit_price"] if room else None,
        icu_days=int(icu["quantity"]) if icu and icu["quantity"] else None,
        line_items=[LLMLineItem(**i) for i in items],
        bill_subtotal=totals["subtotal"],
        taxes=totals["taxes"],
        bill_total=totals["total"],
        field_sources=sources,
        observations=observations,
    )


# ------------------------------------------------------------------------------- agents


def mock_supervisor(messages: Sequence[BaseMessage]) -> Any:
    from app.agents.supervisor import SupervisorChoice

    payload = json.loads(_text(messages[-1]))
    return SupervisorChoice(
        next_steps=list(payload.get("allowed_steps") or []),
        reason="mock: run every step that may run now",
    )


NUMBER_WORDS = {"thirty": 30, "sixty": 60, "ninety": 90, "fifteen": 15}


def _number(raw: str) -> int | None:
    raw = raw.lower()
    return int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw)


def waiting_period(check: str, content: str, diagnosis_key: str | None) -> tuple[int, str] | None:
    """The waiting period a clause states: (value, unit), or None when it states none."""
    text = content.replace("\n", " ")
    if check == "initial_waiting":
        match = re.search(r"first\s+(\w+)\s+days", text, re.I) or re.search(r"(\d+)\s*days", text)
        value = _number(match.group(1)) if match else None
        return (value, "days") if value else None
    if check == "specific_disease_waiting" and diagnosis_key:
        words = diagnosis_key.replace("_", " ")
        match = re.search(rf"{re.escape(words)}\s*:\s*(\d+)\s*months", text, re.I)
        if match:
            return int(match.group(1)), "months"
    match = re.search(r"after\s+(\d+)\s+months", text, re.I)
    return (int(match.group(1)), "months") if match else None


def _clauses_of(tool_message: ToolMessage) -> list[dict[str, Any]]:
    try:
        data = json.loads(_text(tool_message))
    except ValueError:
        return []
    return [c for c in data if isinstance(c, dict)] if isinstance(data, list) else []


def mock_coverage(messages: Sequence[BaseMessage]) -> Any:
    from app.agents.nodes.coverage import CoverageCheckLLM, CoverageLLMOutput

    prompt = next(_text(m) for m in messages if "Claim facts:" in _text(m))
    facts = _json_after(prompt, "Claim facts:\n")
    wanted = _json_after(
        prompt, "checks_to_evaluate (exactly these; search with the suggested query first):\n"
    )
    searches = [_clauses_of(m) for m in messages if isinstance(m, ToolMessage)]
    checks = []
    for position, item in enumerate(wanted):
        name = item["check"]
        clauses = searches[position] if position < len(searches) else []
        top = clauses[0] if clauses else None
        wording = waiting_period(name, top["content"], facts.get("diagnosis_key")) if top else None
        result = "pass"
        if name == "exclusion" and top and re.search(r"exclud", top["content"], re.I):
            covered = re.search(r"are covered after|is covered", top["content"], re.I)
            result = "pass" if covered else "fail"
        checks.append(
            CoverageCheckLLM(
                check=name,
                applies=True,
                result=result,
                explanation=(
                    f"The clause {top['section_path']} states the rule for "
                    f"{name.replace('_', ' ')}."
                    if top
                    else "No clause was found for this check."
                ),
                chunk_ids=[top["ref"]] if top and top.get("ref") else [],
                wording_value=wording[0] if wording else None,
                wording_unit=wording[1] if wording else None,
            )
        )
    return CoverageLLMOutput(checks=checks)


class _MockReact:
    """The coverage tool loop: one search per check, then a closing message."""

    MAX_SEARCHES = 6

    async def ainvoke(self, messages: Sequence[BaseMessage], **_: Any) -> AIMessage:
        prompt = next((_text(m) for m in messages if "Claim facts:" in _text(m)), "")
        wanted = (
            _json_after(
                prompt,
                "checks_to_evaluate (exactly these; search with the suggested query first):\n",
            )
            if prompt
            else []
        )
        done = sum(isinstance(m, ToolMessage) for m in messages)
        if done < min(len(wanted), self.MAX_SEARCHES):
            policy_code = _json_after(prompt, "Claim facts:\n").get("policy_code")
            query = wanted[done]["suggested_query"]
            args: dict[str, Any] = {"query": query}
            if policy_code:
                args["policy_code"] = policy_code
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_policy",
                        "args": args,
                        "id": f"mock-{done}",
                        "type": "tool_call",
                    }
                ],
                **_meta(_size(messages), 20),
            )
        return AIMessage(content="I have the clauses I need.", **_meta(_size(messages), 10))


def mock_risk(messages: Sequence[BaseMessage]) -> Any:
    from app.agents.nodes.risk import RiskLLMOutput

    facts = json.loads(_text(messages[-1]))
    flags = facts.get("rule_flags") or []
    if not flags:
        return RiskLLMOutput(summary="The fixed rules found no risk signals.", extra_flags=[])
    codes = ", ".join(str(f.get("code")) for f in flags)
    return RiskLLMOutput(
        summary=f"The fixed rules raised {len(flags)} flag(s): {codes}. A reviewer should check.",
        extra_flags=[],
    )


def mock_reasons(messages: Sequence[BaseMessage]) -> Any:
    from app.agents.nodes.decision import ReasonsLLM

    try:
        given = json.loads(_text(messages[-1]))  # the decision node sends its facts as JSON
    except ValueError:
        given = {}
    facts = [str(f) for f in given.get("facts") or []]  # already worded from the numbers they hold
    reasons = facts[:4] or ["The decision follows the fixed rules and the cited policy clauses."]
    return ReasonsLLM(reasons=reasons)


# ------------------------------------------------------------------------------- policy answers

CLAUSE_RE = re.compile(
    r'<clause id="[^"]*" n="(\d+)" policy="([^"]*)" section="([^"]*)"[^>]*>\n(.*?)\n</clause>', re.S
)


def _best_clause(
    question: str, clauses: list[tuple[int, str, str, str]]
) -> tuple[int, str, str] | None:
    wanted = set(_words(question))
    best: tuple[int, int, str, str] | None = None
    for number, _policy, section, body in clauses:
        sentences = [
            s.strip() for s in re.split(r"(?<=[.;])\s+", body.replace("\n", " ")) if s.strip()
        ]
        for sentence in sentences:
            score = len(wanted & set(_words(f"{sentence} {section}")))
            if best is None or score > best[0]:
                best = (score, number, section, sentence)
    if best is None or best[0] < min(2, max(len(wanted), 1)):
        return None
    return best[1], best[2], best[3]


def _answer_parts(messages: Sequence[BaseMessage]) -> tuple[str, tuple[int, str, str] | None]:
    human = _text(messages[-1])
    question = human.rsplit("Question:", 1)[-1].strip()
    clauses = [(int(n), p, s, b) for n, p, s, b in CLAUSE_RE.findall(human)]
    return question, _best_clause(question, clauses)


def mock_policy_answer(messages: Sequence[BaseMessage]) -> Any:
    from app.rag.answerer import INSUFFICIENT_TEXT
    from app.rag.schemas import LLMCitation, LLMPolicyAnswer

    _, best = _answer_parts(messages)
    if best is None:
        return LLMPolicyAnswer(
            answer=INSUFFICIENT_TEXT, citations=[], insufficient_context=True, confidence="low"
        )
    number, _section, sentence = best
    return LLMPolicyAnswer(
        answer=sentence,
        citations=[LLMCitation(clause_n=number, supporting_text=sentence[:200])],
        insufficient_context=False,
        confidence="high",
    )


# ------------------------------------------------------------------------------- the chat model

HANDLERS = {
    "LLMExtraction": mock_extraction,
    "SupervisorChoice": mock_supervisor,
    "CoverageLLMOutput": mock_coverage,
    "RiskLLMOutput": mock_risk,
    "ReasonsLLM": mock_reasons,
    "LLMPolicyAnswer": mock_policy_answer,
}


class _Structured:
    def __init__(self, handler: Any, include_raw: bool) -> None:
        self.handler, self.include_raw = handler, include_raw

    async def ainvoke(self, messages: Sequence[BaseMessage], **_: Any) -> Any:
        parsed = self.handler(messages)
        if not self.include_raw:
            return parsed
        raw = AIMessage(content="", **_meta(_size(messages), 60))
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


class MockChat:
    """A chat model that answers from its prompt. Reports itself as `mock-llm` (priced at zero)."""

    model_name = MODEL_NAME

    def with_structured_output(
        self, schema: type, include_raw: bool = False, **_: Any
    ) -> _Structured:
        handler = HANDLERS.get(getattr(schema, "__name__", ""))
        if handler is None:
            raise NotImplementedError(f"MOCK_LLM has no answer for {schema!r}")
        return _Structured(handler, include_raw)

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> _MockReact:
        return _MockReact()

    async def ainvoke(self, messages: Sequence[BaseMessage], **_: Any) -> AIMessage:
        system = _text(messages[0]) if messages else ""
        human = _text(messages[-1]) if messages else ""
        if "Classify the document" in system:
            lowered = human.lower()
            if "discharge summary" in lowered:
                word = "discharge_summary"
            elif "inpatient bill" in lowered or "sub total" in lowered:
                word = "hospital_bill"
            elif "claim form" in lowered:
                word = "claim_form"
            else:
                word = "other"
            return AIMessage(content=word, **_meta(_size(messages), 2))
        if "Latest question:" in human:  # a follow-up rewritten to stand alone: keep the question
            return AIMessage(
                content=human.rsplit("Latest question:", 1)[-1].strip(),
                **_meta(_size(messages), 20),
            )
        return AIMessage(content="", **_meta(_size(messages), 1))

    async def astream(
        self, messages: Sequence[BaseMessage], **_: Any
    ) -> AsyncIterator[AIMessageChunk]:
        """A streamed policy answer: the best sentence with its clause marker, word by word."""
        from app.rag.answerer import INSUFFICIENT_TEXT, STREAM_SENTINEL

        _, best = _answer_parts(messages)
        text = (
            f"{STREAM_SENTINEL} {INSUFFICIENT_TEXT}" if best is None else f"{best[2]} [{best[0]}]"
        )
        pieces = re.findall(r"\S+\s*", text)
        for index, piece in enumerate(pieces):
            yield AIMessageChunk(
                content=piece,
                **(_meta(_size(messages), len(pieces)) if index == 0 else {}),
            )


def assert_allowed(is_production: bool) -> None:
    if is_production:
        raise RuntimeError("MOCK_LLM must never be used in production")
