"""Reversible PII pseudonymisation. Runs BEFORE any claim text is sent to an LLM.

Detection = Microsoft Presidio (spaCy NER for names, e-mail) + custom recognisers for Indian
formats (phone, Aadhaar, PAN, UHID, pincode addresses, titled names) + EXACT matching of values we
already know from the claim form / customer record (high precision, catches names NER misses).

Placeholders (<PERSON_1>, <PHONE_1> ...) are stable inside one masker. The mapping lives only in
the PIIMasker object (memory, one request): it is never logged or stored.

NOT masked, because extraction needs them: policy numbers, bill numbers, dates, amounts,
diagnosis, hospital name / GSTIN / registration number, ICD-10 codes.
"""

import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.recognizer_result import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.core.redaction import AADHAAR_RE, PAN_RE, PHONE_RE

SPACY_MODEL = "en_core_web_sm"

# entity type -> placeholder prefix
PLACEHOLDER_PREFIX = {
    "PERSON": "PERSON",
    "PHONE": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "ADDRESS": "ADDRESS",
    "AADHAAR": "AADHAAR",
    "PAN": "PAN",
    "UHID": "ID",
}

# --- spans that must survive masking (checked first; PII spans overlapping them are dropped) ---
PROTECTED_PATTERNS = [
    re.compile(r"\b[A-Z]{2,}(?:-[A-Z]+)?/\d{4}/\d{4,}\b"),  # policy number POL-PREM/2025/6045547
    re.compile(r"\b[A-Z]{2,}/\d{4}/\d{3,}\b"),  # BILL/2025/36987
    re.compile(r"\bREG/\d+/\d{4}\b"),  # hospital registration number
    re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b"),  # GSTIN (looks like a PAN inside)
    re.compile(r"\b[A-TV-Z]\d{2}(?:\.\d{1,4})?\b"),  # ICD-10 code
    re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b"),  # dates 12-11-2025
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # ISO dates
]

UHID_RE = re.compile(r"\bUH\d{5,}\b")
# a line fragment with a comma and a 6-digit pincode: "48, Bhasin Path, Junagadh 828219"
ADDRESS_RE = re.compile(r"[A-Za-z0-9][^\n|]{4,90},[^\n|]{2,90}\b[1-9]\d{5}\b")
TITLED_NAME_RE = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Miss|Shri|Smt|Prof)\.?[ \t]+[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){0,2}\b"
)
_NAME_WORD = r"[A-Z][A-Za-z.'-]+"
LABELED_NAME_RE = re.compile(
    r"(?i:patient\s+name|name\s+of\s+(?:the\s+)?(?:patient|insured|claimant)|insured\s+name"
    r"|claimant\s+name)\s*[:\-]?\s*\n?\s*"
    rf"(?P<name>{_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,3}})"
)
# words that start the NEXT field on the same row ("Patient Name Suhani Bala Bill No. ...")
NEXT_LABEL_WORDS = frozenset(
    "bill age sex uhid date phone treating diagnosis policy hospital doctor address email id "
    "total gender dob mobile".split()
)

# Words that make an NER "PERSON" hit a medical / billing phrase, not a person.
NOT_A_NAME_WORDS = frozenset(
    "delivery fracture surgery replacement fever dengue hernia cataract appendicitis pneumonia "
    "infection bill summary discharge hospital nursing home clinic charges room rent ward icu "
    "ot theatre injection tablet laboratory radiology consultation diagnosis procedure treatment "
    "admission patient doctor surgeon anaesthetist final inpatient normal vaginal caesarean "
    "arthroscopy angioplasty knee hip total sub gst "
    # billing / lab / ward vocabulary that a small NER model mistakes for names
    "complete count blood grouping gas arterial culture sputum chest ray abdomen check up "
    "anaesthetic anesthetic pre post dressing material oral rehydration supplements fluids "
    "cannula syringes infusion sets disposables consumables registration service imaging scan "
    "ultrasound usg ct mri ecg lab test profile panel kit bed deluxe private general semi "
    "operation assistant physician resident officer laparoscopic cholecystectomy consultant "
    "obstetric radiology pathology pharmacy medicines drugs amount rate qty description "
    "uhid age sex bill date phone treating policy address email gender dob mobile "
    "intraocular lens implant stent graft mesh valve".split()
)

_engine_lock = threading.Lock()
_engines: tuple[AnalyzerEngine, AnonymizerEngine] | None = None


def _custom_recognizers() -> list[PatternRecognizer]:
    def pattern(name: str, regex: str, score: float) -> list[Pattern]:
        return [Pattern(name, regex, score)]

    return [
        PatternRecognizer("PHONE", patterns=pattern("in_phone", PHONE_RE.pattern, 0.85)),
        PatternRecognizer("AADHAAR", patterns=pattern("in_aadhaar", AADHAAR_RE.pattern, 0.85)),
        PatternRecognizer("PAN", patterns=pattern("in_pan", PAN_RE.pattern, 0.85)),
        PatternRecognizer("UHID", patterns=pattern("uhid", UHID_RE.pattern, 0.9)),
        PatternRecognizer("ADDRESS", patterns=pattern("in_address", ADDRESS_RE.pattern, 0.7)),
        PatternRecognizer("PERSON", patterns=pattern("titled_name", TITLED_NAME_RE.pattern, 0.8)),
    ]


def get_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """Presidio engines, built once per process (loading spaCy takes a second or two)."""
    global _engines
    with _engine_lock:
        if _engines is None:
            nlp = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
                }
            ).create_engine()
            registry = RecognizerRegistry(supported_languages=["en"])
            registry.load_predefined_recognizers(nlp_engine=nlp, languages=["en"])
            for recognizer in _custom_recognizers():
                registry.add_recognizer(recognizer)
            analyzer = AnalyzerEngine(nlp_engine=nlp, registry=registry, supported_languages=["en"])
            _engines = (analyzer, AnonymizerEngine())
        return _engines


# ------------------------------------------------------------------------- masker


def _normalise(value: str) -> str:
    return " ".join(value.lower().split())


def looks_like_a_name(text: str) -> bool:
    """Filter for NER PERSON hits: 1-4 alphabetic capitalised words, none a medical/billing term."""
    if "\n" in text.strip():  # a name is on one line; NER spans that cross lines swallow labels
        return False
    if "(" in text or ")" in text:  # "Intraocular Lens (IOL)": a product, not a person
        return False
    words = text.replace(".", " ").split()
    if not 1 <= len(words) <= 4:
        return False
    if any(len(w) > 1 and w.isupper() for w in words):  # abbreviations (IOL, UHID, GST)
        return False
    if any(not w[0].isalpha() or any(c.isdigit() for c in w) for w in words):
        return False
    return not any(w.lower() in NOT_A_NAME_WORDS for w in words)


@dataclass
class PIIMasker:
    """Mask text for the LLM and restore real values afterwards. One instance per request.

    `known` maps an entity type (PERSON, PHONE, EMAIL_ADDRESS, ADDRESS) to exact values taken from
    the claim form / customer record; they are masked wherever they appear, in any letter case.
    """

    known: Mapping[str, Sequence[str]] = field(default_factory=dict)
    _placeholders: dict[tuple[str, str], str] = field(default_factory=dict, init=False, repr=False)
    _originals: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _counters: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __repr__(self) -> str:  # never print the mapping, even by accident
        return f"PIIMasker(placeholders={len(self._originals)})"

    # -- placeholders ---------------------------------------------------------------------

    def placeholder_for(self, entity: str, original: str) -> str:
        key = (entity, _normalise(original))
        if key not in self._placeholders:
            prefix = PLACEHOLDER_PREFIX.get(entity, entity)
            self._counters[prefix] = self._counters.get(prefix, 0) + 1
            placeholder = f"<{prefix}_{self._counters[prefix]}>"
            self._placeholders[key] = placeholder
            # a value we know from the claim form is restored in ITS spelling, not in whatever
            # letter case / spacing the document happened to print it
            self._originals[placeholder] = self._canonical().get(key, original.strip())
        return self._placeholders[key]

    def _canonical(self) -> dict[tuple[str, str], str]:
        return {
            (entity, _normalise(value)): value.strip()
            for entity, values in self.known.items()
            for value in values
            if value
        }

    @property
    def count(self) -> int:
        return len(self._originals)

    # -- detection ------------------------------------------------------------------------

    def _known_spans(self, text: str) -> list[RecognizerResult]:
        spans: list[RecognizerResult] = []
        for entity, values in self.known.items():
            for value in values:
                if not value or len(value.strip()) < 3:
                    continue
                parts = [re.escape(p) for p in value.split()]
                pattern = re.compile(r"\s+".join(parts), re.IGNORECASE)
                spans += [
                    RecognizerResult(entity, m.start(), m.end(), 1.0)
                    for m in pattern.finditer(text)
                ]
        return spans

    @staticmethod
    def _trim(text: str, start: int, end: int) -> tuple[int, int]:
        """The span without leading / trailing whitespace (NER spans can swallow a newline)."""
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    @staticmethod
    def _labeled_names(text: str) -> list[RecognizerResult]:
        results = []
        for m in LABELED_NAME_RE.finditer(text):
            end = m.start("name")
            for word in re.finditer(r"\S+", m.group("name")):
                if word.group().strip(".").lower() in NEXT_LABEL_WORDS:
                    break
                end = m.start("name") + word.end()
            if end > m.start("name"):
                results.append(RecognizerResult("PERSON", m.start("name"), end, 0.9))
        return results

    @staticmethod
    def _protected_spans(text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for p in PROTECTED_PATTERNS for m in p.finditer(text)]

    def detect(self, text: str) -> list[RecognizerResult]:
        analyzer, _ = get_engines()
        found = analyzer.analyze(
            text=text, language="en", entities=["PERSON", "EMAIL_ADDRESS", *PLACEHOLDER_PREFIX]
        )
        results = []
        for r in found:
            if r.entity_type == "PERSON":
                start, end = self._trim(text, r.start, r.end)
                if r.score < 0.8 or not looks_like_a_name(text[start:end]):
                    continue
                r = RecognizerResult("PERSON", start, end, r.score)
            results.append(r)
        results += self._known_spans(text) + self._labeled_names(text)
        protected = self._protected_spans(text)
        known_starts = {(r.start, r.end) for r in results if r.score >= 1.0}
        kept = []
        for r in results:
            overlaps_protected = any(r.start < end and start < r.end for start, end in protected)
            if overlaps_protected and (r.start, r.end) not in known_starts:
                continue
            kept.append(r)
        return kept

    # -- mask / unmask --------------------------------------------------------------------

    def mask(self, text: str) -> str:
        if not text.strip():
            return text
        results = self.detect(text)
        if not results:
            return text
        _, anonymizer = get_engines()

        def operator(entity: str) -> OperatorConfig:
            return OperatorConfig(
                "custom", {"lambda": lambda value: self.placeholder_for(entity, value)}
            )

        entities = {r.entity_type for r in results}
        masked = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={e: operator(e) for e in entities},
        )
        return masked.text

    def unmask(self, text: str) -> str:
        """Put the real values back. Tolerates the model changing brackets or letter case."""

        def replace(match: re.Match[str]) -> str:
            placeholder = f"<{match.group(1).upper()}_{match.group(2)}>"
            return self._originals.get(placeholder, match.group(0))

        return re.sub(r"<\s*([A-Za-z]+)_(\d+)\s*>", replace, text)

    def unmask_value(self, value: str | None) -> str | None:
        return None if value is None else self.unmask(value)


def iter_known(*groups: tuple[str, Iterable[str | None]]) -> dict[str, list[str]]:
    """Helper: build the `known` mapping, skipping empty values."""
    return {entity: [v for v in values if v] for entity, values in groups}
