"""Fakes and fixtures for the intake tests: no real LLM, OCR service or network is used."""

import re
from collections.abc import Callable

import cv2
import numpy as np
import pymupdf
from langchain_core.messages import AIMessage

from app.intake.config import IntakeSettings
from app.intake.ocr import OCRResult
from app.intake.schemas import (
    DocumentText,
    LLMExtraction,
    LLMFieldSource,
    LLMLineItem,
    LLMObservation,
    PageText,
)
from app.models.enums import DocType


def intake_settings(**overrides) -> IntakeSettings:
    return IntakeSettings(_env_file=None, **overrides)


class FakeOCR:
    """Counts calls; returns fixed text."""

    name = "fake"

    def __init__(self, text: str = "OCR TEXT total 1000", confidence: float | None = 85.0):
        self.text = text
        self.confidence = confidence
        self.calls = 0

    def ocr_image(self, image: np.ndarray) -> OCRResult:
        self.calls += 1
        return OCRResult(self.text, self.confidence)


def make_pdf(pages: list[str | None]) -> bytes:
    """PDF bytes: a str page gets that text layer, None gives a page with only a drawn box."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text is None:
            page.draw_rect(pymupdf.Rect(100, 100, 400, 400), fill=(0, 0, 0))
        else:
            page.insert_text((72, 100), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def png_bytes(width: int = 200, height: int = 100) -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((height, width, 3), 255, np.uint8))
    assert ok
    return encoded.tobytes()


def jpg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((80, 120, 3), 200, np.uint8))
    assert ok
    return encoded.tobytes()


def noisy_text_image(
    text: str = "PATIENT NAME SUHANI BALA TOTAL 77917",
    angle: float = 3.0,
    noise: float = 18.0,
    contrast: float = 0.55,
    seed: int = 1,
) -> np.ndarray:
    """A skewed, noisy, low-contrast 'scan' of one line of text."""
    rng = np.random.default_rng(seed)
    image = np.full((420, 1100, 3), 255, np.uint8)
    cv2.putText(image, text, (40, 220), cv2.FONT_HERSHEY_DUPLEX, 1.6, (20, 20, 20), 3, cv2.LINE_AA)
    matrix = cv2.getRotationMatrix2D((550, 210), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (1100, 420), borderValue=(255, 255, 255))
    image = image.astype(np.float32) * contrast + (255 * (1 - contrast)) * 0.8
    image += rng.normal(0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def document(
    doc_type: DocType, pages: list[tuple[str, str, float | None]], name: str | None = None
) -> DocumentText:
    """DocumentText from (text, method, ocr_confidence) per page."""
    return DocumentText(
        document_id=f"doc-{doc_type.value}",
        doc_type=doc_type,
        file_name=name or f"{doc_type.value}.pdf",
        is_scanned=any(method == "ocr" for _, method, _ in pages),
        pages=[
            PageText(page=i, text=text, method=method, ocr_confidence=conf)  # type: ignore[arg-type]
            for i, (text, method, conf) in enumerate(pages, start=1)
        ],
    )


BILL_TEXT = """Prakash Nursing & Maternity Home
55/350 Goyal Road, Indore
Phone: 3806576302 | GSTIN: 31PLCZA4438Z1ZR | Reg. No: REG/4714/2006
FINAL INPATIENT BILL
Patient Name Suhani Bala Bill No. BILL/2025/36987
Age / Sex 24 / Female UHID UH6627725
Date of Admission 12-11-2025 Treating Doctor Dr. Veer Mand
Date of Discharge 14-11-2025 Diagnosis Normal Vaginal Delivery
Patient Phone 07649716752
1 Room Rent - Private Room 2 6,000 12,000
2 Surgeon Fee - Normal Vaginal Delivery 1 6,234 6,234
3 Anaesthetist Fee 1 4,115 4,115
Sub Total 22,349
GST @5% (room rent) 600
TOTAL 22,949"""

SUMMARY_TEXT = """DISCHARGE SUMMARY
Patient Name Suhani Bala UHID UH6627725
Date of Admission 12-11-2025 Date of Discharge 14-11-2025
Diagnosis Normal Vaginal Delivery (ICD-10: O80)
Procedure Performed Normal Vaginal Delivery
Condition at Discharge Stable. Discharge Advice follow-up after 7 days."""


def good_llm_extraction(**overrides) -> LLMExtraction:
    """A correct extraction of BILL_TEXT / SUMMARY_TEXT (patient name left as the placeholder
    unless overridden)."""
    base = dict(
        patient_name="Suhani Bala",
        patient_age=24,
        gender="Female",
        hospital_name="Prakash Nursing & Maternity Home",
        admission_date="2025-11-12",
        discharge_date="2025-11-14",
        diagnosis_text="Normal Vaginal Delivery",
        icd10_code="O80",
        procedure="Normal Vaginal Delivery",
        line_items=[
            LLMLineItem(
                description="Room Rent - Private Room",
                category="room",
                quantity=2,
                unit_price=6000,
                amount=12000,
            ),
            LLMLineItem(
                description="Surgeon Fee - Normal Vaginal Delivery",
                category="surgeon",
                quantity=1,
                unit_price=6234,
                amount=6234,
            ),
            LLMLineItem(
                description="Anaesthetist Fee",
                category="surgeon",
                quantity=1,
                unit_price=4115,
                amount=4115,
            ),
        ],
        bill_subtotal=22349,
        taxes=600,
        bill_total=22949,
        field_sources=[
            LLMFieldSource(
                field="admission_date", doc_type="discharge_summary", page=1, confidence="high"
            ),
            LLMFieldSource(field="bill_total", doc_type="hospital_bill", page=1, confidence="high"),
        ],
        observations=[
            LLMObservation(
                doc_type="hospital_bill",
                patient_name="Suhani Bala",
                admission_date="2025-11-12",
                discharge_date="2025-11-14",
                total_amount=22949,
            ),
            LLMObservation(
                doc_type="discharge_summary",
                patient_name="Suhani Bala",
                admission_date="2025-11-12",
                discharge_date="2025-11-14",
            ),
        ],
    )
    base.update(overrides)
    return LLMExtraction(**base)


class ScriptedChat:
    """Chat model whose structured answers come from a script (one entry per call).

    Entries are LLMExtraction objects or callables(messages) -> LLMExtraction. `answers` may also
    hold plain strings for ainvoke (classification)."""

    def __init__(
        self,
        script: list[LLMExtraction | Callable[[list], LLMExtraction]] | None = None,
        text: str = "other",
        model_name: str = "fake-model",
    ):
        self.script = list(script or [])
        self.text = text
        self.model_name = model_name
        self.calls: list[list] = []  # messages of every structured call
        self.text_calls: list[list] = []

    def __call__(self) -> "ScriptedChat":
        return self

    def with_structured_output(self, schema, include_raw=True, method=None):
        parent = self

        class Runner:
            async def ainvoke(self, messages):
                parent.calls.append(list(messages))
                index = min(len(parent.calls), len(parent.script)) - 1
                item = parent.script[index]
                parsed = item(messages) if callable(item) else item
                raw = AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                    response_metadata={"model_name": parent.model_name},
                )
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return Runner()

    async def ainvoke(self, messages):
        self.text_calls.append(list(messages))
        return AIMessage(content=self.text)

    def all_prompt_text(self) -> str:
        return "\n".join(str(m.content) for call in [*self.calls, *self.text_calls] for m in call)


def placeholder_after(label: str, messages: list) -> str | None:
    """The <PERSON_n> placeholder that follows `label` in the user message (None if absent)."""
    text = "\n".join(str(m.content) for m in messages)
    match = re.search(rf"{re.escape(label)}\s*(<PERSON_\d+>)", text)
    return match.group(1) if match else None
