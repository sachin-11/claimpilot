"""Text of a stored document: PDF text layer per page, OCR for scanned pages and images."""

import json
import logging

import cv2
import numpy as np
import pymupdf

from app.intake.config import IntakeSettings
from app.intake.ocr import OCRProvider
from app.intake.schemas import PageText

logger = logging.getLogger(__name__)


def decode_image(data: bytes) -> np.ndarray:
    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError("could not decode image")
    return array


def render_page(page: pymupdf.Page, dpi: int) -> np.ndarray:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if pixmap.n == 3 else image.copy()


def page_text_layer(page: pymupdf.Page, sort: bool = False) -> str:
    """The text layer in the PDF's own order. sort=True (row by row) is NOT the default: it splits
    a wrapped label from its value ("Date of  19-09-2026" / "Admission" on the next line), which
    made the model take the wrong document's date."""
    return page.get_text("text", sort=sort).strip()


def pdf_needs_ocr(data: bytes, min_chars: int) -> bool:
    """True when at least half of the pages have (almost) no text layer: a scanned PDF."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        low = sum(1 for page in doc if len(page_text_layer(page)) < min_chars)
        return doc.page_count > 0 and low * 2 >= doc.page_count


def extract_pdf(data: bytes, settings: IntakeSettings, ocr: OCRProvider) -> list[PageText]:
    """Per page: the text layer if it has >= ocr_min_chars characters, else OCR of that page."""
    pages: list[PageText] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for number, page in enumerate(doc, start=1):
            text = page_text_layer(page)
            if len(text) >= settings.ocr_min_chars:
                pages.append(PageText(page=number, text=text, method="text_layer"))
                continue
            logger.info("page_routed_to_ocr", extra={"page": number, "chars": len(text)})
            result = ocr.ocr_image(render_page(page, settings.ocr_render_dpi))
            pages.append(
                PageText(
                    page=number, text=result.text, method="ocr", ocr_confidence=result.confidence
                )
            )
    return pages


def extract_image(data: bytes, ocr: OCRProvider) -> list[PageText]:
    """An uploaded image is always OCR'd (page 1)."""
    result = ocr.ocr_image(decode_image(data))
    return [PageText(page=1, text=result.text, method="ocr", ocr_confidence=result.confidence)]


def extract_claim_form_json(data: bytes) -> list[PageText]:
    """Seeded claim forms are JSON: rendered as 'key: value' lines, no OCR and no LLM needed."""
    form = json.loads(data.decode("utf-8"))
    lines: list[str] = []

    def walk(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                walk(f"{prefix}{key}.", inner)
        else:
            lines.append(f"{prefix.rstrip('.')}: {value}")

    walk("", form)
    return [PageText(page=1, text="\n".join(lines), method="structured_json")]


def extract_text(
    data: bytes, ext: str, settings: IntakeSettings, ocr: OCRProvider
) -> list[PageText]:
    if ext == "pdf":
        return extract_pdf(data, settings, ocr)
    if ext in ("png", "jpg"):
        return extract_image(data, ocr)
    if ext == "json":
        return extract_claim_form_json(data)
    raise ValueError(f"unsupported file type {ext!r}")
