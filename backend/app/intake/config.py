from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import get_settings


def _default_storage_dir() -> Path:
    # <repo>/storage on the host, /storage in the container (mounted); same layout as data/.
    return get_settings().data_dir.parent / "storage"


class IntakeSettings(BaseSettings):
    """Claim intake knobs (upload limits, OCR, extraction). All overridable from .env."""

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # Storage / upload
    storage_dir: Path = Field(default_factory=_default_storage_dir)
    max_upload_mb: int = 10
    max_files_per_claim: int = 10
    max_pdf_pages: int = 30  # more pages than this are refused (OCR cost / DoS guard)
    max_image_pixels: int = 40_000_000  # decompression-bomb guard for PNG/JPG

    # OCR
    ocr_provider: Literal["tesseract", "textract"] = "tesseract"
    ocr_langs: str = "eng+hin"  # languages that are not installed are dropped automatically
    ocr_min_chars: int = 50  # a PDF page with fewer text-layer characters is OCR'd
    ocr_low_conf: float = 60.0  # mean word confidence below this -> fields marked low confidence
    ocr_render_dpi: int = 200  # DPI used to rasterise PDF pages for OCR
    ocr_min_width: int = 1600  # smaller images are upscaled to roughly this width

    # Extraction
    extraction_max_retries: int = 2  # extra LLM attempts after a validation failure
    # sum(line items) vs subtotal, subtotal + taxes vs total: the LARGER of pct and abs.
    # 0.1%, not 1%: on real data 1% (about Rs 950 on a Rs 94,000 bill) hid an OCR misread of 800.
    amount_tolerance_pct: float = 0.001
    amount_tolerance_abs: float = 10.0
    max_document_chars: int = 60_000  # text sent to the LLM is truncated beyond this

    # Consistency
    name_match_threshold: float = 0.75  # fuzzy ratio below this = names differ


@lru_cache
def get_intake_settings() -> IntakeSettings:
    return IntakeSettings()
