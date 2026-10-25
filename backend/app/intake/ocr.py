"""Pluggable OCR providers + image preprocessing.

Tesseract is the default (local, free). Textract is a stub behind the same interface: it is only
used when OCR_PROVIDER=textract AND AWS credentials exist, otherwise we fall back to Tesseract.
"""

import logging
import os
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
import pytesseract

from app.intake.config import IntakeSettings

logger = logging.getLogger(__name__)

TESSERACT_CONFIG = "--oem 3 --psm 6"  # one uniform block of text: bills and summaries are forms


@dataclass
class OCRResult:
    text: str
    confidence: float | None  # mean word confidence 0-100; None if nothing was recognised


class OCRProvider(Protocol):
    name: str

    def ocr_image(self, image: np.ndarray) -> OCRResult:
        """OCR one page image (BGR, grayscale or RGBA numpy array)."""


class OCRUnavailableError(RuntimeError):
    pass


# ------------------------------------------------------------------------- preprocessing


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def upscale_if_small(gray: np.ndarray, min_width: int) -> np.ndarray:
    height, width = gray.shape[:2]
    if width >= min_width:
        return gray
    scale = min(3.0, min_width / width)
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0) -> float:
    """Skew angle in degrees (positive = text runs counter-clockwise), found by maximising the
    variance of the horizontal projection profile of the inked pixels (sharp rows = straight)."""
    small_width = 700
    scale = small_width / gray.shape[1] if gray.shape[1] > small_width else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(ink) < 50:
        return 0.0
    height, width = ink.shape
    center = (width / 2, height / 2)

    def score(angle: float) -> float:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(ink, matrix, (width, height), flags=cv2.INTER_NEAREST)
        return float(np.var(rotated.sum(axis=1, dtype=np.float64)))

    coarse = max(np.arange(-max_angle, max_angle + 0.01, 0.5), key=score)
    fine = max(np.arange(coarse - 0.5, coarse + 0.51, 0.1), key=score)
    return float(fine)


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess(image: np.ndarray, settings: IntakeSettings) -> np.ndarray:
    """grayscale -> upscale small images -> denoise -> deskew -> adaptive threshold."""
    gray = to_gray(image)
    gray = upscale_if_small(gray, settings.ocr_min_width)
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=15)
    angle = estimate_skew(gray)
    if abs(angle) >= 0.3:
        gray = rotate(gray, angle)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=35, C=15
    )


# ------------------------------------------------------------------------- providers


def installed_tesseract_languages() -> set[str]:
    try:
        return set(pytesseract.get_languages(config=""))
    except Exception:  # noqa: BLE001 - tesseract missing: reported when OCR is attempted
        return set()


def usable_languages(requested: str, installed: set[str]) -> str:
    """'eng+hin' -> only the parts that are installed ('eng' if hin data is missing)."""
    wanted = [lang for lang in requested.split("+") if lang]
    available = [lang for lang in wanted if lang in installed]
    return "+".join(available) or "eng"


class TesseractOCR:
    name = "tesseract"

    def __init__(self, settings: IntakeSettings) -> None:
        self.settings = settings
        self.langs = usable_languages(settings.ocr_langs, installed_tesseract_languages())

    def ocr_image(self, image: np.ndarray) -> OCRResult:
        prepared = preprocess(image, self.settings)
        data = pytesseract.image_to_data(
            prepared,
            lang=self.langs,
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT,
            timeout=self.settings.ocr_timeout_seconds,  # raises RuntimeError when exceeded
        )
        lines: dict[tuple[int, int, int], list[str]] = {}
        confidences: list[float] = []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            conf = float(data["conf"][i])
            if not word or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            confidences.append(conf)
        text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
        mean_conf = sum(confidences) / len(confidences) if confidences else None
        return OCRResult(text=text, confidence=mean_conf)


class TextractOCR:
    """AWS Textract behind the same interface. Stub: wired up in the AWS deployment (Module 12)."""

    name = "textract"

    def ocr_image(self, image: np.ndarray) -> OCRResult:
        raise NotImplementedError("TextractOCR is a stub (implemented with the AWS deployment)")


def aws_credentials_present() -> bool:
    return bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))


def get_ocr_provider(settings: IntakeSettings) -> OCRProvider:
    if settings.ocr_provider == "textract":
        if aws_credentials_present():
            return TextractOCR()
        logger.warning(
            "ocr_fallback", extra={"reason": "textract selected without AWS credentials"}
        )
    return TesseractOCR(settings)
