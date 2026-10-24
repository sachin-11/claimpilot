"""Upload validation and file storage.

Files are validated by their MAGIC BYTES (never the extension or the client's content type),
opened with the real parser (PyMuPDF / Pillow) to prove they are not corrupt, and stored under
claims/<claim_number>/<sha256>.<ext>. The file name chosen by the client is never used.
"""

import hashlib
import io
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pymupdf
from PIL import Image, UnidentifiedImageError

from app.core.errors import AppError
from app.intake.config import IntakeSettings

# extension -> (mime type, magic bytes)
FILE_TYPES: dict[str, tuple[str, bytes]] = {
    "pdf": ("application/pdf", b"%PDF-"),
    "png": ("image/png", b"\x89PNG\r\n\x1a\n"),
    "jpg": ("image/jpeg", b"\xff\xd8\xff"),
}
CLAIM_NUMBER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,49}$")


class UploadError(AppError):
    status_code = 400
    code = "invalid_upload"


class FileTooLargeError(UploadError):
    status_code = 413
    code = "file_too_large"


class TooManyFilesError(UploadError):
    status_code = 422
    code = "too_many_files"


@dataclass(frozen=True)
class ValidatedFile:
    data: bytes
    sha256: str
    ext: str
    mime: str
    page_count: int
    is_image: bool


def detect_type(data: bytes) -> tuple[str, str] | None:
    """(extension, mime) from the first bytes, or None if it is not PDF/PNG/JPEG."""
    for ext, (mime, magic) in FILE_TYPES.items():
        if data.startswith(magic):
            return ext, mime
    return None


def validate_upload(filename: str, data: bytes, settings: IntakeSettings) -> ValidatedFile:
    """Raise UploadError unless `data` is a readable PDF, PNG or JPEG within the limits."""
    if not data:
        raise UploadError(f"{filename!r} is empty")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise FileTooLargeError(f"{filename!r} is larger than {settings.max_upload_mb} MB")

    detected = detect_type(data)
    if detected is None:
        raise UploadError(f"{filename!r} is not a PDF, PNG or JPEG file (checked by content)")
    ext, mime = detected

    if ext == "pdf":
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise UploadError(f"{filename!r} is password protected")
                page_count = doc.page_count
        except UploadError:
            raise
        except Exception as exc:  # noqa: BLE001 - any parser failure means "not a usable PDF"
            raise UploadError(f"{filename!r} is not a readable PDF") from exc
        if page_count == 0:
            raise UploadError(f"{filename!r} has no pages")
        if page_count > settings.max_pdf_pages:
            raise UploadError(f"{filename!r} has {page_count} pages (max {settings.max_pdf_pages})")
        return ValidatedFile(data, hashlib.sha256(data).hexdigest(), ext, mime, page_count, False)

    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > settings.max_image_pixels:
                raise UploadError(f"{filename!r} is too large in pixels")
            image.verify()
    except UploadError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise UploadError(f"{filename!r} is not a readable image") from exc
    return ValidatedFile(data, hashlib.sha256(data).hexdigest(), ext, mime, 1, True)


def check_file_count(existing: int, adding: int, settings: IntakeSettings) -> None:
    if existing + adding > settings.max_files_per_claim:
        raise TooManyFilesError(
            f"a claim can hold at most {settings.max_files_per_claim} files "
            f"({existing} already stored, {adding} new)"
        )


# ------------------------------------------------------------------------- storage backends


class Storage(Protocol):
    """Where claim files live. LocalStorage now; S3Storage later (same interface)."""

    def save(self, claim_number: str, sha256: str, ext: str, data: bytes) -> str:
        """Store the bytes, return the storage key (a relative path)."""

    def read(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


class LocalStorage:
    """Files under `root`; keys are paths relative to it. `fallback_roots` are searched too for
    reading (seeded claims keep their files in data/, not in the storage dir)."""

    def __init__(self, root: Path, fallback_roots: tuple[Path, ...] = ()) -> None:
        self.root = root
        self.fallback_roots = fallback_roots

    def save(self, claim_number: str, sha256: str, ext: str, data: bytes) -> str:
        if not CLAIM_NUMBER_RE.match(claim_number):
            raise UploadError(f"invalid claim number {claim_number!r}")
        if ext not in FILE_TYPES or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise UploadError("invalid file extension or hash")
        key = f"claims/{claim_number}/{sha256}.{ext}"
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".part")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                os.replace(tmp_name, target)  # atomic: readers never see half a file
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
        return key

    def _resolve(self, key: str) -> Path | None:
        for root in (self.root, *self.fallback_roots):
            candidate = (root / key).resolve()
            # refuse "../" tricks: the resolved file must stay inside the root
            if candidate.is_relative_to(root.resolve()) and candidate.is_file():
                return candidate
        return None

    def read(self, key: str) -> bytes:
        path = self._resolve(key)
        if path is None:
            raise FileNotFoundError(key)
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._resolve(key) is not None


class S3Storage:
    """Placeholder for the AWS deployment (Module 12). Same interface as LocalStorage."""

    def __init__(self, bucket: str) -> None:
        self.bucket = bucket

    def save(self, claim_number: str, sha256: str, ext: str, data: bytes) -> str:
        raise NotImplementedError("S3Storage is a stub; use LocalStorage")

    def read(self, key: str) -> bytes:
        raise NotImplementedError("S3Storage is a stub; use LocalStorage")

    def exists(self, key: str) -> bool:
        raise NotImplementedError("S3Storage is a stub; use LocalStorage")
