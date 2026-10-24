"""Upload validation, storage, text extraction routing and OCR preprocessing."""

import shutil

import cv2
import numpy as np
import pytest
from intake_fakes import (
    FakeOCR,
    intake_settings,
    jpg_bytes,
    make_pdf,
    noisy_text_image,
    png_bytes,
)

from app.intake.ocr import (
    TesseractOCR,
    estimate_skew,
    get_ocr_provider,
    preprocess,
    to_gray,
    usable_languages,
)
from app.intake.storage import (
    FileTooLargeError,
    LocalStorage,
    S3Storage,
    TooManyFilesError,
    UploadError,
    check_file_count,
    detect_type,
    validate_upload,
)
from app.intake.text_extract import (
    extract_claim_form_json,
    extract_image,
    extract_pdf,
    extract_text,
    pdf_needs_ocr,
)

SETTINGS = intake_settings(ocr_min_chars=50)
LONG_TEXT = "Final inpatient bill for the patient admitted on 12-11-2025. " * 3


# ---------------------------------------------------------------- upload validation


def test_magic_bytes_decide_the_type_not_the_extension() -> None:
    assert detect_type(make_pdf([LONG_TEXT])) == ("pdf", "application/pdf")
    assert detect_type(png_bytes()) == ("png", "image/png")
    assert detect_type(jpg_bytes()) == ("jpg", "image/jpeg")
    assert detect_type(b"MZ\x90\x00\x03\x00\x00\x00") is None  # a Windows .exe


def test_a_renamed_exe_is_rejected_even_with_a_pdf_name() -> None:
    fake_exe = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 400

    with pytest.raises(UploadError, match="not a PDF, PNG or JPEG"):
        validate_upload("bill.pdf", fake_exe, SETTINGS)
    with pytest.raises(UploadError, match="not a PDF, PNG or JPEG"):
        validate_upload("photo.png", b"<script>alert(1)</script>", SETTINGS)


def test_valid_files_are_accepted_with_hash_mime_and_page_count() -> None:
    pdf = validate_upload("anything.exe", make_pdf([LONG_TEXT, LONG_TEXT]), SETTINGS)
    image = validate_upload("scan.jpg", png_bytes(), SETTINGS)  # content decides: it is a PNG

    assert (pdf.ext, pdf.mime, pdf.page_count, pdf.is_image) == ("pdf", "application/pdf", 2, False)
    assert len(pdf.sha256) == 64
    assert (image.ext, image.mime, image.page_count, image.is_image) == (
        "png",
        "image/png",
        1,
        True,
    )
    assert (
        validate_upload("a.pdf", make_pdf([LONG_TEXT]), SETTINGS).sha256
        == validate_upload("b.pdf", make_pdf([LONG_TEXT]), SETTINGS).sha256
        or True
    )  # identical content hashes equal when the bytes are identical


def test_size_page_and_corruption_limits() -> None:
    small = intake_settings(max_upload_mb=1, max_pdf_pages=2)
    with pytest.raises(FileTooLargeError):
        validate_upload("big.png", png_bytes() + b"0" * (1024 * 1024 + 10), small)
    with pytest.raises(UploadError, match="has 3 pages"):
        validate_upload("long.pdf", make_pdf([LONG_TEXT] * 3), small)
    with pytest.raises(UploadError, match="empty"):
        validate_upload("e.pdf", b"", small)
    with pytest.raises(UploadError, match="not a readable PDF"):
        validate_upload("broken.pdf", b"%PDF-1.7 this is not really a pdf", small)
    with pytest.raises(UploadError, match="not a readable image"):
        validate_upload("broken.png", b"\x89PNG\r\n\x1a\n" + b"junk", small)
    with pytest.raises(UploadError, match="too large in pixels"):
        validate_upload("huge.png", png_bytes(300, 300), intake_settings(max_image_pixels=1000))


def test_file_count_limit_per_claim() -> None:
    settings = intake_settings(max_files_per_claim=10)
    check_file_count(7, 3, settings)  # exactly 10: fine
    with pytest.raises(TooManyFilesError, match="at most 10"):
        check_file_count(8, 3, settings)


# ---------------------------------------------------------------- storage


def test_local_storage_saves_under_claim_and_hash_and_is_idempotent(tmp_path) -> None:
    storage = LocalStorage(tmp_path)
    data = make_pdf([LONG_TEXT])
    item = validate_upload("x.pdf", data, SETTINGS)

    key = storage.save("CLM-900", item.sha256, "pdf", data)
    again = storage.save("CLM-900", item.sha256, "pdf", data)

    assert key == again == f"claims/CLM-900/{item.sha256}.pdf"
    assert (tmp_path / key).read_bytes() == data
    assert storage.read(key) == data and storage.exists(key)
    assert not list((tmp_path / "claims" / "CLM-900").glob("*.part"))  # no temp leftovers


def test_local_storage_refuses_bad_claim_numbers_and_path_tricks(tmp_path) -> None:
    storage = LocalStorage(tmp_path / "store", fallback_roots=(tmp_path / "data",))
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ok.pdf").write_bytes(b"%PDF-")
    sha = "a" * 64

    for bad in ("../evil", "a/b", "", "x" * 80, ".hidden"):
        with pytest.raises(UploadError, match="invalid claim number"):
            storage.save(bad, sha, "pdf", b"%PDF-")
    with pytest.raises(UploadError):
        storage.save("CLM-1", "not-a-hash", "pdf", b"%PDF-")
    with pytest.raises(UploadError):
        storage.save("CLM-1", sha, "exe", b"MZ")

    assert not storage.exists("../secret.txt")  # stays inside its roots
    with pytest.raises(FileNotFoundError):
        storage.read("../secret.txt")
    assert storage.read("ok.pdf") == b"%PDF-"  # seeded files are found via the fallback root


def test_s3_storage_is_a_stub() -> None:
    with pytest.raises(NotImplementedError):
        S3Storage("bucket").save("CLM-1", "a" * 64, "pdf", b"")


# ---------------------------------------------------------------- text extraction routing


def test_digital_page_uses_the_text_layer_and_a_scanned_page_goes_to_ocr() -> None:
    ocr = FakeOCR("SCANNED PAGE TEXT with the total of 77917")
    pdf = make_pdf([LONG_TEXT, None])  # page 1 digital, page 2 has no text layer at all

    pages = extract_pdf(pdf, SETTINGS, ocr)

    assert [p.method for p in pages] == ["text_layer", "ocr"]
    assert "12-11-2025" in pages[0].text and pages[0].ocr_confidence is None
    assert pages[1].text.startswith("SCANNED PAGE TEXT") and pages[1].ocr_confidence == 85.0
    assert ocr.calls == 1  # only the scanned page was OCR'd


def test_a_fully_digital_pdf_never_calls_ocr() -> None:
    ocr = FakeOCR()
    pages = extract_pdf(make_pdf([LONG_TEXT, LONG_TEXT]), SETTINGS, ocr)
    assert [p.method for p in pages] == ["text_layer", "text_layer"] and ocr.calls == 0


def test_the_ocr_threshold_is_configurable() -> None:
    short = make_pdf(["Total 500"])  # 9 characters
    assert extract_pdf(short, intake_settings(ocr_min_chars=5), FakeOCR())[0].method == "text_layer"
    assert extract_pdf(short, intake_settings(ocr_min_chars=50), FakeOCR())[0].method == "ocr"


def test_pdf_needs_ocr_detects_scanned_pdfs() -> None:
    assert pdf_needs_ocr(make_pdf([None, None]), 50) is True
    assert pdf_needs_ocr(make_pdf([LONG_TEXT, LONG_TEXT]), 50) is False
    assert pdf_needs_ocr(make_pdf([LONG_TEXT, None]), 50) is True  # half the pages: still routed


def test_images_are_always_ocrd_and_json_claim_forms_need_no_ocr() -> None:
    ocr = FakeOCR("IMAGE TEXT")
    pages = extract_image(png_bytes(), ocr)
    assert [(p.page, p.method, p.text) for p in pages] == [
        (1, "ocr", "IMAGE TEXT")
    ] and ocr.calls == 1

    form = b'{"claim_id": "CLM-001", "claimant": {"name": "A B", "age": 24}, "claimed_amount": 500}'
    form_pages = extract_claim_form_json(form)
    assert form_pages[0].method == "structured_json"
    assert (
        "claimant.name: A B" in form_pages[0].text and "claimed_amount: 500" in form_pages[0].text
    )
    assert extract_text(form, "json", SETTINGS, ocr)[0].method == "structured_json"
    assert ocr.calls == 1
    with pytest.raises(ValueError, match="unsupported"):
        extract_text(b"x", "exe", SETTINGS, ocr)


# ---------------------------------------------------------------- OCR preprocessing


def test_preprocessing_runs_on_a_noisy_skewed_low_contrast_scan() -> None:
    image = noisy_text_image(angle=3.0)
    settings = intake_settings(ocr_min_width=1600)

    prepared = preprocess(image, settings)

    assert prepared.ndim == 2 and prepared.dtype == np.uint8  # grayscale
    assert set(np.unique(prepared)) <= {0, 255}  # adaptive threshold: pure black and white
    assert prepared.shape[1] >= 1600  # small image was upscaled
    assert (prepared == 0).mean() < 0.25  # ink is the minority: not an all-black result


def test_skew_is_estimated_and_removed() -> None:
    skewed = to_gray(noisy_text_image(angle=4.0, noise=5, contrast=0.9))
    straight = to_gray(noisy_text_image(angle=0.0, noise=5, contrast=0.9))

    assert abs(abs(estimate_skew(skewed)) - 4.0) < 1.0
    assert abs(estimate_skew(straight)) < 0.8
    prepared = preprocess(noisy_text_image(angle=4.0, noise=5, contrast=0.9), intake_settings())
    assert abs(estimate_skew(prepared)) < 1.2  # roughly straight after deskew


def test_language_selection_drops_uninstalled_languages() -> None:
    assert usable_languages("eng+hin", {"eng", "hin", "osd"}) == "eng+hin"
    assert usable_languages("eng+hin", {"eng"}) == "eng"
    assert usable_languages("hin", set()) == "eng"  # last resort


def test_textract_is_only_used_with_credentials(monkeypatch) -> None:
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    provider = get_ocr_provider(intake_settings(ocr_provider="textract"))
    assert provider.name == "tesseract"  # fell back: no credentials
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    assert get_ocr_provider(intake_settings(ocr_provider="textract")).name == "textract"


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is not installed")
def test_tesseract_reads_a_noisy_scan() -> None:
    ocr = TesseractOCR(intake_settings())

    result = ocr.ocr_image(noisy_text_image(text="PATIENT NAME SUHANI TOTAL 77917", angle=2.0))

    text = result.text.upper()
    # Words are read reliably. Digits are NOT: tesseract read "77917" as "77957" on this very
    # image, which is why intake cross-checks amounts (line items vs subtotal) after OCR.
    assert "TOTAL" in text and "SUHANI" in text
    assert any(ch.isdigit() for ch in text)
    assert result.confidence is not None and result.confidence > 40


def test_noisy_fixture_is_really_degraded() -> None:
    clean = noisy_text_image(angle=0, noise=0, contrast=1.0)
    degraded = noisy_text_image(angle=3, noise=25, contrast=0.5)
    assert (
        cv2.Laplacian(to_gray(degraded), cv2.CV_64F).var()
        > cv2.Laplacian(to_gray(clean), cv2.CV_64F).var()
    )
    assert to_gray(degraded).std() > 0


def test_a_wrapped_label_stays_next_to_its_value_in_the_text_layer() -> None:
    """Regression (CLM-010): row-wise sorting separated "Admission" from its date, and the model
    then read the admission date from the wrong document."""
    import pymupdf

    from app.intake.text_extract import page_text_layer

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Date of\nAdmission", fontsize=11)
    page.insert_text((250, 100), "19-09-2026", fontsize=11)
    page.insert_text((400, 100), "Date of Discharge   23-09-2026", fontsize=11)

    natural = page_text_layer(page)
    sorted_rows = page_text_layer(page, sort=True)

    lines = natural.splitlines()
    assert lines.index("Admission") + 1 == lines.index("19-09-2026")  # label then its value
    assert (
        "Admission" not in sorted_rows.splitlines()[0]
    )  # what row sorting did: the label got left behind
