"""Test helpers: a small synthetic policy PDF (reportlab) and a deterministic fake embedder."""

import hashlib
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.ingestion.config import IngestionSettings
from app.models.policy import EMBEDDING_DIM

H1 = ParagraphStyle(
    "h1", fontName="Helvetica-Bold", fontSize=14, leading=18, spaceBefore=14, spaceAfter=10
)
H2 = ParagraphStyle(
    "h2", fontName="Helvetica-Bold", fontSize=12, leading=16, spaceBefore=10, spaceAfter=8
)
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=13, spaceAfter=10)

HEADER = "ACME HEALTH INSURANCE - Policy Wording"
UIN = "UIN: ACMHLIP12345V012021"


def whitespace_tokens(text: str) -> int:
    """Token counter for tests: one token per word, no tiktoken download needed."""
    return len(text.split())


def ingestion_settings(tmp_path: Path, **overrides) -> IngestionSettings:
    values = {
        "policies_dir": tmp_path,
        "debug_dir": tmp_path / "debug",
        "chunk_max_tokens": 120,
        "chunk_overlap_tokens": 30,
        "chunk_min_tokens": 15,
    }
    values.update(overrides)
    return IngestionSettings(_env_file=None, **values)


def _decorate(c: canvas.Canvas, doc) -> None:
    width, height = A4
    c.setFont("Helvetica", 8)
    c.drawString(50, height - 30, HEADER)
    c.drawString(50, 40, f"Page {doc.page}")
    c.drawRightString(width - 50, 40, UIN)


def make_policy_pdf(path: Path, *, cataract_months: int = 24) -> Path:
    """4-page policy: definitions, exclusions (with a waiting-period table), long claims section.

    A running header, "Page n" and a UIN footer appear on every page.
    """
    story = [
        Paragraph("1. Definitions", H1),
        Paragraph(
            "Hospital means an institution with at least ten inpatient beds that is registered "
            "with the local authority and employs qualified nursing staff round the clock.",
            BODY,
        ),
        Paragraph("2. Exclusions", H1),
        Paragraph("2.1 Permanent Exclusions", H2),
        Paragraph("a) Cosmetic surgery unless required after an accident.", BODY),
        Paragraph("b) Treatment of obesity or weight control.", BODY),
        PageBreak(),
        Paragraph("2.2 Specific Waiting Periods", H2),
        Paragraph("The following conditions are covered only after the waiting period:", BODY),
    ]
    table = Table(
        [
            ["Disease", "Waiting period"],
            ["Cataract", f"{cataract_months} months"],
            ["Hernia", "24 months"],
        ]
    )
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black)]))
    story += [table, Spacer(1, 12), PageBreak(), Paragraph("3. Claims Procedure", H1)]
    for i in range(1, 15):
        story.append(
            Paragraph(
                f"Clause {i}: The insurer shall assess claim number {i} within thirty days of "
                f"receiving all documents. A payment of Rs. {i},000 is made to the insured. "
                f"Any dispute about clause {i} is referred to the ombudsman.",
                BODY,
            )
        )
    SimpleDocTemplate(str(path), pagesize=A4).build(
        story, onFirstPage=_decorate, onLaterPages=_decorate
    )
    return path


def make_pdf_with_blank_page(path: Path) -> Path:
    """Page 1 has text, page 2 has no text layer at all (like a scanned page)."""
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont("Helvetica", 11)
    c.drawString(60, 700, "1. Scope")
    c.drawString(60, 680, "This page has plenty of extractable text for the parser to find.")
    c.showPage()
    c.rect(100, 100, 300, 300, fill=1)  # an "image" only
    c.showPage()
    c.save()
    return path


def fake_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()  # 32 bytes
    return [b / 255 for b in digest] * (EMBEDDING_DIM // 32)


class FakeEmbedder:
    """Deterministic stand-in for OpenAI embeddings. Records every call."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("embedding service down")
        return [fake_vector(t) for t in texts]

    @property
    def texts_embedded(self) -> int:
        return sum(len(c) for c in self.calls)
