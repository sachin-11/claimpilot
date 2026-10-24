"""PDF -> structured blocks.

Parsing is two phases, with the cleaner in between:

1. ``extract_blocks``  : PyMuPDF text/table blocks with page, font size, bold flag and bbox.
2. (cleaner removes running headers/footers/page numbers, otherwise they look like headings)
3. ``assign_structure``: detects headings, builds the heading hierarchy and returns ``Block``s
   that each know their ``section_path``.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pymupdf  # PyMuPDF

logger = logging.getLogger(__name__)


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"


@dataclass
class RawBlock:
    """A text block (or a whole table, as markdown) straight from the PDF."""

    text: str
    page: int  # 1-based
    bbox: tuple[float, float, float, float]
    page_height: float
    font_size: float
    bold: bool
    is_table: bool = False


@dataclass
class Block:
    text: str
    page: int
    section_path: str
    block_type: BlockType
    page_end: int | None = None  # set when a sentence continues onto the next page


@dataclass
class ParsedDocument:
    blocks: list[RawBlock]
    page_count: int
    skipped_pages: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------------------
# Phase 1: extraction
# ---------------------------------------------------------------------------------------

_BOLD_NAME_HINTS = ("bold", "black", "heavy", "semibold", "demi")


def _span_is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or any(h in span["font"].lower() for h in _BOLD_NAME_HINTS)


def table_to_markdown(rows: list[list[str | None]]) -> str:
    """Markdown for one table. The first row is used as the header."""

    def cell(value: str | None) -> str:
        return " ".join((value or "").split()).replace("|", "\\|")

    cleaned = [[cell(c) for c in row] for row in rows]
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    lines = ["| " + " | ".join(cleaned[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in cleaned[1:]]
    return "\n".join(lines)


def _line_style(line: dict) -> tuple[str, float, bool] | None:
    """(text, dominant font size, bold) for one PyMuPDF line, or None if it is blank."""
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans:
        return None
    text = "".join(s["text"] for s in line["spans"]).rstrip()
    weight: Counter[float] = Counter()
    bold_chars = 0
    for span in spans:
        n = len(span["text"].strip())
        weight[round(span["size"], 1)] += n
        if _span_is_bold(span):
            bold_chars += n
    total = sum(len(s["text"].strip()) for s in spans)
    return text, weight.most_common(1)[0][0], bold_chars * 2 >= total


def _inside(bbox: tuple[float, float, float, float], area: pymupdf.Rect) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return area.x0 - 2 <= cx <= area.x1 + 2 and area.y0 - 2 <= cy <= area.y1 + 2


def _table_rows(page: pymupdf.Page, table: "pymupdf.table.Table") -> list[list[str | None]]:
    """Cell texts per row.

    Table.extract() joins words inside a cell without spaces ("24months"), which would corrupt
    waiting periods and limits, so each cell is read with get_text(clip=cell) instead.
    """
    rows: list[list[str | None]] = []
    for row in table.rows:
        rows.append(
            [
                " ".join(page.get_text("text", clip=pymupdf.Rect(cell)).split()) if cell else None
                for cell in row.cells
            ]
        )
    return rows


# A physical line that is just a clause heading such as "6.1. Pre-Existing Diseases" or
# "10.17 Premium Payment in Installments". Real wordings often print these in the body font,
# directly followed by body lines, so font-based detection alone would miss them.
CLAUSE_LINE_RE = re.compile(r"^\d{1,2}(?:\.\d{1,3})+\.?\s+[A-Z]")
MAX_CLAUSE_LINE_CHARS = 70
# "(i)", "(iv)", "a)", "ii." or a bare clause number "8": printed as a block of its own
LONE_MARKER_RE = re.compile(r"^(?:\([A-Za-z0-9]{1,5}\)|[A-Za-z0-9]{1,5}[.)]|\d{1,2})$")
# first line of a numbered clause ("7.5 Cosmetic ...", "1.1Procedure ..."): starts a new paragraph
CLAUSE_START_RE = re.compile(r"^\d{1,2}(?:\.\d{1,3})+\.?\s*[A-Z(]")


def is_clause_heading_line(text: str) -> bool:
    text = text.strip()
    return (
        len(text) <= MAX_CLAUSE_LINE_CHARS
        and bool(CLAUSE_LINE_RE.match(text))
        and not text.endswith((".", ";", ","))
        and " means " not in text
    )


def merge_lone_markers(blocks: list[RawBlock]) -> list[RawBlock]:
    """Glue a lone list marker / clause number to the block printed on the same line.

    "(ii)" + "Assisted Reproduction ..." or "8" + "Moratorium Period:". The partner may sort
    before or after the marker (its top edge can be a point higher), so search both ways.
    """
    consumed: set[int] = set()
    replacement: dict[int, RawBlock] = {}
    for i, marker in enumerate(blocks):
        if marker.is_table or not LONE_MARKER_RE.match(marker.text.strip()):
            continue
        candidates = [
            (abs(other.bbox[1] - marker.bbox[1]), j)
            for j, other in enumerate(blocks)
            if j != i
            and j not in consumed
            and not other.is_table
            and other.page == marker.page
            and abs(other.bbox[1] - marker.bbox[1]) <= 6
            and other.bbox[0] > marker.bbox[0]
            and not LONE_MARKER_RE.match(other.text.strip())
        ]
        if not candidates:
            continue
        _, j = min(candidates)
        partner = blocks[j]
        consumed.add(i)
        replacement[j] = RawBlock(
            text=f"{marker.text.strip()} {partner.text}",
            page=partner.page,
            bbox=(
                min(marker.bbox[0], partner.bbox[0]),
                min(marker.bbox[1], partner.bbox[1]),
                max(marker.bbox[2], partner.bbox[2]),
                max(marker.bbox[3], partner.bbox[3]),
            ),
            page_height=partner.page_height,
            font_size=partner.font_size,
            bold=partner.bold,
        )
    return [replacement.get(i, block) for i, block in enumerate(blocks) if i not in consumed]


# a line that is only "7.4", "10.17", "(i)", "iv." ... (number and text sit on separate lines)
LONE_LINE_RE = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,3})+\.?|\([A-Za-z0-9]{1,5}\)|[A-Za-z0-9]{1,5}[.)]|\d{1,2})$"
)


def _merge_marker_lines(
    lines: list[dict],
) -> list[tuple[str, tuple[float, float, float, float], float, bool]]:
    """Lines as (text, bbox, size, bold), with a lone marker/number joined to the next line.

    Many wordings print hanging indents so that "7.4" and "Change-of-Gender treatments" come out
    of the PDF as two lines; the text line's style wins.
    """
    infos = [(i, tuple(line["bbox"])) for line in lines if (i := _line_style(line)) is not None]
    merged: list[tuple[str, tuple[float, float, float, float], float, bool]] = []
    pending: tuple[str, tuple[float, float, float, float]] | None = None
    for position, ((text, size, bold), bbox) in enumerate(infos):
        if pending is not None:
            text = f"{pending[0]} {text}"
            bbox = (
                min(pending[1][0], bbox[0]),
                min(pending[1][1], bbox[1]),
                max(pending[1][2], bbox[2]),
                max(pending[1][3], bbox[3]),
            )
            pending = None
        if LONE_LINE_RE.match(text.strip()) and position < len(infos) - 1:
            pending = (text.strip(), bbox)
            continue
        merged.append((text, bbox, size, bold))
    return merged


def _group_lines(
    lines: list[dict],
) -> list[tuple[str, tuple[float, float, float, float], float, bool]]:
    """Group the lines of one PyMuPDF block into runs of identical style.

    One PyMuPDF block can hold a heading line followed by body lines; splitting wherever the
    style changes makes headings blocks of their own.
    """
    groups: list[tuple[str, tuple[float, float, float, float], float, bool]] = []
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    style: tuple[float, bool] | None = None

    def close() -> None:
        if texts and style is not None:
            bbox = (
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            )
            groups.append(("\n".join(texts), bbox, style[0], style[1]))

    for text, bbox, size, bold in _merge_marker_lines(lines):
        if is_clause_heading_line(text):
            # its own block, flagged bold so heading detection treats it as emphasised
            close()
            texts, boxes, style = [], [], None
            groups.append((text, bbox, size, True))
            continue
        if style is not None and (size, bold) != style:
            close()
            texts, boxes = [], []
        style = (size, bold)
        texts.append(text)
        boxes.append(bbox)
    close()
    return groups


def _extract_page(page: pymupdf.Page, number: int) -> list[RawBlock]:
    height = page.rect.height
    blocks: list[RawBlock] = []

    tables: list[pymupdf.Rect] = []
    for table in page.find_tables().tables:
        rows = _table_rows(page, table)
        if len(rows) < 2 or max(len(r) for r in rows) < 2:  # a ruled box is not a table
            continue
        rect = pymupdf.Rect(table.bbox)
        tables.append(rect)
        blocks.append(
            RawBlock(
                text=table_to_markdown(rows),
                page=number,
                bbox=tuple(rect),
                page_height=height,
                font_size=0.0,
                bold=False,
                is_table=True,
            )
        )

    for raw in page.get_text("dict", sort=True)["blocks"]:
        if raw["type"] != 0 or any(_inside(raw["bbox"], t) for t in tables):
            continue
        for text, bbox, size, bold in _group_lines(raw["lines"]):
            blocks.append(
                RawBlock(
                    text=text,
                    page=number,
                    bbox=bbox,
                    page_height=height,
                    font_size=size,
                    bold=bold,
                )
            )

    blocks.sort(key=lambda b: (round(b.bbox[1]), b.bbox[0]))
    return merge_lone_markers(blocks)


def extract_blocks(pdf_path: Path, *, min_page_chars: int = 30) -> ParsedDocument:
    """All text blocks and tables of a PDF in reading order.

    Pages with (almost) no extractable text are scanned images: they are skipped with a warning
    (OCR is Module 6).
    """
    blocks: list[RawBlock] = []
    skipped: list[int] = []
    with pymupdf.open(pdf_path) as doc:
        page_count = doc.page_count
        for index, page in enumerate(doc, start=1):
            page_blocks = _extract_page(page, index)
            chars = sum(len(b.text.strip()) for b in page_blocks)
            if chars < min_page_chars:
                logger.warning(
                    "page_skipped_scanned",
                    extra={"pdf": pdf_path.name, "page": index, "chars": chars},
                )
                skipped.append(index)
                continue
            blocks.extend(page_blocks)
    return ParsedDocument(blocks=blocks, page_count=page_count, skipped_pages=skipped)


# ---------------------------------------------------------------------------------------
# Phase 2: structure (headings -> section_path)
# ---------------------------------------------------------------------------------------

MAX_HEADING_CHARS = 150

# "Section 4", "Clause 4.2: Title", "Article IV"
SECTION_RE = re.compile(
    r"^(?:section|clause|article|part)\s+(?P<num>[0-9]+(?:\.[0-9]+)*|[IVXivx]+|[A-Z])\b[\s:.\-]*",
    re.IGNORECASE,
)
# "4.", "4.2", "4.2.1 Title" (small numbers only, so amounts like "5000 rupees" are not headings)
NUMBERED_RE = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+(?P<title>\S.*)$")
# list markers: bullets, "a)", "(i)", "1)", "(a)"
LIST_RE = re.compile(r"^\s*(?:[-*•▪●◦■]|\(?[a-zA-Z]\)|\(?[ivxIVX]{1,4}\)|\d{1,2}\))\s+")

# Inside a body block, "1. text", "a. text", "ii. text" also start a list item.
LIST_ITEM_LINE_RE = re.compile(
    r"^\s*(?:[-*•▪●◦■]|\(?[a-zA-Z]\)|\(?[ivxIVX]{1,4}\)|\d{1,2}\)|\d{1,2}\.|[a-zA-Z]\.|[ivxIVX]{1,4}\.)\s+\S"
)

_DOT_LEADERS_RE = re.compile(r"\.{3,}|…")


@dataclass
class _Heading:
    level: int | None  # None = unnumbered: level comes from the stack
    numbered: bool


def _is_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 3 and text.upper() == text


def detect_heading(text: str, font_size: float, bold: bool, body_size: float) -> _Heading | None:
    """Is this block a heading? Uses numbering, ALL CAPS, and font size / bold vs the body."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    joined = " ".join(lines)
    if not joined or len(lines) > 3 or len(joined) > MAX_HEADING_CHARS:
        return None
    if _DOT_LEADERS_RE.search(joined):  # table-of-contents entry
        return None
    if LONE_MARKER_RE.match(joined) and not joined.isdigit():  # "(ii)" on its own
        return None
    # several list items / clauses in one block are a list, not one long heading
    if any(LIST_ITEM_LINE_RE.match(ln) or CLAUSE_START_RE.match(ln) for ln in lines[1:]):
        return None

    emphasised = bold or font_size >= body_size * 1.15
    ends_like_sentence = joined.endswith((".", ";", ","))

    if m := SECTION_RE.match(joined):
        num = m.group("num")
        level = num.count(".") + 1 if num[0].isdigit() else 1
        return _Heading(level=level, numbered=True)

    if m := NUMBERED_RE.match(joined):
        title = m.group("title")
        short_unterminated = len(joined) <= 100 and not ends_like_sentence and title[0].isupper()
        if emphasised or short_unterminated:
            return _Heading(level=m.group("num").count(".") + 1, numbered=True)
        return None

    if LIST_RE.match(joined):  # "a)", "(i)" are list items unless clearly styled as headings
        return _Heading(level=None, numbered=False) if emphasised and len(joined) <= 80 else None

    if len(joined) <= 80 and _is_all_caps(joined):
        return _Heading(level=None, numbered=False)
    if emphasised and len(joined) <= 100 and not ends_like_sentence:
        return _Heading(level=None, numbered=False)
    return None


def _body_sizes(blocks: list[RawBlock]) -> tuple[dict[int, float], float]:
    """Body font size per page (character-weighted mode) plus the document-wide fallback."""
    per_page: dict[int, Counter[float]] = {}
    overall: Counter[float] = Counter()
    for b in blocks:
        if b.is_table:
            continue
        weight = len(b.text)
        per_page.setdefault(b.page, Counter())[b.font_size] += weight
        overall[b.font_size] += weight
    doc_body = overall.most_common(1)[0][0] if overall else 10.0
    sizes = {
        page: counter.most_common(1)[0][0]
        for page, counter in per_page.items()
        if sum(counter.values()) >= 300  # enough text to trust the page's own body size
    }
    return sizes, doc_body


def _split_body(text: str, page: int, path: str) -> list[Block]:
    """Turn a body block into paragraph / list_item blocks (one list item per marker line)."""
    blocks: list[Block] = []
    current: list[str] = []
    kind = BlockType.PARAGRAPH

    def flush() -> None:
        if current:
            blocks.append(Block(" ".join(current), page, path, kind))
        current.clear()

    for line in (ln.strip() for ln in text.split("\n")):
        if not line:
            continue
        if CLAUSE_START_RE.match(line):
            flush()
            kind = BlockType.PARAGRAPH
        elif LIST_ITEM_LINE_RE.match(line):
            flush()
            kind = BlockType.LIST_ITEM
        elif not current:
            kind = BlockType.PARAGRAPH
        current.append(line)
    flush()
    return blocks


_SENTENCE_END = ".:;?!)"


def _append_body(result: list[Block], new_blocks: list[Block]) -> None:
    """Append body blocks, gluing a block that continues the previous sentence (it starts in
    lower case and the previous one has no closing punctuation), also across pages."""
    for block in new_blocks:
        prev = result[-1] if result else None
        if (
            prev is not None
            and prev.block_type in (BlockType.PARAGRAPH, BlockType.LIST_ITEM)
            and block.block_type == BlockType.PARAGRAPH
            and prev.section_path == block.section_path
            and block.text[:1].islower()
            and prev.text.rstrip()[-1:] not in _SENTENCE_END
        ):
            prev.text = f"{prev.text} {block.text}"
            prev.page_end = max(prev.page_end or prev.page, block.page)
        else:
            result.append(block)


def assign_structure(blocks: list[RawBlock]) -> list[Block]:
    """Detect headings and tag every block with its section_path ("4. Exclusions > 4.2 ...")."""
    page_body, doc_body = _body_sizes(blocks)
    # stack entries: (level, title, strength) with strength = (font size, bold)
    stack: list[tuple[int, str, tuple[float, bool]]] = []
    result: list[Block] = []

    def path() -> str:
        return " > ".join(title for _, title, _ in stack)

    for raw in blocks:
        if raw.is_table:
            result.append(Block(raw.text, raw.page, path(), BlockType.TABLE))
            continue

        body = page_body.get(raw.page, doc_body)
        heading = detect_heading(raw.text, raw.font_size, raw.bold, body)
        if heading is None:
            _append_body(result, _split_body(raw.text, raw.page, path()))
            continue

        title = " ".join(raw.text.split())
        strength = (raw.font_size, raw.bold)
        if heading.numbered and heading.level is not None:
            level = heading.level
            while stack and stack[-1][0] >= level:
                stack.pop()
        else:
            # Unnumbered: same or weaker style than the top of the stack means sibling/ancestor.
            while stack and stack[-1][2] <= strength:
                stack.pop()
            level = stack[-1][0] + 1 if stack else 1
        stack.append((level, title, strength))
        result.append(Block(title, raw.page, path(), BlockType.HEADING))
    return result
