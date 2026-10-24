"""Noise removal between extraction and structure detection.

Numbers, percentages and amounts are never rewritten: only whitespace, ligatures, quotes,
bullets and line-break hyphenation are normalised.
"""

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace

from app.ingestion.parser import RawBlock

logger = logging.getLogger(__name__)

EDGE_FRACTION = 0.18  # top / bottom share of the page where headers and footers live

PAGE_X_OF_Y_RE = re.compile(r"^\s*page\s*\d+\s*(?:of|/)\s*\d+\s*$", re.IGNORECASE)
BARE_PAGE_NUMBER_RE = re.compile(r"^\s*[-–]?\s*\d{1,4}\s*[-–]?\s*$")
# UIN / registration / CIN boilerplate lines, e.g. "UIN: SHAHLIP22001V012122"
BOILERPLATE_RE = re.compile(
    r"^\s*(?:UIN\s*[:\-]?\s*[A-Z0-9]{8,}|CIN\s*[:\-]?\s*[A-Z0-9]{10,}|"
    r"IRDA(?:I)?\s*(?:Reg(?:istration|n)?\.?)?\s*(?:No\.?|Number)\s*[:\-]?\s*\d+)",
    re.IGNORECASE,
)

_CHAR_MAP = str.maketrans(
    {
        "­": None,  # soft hyphen
        "​": None,
        "‌": None,
        "‍": None,
        "﻿": None,
        " ": " ",
        " ": " ",
        " ": " ",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "•": "-",
        "▪": "-",
        "●": "-",
        "◦": "-",
        "■": "-",
        "∙": "-",
    }
)

# A hyphen before a line break is kept (not a soft wrap) after these, e.g. "pre-\nexisting".
_KEEP_HYPHEN_PREFIXES = {
    "pre", "post", "non", "co", "self", "anti", "semi", "multi", "cross", "in", "out",
    "day", "well", "life", "long", "short", "sub", "inter", "intra", "re", "de", "ex",
}  # fmt: skip
_HYPHEN_BREAK_RE = re.compile(r"(\w+)-[ \t]*\n[ \t]*(\w+)")


def _join_hyphenated(match: re.Match[str]) -> str:
    left, right = match.group(1), match.group(2)
    keep = (
        right[0].isupper()
        or right[0].isdigit()
        or left[-1].isdigit()
        or left.lower() in _KEEP_HYPHEN_PREFIXES
    )
    return f"{left}-{right}" if keep else f"{left}{right}"


def fix_hyphenation(text: str) -> str:
    """'hospitali-\\nzation' -> 'hospitalization', but 'pre-\\nexisting' -> 'pre-existing'."""
    return _HYPHEN_BREAK_RE.sub(_join_hyphenated, text)


def normalize_text(text: str) -> str:
    """Unicode and whitespace clean-up that never touches digits, %, Rs. or the rupee sign."""
    text = text.translate(_CHAR_MAP)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _repeat_key(block: RawBlock) -> tuple[str, str]:
    """Text with digits masked + top/bottom zone. Exact y is deliberately not part of the key:
    real footers drift by a few points between pages."""
    digits_masked = re.sub(r"\d+", "#", " ".join(block.text.lower().split()))
    zone = "top" if block.bbox[1] < block.page_height * EDGE_FRACTION else "bottom"
    return digits_masked, zone


def _in_edge_zone(block: RawBlock) -> bool:
    top = block.bbox[1] < block.page_height * EDGE_FRACTION
    bottom = block.bbox[3] > block.page_height * (1 - EDGE_FRACTION)
    return top or bottom


def find_repeated_edge_blocks(blocks: list[RawBlock]) -> set[int]:
    """ids of header/footer blocks: same text (digits masked) at the same height on most pages."""
    pages = {b.page for b in blocks}
    if len(pages) < 3:
        return set()
    needed = max(3, math.ceil(len(pages) * 0.5))
    seen: dict[tuple[str, str], set[int]] = defaultdict(set)
    for b in blocks:
        if not b.is_table and _in_edge_zone(b):
            seen[_repeat_key(b)].add(b.page)
    return {
        id(b)
        for b in blocks
        if not b.is_table and _in_edge_zone(b) and len(seen[_repeat_key(b)]) >= needed
    }


@dataclass
class CleanReport:
    repeated_removed: int = 0
    lines_removed: int = 0


def clean_blocks(blocks: list[RawBlock]) -> tuple[list[RawBlock], CleanReport]:
    """Drop headers/footers, page numbers and boilerplate; normalise the remaining text."""
    report = CleanReport()
    repeated = find_repeated_edge_blocks(blocks)
    cleaned: list[RawBlock] = []

    for block in blocks:
        if id(block) in repeated:
            report.repeated_removed += 1
            continue
        if block.is_table:
            cleaned.append(replace(block, text=block.text.translate(_CHAR_MAP)))
            continue

        text = normalize_text(fix_hyphenation(block.text))
        kept: list[str] = []
        lines = text.split("\n")
        for line in lines:
            noise = (
                PAGE_X_OF_Y_RE.match(line)
                or BOILERPLATE_RE.match(line)
                # a bare number is a page number only when it is the whole block: "9" above
                # "CLAIM PROCEDURE" is a section number
                or (len(lines) == 1 and _in_edge_zone(block) and BARE_PAGE_NUMBER_RE.match(line))
            )
            if noise:
                report.lines_removed += 1
            else:
                kept.append(line)
        if kept:
            cleaned.append(replace(block, text="\n".join(kept)))

    logger.info(
        "clean_done",
        extra={
            "blocks_in": len(blocks),
            "blocks_out": len(cleaned),
            "repeated_removed": report.repeated_removed,
            "lines_removed": report.lines_removed,
        },
    )
    return cleaned, report
