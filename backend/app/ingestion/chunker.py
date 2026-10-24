"""Structure-aware chunking: one chunk per section, split/merged only when needed.

Rules (all limits come from IngestionSettings so Module 5 can tune them):
- a heading starts a new section; a section is its heading plus the blocks below it
- sections with no body (a parent heading directly followed by a child) are dropped: the
  parent title is already part of every child's section_path
- sections > max tokens are split on paragraph / list-item / sentence boundaries with overlap
- tables are atomic: never split, never used as overlap
- tiny sections (< min tokens) are merged with a related neighbour (same parent, or ancestor)
"""

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from app.ingestion.config import IngestionSettings
from app.ingestion.parser import Block, BlockType

logger = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]

PATH_SEP = " > "
_ABBREVIATIONS = {
    "rs", "no", "nos", "e.g", "i.e", "etc", "vs", "dr", "mr", "mrs", "ms", "sec", "cl", "art",
    "approx", "max", "min", "sr", "jr", "inc", "ltd", "co", "viz",
}  # fmt: skip
_CLAUSE_NUMBER_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,3})*)\.?(?=\s|[A-Z(])")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'₹])")


_encoding_cache: dict[str, object] = {}


def _get_encoding():  # type: ignore[no-untyped-def]
    """The cl100k_base encoder, or None if it cannot be loaded (cached, so we try only once)."""
    if "enc" not in _encoding_cache:
        try:
            import tiktoken

            _encoding_cache["enc"] = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 - e.g. offline: the BPE file is downloaded on first use
            logger.warning("tiktoken_unavailable_using_estimate")
            _encoding_cache["enc"] = None
    return _encoding_cache["enc"]


def count_tokens(text: str) -> int:
    """Token count with tiktoken cl100k_base (~4 characters per token if it is unusable)."""
    encoding = _get_encoding()
    if encoding is None:
        return max(1, len(text) // 4)
    return len(encoding.encode(text, disallowed_special=()))  # type: ignore[attr-defined]


@dataclass
class Chunk:
    content: str
    section_path: str
    page_start: int
    page_end: int
    token_count: int
    content_hash: str
    block_types: list[str]
    has_table: bool
    chunk_index: int = 0
    headings: list[str] = field(default_factory=list)  # heading lines inside the chunk
    clauses: list[str] = field(default_factory=list)  # clause numbers inside, e.g. ["7.4", "7.5"]


@dataclass
class _Unit:
    text: str
    page: int
    block_type: BlockType
    tokens: int
    page_end: int = 0  # last page the unit touches (0 = same as page)

    def __post_init__(self) -> None:
        self.page_end = self.page_end or self.page


@dataclass
class _Section:
    path: str
    units: list[_Unit] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return _joined_tokens(self.units)

    @property
    def has_body(self) -> bool:
        return any(u.block_type != BlockType.HEADING for u in self.units)


def normalized_hash(content: str) -> str:
    """sha256 of the content with case and whitespace folded, so cosmetic changes don't re-embed."""
    return hashlib.sha256(" ".join(content.lower().split()).encode()).hexdigest()


def _joined_tokens(units: list[_Unit]) -> int:
    return sum(u.tokens for u in units) + max(0, len(units) - 1)  # ~1 token per separator


def _join(units: list[_Unit]) -> str:
    parts: list[str] = []
    for i, unit in enumerate(units):
        if i:
            both_list = (
                unit.block_type == BlockType.LIST_ITEM
                and units[i - 1].block_type == BlockType.LIST_ITEM
            )
            parts.append("\n" if both_list else "\n\n")
        parts.append(unit.text)
    return "".join(parts)


def split_sentences(text: str) -> list[str]:
    """Sentence split that does not break after 'Rs.', 'No.', 'e.g.' and similar."""
    pieces = _SENTENCE_BOUNDARY_RE.split(text)
    sentences: list[str] = []
    for piece in pieces:
        previous = sentences[-1] if sentences else ""
        last_word = previous.rsplit(None, 1)[-1].rstrip(".").lower() if previous else ""
        if previous and previous.endswith(".") and last_word in _ABBREVIATIONS:
            sentences[-1] = f"{previous} {piece}"
        else:
            sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]


def _parent(path: str) -> str:
    return path.rsplit(PATH_SEP, 1)[0] if PATH_SEP in path else ""


def _related(a: str, b: str) -> bool:
    """Same (non-root) parent heading, or one is an ancestor of the other.

    Top-level sections ("5 Cumulative Bonus", "6 Waiting Period") are never merged together.
    """
    if _parent(a) == _parent(b):
        return _parent(a) != ""
    return a.startswith(b + PATH_SEP) or b.startswith(a + PATH_SEP)


def _common_prefix(paths: list[str]) -> str:
    split = [p.split(PATH_SEP) if p else [] for p in paths]
    prefix: list[str] = []
    for parts in zip(*split, strict=False):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    return PATH_SEP.join(prefix)


def _build_sections(blocks: list[Block], tokens: TokenCounter) -> list[_Section]:
    sections: list[_Section] = []
    for block in blocks:
        unit = _Unit(
            block.text, block.page, block.block_type, tokens(block.text), block.page_end or 0
        )
        if block.block_type == BlockType.HEADING or not sections:
            sections.append(_Section(path=block.section_path))
        sections[-1].units.append(unit)
    return sections


def _merge_small(sections: list[_Section], settings: IngestionSettings) -> list[_Section]:
    merged: list[_Section] = []
    for section in sections:
        if merged:
            prev = merged[-1]
            small = min(prev.tokens, section.tokens) < settings.chunk_min_tokens
            fits = prev.tokens + section.tokens + 1 <= settings.chunk_max_tokens
            if small and fits and _related(prev.path, section.path):
                merged[-1] = _Section(
                    path=_common_prefix([prev.path, section.path]),
                    units=prev.units + section.units,
                )
                continue
        merged.append(section)
    return merged


def _explode_paragraph(unit: _Unit, tokens: TokenCounter, max_tokens: int) -> list[_Unit]:
    """A paragraph larger than a whole chunk becomes sentence units."""
    if unit.block_type == BlockType.TABLE or unit.tokens <= max_tokens:
        return [unit]
    sentences = split_sentences(unit.text)
    return [_Unit(s, unit.page, unit.block_type, tokens(s), unit.page_end) for s in sentences] or [
        unit
    ]


def _overlap_tail(units: list[_Unit], overlap_tokens: int) -> list[_Unit]:
    """Whole trailing units (never tables or headings) worth at most overlap_tokens."""
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        if unit.block_type in (BlockType.TABLE, BlockType.HEADING):
            break
        if total + unit.tokens + 1 > overlap_tokens:
            break
        tail.insert(0, unit)
        total += unit.tokens + 1
    return tail


def _split_section(
    section: _Section, tokens: TokenCounter, settings: IngestionSettings
) -> list[list[_Unit]]:
    max_tokens = settings.chunk_max_tokens
    units: list[_Unit] = []
    for unit in section.units:
        units.extend(_explode_paragraph(unit, tokens, max_tokens))

    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    for unit in units:
        has_body = any(u.block_type != BlockType.HEADING for u in current)
        # A chunk that holds only headings is never emitted: the heading stays with its content
        # even if that makes the chunk too big (e.g. heading + one oversized table).
        if has_body and _joined_tokens([*current, unit]) > max_tokens:
            # never leave a heading dangling at the end of a chunk
            carry: list[_Unit] = []
            while len(current) > 1 and current[-1].block_type == BlockType.HEADING:
                carry.insert(0, current.pop())
            groups.append(current)
            tail = [] if carry else _overlap_tail(current, settings.chunk_overlap_tokens)
            while tail and _joined_tokens([*tail, *carry, unit]) > max_tokens:
                tail.pop(0)
            current = [*tail, *carry]
        current.append(unit)
    if current:
        groups.append(current)
    return groups


def _clause_numbers(units: list[_Unit]) -> list[str]:
    """Clause numbers that start a unit ("4.1", "4.2", ...), for citations.

    A plain "1" only counts for headings: in a paragraph it is a numbered list item.
    """
    numbers: list[str] = []
    for unit in units:
        if unit.block_type not in (BlockType.HEADING, BlockType.PARAGRAPH):
            continue
        m = _CLAUSE_NUMBER_RE.match(unit.text)
        if not m or m.group(1) in numbers:
            continue
        if unit.block_type == BlockType.PARAGRAPH and "." not in m.group(1):
            continue
        numbers.append(m.group(1))
    return numbers


def _drop_excluded_sections(blocks: list[Block], settings: IngestionSettings) -> list[Block]:
    patterns = [re.compile(p, re.IGNORECASE) for p in settings.exclude_section_patterns]
    if not patterns:
        return blocks
    kept = [
        b
        for b in blocks
        if not any(p.search(b.section_path.split(PATH_SEP, 1)[0]) for p in patterns)
    ]
    if len(kept) != len(blocks):
        logger.info("sections_excluded", extra={"blocks_dropped": len(blocks) - len(kept)})
    return kept


def chunk_blocks(
    blocks: list[Block],
    settings: IngestionSettings,
    tokens: TokenCounter = count_tokens,
) -> list[Chunk]:
    """Blocks (with section paths) -> citation-ready chunks."""
    blocks = _drop_excluded_sections(blocks, settings)
    sections = [s for s in _build_sections(blocks, tokens) if s.has_body]
    sections = _merge_small(sections, settings)

    chunks: list[Chunk] = []
    seen_hashes: set[str] = set()
    for section in sections:
        for units in _split_section(section, tokens, settings):
            content = _join(units)
            digest = normalized_hash(content)
            if digest in seen_hashes:
                logger.info("chunk_duplicate_dropped", extra={"section": section.path})
                continue
            seen_hashes.add(digest)

            count = tokens(content)
            if count > settings.chunk_max_tokens:
                logger.warning(
                    "chunk_oversized",
                    extra={
                        "section": section.path,
                        "tokens": count,
                        "limit": settings.chunk_max_tokens,
                        "has_table": any(u.block_type == BlockType.TABLE for u in units),
                    },
                )
            chunks.append(
                Chunk(
                    content=content,
                    section_path=section.path,
                    page_start=min(u.page for u in units),
                    page_end=max(u.page_end for u in units),
                    token_count=count,
                    content_hash=digest,
                    block_types=sorted({u.block_type.value for u in units}),
                    has_table=any(u.block_type == BlockType.TABLE for u in units),
                    headings=[u.text for u in units if u.block_type == BlockType.HEADING],
                    clauses=_clause_numbers(units),
                )
            )
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index
    return chunks


def embed_text(policy_name: str, chunk: Chunk) -> str:
    """Contextual text that is embedded instead of the bare content."""
    section = chunk.section_path or "(preamble)"
    return f"Policy: {policy_name} | Section: {section}\n\n{chunk.content}"
