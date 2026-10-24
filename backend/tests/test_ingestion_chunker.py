import logging

import pytest
from pdf_fixtures import ingestion_settings, whitespace_tokens

from app.ingestion.chunker import (
    Chunk,
    chunk_blocks,
    embed_text,
    normalized_hash,
    split_sentences,
)
from app.ingestion.parser import Block, BlockType

H, P, T, L = (BlockType.HEADING, BlockType.PARAGRAPH, BlockType.TABLE, BlockType.LIST_ITEM)
TABLE_MD = "| Disease | Waiting |\n|---|---|\n| Cataract | 24 months |\n| Hernia | 24 months |"


def sentence(i: int, words: int = 12) -> str:
    return f"Sentence number {i} " + " ".join(f"w{i}x{j}" for j in range(words - 3)) + "."


def paragraph(start: int, count: int) -> str:
    return " ".join(sentence(i) for i in range(start, start + count))


def blocks_for(path: str, *items: tuple[BlockType, str]) -> list[Block]:
    return [Block(text, 1, path, kind) for kind, text in items]


def settings(tmp_path, **kw):
    return ingestion_settings(tmp_path, **kw)


def test_one_chunk_per_section_when_it_fits(tmp_path) -> None:
    blocks = [
        Block("1. Definitions", 1, "1. Definitions", H),
        Block(paragraph(0, 4), 1, "1. Definitions", P),
        Block("2. Exclusions", 2, "2. Exclusions", H),
        Block(paragraph(10, 4), 2, "2. Exclusions", P),
    ]
    chunks = chunk_blocks(blocks, settings(tmp_path, chunk_min_tokens=5), whitespace_tokens)

    assert [c.section_path for c in chunks] == ["1. Definitions", "2. Exclusions"]
    assert chunks[0].content.startswith("1. Definitions\n\nSentence number 0")
    assert (chunks[1].page_start, chunks[1].page_end) == (2, 2)
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert chunks[0].token_count == whitespace_tokens(chunks[0].content)


def test_tables_are_never_split_and_stay_with_their_heading(tmp_path) -> None:
    path = "2. Exclusions > 2.2 Waiting"
    blocks = [
        Block("2.2 Waiting", 1, path, H),
        Block(paragraph(0, 6), 1, path, P),
        Block(paragraph(6, 6), 1, path, P),
        Block(TABLE_MD, 2, path, T),
        Block(paragraph(12, 6), 2, path, P),
    ]
    cfg = settings(tmp_path, chunk_max_tokens=90, chunk_overlap_tokens=20, chunk_min_tokens=5)

    chunks = chunk_blocks(blocks, cfg, whitespace_tokens)

    holders = [c for c in chunks if "| Disease |" in c.content]
    assert len(holders) == 1
    assert TABLE_MD in holders[0].content  # whole table in one chunk
    assert holders[0].has_table and "table" in holders[0].block_types
    assert len(chunks) >= 3  # the section really was split around the table
    # the table is never copied into a neighbouring chunk as overlap
    assert sum("Cataract" in c.content for c in chunks) == 1


def test_long_section_splits_with_overlap_inside_limits(tmp_path) -> None:
    blocks = [
        Block("3. Claims", 1, "3. Claims", H),
        Block(paragraph(0, 30), 1, "3. Claims", P),
    ]
    cfg = settings(tmp_path, chunk_max_tokens=100, chunk_overlap_tokens=30, chunk_min_tokens=5)

    chunks = chunk_blocks(blocks, cfg, whitespace_tokens)

    assert len(chunks) > 3
    assert all(c.token_count <= 100 for c in chunks)
    for chunk in chunks:
        assert chunk.content.rstrip().endswith(".")  # never cut mid-sentence
        assert chunk.section_path == "3. Claims"
    for previous, following in zip(chunks, chunks[1:], strict=False):
        last_sentence = split_sentences(previous.content)[-1]
        assert last_sentence in following.content  # overlap
        overlap_tokens = whitespace_tokens(last_sentence)
        assert overlap_tokens <= 30
    covered = " ".join(c.content for c in chunks)
    assert all(f"Sentence number {i} " in covered for i in range(30))  # nothing lost


def test_heading_is_not_left_dangling_at_chunk_end(tmp_path) -> None:
    path = "4. Section"
    blocks = [
        Block("4. Section", 1, path, H),
        Block(paragraph(0, 7), 1, path, P),  # ~84 tokens, nearly fills the chunk
        Block("Sub heading", 1, path, H),
        Block(paragraph(7, 4), 1, path, P),
    ]
    cfg = settings(tmp_path, chunk_max_tokens=95, chunk_overlap_tokens=0, chunk_min_tokens=5)

    chunks = chunk_blocks(blocks, cfg, whitespace_tokens)

    assert not any(c.content.rstrip().endswith("Sub heading") for c in chunks)
    assert any(c.content.startswith("Sub heading") for c in chunks)


def test_oversized_table_is_kept_whole_and_logged(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    big = "| A | B |\n|---|---|\n" + "\n".join(f"| row{i} | value{i} |" for i in range(80))
    blocks = blocks_for("5. Annex", (H, "5. Annex"), (T, big))
    cfg = settings(tmp_path, chunk_max_tokens=50, chunk_min_tokens=5)

    with caplog.at_level(logging.WARNING, logger="app.ingestion.chunker"):
        chunks = chunk_blocks(blocks, cfg, whitespace_tokens)

    assert len(chunks) == 1 and big in chunks[0].content
    assert any(r.getMessage() == "chunk_oversized" for r in caplog.records)


def test_tiny_sections_are_merged_under_their_parent(tmp_path) -> None:
    blocks = [
        Block("2. Exclusions", 1, "2. Exclusions", H),  # heading only: dropped
        Block("2.1 Dental", 1, "2. Exclusions > 2.1 Dental", H),
        Block("Dental treatment is excluded.", 1, "2. Exclusions > 2.1 Dental", P),
        Block("2.2 Cosmetic", 1, "2. Exclusions > 2.2 Cosmetic", H),
        Block("Cosmetic surgery is excluded.", 1, "2. Exclusions > 2.2 Cosmetic", P),
        Block("3. Claims", 2, "3. Claims", H),
        Block(paragraph(0, 8), 2, "3. Claims", P),
    ]
    chunks = chunk_blocks(blocks, settings(tmp_path, chunk_min_tokens=15), whitespace_tokens)

    assert [c.section_path for c in chunks] == ["2. Exclusions", "3. Claims"]
    assert "Dental treatment is excluded." in chunks[0].content
    assert "Cosmetic surgery is excluded." in chunks[0].content
    assert not any(c.content.strip() == "2. Exclusions" for c in chunks)


def test_duplicate_chunks_are_dropped_and_indices_stay_dense(tmp_path) -> None:
    same = paragraph(0, 4)
    blocks = [
        Block("A", 1, "A", H),
        Block(same, 1, "A", P),
        Block("A", 2, "B", H),  # identical heading text + body under another path
        Block(same, 2, "B", P),
        Block("C", 3, "C", H),
        Block(paragraph(50, 4), 3, "C", P),
    ]
    chunks = chunk_blocks(blocks, settings(tmp_path, chunk_min_tokens=1), whitespace_tokens)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.content_hash for c in chunks}) == len(chunks)


def test_list_items_are_joined_with_single_newlines(tmp_path) -> None:
    blocks = blocks_for(
        "2.1 Exclusions",
        (H, "2.1 Exclusions"),
        (L, "a) Cosmetic surgery " + "x " * 20),
        (L, "b) Obesity treatment " + "y " * 20),
    )
    chunks = chunk_blocks(blocks, settings(tmp_path), whitespace_tokens)
    assert "\na) Cosmetic" not in chunks[0].content.split("\n\n")[0]
    assert "\nb) Obesity" in chunks[0].content


def test_split_sentences_respects_abbreviations() -> None:
    text = "The limit is Rs. 5,000 per day. Room rent e.g. single room is covered. Next one."
    assert split_sentences(text) == [
        "The limit is Rs. 5,000 per day.",
        "Room rent e.g. single room is covered.",
        "Next one.",
    ]


def test_hash_ignores_case_and_whitespace_but_not_numbers() -> None:
    assert normalized_hash("Cataract  24 months") == normalized_hash("cataract 24\nmonths")
    assert normalized_hash("Cataract 24 months") != normalized_hash("Cataract 36 months")


def test_embed_text_adds_policy_and_section_context() -> None:
    chunk = Chunk(
        content="Cataract: 24 months.",
        section_path="2. Exclusions > 2.2 Waiting",
        page_start=1,
        page_end=1,
        token_count=4,
        content_hash="h",
        block_types=["paragraph"],
        has_table=False,
    )
    assert embed_text("Test Policy", chunk) == (
        "Policy: Test Policy | Section: 2. Exclusions > 2.2 Waiting\n\nCataract: 24 months."
    )
    assert chunk.content == "Cataract: 24 months."  # content itself is stored as is


def test_top_level_sections_are_never_merged_together(tmp_path) -> None:
    blocks = [
        Block("5 Bonus", 1, "5 Bonus", H),
        Block("Short bonus text.", 1, "5 Bonus", P),
        Block("6 Waiting", 1, "6 Waiting", H),
        Block("Short waiting text.", 1, "6 Waiting", P),
    ]
    chunks = chunk_blocks(blocks, settings(tmp_path, chunk_min_tokens=50), whitespace_tokens)
    assert [c.section_path for c in chunks] == ["5 Bonus", "6 Waiting"]


def test_chunks_record_headings_and_clause_numbers_for_citation(tmp_path) -> None:
    blocks = [
        Block("7 EXCLUSIONS", 1, "7 EXCLUSIONS", H),
        Block("7.5 Cosmetic surgery: (Code- Excl08)", 1, "7 EXCLUSIONS > 7.5 Cosmetic", H),
        Block("Expenses for cosmetic surgery are excluded.", 1, "7 EXCLUSIONS > 7.5 Cosmetic", P),
        Block("7.6 Sports (Code- Excl09)", 1, "7 EXCLUSIONS > 7.6 Sports", H),
        Block("7.6 Sports expenses are excluded.", 1, "7 EXCLUSIONS > 7.6 Sports", P),
        Block("1. a list item, not a clause", 1, "7 EXCLUSIONS > 7.6 Sports", P),
    ]
    chunks = chunk_blocks(blocks, settings(tmp_path), whitespace_tokens)
    assert len(chunks) == 1
    assert chunks[0].section_path == "7 EXCLUSIONS"
    assert chunks[0].clauses == ["7.5", "7.6"]
    assert chunks[0].headings == [
        "7.5 Cosmetic surgery: (Code- Excl08)",
        "7.6 Sports (Code- Excl09)",
    ]
