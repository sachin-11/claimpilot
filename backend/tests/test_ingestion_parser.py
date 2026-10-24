import logging
from pathlib import Path

import pytest
from pdf_fixtures import make_pdf_with_blank_page, make_policy_pdf

from app.ingestion.cleaner import clean_blocks
from app.ingestion.parser import (
    BlockType,
    RawBlock,
    assign_structure,
    detect_heading,
    extract_blocks,
    table_to_markdown,
)


def raw(text: str, *, size: float = 10, bold: bool = False, page: int = 1, y: float = 300):
    return RawBlock(
        text=text,
        page=page,
        bbox=(50, y, 500, y + 12),
        page_height=800,
        font_size=size,
        bold=bold,
    )


def test_numbered_sections_build_section_paths() -> None:
    blocks = [
        raw("1. Definitions", size=14, bold=True),
        raw("Hospital means an institution with beds."),
        raw("2. Exclusions", size=14, bold=True),
        raw("2.2 Specific Waiting Periods", size=12, bold=True),
        raw("Cataract", bold=True),  # unnumbered sub-heading
        raw("Cataract is covered after 24 months."),
        raw("Hernia", bold=True),  # sibling of Cataract, not its child
        raw("Hernia is covered after 24 months."),
        raw("2.3 General Exclusions", size=12, bold=True),
        raw("a) Cosmetic surgery"),
        raw("(i) Dental treatment"),
        raw("3. Claims", size=14, bold=True),
        raw("Notify the insurer."),
    ]

    structured = assign_structure(blocks)
    paths = {b.text: b.section_path for b in structured}

    assert paths["Hospital means an institution with beds."] == "1. Definitions"
    assert paths["Cataract"] == "2. Exclusions > 2.2 Specific Waiting Periods > Cataract"
    assert paths["Cataract is covered after 24 months."] == (
        "2. Exclusions > 2.2 Specific Waiting Periods > Cataract"
    )
    assert paths["Hernia"] == "2. Exclusions > 2.2 Specific Waiting Periods > Hernia"
    assert paths["a) Cosmetic surgery"] == "2. Exclusions > 2.3 General Exclusions"
    assert paths["Notify the insurer."] == "3. Claims"
    types = {b.text: b.block_type for b in structured}
    assert types["2.3 General Exclusions"] == BlockType.HEADING
    assert types["a) Cosmetic surgery"] == BlockType.LIST_ITEM
    assert types["(i) Dental treatment"] == BlockType.LIST_ITEM


def test_all_caps_and_long_numbered_sentences() -> None:
    body = 10
    assert detect_heading("GENERAL CONDITIONS", body, False, body) is not None
    # a numbered sentence is body text, not a heading
    long_clause = (
        "1. The insurer shall pay all admissible claims within thirty days of receiving "
        "every document that was asked for."
    )
    assert detect_heading(long_clause, body, False, body) is None
    # table-of-contents entries are not headings
    assert detect_heading("4. Exclusions ........ 12", body, False, body) is None
    # an amount at the start of a sentence is not a numbered heading
    assert detect_heading("5000 rupees are payable per day.", body, False, body) is None


def test_multiline_block_splits_into_list_items() -> None:
    blocks = [
        raw("1. Scope", size=14, bold=True),
        raw("a) first item\ncontinues here\nb) second item"),
    ]
    structured = assign_structure(blocks)
    items = [b for b in structured if b.block_type == BlockType.LIST_ITEM]
    assert [b.text for b in items] == ["a) first item continues here", "b) second item"]


def test_table_to_markdown() -> None:
    md = table_to_markdown([["Disease", "Months"], ["Cataract", "24"], ["A|B", None]])
    assert md.splitlines() == [
        "| Disease | Months |",
        "|---|---|",
        "| Cataract | 24 |",
        "| A\\|B |  |",
    ]


def test_parse_fixture_pdf_end_to_end(tmp_path: Path) -> None:
    pdf = make_policy_pdf(tmp_path / "policy.pdf")

    parsed = extract_blocks(pdf)
    assert parsed.skipped_pages == []
    tables = [b for b in parsed.blocks if b.is_table]
    assert len(tables) == 1
    assert "| Cataract | 24 months |" in tables[0].text
    raw_text = "\n".join(b.text for b in parsed.blocks)
    assert "ACME HEALTH INSURANCE" in raw_text  # still there before cleaning

    cleaned, report = clean_blocks(parsed.blocks)
    text = "\n".join(b.text for b in cleaned)
    assert "ACME HEALTH INSURANCE" not in text
    assert "UIN:" not in text
    assert "Page 2" not in text
    assert report.repeated_removed >= parsed.page_count

    structured = assign_structure(cleaned)
    table = next(b for b in structured if b.block_type == BlockType.TABLE)
    assert table.section_path == "2. Exclusions > 2.2 Specific Waiting Periods"
    clause = next(b for b in structured if b.text.startswith("Clause 3:"))
    assert clause.section_path == "3. Claims Procedure"
    first_exclusion = next(b for b in structured if b.text.startswith("a) Cosmetic"))
    assert first_exclusion.section_path == "2. Exclusions > 2.1 Permanent Exclusions"
    assert first_exclusion.block_type == BlockType.LIST_ITEM


def test_scanned_page_is_skipped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf = make_pdf_with_blank_page(tmp_path / "scan.pdf")

    with caplog.at_level(logging.WARNING, logger="app.ingestion.parser"):
        parsed = extract_blocks(pdf)

    assert parsed.skipped_pages == [2]
    assert parsed.page_count == 2
    assert {b.page for b in parsed.blocks} == {1}
    warning = next(r for r in caplog.records if r.getMessage() == "page_skipped_scanned")
    assert warning.page == 2


# ---- regressions found on the real Arogya Sanjeevani wording ---------------------------------


def test_clause_heading_lines_are_detected_inside_body_text() -> None:
    from app.ingestion.parser import is_clause_heading_line

    assert is_clause_heading_line("6.1. Pre-Existing Diseases(Code- Excl01)")
    assert is_clause_heading_line("10.17 Premium Payment in Installments")
    assert not is_clause_heading_line("3.12. Cashless Facility means a facility extended by us.")
    assert not is_clause_heading_line("7.5 dioptres is the limit")  # lower-case continuation
    assert not is_clause_heading_line("1. Benign ENT disorders")  # list item, no inner dot


def test_lone_markers_merge_with_partner_in_either_sort_order() -> None:
    from app.ingestion.parser import merge_lone_markers

    marker = raw("(ii)", bold=True, y=120)
    marker.bbox = (68, 120, 82, 133)
    partner = raw("Assisted Reproduction services", y=119)  # 1pt higher: sorts first
    partner.bbox = (104, 119, 400, 132)
    number = raw("8", y=300)
    number.bbox = (68, 300, 76, 312)
    title = raw("Moratorium Period:", bold=True, y=300)
    title.bbox = (86, 300, 170, 312)

    merged = merge_lone_markers([partner, marker, number, title])

    assert [b.text for b in merged] == [
        "(ii) Assisted Reproduction services",
        "8 Moratorium Period:",
    ]


def test_a_marker_alone_is_never_a_heading() -> None:
    assert detect_heading("(iv)", 11, True, 11) is None


def test_sentence_continuing_on_next_block_or_page_is_glued() -> None:
    blocks = [
        raw("6.3 Specific Waiting Period", size=12, bold=True, page=8),
        raw("a) Expenses related to the listed conditions shall be excluded until the", page=8),
        raw("expiry of 24/36 months of continuous coverage.", page=9),
    ]

    structured = assign_structure(blocks)

    item = structured[-1]
    assert item.text == (
        "a) Expenses related to the listed conditions shall be excluded until the "
        "expiry of 24/36 months of continuous coverage."
    )
    assert (item.page, item.page_end) == (8, 9)


def test_numbered_lines_inside_a_block_become_list_items() -> None:
    blocks = [
        raw("6. Waiting", size=14, bold=True),
        raw("1. Benign ENT disorders\n2. Tonsillectomy\nii. 48 Months waiting period"),
    ]
    items = [b.text for b in assign_structure(blocks) if b.block_type == BlockType.LIST_ITEM]
    assert items == ["1. Benign ENT disorders", "2. Tonsillectomy", "ii. 48 Months waiting period"]
