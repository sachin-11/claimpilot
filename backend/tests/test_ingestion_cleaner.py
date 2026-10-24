from app.ingestion.cleaner import clean_blocks, fix_hyphenation, normalize_text
from app.ingestion.parser import RawBlock


def block(text: str, page: int, y: float, *, height: float = 800) -> RawBlock:
    return RawBlock(
        text=text,
        page=page,
        bbox=(50, y, 500, y + 10),
        page_height=height,
        font_size=10,
        bold=False,
    )


def test_repeated_headers_footers_and_page_numbers_are_removed() -> None:
    blocks: list[RawBlock] = []
    for page in range(1, 6):
        blocks += [
            block("ACME INSURANCE Policy Wording", page, 20),  # running header
            block(f"Body text of page {page} about coverage limits.", page, 300),
            block(f"Page {page} of 5", page, 770),  # footer
            block("UIN: ACMHLIP12345V012021", page, 785),
        ]
    blocks.append(block("Header that appears only once", 3, 25))
    blocks.append(block("7", 4, 790))  # bare page number in the footer zone
    blocks.append(block("5", 4, 400))  # a number in the body is content, keep it

    cleaned, report = clean_blocks(blocks)
    texts = [b.text for b in cleaned]

    assert not any("ACME INSURANCE" in t for t in texts)
    assert not any("Page " in t and " of 5" in t for t in texts)
    assert not any("UIN" in t for t in texts)
    assert "7" not in texts
    assert "5" in texts
    assert "Header that appears only once" in texts
    assert sum(t.startswith("Body text of page") for t in texts) == 5
    assert report.repeated_removed >= 10


def test_line_level_boilerplate_inside_a_block() -> None:
    blocks = [block("Cover limit applies.\nPage 3 of 12\nUIN: SHAHLIP22001V012122", 1, 300)]
    cleaned, _ = clean_blocks(blocks)
    assert cleaned[0].text == "Cover limit applies."


def test_hyphenation_fix() -> None:
    assert fix_hyphenation("hospitali-\nzation expenses") == "hospitalization expenses"
    assert fix_hyphenation("pre-\nexisting disease") == "pre-existing disease"
    assert fix_hyphenation("Co-\nOperative society") == "Co-Operative society"
    assert fix_hyphenation("clause 4-\n5 applies") == "clause 4-5 applies"
    assert fix_hyphenation("a well-known term") == "a well-known term"  # no line break: untouched


def test_numbers_and_amounts_survive_normalisation() -> None:
    text = "Rs. 5,000 per day, ₹ 1,00,000 sum insured, 2.5% co-pay, 30-60 days, 24 months."
    assert normalize_text(text) == text


def test_unicode_and_whitespace_normalisation() -> None:
    raw_text = "•  Room   rent limit “at” ‘most’\n\nﬁnal​  clause"
    assert normalize_text(raw_text) == "- Room rent limit \"at\" 'most'\nfinal clause"


def test_footer_that_drifts_between_pages_is_still_removed() -> None:
    blocks = []
    for page in range(1, 11):  # footer y drifts 784 -> 791 like the real wording
        blocks.append(
            block("Policy Wording - Arogya Sanjeevani UIN: X", page, 784 if page < 4 else 791)
        )
        blocks.append(block(f"Body of page {page}", page, 300))

    cleaned, _ = clean_blocks(blocks)

    assert [b.text for b in cleaned] == [f"Body of page {p}" for p in range(1, 11)]


def test_section_number_above_a_heading_is_not_a_page_number() -> None:
    top = block("9\nCLAIM PROCEDURE", 3, 60)  # top of the page, but two lines: a heading
    lone = block("12", 3, 790)  # the real page number
    cleaned, _ = clean_blocks([top, lone])
    assert [b.text for b in cleaned] == ["9\nCLAIM PROCEDURE"]
