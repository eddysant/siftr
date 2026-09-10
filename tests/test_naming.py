"""Bracket-tag parsing and formatting in filenames."""

import pytest

from siftr.naming import (
    InvalidTagName,
    format_stem,
    needs_rename,
    parse_stem,
    retag_name,
    tags_in_name,
    validate_tag,
)

KNOWN = ["glaze", "outdoor", "wide shot"]


# ------------------------------------------------------------------ validation


def test_validate_collapses_whitespace():
    assert validate_tag("  wide   shot  ") == "wide shot"


@pytest.mark.parametrize("bad", ["", "   ", "a[b", "a]b", "a/b", "a\\b", ".hidden"])
def test_validate_rejects_unusable_names(bad):
    with pytest.raises(InvalidTagName):
        validate_tag(bad)


# ---------------------------------------------------------------------- parsing


def test_parse_extracts_known_tags():
    assert parse_stem("beach [glaze]", KNOWN) == ("beach", ["glaze"])


def test_parse_leaves_unknown_brackets_in_place():
    """mediate writes [2] and [site 3]; siftr must not eat them."""
    base, tags = parse_stem("holiday [2] [glaze]", KNOWN)
    assert base == "holiday [2]"
    assert tags == ["glaze"]


def test_parse_preserves_position_of_unmanaged_brackets():
    base, tags = parse_stem("a [site 3] b [outdoor] c", KNOWN)
    assert base == "a [site 3] b c"
    assert tags == ["outdoor"]


def test_parse_is_case_insensitive_but_returns_canonical_case():
    assert parse_stem("x [GLAZE]", KNOWN) == ("x", ["glaze"])


def test_parse_handles_multiword_tags():
    assert parse_stem("x [wide shot]", KNOWN) == ("x", ["wide shot"])


def test_parse_deduplicates_repeats():
    assert parse_stem("x [glaze] [glaze]", KNOWN) == ("x", ["glaze"])


def test_parse_with_no_brackets():
    assert parse_stem("plain name", KNOWN) == ("plain name", [])


def test_parse_with_no_known_tags_keeps_everything():
    assert parse_stem("x [glaze]", []) == ("x [glaze]", [])


# -------------------------------------------------------------------- formatting


def test_format_sorts_tags_canonically():
    """Two runs producing the same set must produce identical names."""
    assert format_stem("x", ["outdoor", "glaze"]) == format_stem("x", ["glaze", "outdoor"])


def test_format_with_no_tags_returns_bare_base():
    assert format_stem("x", []) == "x"


def test_format_deduplicates():
    assert format_stem("x", ["glaze", "glaze"]) == "x [glaze]"


# ----------------------------------------------------------------- retag_name


def test_retag_adds_tags_and_keeps_extension():
    assert retag_name("IMG_1.jpg", ["glaze"], KNOWN) == "IMG_1 [glaze].jpg"


def test_retag_is_idempotent():
    once = retag_name("IMG_1.jpg", ["glaze"], KNOWN)
    assert retag_name(once, ["glaze"], KNOWN) == once


def test_retag_removes_a_tag_that_no_longer_matches():
    assert retag_name("IMG_1 [glaze].jpg", [], KNOWN) == "IMG_1.jpg"


def test_retag_replaces_rather_than_appends():
    assert retag_name("IMG_1 [glaze].jpg", ["outdoor"], KNOWN) == "IMG_1 [outdoor].jpg"


def test_retag_preserves_foreign_brackets_while_changing_its_own():
    assert (
        retag_name("IMG_1 [2] [glaze].jpg", ["outdoor"], KNOWN) == "IMG_1 [2] [outdoor].jpg"
    )


def test_retag_only_keeps_the_final_extension():
    assert retag_name("archive.tar.gz", ["glaze"], KNOWN) == "archive.tar [glaze].gz"


def test_retag_never_produces_a_bare_extension():
    """A file called '[glaze].jpg' must not become '.jpg'."""
    assert retag_name("[glaze].jpg", [], KNOWN) == "[glaze].jpg"


def test_retag_unknown_tag_argument_still_written():
    assert retag_name("a.jpg", ["brand new"], [*KNOWN, "brand new"]) == "a [brand new].jpg"


def test_tags_in_name_reads_back_what_retag_wrote():
    name = retag_name("a.jpg", ["outdoor", "glaze"], KNOWN)
    assert tags_in_name(name, KNOWN) == ["glaze", "outdoor"]


def test_needs_rename_is_false_when_already_correct():
    assert not needs_rename("a [glaze].jpg", ["glaze"], KNOWN)
    assert needs_rename("a.jpg", ["glaze"], KNOWN)
