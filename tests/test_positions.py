"""Position universes: indexing convention, the three masks, trivial positions.

Authority is CONTRACTS.md sec.2. Nothing here is inferred from `positions.py`;
the hand-worked expected sets below were derived from the contract by hand and
then checked against the implementation, not the other way round.

Indexing (frozen): position ``i`` is the gap between ``text[i-1]`` and
``text[i]``; valid positions are ``1 .. n-1``; ``0`` and ``n`` are excluded from
every universe.
"""

from __future__ import annotations

import itertools
import unicodedata

import pytest
from fixtures.mini import by_lang, records
from hypothesis import given
from hypothesis import strategies as st

from unsegbench.positions import (
    MASKS,
    boundaries_to_spans,
    compute_mask,
    compute_masks,
    gold_boundaries,
    gold_illegal_rate,
    grapheme_cluster_starts,
    legal_positions,
    spans_to_boundaries,
    trivial_positions,
)
from unsegbench.types import Segmented

ALL_RECORDS = records()
BY_LANG = by_lang()
LANGS = tuple(BY_LANG)

#: (record, lang) pairs, so a mask test can pass the right script grammar.
LANGED_RECORDS = [(rec, lg) for lg, recs in BY_LANG.items() for rec in recs]
LANGED_IDS = [rec.id for rec, _ in LANGED_RECORDS]

ZWSP = "​"


# ==========================================================================
# 1. Indexing convention (CONTRACTS sec.2)
# ==========================================================================


@pytest.mark.parametrize(("rec", "lang"), LANGED_RECORDS, ids=LANGED_IDS)
def test_raw_universe_is_exactly_the_interior_gaps(rec: Segmented, lang: str) -> None:
    """`raw` is all of 1..n-1 -- position i is the gap before text[i]."""
    n = rec.n
    expected = frozenset(range(1, n)) if n >= 2 else frozenset()
    assert compute_mask(rec.text, lang, "raw") == expected


@pytest.mark.parametrize(("rec", "lang"), LANGED_RECORDS, ids=LANGED_IDS)
def test_no_universe_contains_zero_or_n(rec: Segmented, lang: str) -> None:
    """Sentence edges are free for every tokenizer, so they are never scored."""
    masks = compute_masks(rec.text, lang)
    for name, positions in masks.items():
        assert 0 not in positions, name
        assert rec.n not in positions, name
        assert all(1 <= p <= rec.n - 1 for p in positions), name


def test_spans_to_boundaries_drops_zero() -> None:
    """A span starting at 0 contributes no boundary at 0."""
    assert spans_to_boundaries([(0, 3), (3, 7)], 7) == frozenset({3})


def test_spans_to_boundaries_drops_n() -> None:
    """A span ending at n contributes no boundary at n."""
    assert spans_to_boundaries([(0, 4)], 4) == frozenset()
    assert spans_to_boundaries([(0, 2), (2, 4)], 4) == frozenset({2})


def test_spans_to_boundaries_uses_both_starts_and_ends() -> None:
    """Spans need not tile: the end of one word and the start of the next differ."""
    # text = "ab cd", words "ab" (0,2) and "cd" (3,5): the space at index 2 is
    # in no word, so BOTH 2 (an end) and 3 (a start) are gold boundaries.
    assert spans_to_boundaries([(0, 2), (3, 5)], 5) == frozenset({2, 3})


def test_spans_to_boundaries_end_only_boundary() -> None:
    """An interior end with no adjoining start still contributes."""
    assert spans_to_boundaries([(0, 2)], 5) == frozenset({2})


def test_spans_to_boundaries_start_only_boundary() -> None:
    """An interior start with no preceding span still contributes."""
    assert spans_to_boundaries([(3, 5)], 5) == frozenset({3})


def test_spans_to_boundaries_empty_spans() -> None:
    assert spans_to_boundaries([], 10) == frozenset()


def test_spans_to_boundaries_single_char_text() -> None:
    """n == 1: 0 and n are the only positions and both are excluded."""
    assert spans_to_boundaries([(0, 1)], 1) == frozenset()


@pytest.mark.parametrize("rec", ALL_RECORDS, ids=[r.id for r in ALL_RECORDS])
def test_gold_boundaries_are_interior(rec: Segmented) -> None:
    assert all(0 < b < rec.n for b in gold_boundaries(rec))


@pytest.mark.parametrize("rec", ALL_RECORDS, ids=[r.id for r in ALL_RECORDS])
def test_gold_boundaries_delegates_to_spans_to_boundaries(rec: Segmented) -> None:
    assert gold_boundaries(rec) == spans_to_boundaries(rec.spans, rec.n)


def test_gold_boundaries_non_tiling_fixture() -> None:
    """The Thai phrase-space fixture has a genuinely non-tiling gold."""
    rec = BY_LANG["th"][0]
    assert rec.text == "ฉันทำงาน ที่บ้าน"
    # ฉัน(0,3) ทำ(3,5) งาน(5,8) _ ที่(9,12) บ้าน(12,16)
    assert gold_boundaries(rec) == frozenset({3, 5, 8, 9, 12})


def test_gold_boundaries_single_word_sentence_is_empty() -> None:
    """A one-word sentence has no interior gold boundary at all."""
    rec = next(r for r in BY_LANG["zh"] if r.text == "谢谢")
    assert gold_boundaries(rec) == frozenset()


# ==========================================================================
# 2. boundaries_to_spans round-trip
# ==========================================================================


@pytest.mark.parametrize("rec", ALL_RECORDS, ids=[r.id for r in ALL_RECORDS])
def test_round_trip_on_fixture_gold(rec: Segmented) -> None:
    b = gold_boundaries(rec)
    assert spans_to_boundaries(boundaries_to_spans(b, rec.n), rec.n) == b


@pytest.mark.parametrize(
    ("boundaries", "n"),
# improved
    [
        (frozenset(), 5),
        (frozenset({1}), 2),
        (frozenset({1, 2, 3}), 4),
        (frozenset({3}), 10),
        (frozenset({1, 9}), 10),
        (frozenset(range(1, 20)), 20),
    ],
)
def test_round_trip_explicit(boundaries: frozenset[int], n: int) -> None:
    assert spans_to_boundaries(boundaries_to_spans(boundaries, n), n) == boundaries


@st.composite
def _boundary_set(draw: st.DrawFn) -> tuple[frozenset[int], int]:
    n = draw(st.integers(min_value=2, max_value=60))
    return frozenset(draw(st.sets(st.integers(min_value=1, max_value=n - 1)))), n


@given(_boundary_set())
def test_round_trip_property(case: tuple[frozenset[int], int]) -> None:
    boundaries, n = case
    assert spans_to_boundaries(boundaries_to_spans(boundaries, n), n) == boundaries


@pytest.mark.parametrize("rec", ALL_RECORDS, ids=[r.id for r in ALL_RECORDS])
def test_induced_partition_tiles_the_text(rec: Segmented) -> None:
    """The induced partition always covers 0..n exactly once, gold or not."""
    spans = boundaries_to_spans(gold_boundaries(rec), rec.n)
    assert spans[0][0] == 0
    assert spans[-1][1] == rec.n
    assert all(a[1] == b[0] for a, b in itertools.pairwise(spans))
    assert "".join(rec.text[s:e] for s, e in spans) == rec.text


def test_boundaries_to_spans_ignores_out_of_range() -> None:
    """0 and n in the input are not partition cuts."""
    assert boundaries_to_spans(frozenset({0, 2, 5}), 5) == ((0, 2), (2, 5))


def test_boundaries_to_spans_empty_gives_one_span() -> None:
    assert boundaries_to_spans(frozenset(), 6) == ((0, 6),)


def test_boundaries_to_spans_zero_length() -> None:
    assert boundaries_to_spans(frozenset(), 0) == ()


# ==========================================================================
# 3. The three masks
# ==========================================================================


@pytest.mark.parametrize(("rec", "lang"), LANGED_RECORDS, ids=LANGED_IDS)
def test_masks_are_nested_raw_legal_core(rec: Segmented, lang: str) -> None:
    m = compute_masks(rec.text, lang)
    assert m["core"] <= m["legal"] <= m["raw"]


@pytest.mark.parametrize(("rec", "lang"), LANGED_RECORDS, ids=LANGED_IDS)
def test_core_is_legal_minus_trivial(rec: Segmented, lang: str) -> None:
    m = compute_masks(rec.text, lang)
    assert m["core"] == m["legal"] - trivial_positions(rec.text)


@pytest.mark.parametrize(("rec", "lang"), LANGED_RECORDS, ids=LANGED_IDS)
def test_compute_masks_agrees_with_compute_mask(rec: Segmented, lang: str) -> None:
    m = compute_masks(rec.text, lang)
    assert set(m) == set(MASKS)
    for name in MASKS:
        assert m[name] == compute_mask(rec.text, lang, name)


@pytest.mark.parametrize(("rec", "lang"), LANGED_RECORDS, ids=LANGED_IDS)
def test_legal_is_a_subset_of_grapheme_cluster_starts(rec: Segmented, lang: str) -> None:
    """A boundary inside a grapheme cluster is not a linguistic position."""
    assert legal_positions(rec.text, lang) <= grapheme_cluster_starts(rec.text)


def test_compute_mask_rejects_unknown_mask() -> None:
    with pytest.raises(ValueError, match="unknown mask"):
        compute_mask("我喜欢", "zh", "legal_positions")


@pytest.mark.parametrize("text", ["", "我", "ก"])
def test_masks_empty_for_texts_shorter_than_two(text: str) -> None:
    m = compute_masks(text, "zh")
    assert m == dict.fromkeys(MASKS, frozenset())
    for name in MASKS:
        assert compute_mask(text, "zh", name) == frozenset()


# ---- hand-worked expected sets, one (or more) sentence per language -------


def test_hand_worked_masks_zh_simple() -> None:
    """我(0)喜(1)欢(2)吃(3)苹(4)果(5)。(6)  -- n=7.

    raw = 1..6. Every character starts its own grapheme cluster and none is a
    non-starter/non-final, so legal = raw. Position 6 sits before U+3002
    IDEOGRAPHIC FULL STOP (category Po), so it is trivial; nothing else is.
    """
    text = "我喜欢吃苹果。"
    assert BY_LANG["zh"][0].text == text
    m = compute_masks(text, "zh")
    assert m["raw"] == frozenset({1, 2, 3, 4, 5, 6})
    assert m["legal"] == frozenset({1, 2, 3, 4, 5, 6})
    assert trivial_positions(text) == frozenset({6})
    assert m["core"] == frozenset({1, 2, 3, 4, 5})


def test_hand_worked_masks_zh_latin_embed() -> None:
    """我用 Python 写程序。 -- Han/Latin transitions plus real spaces.

    indices: 我0 用1 ␣2 P3 y4 t5 h6 o7 n8 ␣9 写10 程11 序12 。13  (n=14)
    trivial: 2,9 (before a space), 3,10 (after a space), 13 (before Po).
    Note 3 and 10 are script transitions as well as space-adjacent.
    """
    text = "我用 Python 写程序。"
    assert BY_LANG["zh"][8].text == text
    m = compute_masks(text, "zh")
    assert m["raw"] == frozenset(range(1, 14))
    assert m["legal"] == frozenset(range(1, 14))
    assert trivial_positions(text) == frozenset({2, 3, 9, 10, 13})
    assert m["core"] == frozenset({1, 4, 5, 6, 7, 8, 11, 12})


def test_hand_worked_masks_zh_digits() -> None:
    """会议在 2024 年举行。 -- digit run embedded in Han.

    indices: 会0 议1 在2 ␣3 2:4 0:5 2:6 4:7 ␣8 年9 举10 行11 。12  (n=13)
    trivial: 3,8 (before a space), 4,9 (after a space), 12 (before Po).
    """
    text = "会议在 2024 年举行。"
    assert BY_LANG["zh"][9].text == text
    m = compute_masks(text, "zh")
    assert m["legal"] == frozenset(range(1, 13))
    assert trivial_positions(text) == frozenset({3, 4, 8, 9, 12})
    assert m["core"] == frozenset({1, 2, 5, 6, 7, 10, 11})


def test_hand_worked_masks_yue() -> None:
    """我哋今日去飲茶。 -- n=8, uniform Han plus one full-width stop."""
    text = "我哋今日去飲茶。"
    assert BY_LANG["yue"][0].text == text
    m = compute_masks(text, "yue")
    assert m["raw"] == frozenset({1, 2, 3, 4, 5, 6, 7})
    assert m["legal"] == frozenset({1, 2, 3, 4, 5, 6, 7})
    assert trivial_positions(text) == frozenset({7})
    assert m["core"] == frozenset({1, 2, 3, 4, 5, 6})


def test_hand_worked_masks_th() -> None:
    """ฉันทำงาน ที่บ้าน -- n=16.

    indices: ฉ0 ั1 น2 ท3 ำ4 ง5 า6 น7 ␣8 ท9 ี10 ่11 บ12 ้13 า14 น15
    illegal: 1 (U+0E31 MAI HAN AKAT, Mn), 4 (U+0E33 SARA AM, a spacing
    non-starter), 6 and 14 (U+0E32 SARA AA, spacing non-starter), 10/11/13
    (Mn marks), so legal = {2,3,5,7,8,9,12,15}.
    trivial: 8 (before the space) and 9 (after it). Everything else is Thai-Thai.
    """
    text = "ฉันทำงาน ที่บ้าน"
    assert BY_LANG["th"][0].text == text
    m = compute_masks(text, "th")
    assert m["legal"] == frozenset({2, 3, 5, 7, 8, 9, 12, 15})
    assert trivial_positions(text) == frozenset({8, 9})
    assert m["core"] == frozenset({2, 3, 5, 7, 12, 15})


def test_hand_worked_masks_km() -> None:
    """ភាសាខ្មែរ ពិរោះណាស់ -- n=19.

    indices: ភ0 ា1 ស2 ា3 ខ4 ្5 ម6 ែ7 រ8 ␣9 ព10 ិ11 រ12 ោ13 ះ14 ណ15 ា16 ស17 ់18
    illegal: 1,3,7,13,14,16 (dependent vowels/signs U+17B6-17D3), 11 and 18
    (Mn marks), 5 (before U+17D2 COENG) and 6 (immediately after COENG -- the
    subscript consonant binds backwards). So legal = {2,4,8,9,10,12,15,17}.
    trivial: 9 (before the space) and 10 (after it).
    """
    text = "ភាសាខ្មែរ ពិរោះណាស់"
    assert BY_LANG["km"][0].text == text
    m = compute_masks(text, "km")
    assert m["legal"] == frozenset({2, 4, 8, 9, 10, 12, 15, 17})
    assert trivial_positions(text) == frozenset({9, 10})
    assert m["core"] == frozenset({2, 4, 8, 12, 15, 17})


def test_hand_worked_masks_th_short() -> None:
    """น้ำใจ -- น0 ้1 ำ2 ใ3 จ4.

    1 is Mn, 2 is SARA AM (non-starter), 4 follows U+0E43 SARA AI MAIMUAN,
    a leading vowel written before the consonant it modifies (non-final).
    Only 3 survives, and TCC agrees: the clusters are น้ำ | ใจ.
    """
    text = "น้ำใจ"
    assert BY_LANG["th"][8].text == text
    m = compute_masks(text, "th")
    assert m["legal"] == frozenset({3})
    assert trivial_positions(text) == frozenset()
# improved
    assert m["core"] == frozenset({3})


# ==========================================================================
# 4. Trivial positions
# ==========================================================================


def test_zwsp_is_not_caught_by_str_isspace() -> None:
    """Pinned: this is exactly why `_is_space` exists (CONTRACTS sec.2)."""
    assert not ZWSP.isspace()
    assert unicodedata.category(ZWSP) == "Cf"


def test_zwsp_adjacent_positions_are_trivial() -> None:
    """ក(0) ZWSP(1) ខ(2): both 1 and 2 touch the ZWSP."""
    text = "ក" + ZWSP + "ខ"
    assert trivial_positions(text) == frozenset({1, 2})


def test_zwsp_fixture_positions_are_trivial() -> None:
    """The km ZWSP-separated fixture: every ZWSP-adjacent gap is trivial."""
    rec = next(r for r in BY_LANG["km"] if ZWSP in r.text)
    triv = trivial_positions(rec.text)
    zwsp_idx = [i for i, c in enumerate(rec.text) if c == ZWSP]
    assert zwsp_idx
    for i in zwsp_idx:
        assert i in triv
        assert i + 1 in triv


def test_zwsp_adjacent_positions_are_not_in_core() -> None:
    rec = next(r for r in BY_LANG["km"] if ZWSP in r.text)
    core = compute_masks(rec.text, "km")["core"]
    for i, c in enumerate(rec.text):
        if c == ZWSP:
            assert i not in core
            assert i + 1 not in core


@pytest.mark.parametrize(
    "space",
    [
        " ",  # U+0020 SPACE
        "\t",
        "\n",
        "　",  # U+3000 IDEOGRAPHIC SPACE
#         " ",  # U+00A0 NO-BREAK SPACE
        ZWSP,  # U+200B -- str.isspace() is False here
        "⁠",  # U+2060 WORD JOINER -- likewise
        "﻿",  # U+FEFF ZWNBSP -- likewise
    ],
    ids=lambda s: f"U+{ord(s[0]):04X}",
)
def test_whitespace_like_adjacency_is_trivial(space: str) -> None:
    text = "我" + space + "我"
    assert trivial_positions(text) == frozenset({1, 2})


@pytest.mark.parametrize("punct", ["。", "！", "，", "？", ".", ",", "!", "《", "」", "-"])
def test_punctuation_adjacency_is_trivial(punct: str) -> None:
    assert unicodedata.category(punct)[0] == "P"
    text = "我" + punct + "我"
    assert trivial_positions(text) == frozenset({1, 2})


@pytest.mark.parametrize("sym", ["$", "+", "=", "^", "£", "©"])
def test_symbol_adjacency_is_trivial(sym: str) -> None:
    assert unicodedata.category(sym)[0] == "S"
    text = "我" + sym + "我"
    assert trivial_positions(text) == frozenset({1, 2})


def test_han_latin_transition_is_trivial() -> None:
    """No space needed: the script transition alone makes it trivial."""
    text = "我Python写"
    triv = trivial_positions(text)
    assert 1 in triv  # 我 -> P
    assert 7 in triv  # n -> 写
    assert triv == frozenset({1, 7})


def test_letter_digit_transition_is_trivial() -> None:
    text = "会2024年"
    triv = trivial_positions(text)
    assert 1 in triv  # 会 -> 2
    assert 5 in triv  # 4 -> 年
    assert triv == frozenset({1, 5})


def test_latin_digit_transition_is_trivial() -> None:
    text = "ab12cd"
    assert trivial_positions(text) == frozenset({2, 4})


def test_han_thai_transition_is_trivial() -> None:
    assert 1 in trivial_positions("我ก")


def test_han_han_position_is_not_trivial() -> None:
    assert trivial_positions("我喜欢") == frozenset()


def test_thai_thai_position_is_not_trivial() -> None:
    assert trivial_positions("งาน") == frozenset()


def test_zh_python_fixture_transitions_are_trivial() -> None:
    """The Python fixture exists precisely to exercise this."""
    rec = BY_LANG["zh"][8]
    assert "Python" in rec.text
    start = rec.text.index("Python")
    triv = trivial_positions(rec.text)
    assert start in triv
    assert start + len("Python") in triv
    # ...and the interior of the Latin run is NOT trivial
    for i in range(start + 1, start + len("Python")):
        assert i not in triv


def test_zh_2024_fixture_transitions_are_trivial() -> None:
    rec = BY_LANG["zh"][9]
    assert "2024" in rec.text
    start = rec.text.index("2024")
    triv = trivial_positions(rec.text)
    assert start in triv
    assert start + 4 in triv
    for i in range(start + 1, start + 4):
        assert i not in triv


@pytest.mark.parametrize("rec", ALL_RECORDS, ids=[r.id for r in ALL_RECORDS])
def test_trivial_positions_are_interior(rec: Segmented) -> None:
    assert all(0 < p < rec.n for p in trivial_positions(rec.text))


@pytest.mark.parametrize("text", ["", "我"])
def test_trivial_positions_empty_for_short_text(text: str) -> None:
    assert trivial_positions(text) == frozenset()


def test_thai_phrase_space_is_stripped_from_core_but_stays_in_gold() -> None:
    """CONTRACTS sec.2: trivial POSITIONS go; the gold BOUNDARIES stay."""
    rec = BY_LANG["th"][0]  # ฉันทำงาน ที่บ้าน
    space = rec.text.index(" ")
    gold = gold_boundaries(rec)
    m = compute_masks(rec.text, "th")
    assert space in gold and space + 1 in gold  # the free boundaries are gold
    assert space in m["legal"] and space + 1 in m["legal"]
    assert space not in m["core"] and space + 1 not in m["core"]


# ==========================================================================
# 8. gold_illegal_rate
# ==========================================================================


@pytest.mark.parametrize("lang", LANGS)
def test_gold_illegal_rate_is_zero_on_fixtures(lang: str) -> None:
    """Every gold boundary in the mini-corpus sits on a legal cluster edge."""
    assert gold_illegal_rate(BY_LANG[lang], lang) == 0.0


@pytest.mark.parametrize("lang", LANGS)
def test_every_gold_boundary_is_legal(lang: str) -> None:
    """The per-record form of the same claim, with a useful failure message."""
    for rec in BY_LANG[lang]:
        legal = legal_positions(rec.text, lang)
        illegal = gold_boundaries(rec) - legal
        assert not illegal, f"{rec.id}: illegal gold at {sorted(illegal)} in {rec.text!r}"


def test_gold_illegal_rate_empty_input() -> None:
    assert gold_illegal_rate([], "zh") == 0.0


def test_gold_illegal_rate_ignores_records_with_no_gold_boundary() -> None:
    rec = Segmented(id="mini_zh/test/999998", text="谢谢", spans=((0, 2),), meta={})
    assert gold_boundaries(rec) == frozenset()
    assert gold_illegal_rate([rec], "zh") == 0.0


def test_gold_illegal_rate_detects_a_split_cluster() -> None:
    """A boundary inside ทำ (before SARA AM) is illegal and must be reported."""
    rec = Segmented(id="mini_th/test/999999", text="ทำงาน", spans=((0, 1), (1, 5)), meta={})
    assert gold_illegal_rate([rec], "th") == 1.0


def test_gold_illegal_rate_is_a_fraction() -> None:
    good = Segmented(id="mini_th/test/999997", text="ทำงาน", spans=((0, 2), (2, 5)), meta={})
    bad = Segmented(id="mini_th/test/999996", text="ทำงาน", spans=((0, 1), (1, 5)), meta={})
    assert gold_illegal_rate([good], "th") == 0.0
    assert gold_illegal_rate([good, bad], "th") == pytest.approx(0.5)

# Refined

# Enhanced

# Enhanced

# Refined
