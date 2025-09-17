"""Offset integrity: the adapter layer's guard table, unit-tested.

CONTRACTS.md sec.4 is the spec these tests encode. The risk in this module is not
that it crashes -- it is that it silently succeeds, so nearly every test here
pins an exact integer rather than a "did not blow up".

Three groups:

* **Offline, synthetic** -- `accept_spans` and `spans_from_byte_ends` driven by
  hand-written offset lists. No tokenizer, no network. This is where the guard
  table itself is checked, including the forward-gap case that an earlier draft
  of the contract got wrong.
* **Offline, fixtures** -- the three builtin baselines over every sentence of
  `fixtures.mini`, checking the structural contract on `EncodeResult`.
* **Network** (`-m network`) -- real tiktoken and HuggingFace tokenizers, where
  the interesting claims are cross-implementation agreement and the mid-codepoint
  rate on Khmer.
"""

from __future__ import annotations

import pytest
from fixtures.mini import records

from unsegbench.errors import TokenizerUnavailable
from unsegbench.positions import compute_mask
from unsegbench.tok.base import TokenizerSpec
from unsegbench.tok.loader import get_adapter
from unsegbench.tok.offsets import (
    accept_spans,
    byte_to_char_map,
    dropped_chars,
    spans_from_byte_ends,
)
from unsegbench.tok.registry import get_tokenizer_spec
from unsegbench.types import FLAG_KEYS, EncodeResult, Span

# --------------------------------------------------------------------------
# Shared material
# --------------------------------------------------------------------------

#: The contract's own worked example (CONTRACTS.md sec.4): five Thai words
#: separated by four single spaces.
THAI_GAPPED = "ฉัน ทำ งาน ที่ บ้าน"

#: What XLM-R (Metaspace) actually returns for it -- the delimiter spaces are
#: absent from the offsets, so consecutive tokens are separated by a one-char gap.
XLMR_THAI_OFFSETS: list[Span] = [(0, 3), (4, 6), (7, 10), (11, 14), (15, 19)]

#: Pure Khmer, no spaces: every codepoint is 3 bytes in UTF-8, so any overlap
#: inside it is explainable as HuggingFace's mid-codepoint collapse.
KHMER = "ភាសាខ្មែរ"

#: Pure Thai, no spaces.
THAI = "ทำงาน"

ALL_RECORDS = records()
RECORD_IDS = [r.id for r in ALL_RECORDS]
BASELINES = ("char", "whole", "whitespace")
LANGS = ("zh", "yue", "th", "km")


def baseline(ref: str, lang: str):
    """A loaded builtin baseline, via the real factory (network-free)."""
    return get_adapter(get_tokenizer_spec(ref), lang)


def assert_well_formed(spans: tuple[Span, ...], text: str) -> None:
    """The structural half of the `encode` contract."""
    prev_end = 0
    for s, e in spans:
        assert 0 <= s < e <= len(text), f"span ({s},{e}) out of range for n={len(text)}"
        assert s >= prev_end, f"span ({s},{e}) overlaps previous end {prev_end}"
        assert text[s:e] != ""
        prev_end = e


def covered(spans: tuple[Span, ...]) -> int:
    return sum(e - s for s, e in spans)


# ==========================================================================
# 1. The guard table -- contiguous
# ==========================================================================


def test_contiguous_pair_is_accepted():
    spans, flags = accept_spans([(0, 3), (3, 6)], "abcdef")
    assert spans == ((0, 3), (3, 6))
    assert flags["overlap_rejected"] == 0
    assert flags["midcodepoint_split"] == 0


def test_contiguous_run_is_returned_unchanged():
    raw = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    spans, _ = accept_spans(raw, "abcde")
    assert spans == tuple(raw)


def test_contiguous_run_leaves_nothing_dropped():
    _, flags = accept_spans([(0, 2), (2, 5)], "abcde")
    assert flags["dropped_chars"] == 0


def test_contiguous_boundary_is_the_meeting_point():
    spans, _ = accept_spans([(0, 3), (3, 6)], "abcdef")
    assert EncodeResult(spans=spans, n_tokens=2).boundaries == frozenset({3})


def test_single_span_covering_everything_has_no_interior_boundary():
    spans, flags = accept_spans([(0, 6)], "abcdef")
    assert spans == ((0, 6),)
    assert EncodeResult(spans=spans, n_tokens=1).boundaries == frozenset()
    assert flags["dropped_chars"] == 0


def test_empty_offset_list_yields_no_spans():
    spans, _ = accept_spans([], "abc")
    assert spans == ()


def test_empty_offset_list_drops_the_whole_text():
    _, flags = accept_spans([], "abc")
    assert flags["dropped_chars"] == 3


def test_empty_text_yields_no_spans_and_nothing_dropped():
    spans, flags = accept_spans([], "")
    assert spans == ()
    assert flags["dropped_chars"] == 0


def test_span_against_empty_text_is_rejected_not_accepted():
    spans, flags = accept_spans([(0, 1)], "")
    assert spans == ()
    assert flags["overlap_rejected"] == 1


# ==========================================================================
# 2. Zero-width entries (special tokens, neutralised prefix-space markers)
# ==========================================================================


def test_zero_width_span_is_skipped():
    spans, _ = accept_spans([(0, 0), (0, 3), (3, 3), (3, 6)], "abcdef")
    assert spans == ((0, 3), (3, 6))


def test_zero_width_span_is_not_a_defect():
    _, flags = accept_spans([(0, 0), (0, 3), (3, 3), (3, 6)], "abcdef")
    assert flags["overlap_rejected"] == 0
    assert flags["midcodepoint_split"] == 0
    assert flags["dropped_chars"] == 0


def test_zero_width_span_does_not_advance_prev_end():
    # A trailing zero-width entry must not make the following span look gapped.
    spans, flags = accept_spans([(0, 2), (2, 2), (2, 4)], "abcd")
    assert spans == ((0, 2), (2, 4))
    assert flags["dropped_chars"] == 0


def test_zero_width_span_beyond_prev_end_is_still_ignored():
    spans, flags = accept_spans([(0, 2), (9, 9), (2, 4)], "abcd")
    assert spans == ((0, 2), (2, 4))
    assert flags["overlap_rejected"] == 0


# ==========================================================================
# 3. The guard table -- FORWARD GAP. start_i > end_{i-1} is an ACCEPT.
# ==========================================================================


def test_forward_gap_is_accepted_not_rejected():
    spans, _ = accept_spans([(0, 3), (4, 7)], "abcdefg")
    assert spans == ((0, 3), (4, 7))


def test_forward_gap_skipped_codepoints_go_to_dropped_chars():
# improved
    _, flags = accept_spans([(0, 3), (4, 7)], "abcdefg")
    assert flags["dropped_chars"] == 1


def test_forward_gap_raises_no_rejection_flag():
    _, flags = accept_spans([(0, 3), (4, 7)], "abcdefg")
    assert flags["overlap_rejected"] == 0
    assert flags["midcodepoint_split"] == 0


def test_forward_gap_plants_the_boundary_at_the_span_start():
    spans, _ = accept_spans([(0, 3), (4, 7)], "abcdefg")
    assert EncodeResult(spans=spans, n_tokens=2).boundaries == frozenset({4})


def test_multi_character_forward_gap_is_accepted_and_counted():
    spans, flags = accept_spans([(0, 2), (6, 8)], "abcdefgh")
    assert spans == ((0, 2), (6, 8))
    assert flags["dropped_chars"] == 4


def test_gap_before_the_first_span_is_accepted_and_counted():
    spans, flags = accept_spans([(2, 4)], "abcdef")
    assert spans == ((2, 4),)
    assert flags["dropped_chars"] == 4


def test_uncovered_tail_is_counted_as_dropped():
    _, flags = accept_spans([(0, 2)], "abcdef")
    assert flags["dropped_chars"] == 4


def test_several_forward_gaps_accumulate_in_dropped_chars():
    _, flags = accept_spans([(0, 1), (2, 3), (4, 5), (6, 7)], "abcdefg")
    assert flags["dropped_chars"] == 3


# ==========================================================================
# 4. Why the forward gap matters: Metaspace / SentencePiece.
#
# An earlier version of CONTRACTS.md sec.4 read the guard literally as
# "accept only when start_i == end_{i-1}". Every Metaspace/SentencePiece
# tokenizer drops the delimiter space from its offsets, so that reading throws
# away a correct segmentation and reports the tokenizer as a catastrophe.
# ==========================================================================


def test_metaspace_forward_gaps_are_a_perfect_thai_segmentation_not_a_defect():
    spans, flags = accept_spans(XLMR_THAI_OFFSETS, THAI_GAPPED)
    assert spans == tuple(XLMR_THAI_OFFSETS)
    assert flags["overlap_rejected"] == 0
    assert flags["midcodepoint_split"] == 0


def test_metaspace_gap_chars_are_exactly_the_four_delimiter_spaces():
    _, flags = accept_spans(XLMR_THAI_OFFSETS, THAI_GAPPED)
    assert flags["dropped_chars"] == THAI_GAPPED.count(" ") == 4


def test_metaspace_accepted_spans_recover_the_five_words():
    spans, _ = accept_spans(XLMR_THAI_OFFSETS, THAI_GAPPED)
    assert [THAI_GAPPED[s:e] for s, e in spans] == THAI_GAPPED.split(" ")


def test_literal_contiguous_reading_would_discard_four_of_the_five_thai_tokens():
    """The regression the contract note exists to prevent.

    ``allow_gaps=False`` is the earlier, literal reading of the guard. On XLM-R's
    (correct) Thai offsets it keeps ONE token out of five and reports four
    rejections -- a near-perfect Thai tokenizer scored as a disaster.
    """
    kept, flags = accept_spans(XLMR_THAI_OFFSETS, THAI_GAPPED, allow_gaps=False)
    accepted_now, _ = accept_spans(XLMR_THAI_OFFSETS, THAI_GAPPED)

    assert len(accepted_now) == 5
    assert len(kept) == 1
    assert len(accepted_now) - len(kept) == 4
    assert flags["overlap_rejected"] == 4


def test_literal_contiguous_reading_loses_every_interior_thai_boundary():
    kept, _ = accept_spans(XLMR_THAI_OFFSETS, THAI_GAPPED, allow_gaps=False)
    kept_bounds = EncodeResult(spans=kept, n_tokens=5).boundaries
    real_bounds = EncodeResult(
        spans=accept_spans(XLMR_THAI_OFFSETS, THAI_GAPPED)[0], n_tokens=5
    ).boundaries

    assert kept_bounds == frozenset()
    assert real_bounds == frozenset({4, 7, 11, 15})


def test_literal_contiguous_reading_inflates_dropped_chars():
    _, literal = accept_spans(XLMR_THAI_OFFSETS, THAI_GAPPED, allow_gaps=False)
    _, contract = accept_spans(XLMR_THAI_OFFSETS, THAI_GAPPED)
    assert literal["dropped_chars"] == 16
    assert contract["dropped_chars"] == 4


def test_allow_gaps_false_still_accepts_contiguous_offsets():
    spans, flags = accept_spans([(0, 3), (3, 6)], "abcdef", allow_gaps=False)
    assert spans == ((0, 3), (3, 6))
    assert flags["overlap_rejected"] == 0


# ==========================================================================
# 5. The guard table -- OVERLAP. start_i < end_{i-1} rejects the boundary.
# ==========================================================================


def test_backward_overlap_rejects_the_boundary():
    spans, _ = accept_spans([(0, 3), (2, 5)], "abcdef")
    assert spans == ((0, 3),)


def test_backward_overlap_is_counted():
    _, flags = accept_spans([(0, 3), (2, 5)], "abcdef")
    assert flags["overlap_rejected"] == 1


def test_rejected_span_does_not_advance_prev_end():
    # (2,5) is rejected, so (3,6) is still judged against prev_end == 3.
    spans, _ = accept_spans([(0, 3), (2, 5), (3, 6)], "abcdef")
    assert spans == ((0, 3), (3, 6))


def test_rejected_span_chars_land_in_dropped_chars():
    _, flags = accept_spans([(0, 3), (2, 5)], "abcdef")
    assert flags["dropped_chars"] == 3


def test_fully_contained_span_is_rejected():
    spans, flags = accept_spans([(0, 5), (1, 3)], "abcdef")
    assert spans == ((0, 5),)
    assert flags["overlap_rejected"] == 1


def test_duplicate_ascii_span_is_rejected():
    spans, flags = accept_spans([(0, 3), (0, 3)], "abcdef")
    assert spans == ((0, 3),)
    assert flags["overlap_rejected"] == 1
# improved


def test_reversed_span_is_rejected():
    spans, flags = accept_spans([(0, 2), (5, 3)], "abcdef")
    assert spans == ((0, 2),)
    assert flags["overlap_rejected"] == 1


def test_negative_start_is_rejected():
    spans, flags = accept_spans([(-1, 2)], "abcdef")
    assert spans == ()
    assert flags["overlap_rejected"] == 1


def test_end_beyond_text_is_rejected():
    spans, flags = accept_spans([(2, 99)], "abcdef")
    assert spans == ()
    assert flags["overlap_rejected"] == 1


def test_out_of_range_spans_never_reach_the_output():
    spans, flags = accept_spans([(0, 2), (5, 3), (-1, 2), (2, 99)], "abcdef")
    assert spans == ((0, 2),)
    assert flags["overlap_rejected"] == 3


# ==========================================================================
# 6. Overlap CLASSIFICATION. midcodepoint_split vs overlap_rejected must not
#    leak into each other -- midcodepoint_split is a headline Tier-0 number.
# ==========================================================================


def test_multibyte_overlap_is_a_midcodepoint_split():
    _, flags = accept_spans([(0, 2), (1, 3)], THAI)
    assert flags["midcodepoint_split"] == 1


def test_multibyte_overlap_is_not_filed_as_disorder():
    _, flags = accept_spans([(0, 2), (1, 3)], THAI)
    assert flags["overlap_rejected"] == 0


def test_ascii_overlap_is_disorder():
    _, flags = accept_spans([(0, 3), (2, 5)], "abcdef")
    assert flags["overlap_rejected"] == 1


def test_ascii_overlap_is_not_filed_as_a_midcodepoint_split():
    _, flags = accept_spans([(0, 3), (2, 5)], "abcdef")
    assert flags["midcodepoint_split"] == 0


def test_hf_collapse_of_one_thai_codepoint_is_two_midcodepoint_splits():
    # Three tokens over the three UTF-8 bytes of one codepoint come back as
    # three IDENTICAL char spans. Two of them are collapse artefacts.
    _, flags = accept_spans([(0, 1), (0, 1), (0, 1), (1, 2)], THAI)
    assert flags["midcodepoint_split"] == 2
    assert flags["overlap_rejected"] == 0


def test_hf_collapse_leaves_one_accepted_span_per_codepoint():
    spans, _ = accept_spans([(0, 1), (0, 1), (0, 1), (1, 2)], THAI)
    assert spans == ((0, 1), (1, 2))


def test_hf_collapse_invents_no_boundary():
    spans, _ = accept_spans([(0, 1), (0, 1), (0, 1)], THAI)
    assert EncodeResult(spans=spans, n_tokens=3).boundaries == frozenset()


def test_khmer_collapse_is_midcodepoint_not_disorder():
    _, flags = accept_spans([(0, 1), (0, 1), (1, 2), (1, 2)], KHMER)
    assert flags["midcodepoint_split"] == 2
    assert flags["overlap_rejected"] == 0


def test_classification_reads_the_overlap_region_not_the_whole_span():
    """Overlap inside ASCII, remainder of the span multi-byte -> disorder.

    ``(2,5)`` covers 'c' plus two Thai codepoints, but the region it overlaps is
    ``text[2:3] == 'c'``, which no UTF-8 artefact can explain.
    """
    text = "abcทำ"
    _, flags = accept_spans([(0, 3), (2, 5)], text)
    assert flags["overlap_rejected"] == 1
    assert flags["midcodepoint_split"] == 0


def test_classification_reads_the_overlap_region_not_the_span_tail():
    """Overlap inside a Thai codepoint, remainder of the span ASCII -> collapse."""
    text = "ทำabc"
    _, flags = accept_spans([(0, 2), (1, 5)], text)
    assert flags["midcodepoint_split"] == 1
    assert flags["overlap_rejected"] == 0


def test_token_straddling_two_codepoints_ends_beyond_the_previous_end():
    """The BLOOM-on-Khmer case: ``(3,5)`` then ``(4,6)``.

    The second token holds the last byte of one codepoint plus the first bytes of
    the next, so it starts INSIDE the previous token and ends BEYOND it. Judging
    it on the end offset would file a routine UTF-8 artefact as disorder and
    deflate the headline mid-codepoint rate.
    """
    raw = [(0, 3), (3, 5), (4, 6)]
    assert raw[2][1] > raw[1][1]  # ends beyond the previous token's end
    _, flags = accept_spans(raw, KHMER)
    assert flags["midcodepoint_split"] == 1
    assert flags["overlap_rejected"] == 0


def test_straddling_token_yields_no_boundary():
    spans, _ = accept_spans([(0, 3), (3, 5), (4, 6)], KHMER)
    assert spans == ((0, 3), (3, 5))
    assert EncodeResult(spans=spans, n_tokens=3).boundaries == frozenset({3})


def test_ascii_straddle_ending_beyond_prev_end_is_disorder():
    _, flags = accept_spans([(0, 3), (3, 5), (4, 6)], "abcdefgh")
    assert flags["overlap_rejected"] == 1
    assert flags["midcodepoint_split"] == 0


def test_the_two_overlap_flags_do_not_leak_into_each_other():
    """One ASCII overlap and one Thai overlap in the same offset list."""
    text = "abcdefทำงาน"
    _, flags = accept_spans([(0, 3), (2, 6), (6, 8), (7, 9)], text)
    assert flags["overlap_rejected"] == 1
    assert flags["midcodepoint_split"] == 1


def test_overlap_region_of_length_zero_cannot_occur_for_a_rejected_span():
    # s < prev_end and s < e, so text[s:min(e, prev_end)] is always non-empty.
    text = "abcทำ"
    for raw in ([(0, 3), (2, 5)], [(0, 5), (4, 5)], [(0, 4), (3, 5)]):
        _, flags = accept_spans(raw, text)
        assert flags["overlap_rejected"] + flags["midcodepoint_split"] == 1


@pytest.mark.parametrize(
    "raw,text",
    [
        ([(0, 3), (2, 5)], "abcdef"),
        ([(0, 1), (0, 1)], THAI),
        ([(0, 3), (4, 7)], "abcdefg"),
        (XLMR_THAI_OFFSETS, THAI_GAPPED),
        ([(0, 3), (3, 5), (4, 6)], KHMER),
        ([(5, 3), (-1, 1)], "abcdef"),
    ],
)
def test_flag_vocabulary_is_closed(raw, text):
    _, flags = accept_spans(raw, text)
    assert set(flags) <= set(FLAG_KEYS)


# ==========================================================================
# 7. Output invariants of accept_spans, whatever the input
# ==========================================================================

MESSY = [
    ([(0, 3), (2, 5), (5, 7), (6, 6), (7, 9), (7, 8)], "abcdefghi"),
    ([(0, 1), (0, 1), (1, 2), (1, 3), (3, 5)], THAI),
    (XLMR_THAI_OFFSETS, THAI_GAPPED),
    ([(4, 6), (0, 2)], "abcdef"),
    ([(0, 3), (3, 5), (4, 6), (6, 9)], KHMER),
    ([], "abcdef"),
]


@pytest.mark.parametrize("raw,text", MESSY, ids=range(len(MESSY)))
def test_accepted_spans_are_well_formed(raw, text):
    spans, _ = accept_spans(raw, text)
    assert_well_formed(spans, text)


@pytest.mark.parametrize("raw,text", MESSY, ids=range(len(MESSY)))
def test_accepted_spans_are_strictly_ascending(raw, text):
    spans, _ = accept_spans(raw, text)
    starts = [s for s, _ in spans]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


@pytest.mark.parametrize("raw,text", MESSY, ids=range(len(MESSY)))
def test_dropped_chars_accounts_for_everything_not_covered(raw, text):
    spans, flags = accept_spans(raw, text)
    assert covered(spans) + flags["dropped_chars"] == len(text)


@pytest.mark.parametrize("raw,text", MESSY, ids=range(len(MESSY)))
def test_dropped_chars_helper_agrees_with_the_flag(raw, text):
    spans, flags = accept_spans(raw, text)
    assert dropped_chars(spans, text) == flags["dropped_chars"]


def test_dropped_chars_is_never_negative():
    for raw, text in MESSY:
        _, flags = accept_spans(raw, text)
        assert flags["dropped_chars"] >= 0


def test_out_of_order_offsets_do_not_reorder_the_output():
    # A later span that precedes prev_end is rejected, never sorted into place.
    spans, flags = accept_spans([(4, 6), (0, 2)], "abcdef")
    assert spans == ((4, 6),)
    assert flags["overlap_rejected"] == 1


# ==========================================================================
# 8. The exact path: per-token byte lengths (tiktoken)
# ==========================================================================


def test_byte_to_char_map_has_one_entry_per_byte_plus_end():
    text = "aทb"
    char_of_byte, is_boundary = byte_to_char_map(text)
    assert len(char_of_byte) == len(text.encode("utf-8")) + 1
    assert len(is_boundary) == len(char_of_byte)


def test_byte_to_char_map_marks_continuation_bytes():
    char_of_byte, is_boundary = byte_to_char_map("aทb")
    assert is_boundary == [True, True, False, False, True, True]
    assert char_of_byte == [0, 1, 1, 1, 2, 3]


def test_byte_to_char_map_end_position_is_a_boundary():
    char_of_byte, is_boundary = byte_to_char_map(THAI)
    assert is_boundary[-1] is True
    assert char_of_byte[-1] == len(THAI)


def test_spans_from_byte_ends_tiles_ascii_exactly():
    spans, flags = spans_from_byte_ends("abcdef", [1, 2, 3])
    assert spans == ((0, 1), (1, 3), (3, 6))
    assert flags["dropped_chars"] == 0
    assert flags["midcodepoint_split"] == 0


def test_spans_from_byte_ends_counts_edges_inside_a_codepoint():
    # The first codepoint of THAI is torn into three 1-byte tokens: the two
    # interior edges fall inside it and yield no boundary, so those three tokens
    # are merged back into the single accepted span (0,1).
    spans, flags = spans_from_byte_ends(THAI, [1, 1, 1, 3])
    assert flags["midcodepoint_split"] == 2
    assert spans == ((0, 1), (1, 2), (2, 5))
    assert flags["dropped_chars"] == 0


def test_spans_from_byte_ends_never_places_a_boundary_inside_a_codepoint():
    spans, _ = spans_from_byte_ends(THAI, [1, 1, 1, 1, 1, 1])
    assert_well_formed(spans, THAI)
    assert EncodeResult(spans=spans, n_tokens=6).boundaries <= {1, 2}


def test_spans_from_byte_ends_on_empty_text():
    spans, flags = spans_from_byte_ends("", [1, 2])
    assert spans == ()
    assert flags == {}


def test_spans_from_byte_ends_truncates_rather_than_fabricating():
    spans, _ = spans_from_byte_ends("abc", [2, 5])
    assert_well_formed(spans, "abc")
    assert spans[-1][1] <= 3


def test_spans_from_byte_ends_accounting_identity():
    for lengths in ([1, 1, 1], [3, 3], [1, 2, 3], [9]):
        spans, flags = spans_from_byte_ends(THAI, lengths)
        assert covered(spans) + flags["dropped_chars"] == len(THAI)


def test_spans_from_byte_ends_flag_vocabulary_is_closed():
    _, flags = spans_from_byte_ends(KHMER, [1, 1, 1, 3, 3])
    assert set(flags) <= set(FLAG_KEYS)


# ==========================================================================
# 9. Baselines -- char / whole / whitespace. No network, no vocabulary.
# ==========================================================================


@pytest.mark.parametrize("lang", LANGS)
def test_char_baseline_boundaries_are_exactly_the_legal_mask(lang):
    adapter = baseline("char", lang)
    for rec in records(lang):
        got = adapter.encode(rec.text).boundaries
        assert got == compute_mask(rec.text, lang, "legal")


@pytest.mark.parametrize("lang", LANGS)
def test_char_baseline_emits_one_token_per_legal_cluster(lang):
    adapter = baseline("char", lang)
    for rec in records(lang):
        res = adapter.encode(rec.text)
        assert res.n_tokens == len(compute_mask(rec.text, lang, "legal")) + 1
        assert res.n_tokens == len(res.spans)


@pytest.mark.parametrize("lang", LANGS)
def test_char_baseline_tiles_the_text(lang):
    adapter = baseline("char", lang)
    for rec in records(lang):
        res = adapter.encode(rec.text)
        assert "".join(rec.text[s:e] for s, e in res.spans) == rec.text
        assert res.flags["dropped_chars"] == 0


@pytest.mark.parametrize("lang", LANGS)
def test_char_baseline_raises_no_flags(lang):
    adapter = baseline("char", lang)
    for rec in records(lang):
        assert dict(adapter.encode(rec.text).flags) == {}


def test_char_baseline_never_splits_a_khmer_coeng_sequence():
    adapter = baseline("char", "km")
    text = "ខ្មែរ"
    res = adapter.encode(text)
    coeng = text.index("្")
    assert coeng + 1 not in res.boundaries


def test_char_baseline_on_empty_text():
    res = baseline("char", "zh").encode("")
    assert res.spans == ()
    assert res.n_tokens == 0
    assert res.boundaries == frozenset()


def test_whole_baseline_emits_exactly_one_token():
    res = baseline("whole", "zh").encode("我喜欢吃苹果")
    assert res.spans == ((0, 6),)
    assert res.n_tokens == 1


def test_whole_baseline_has_no_interior_boundary():
    for rec in ALL_RECORDS:
        res = baseline("whole", rec.meta["lang"]).encode(rec.text)
        assert res.boundaries == frozenset()


def test_whole_baseline_drops_nothing():
    for rec in ALL_RECORDS:
        res = baseline("whole", rec.meta["lang"]).encode(rec.text)
        assert res.flags["dropped_chars"] == 0


def test_whole_baseline_on_empty_text():
    res = baseline("whole", "th").encode("")
    assert res.spans == ()
    assert res.n_tokens == 0


def test_whitespace_baseline_splits_on_ascii_space():
    res = baseline("whitespace", "th").encode(THAI_GAPPED)
    assert [THAI_GAPPED[s:e] for s, e in res.spans] == THAI_GAPPED.split(" ")


def test_whitespace_baseline_delimiters_are_not_tokens():
    res = baseline("whitespace", "th").encode(THAI_GAPPED)
    assert res.flags["dropped_chars"] == 4
    assert covered(res.spans) == len(THAI_GAPPED) - 4


def test_whitespace_baseline_splits_on_zwsp():
    text = "ខ្ញុំ​ស្រឡាញ់​ប្រទេស"
    res = baseline("whitespace", "km").encode(text)
    assert len(res.spans) == 3
    assert all("​" not in text[s:e] for s, e in res.spans)


def test_whitespace_baseline_zwsp_is_not_caught_by_str_isspace():
    # The reason ZWSP needs its own rule at all.
    assert not "​".isspace()
    res = baseline("whitespace", "km").encode("ក​ខ")
    assert res.spans == ((0, 1), (2, 3))


@pytest.mark.parametrize("sep", ["​", "⁠", "﻿", " ", "\t", "\n"])
def test_whitespace_baseline_splits_on_every_declared_separator(sep):
    res = baseline("whitespace", "zh").encode(f"我{sep}你")
    assert res.spans == ((0, 1), (2, 3))
    assert res.boundaries == frozenset({2})


def test_whitespace_baseline_collapses_a_run_of_spaces():
    res = baseline("whitespace", "zh").encode("我   你")
    assert res.spans == ((0, 1), (4, 5))
    assert res.flags["dropped_chars"] == 3


def test_whitespace_baseline_handles_leading_and_trailing_space():
    res = baseline("whitespace", "zh").encode(" 我你 ")
    assert res.spans == ((1, 3),)
    assert res.flags["dropped_chars"] == 2


def test_whitespace_baseline_raises_no_flag_without_whitespace():
    res = baseline("whitespace", "zh").encode("我喜欢吃苹果")
    assert dict(res.flags) == {}
    assert res.spans == ((0, 6),)


def test_whitespace_baseline_on_all_space_text():
    res = baseline("whitespace", "zh").encode("   ")
    assert res.spans == ()
    assert res.n_tokens == 0
    assert res.flags["dropped_chars"] == 3


def test_whitespace_baseline_on_empty_text():
    res = baseline("whitespace", "zh").encode("")
    assert res.spans == ()
    assert res.n_tokens == 0


@pytest.mark.parametrize("ref", BASELINES)
def test_baseline_n_tokens_equals_accepted_span_count(ref):
    for rec in ALL_RECORDS:
        res = baseline(ref, rec.meta["lang"]).encode(rec.text)
        assert res.n_tokens == len(res.spans)


@pytest.mark.parametrize("ref", BASELINES)
def test_baseline_boundaries_exclude_the_sentence_edges(ref):
    for rec in ALL_RECORDS:
        res = baseline(ref, rec.meta["lang"]).encode(rec.text)
        assert 0 not in res.boundaries
        assert rec.n not in res.boundaries
        assert all(0 < b < rec.n for b in res.boundaries)


@pytest.mark.parametrize("ref", BASELINES)
def test_baseline_first_span_start_is_not_a_boundary(ref):
    # CONTRACTS sec.2: sentence edges are free and are excluded. For the
    # whitespace baseline the first span may start at a nonzero index (leading
    # space) and that position is still not reported.
    res = baseline(ref, "zh").encode(" 我你 ")
    assert res.spans[0][0] not in res.boundaries


@pytest.mark.parametrize("ref", BASELINES)
def test_baseline_fingerprint_is_stable_and_lang_aware(ref):
    a = baseline(ref, "th").fingerprint()
    b = baseline(ref, "th").fingerprint()
    c = baseline(ref, "km").fingerprint()
    assert a == b
    assert a != c


def test_unknown_builtin_surfaces_as_tokenizer_unavailable():
    spec = TokenizerSpec("nope", "builtin", "nope")
    with pytest.raises(TokenizerUnavailable):
        get_adapter(spec, "zh")


def test_unknown_source_surfaces_as_tokenizer_unavailable():
    spec = TokenizerSpec("nope", "psychic", "nope")
    with pytest.raises(TokenizerUnavailable):
        get_adapter(spec, "zh")


def test_get_adapter_caches_per_spec_and_lang():
    assert baseline("char", "th") is baseline("char", "th")
    assert baseline("char", "th") is not baseline("char", "km")


# ==========================================================================
# 10. Round-trip / integrity on EVERY fixture sentence, for every baseline
# ==========================================================================


@pytest.mark.parametrize("ref", BASELINES)
@pytest.mark.parametrize("rec", ALL_RECORDS, ids=RECORD_IDS)
def test_baseline_encode_satisfies_the_structural_contract(ref, rec):
    res = baseline(ref, rec.meta["lang"]).encode(rec.text)

    # strictly ascending, non-overlapping, every text[s:e] non-empty
    assert_well_formed(res.spans, rec.text)
    starts = [s for s, _ in res.spans]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)

    # nothing is invented and nothing vanishes
    assert covered(res.spans) + res.flags["dropped_chars"] == len(rec.text)
    assert set(res.flags) <= set(FLAG_KEYS)
    assert all(isinstance(v, int) for v in res.flags.values())


# ==========================================================================
# 11. Network. Real tokenizers; the offsets must survive contact with reality.
# ==========================================================================

GPT2_TIKTOKEN = TokenizerSpec("gpt2-tiktoken", "tiktoken", "gpt2", family="openai")
GPT2_HF = TokenizerSpec("gpt2-hf", "hf", "openai-community/gpt2", family="openai")


def load(tokenizer_id: str, lang: str):
    return get_adapter(get_tokenizer_spec(tokenizer_id), lang).load()


def assert_round_trips(res: EncodeResult, text: str) -> None:
    """Offsets are indices into the ORIGINAL string and account for all of it."""
    assert_well_formed(res.spans, text)
    assert covered(res.spans) + res.flags["dropped_chars"] == len(text)
    pieces = [text[s:e] for s, e in res.spans]
    assert all(pieces)
    if res.flags["dropped_chars"] == 0:
        assert "".join(pieces) == text
    assert res.n_tokens >= len(res.spans)


@pytest.mark.network
@pytest.mark.parametrize("tokenizer_id", ["cl100k_base", "o200k_base", "xlm-r"])
@pytest.mark.parametrize("lang", LANGS)
def test_offsets_round_trip_against_the_original_string(tokenizer_id, lang):
    adapter = load(tokenizer_id, lang)
    for rec in records(lang):
        assert_round_trips(adapter.encode(rec.text), rec.text)


@pytest.mark.network
@pytest.mark.parametrize("tokenizer_id", ["cl100k_base", "o200k_base", "xlm-r"])
@pytest.mark.parametrize("lang", LANGS)
def test_real_tokenizer_flags_stay_in_the_closed_vocabulary(tokenizer_id, lang):
    adapter = load(tokenizer_id, lang)
    for rec in records(lang):
        assert set(adapter.encode(rec.text).flags) <= set(FLAG_KEYS)


@pytest.mark.network
def test_cl100k_splits_khmer_codepoints():
    adapter = load("cl100k_base", "km")
    total = sum(adapter.encode(rec.text).flags["midcodepoint_split"] for rec in records("km"))
    assert total > 0


@pytest.mark.network
def test_cl100k_khmer_split_rate_is_substantial():
    adapter = load("cl100k_base", "km")
    splits = sum(adapter.encode(rec.text).flags["midcodepoint_split"] for rec in records("km"))
    chars = sum(rec.n for rec in records("km"))
    assert splits / chars > 0.25


@pytest.mark.network
def test_o200k_does_not_split_chinese_codepoints():
    adapter = load("o200k_base", "zh")
    total = sum(adapter.encode(rec.text).flags["midcodepoint_split"] for rec in records("zh"))
    assert total == 0


@pytest.mark.network
def test_o200k_splits_fewer_khmer_codepoints_than_cl100k():
    km = records("km")
    old = sum(load("cl100k_base", "km").encode(r.text).flags["midcodepoint_split"] for r in km)
    new = sum(load("o200k_base", "km").encode(r.text).flags["midcodepoint_split"] for r in km)
    assert new < old


@pytest.mark.network
def test_tiktoken_n_tokens_is_the_raw_count_not_the_accepted_count():
    adapter = load("cl100k_base", "km")
    res = adapter.encode(records("km")[0].text)
    assert res.n_tokens > len(res.spans)


@pytest.mark.network
@pytest.mark.parametrize("lang", LANGS)
def test_gpt2_via_tiktoken_and_transformers_agree_on_boundaries(lang):
    """Cross-implementation: the byte path and the char-offset path must agree.

    tiktoken goes through exact per-token byte lengths; transformers goes through
    ``return_offsets_mapping`` and the collapse guard. Same tokenizer, two
    completely different routes to the answer -- disagreement means one of the
    two routes is wrong.
    """
    tt = get_adapter(GPT2_TIKTOKEN, lang).load()
    hf = get_adapter(GPT2_HF, lang).load()
    for rec in records(lang):
        a, b = tt.encode(rec.text), hf.encode(rec.text)
        assert a.boundaries == b.boundaries, rec.id


@pytest.mark.network
@pytest.mark.parametrize("lang", LANGS)
def test_gpt2_via_tiktoken_and_transformers_agree_on_raw_token_count(lang):
    tt = get_adapter(GPT2_TIKTOKEN, lang).load()
    hf = get_adapter(GPT2_HF, lang).load()
    for rec in records(lang):
        assert tt.encode(rec.text).n_tokens == hf.encode(rec.text).n_tokens, rec.id


@pytest.mark.network
@pytest.mark.parametrize("tokenizer_id", ["olmo2", "phi4"])
@pytest.mark.parametrize("lang", LANGS)
def test_cl100k_merge_sharers_produce_identical_boundaries(tokenizer_id, lang):
    """Identity invariance: same merges => byte-identical boundary sets.

    ``olmo2`` and ``phi4`` ship cl100k's merge table and differ only in special
    tokens, which never appear in scored text. A difference here means the
    adapter layer is leaking something that is not the tokenizer.
    """
    ref = load("cl100k_base", lang)
    other = load(tokenizer_id, lang)
    for rec in records(lang):
        assert ref.encode(rec.text).boundaries == other.encode(rec.text).boundaries, rec.id


@pytest.mark.network
@pytest.mark.parametrize("tokenizer_id", ["olmo2", "phi4"])
def test_cl100k_merge_sharers_produce_identical_flags(tokenizer_id):
    ref = load("cl100k_base", "km")
    other = load(tokenizer_id, "km")
    for rec in records("km"):
        a, b = ref.encode(rec.text), other.encode(rec.text)
        assert a.n_tokens == b.n_tokens, rec.id
        assert a.flags["midcodepoint_split"] == b.flags["midcodepoint_split"], rec.id


@pytest.mark.network
def test_xlmr_reproduces_the_contract_example_for_thai():
    res = load("xlm-r", "th").encode(THAI_GAPPED)
    assert res.spans == tuple(XLMR_THAI_OFFSETS)
    assert res.flags["dropped_chars"] == 4
    assert res.flags["overlap_rejected"] == 0


@pytest.mark.network
def test_xlmr_thai_gaps_recover_the_five_contract_words():
    res = load("xlm-r", "th").encode(THAI_GAPPED)
    assert [THAI_GAPPED[s:e] for s, e in res.spans] == THAI_GAPPED.split(" ")


@pytest.mark.network
@pytest.mark.parametrize("lang", ["th", "km"])
def test_xlmr_drops_exactly_the_delimiter_separators(lang):
    """Metaspace's gaps are the separators and nothing else.

    If ``dropped_chars`` ever exceeded the separator count, the guard would be
    silently discarding real content instead of just the delimiters.
    """
    adapter = load("xlm-r", lang)
    for rec in records(lang):
        separators = sum(1 for ch in rec.text if ch.isspace() or ch in "​⁠﻿")
        assert adapter.encode(rec.text).flags["dropped_chars"] == separators, rec.id


@pytest.mark.network
def test_xlmr_bare_prefix_space_marker_is_trimmed():
    """XLM-R gives the bare ``▁`` the first character of the FOLLOWING word."""
    text = "สวัสดี ครับ ผม ชื่อ สมชาย"
    res = load("xlm-r", "th").encode(text)
    assert res.flags["prefix_space_trim"] >= 1


@pytest.mark.network
def test_xlmr_plants_no_boundary_one_char_inside_the_first_word():
    text = "สวัสดี ครับ ผม ชื่อ สมชาย"
    res = load("xlm-r", "th").encode(text)
    assert 1 not in res.boundaries
    assert res.spans[0][0] == 0
    assert res.spans[0][1] > 1


@pytest.mark.network
@pytest.mark.parametrize("lang", ["th", "km"])
def test_xlmr_never_plants_a_boundary_at_position_one_of_a_fixture(lang):
    adapter = load("xlm-r", lang)
    for rec in records(lang):
        res = adapter.encode(rec.text)
        if res.spans:
            assert res.spans[0][0] == 0, rec.id


@pytest.mark.network
def test_xlmr_flags_its_normaliser_mutation_rather_than_scoring_the_copy():
    res = load("xlm-r", "th").encode(THAI_GAPPED)
    # Whether or not the normaliser fires, the offsets index the ORIGINAL text.
    assert_round_trips(res, THAI_GAPPED)
    assert set(res.flags) <= set(FLAG_KEYS)

# Refined

# Refined
