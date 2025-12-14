"""Golden hand-computed examples and the frozen invariants of the metric core.

Everything here is checked against CONTRACTS.md sections 2 and 3, which are the
spec. Where a number is asserted it was computed by hand first and the comment
table above the test is the derivation, not a transcript of what the code
returned.

Fixture data comes from ``tests/fixtures/mini.py`` and nowhere else. No network.
"""

from __future__ import annotations

import itertools
import math

import pytest
from fixtures.mini import by_lang, records

from unsegbench.metrics.core import (
    Counts,
    boundary_counts,
    compute_row,
    crossing_tokens,
    f1,
    informedness,
    markedness,
    phi,
    precision,
    recall,
    word_counts,
    words_intact,
)
from unsegbench.positions import (
    boundaries_to_spans,
    compute_mask,
    compute_masks,
    gold_boundaries,
    legal_positions,
    spans_to_boundaries,
    trivial_positions,
)
from unsegbench.types import MASKS

ALL_RECORDS = records()
BY_LANG = by_lang()


def char_spans(n: int) -> list[tuple[int, int]]:
    """Token spans of the character tokenizer: one span per codepoint."""
    return [(i, i + 1) for i in range(n)]


def lang_of(rec) -> str:
    return rec.meta["lang"]


# ==========================================================================
# GOLDEN 1 -- the fully hand-computed Chinese example
# ==========================================================================
#
#   text        我喜欢吃苹果                n = 6
#
#   idx         0  1  2  3  4  5
#   char        我 喜 欢 吃 苹 果
#   position     1  2  3  4  5              (0 and n are excluded, CONTRACTS s.2)
#
#   gold spans  (0,1)(1,3)(3,4)(4,6)   ->  gold_B {1,3,4}
#               starts 0,1,3,4 / ends 1,3,4,6; drop 0 and 6 (== n)
#   pred spans  (0,2)(2,3)(3,5)(5,6)   ->  pred_B {2,3,5}
#
#   mask = raw = legal = {1,2,3,4,5}    (all Han, no punctuation, no transition)
#   NB the `core` mask equals raw here too: nothing in 我喜欢吃苹果 is trivial.
#
#   g & m = {1,3,4}   p & m = {2,3,5}   g & p = {3}
#   TP = 1   FP = 3-1 = 2   FN = 3-1 = 2   TN = |m| - |g|p| = 5 - 5 = 0
#
#   P  = 1/(1+2) = 1/3
#   R  = 1/(1+2) = 1/3
#   F1 = 2PR/(P+R) = 1/3
#
#   word_counts(gold, pred, mask, 6):
#       induced gold words (0,1)(1,3)(3,4)(4,6)   -> 4
#       induced pred words (0,2)(2,3)(3,5)(5,6)   -> 4
# improved
#       exact matches                              -> 0
#       => (0, 4, 4)
#
#   crossing_tokens([(0,2),(2,3),(3,5),(5,6)], {1,3,4}):
#       (0,2) contains 1  -> cross
#       (2,3) contains -- -> no
#       (3,5) contains 4  -> cross
#       (5,6) contains -- -> no
#       => 2, purity = 1 - 2/4 = 0.5
#
#   words_intact([(0,1),(1,3),(3,4),(4,6)], pred):
#       (0,1) inside (0,2)  yes
#       (1,3) inside ?      no
#       (3,4) inside (3,5)  yes
#       (4,6) inside ?      no
# #       => 2
#
#   phi = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
#       = (1*0 - 2*2) / sqrt(3 * 3 * 2 * 2)
#       = -4 / 6 = -0.666666...   (6dp: -0.666667)
#
#   J = (R - delta_s)/(1 - delta_g) = (1/3 - 3/5)/(1 - 3/5) = -2/3
#   M = (P - delta_g)/(1 - delta_s) = (1/3 - 3/5)/(1 - 3/5) = -2/3
#   phi^2 = 4/9 = J*M, and phi < 0.
#
# --------------------------------------------------------------------------

ZH_TEXT = "我喜欢吃苹果"
ZH_N = 6
ZH_GOLD_SPANS = ((0, 1), (1, 3), (3, 4), (4, 6))
ZH_PRED_SPANS = ((0, 2), (2, 3), (3, 5), (5, 6))
ZH_GOLD_B = frozenset({1, 3, 4})
ZH_PRED_B = frozenset({2, 3, 5})
ZH_MASK = frozenset({1, 2, 3, 4, 5})


def test_golden_zh_is_the_mini_fixture_sentence_without_its_full_stop():
    """The golden really is mini_zh/test/000000, minus the trailing punctuation."""
    rec = BY_LANG["zh"][0]
    assert rec.id == "mini_zh/test/000000"
    assert rec.text == ZH_TEXT + "。"
    assert rec.text[:ZH_N] == ZH_TEXT
    assert rec.spans == ZH_GOLD_SPANS
    assert len(ZH_TEXT) == ZH_N


def test_golden_zh_gold_boundaries():
    assert spans_to_boundaries(ZH_GOLD_SPANS, ZH_N) == ZH_GOLD_B
    assert sorted(ZH_GOLD_B) == [1, 3, 4]


def test_golden_zh_pred_boundaries():
    assert spans_to_boundaries(ZH_PRED_SPANS, ZH_N) == ZH_PRED_B
    assert sorted(ZH_PRED_B) == [2, 3, 5]


def test_golden_zh_edges_zero_and_n_are_never_boundaries():
    assert 0 not in ZH_GOLD_B
    assert ZH_N not in ZH_GOLD_B
    assert 0 not in ZH_PRED_B
    assert ZH_N not in ZH_PRED_B


def test_golden_zh_all_three_masks_coincide():
    masks = compute_masks(ZH_TEXT, "zh")
    assert masks["raw"] == ZH_MASK
    assert masks["legal"] == ZH_MASK
    assert masks["core"] == ZH_MASK
    assert trivial_positions(ZH_TEXT) == frozenset()


def test_golden_zh_contingency_table():
    c = boundary_counts(ZH_GOLD_B, ZH_PRED_B, ZH_MASK)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 2, 2, 0)
    assert c.n == 5


def test_golden_zh_precision_recall_f1_are_all_one_third():
    c = boundary_counts(ZH_GOLD_B, ZH_PRED_B, ZH_MASK)
    assert precision(c) == pytest.approx(1 / 3)
    assert recall(c) == pytest.approx(1 / 3)
    assert f1(c) == pytest.approx(1 / 3)


def test_golden_zh_word_counts():
    assert word_counts(ZH_GOLD_B, ZH_PRED_B, ZH_MASK, ZH_N) == (0, 4, 4)


def test_golden_zh_induced_partitions():
    assert boundaries_to_spans(ZH_GOLD_B, ZH_N) == ZH_GOLD_SPANS
    assert boundaries_to_spans(ZH_PRED_B, ZH_N) == ZH_PRED_SPANS


def test_golden_zh_crossing_tokens_and_purity():
    crossing = crossing_tokens(list(ZH_PRED_SPANS), ZH_GOLD_B)
    assert crossing == 2
    assert 1.0 - crossing / len(ZH_PRED_SPANS) == pytest.approx(0.5)


def test_golden_zh_words_intact():
    assert words_intact(list(ZH_GOLD_SPANS), list(ZH_PRED_SPANS)) == 2


def test_golden_zh_phi_to_six_decimal_places():
    c = boundary_counts(ZH_GOLD_B, ZH_PRED_B, ZH_MASK)
    got = phi(c)
    assert round(got, 6) == -0.666667
    assert got == pytest.approx(-2 / 3, abs=1e-12)
    # and the raw arithmetic, spelled out
    assert got == pytest.approx((1 * 0 - 2 * 2) / math.sqrt(3 * 3 * 2 * 2), abs=1e-12)


def test_golden_zh_informedness_and_markedness():
    c = boundary_counts(ZH_GOLD_B, ZH_PRED_B, ZH_MASK)
    assert informedness(c) == pytest.approx(-2 / 3)
    assert markedness(c) == pytest.approx(-2 / 3)


def test_golden_zh_phi_squared_equals_j_times_m_away_from_a_zero_margin():
    c = boundary_counts(ZH_GOLD_B, ZH_PRED_B, ZH_MASK)
    assert phi(c) ** 2 == pytest.approx(informedness(c) * markedness(c))
    assert phi(c) < 0


def test_golden_zh_full_metric_row():
    c = boundary_counts(ZH_GOLD_B, ZH_PRED_B, ZH_MASK)
    w_tp, w_pred, w_gold = word_counts(ZH_GOLD_B, ZH_PRED_B, ZH_MASK, ZH_N)
    row = compute_row(
        c,
        w_tp=w_tp,
        w_pred=w_pred,
        w_gold=w_gold,
        w_intact=words_intact(list(ZH_GOLD_SPANS), list(ZH_PRED_SPANS)),
        n_tokens=len(ZH_PRED_SPANS),
        n_chars=ZH_N,
        n_gold_words=len(ZH_GOLD_SPANS),
        crossing=crossing_tokens(list(ZH_PRED_SPANS), ZH_GOLD_B),
    )
    assert row.phi == pytest.approx(-2 / 3)
    assert row.b_p == pytest.approx(1 / 3)
    assert row.b_r == pytest.approx(1 / 3)
    assert row.b_f1 == pytest.approx(1 / 3)
    assert row.w_p == pytest.approx(0.0)
    assert row.w_r == pytest.approx(0.0)
    assert row.w_f1 == pytest.approx(0.0)  # P + R == 0 -> 0.0, not nan
    assert row.delta_g == pytest.approx(3 / 5)
    assert row.delta_s == pytest.approx(3 / 5)
    assert row.rho == pytest.approx(1.0)
    assert row.fertility == pytest.approx(1.0)
    assert row.cpt == pytest.approx(6 / 4)
    assert row.purity == pytest.approx(0.5)
    assert row.boundary_miss_rate == pytest.approx(2 / 3)
    assert row.word_exact_rate == pytest.approx(0.0)
    assert row.word_intact_rate == pytest.approx(0.5)


# ==========================================================================
# GOLDEN 2 -- interior space: pins `core` stripping whitespace-adjacent positions
# ==========================================================================
#
#   mini_th/test/000000
#   text   ฉันทำงาน ที่บ้าน            n = 16   (U+0E33 SARA AM lives in ทำ)
#
#   idx    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
#   char   ฉ ั น ท ำ ง า น _ ท  ี ่  บ  ้  า  น
#                            ^ index 8 is U+0020
#
#   gold spans (0,3)(3,5)(5,8)(9,12)(12,16)  -> gold_B {3,5,8,9,12}
#       note the spans do NOT tile: index 8 (the space) belongs to no word,
#       so BOTH 8 (an end) and 9 (a start) are gold boundaries.
#
#   raw    = {1..15}                                    15 positions
#   legal  = {2,3,5,7,8,9,12,15}                        Thai TCC cluster edges
#   core   = legal - trivial = {2,3,5,7,12,15}
#       trivial = {8,9}: position 8 has a space on its right, position 9 has a
#       space on its left. Both are free boundaries for any pre-tokenizer, so
#       `core` removes them from the scored universe -- while leaving them in
#       the gold (CONTRACTS s.2: gold density must not change).
#
#   pred spans (0,2)(2,5)(5,9)(9,12)(12,16) -> pred_B {2,5,9,12}
#
#   -- on `core` = {2,3,5,7,12,15} ------------------------------------------
#   g & m = {3,5,12}    p & m = {2,5,12}    g & p & m = {5,12}
#   TP = 2   FP = 1   FN = 1   TN = 6 - |{2,3,5,12}| = 2
#   P = R = 2/3, F1 = 2/3
#   phi = (2*2 - 1*1)/sqrt(3*3*3*3) = 3/9 = 1/3
#   word_counts: gold (0,3)(3,5)(5,12)(12,16); pred (0,2)(2,5)(5,12)(12,16)
#                exact = {(5,12),(12,16)} -> (2, 4, 4)
#
#   -- on `raw` = {1..15} ---------------------------------------------------
#   TP = 3 (positions 5, 9, 12)   FP = 1 (2)   FN = 2 (3, 8)   TN = 15 - 6 = 9
#   The extra TP is position 9 -- the free post-space boundary. That is exactly
#   the credit `core` exists to withhold.
#   P = 3/4, R = 3/5
#
#   crossing_tokens is mask-independent (it takes the full gold):
#       (0,2) contains --      no      (2,5) contains 3   cross
#       (5,9) contains 8       cross   (9,12) contains -- no
#       (12,16) contains --    no
#       => 2, purity = 1 - 2/5 = 0.6
#
# --------------------------------------------------------------------------

TH_REC = BY_LANG["th"][0]
TH_N = 16
TH_GOLD_B = frozenset({3, 5, 8, 9, 12})
TH_PRED_SPANS = ((0, 2), (2, 5), (5, 9), (9, 12), (12, 16))
TH_PRED_B = frozenset({2, 5, 9, 12})
TH_LEGAL = frozenset({2, 3, 5, 7, 8, 9, 12, 15})
TH_CORE = frozenset({2, 3, 5, 7, 12, 15})


def test_golden_th_fixture_shape():
    assert TH_REC.id == "mini_th/test/000000"
    assert TH_REC.text == "ฉันทำงาน ที่บ้าน"
    assert TH_REC.n == TH_N
    assert TH_REC.text[8] == " "
    assert TH_REC.spans == ((0, 3), (3, 5), (5, 8), (9, 12), (12, 16))
    assert "ำ" in TH_REC.text  # SARA AM, the NFD/NFKD codepoint-count hazard


def test_golden_th_gold_boundaries_include_both_sides_of_the_space():
    assert gold_boundaries(TH_REC) == TH_GOLD_B
    assert 8 in TH_GOLD_B and 9 in TH_GOLD_B


def test_golden_th_trivial_positions_are_exactly_the_space_neighbours():
    assert trivial_positions(TH_REC.text) == frozenset({8, 9})


def test_golden_th_masks():
    masks = compute_masks(TH_REC.text, "th")
    assert masks["raw"] == frozenset(range(1, TH_N))
    assert masks["legal"] == TH_LEGAL
    assert masks["core"] == TH_CORE
    assert masks["core"] == masks["legal"] - trivial_positions(TH_REC.text)


def test_golden_th_core_strips_the_whitespace_adjacent_positions_from_the_mask_only():
    """The positions leave the universe; the gold boundaries do not leave the gold."""
    assert 8 in TH_LEGAL and 9 in TH_LEGAL
    assert 8 not in TH_CORE and 9 not in TH_CORE
    assert 8 in TH_GOLD_B and 9 in TH_GOLD_B


def test_golden_th_pred_boundaries():
    assert spans_to_boundaries(TH_PRED_SPANS, TH_N) == TH_PRED_B


def test_golden_th_core_contingency_and_scalars():
    c = boundary_counts(TH_GOLD_B, TH_PRED_B, TH_CORE)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 2)
    assert precision(c) == pytest.approx(2 / 3)
    assert recall(c) == pytest.approx(2 / 3)
    assert f1(c) == pytest.approx(2 / 3)
    assert phi(c) == pytest.approx(1 / 3, abs=1e-12)
    assert round(phi(c), 6) == 0.333333


def test_golden_th_raw_contingency_grants_the_free_post_space_boundary():
    c = boundary_counts(TH_GOLD_B, TH_PRED_B, frozenset(range(1, TH_N)))
    assert (c.tp, c.fp, c.fn, c.tn) == (3, 1, 2, 9)
    assert precision(c) == pytest.approx(3 / 4)
    assert recall(c) == pytest.approx(3 / 5)
    # position 9 is a true positive under raw and simply absent under core
    assert 9 in (TH_GOLD_B & TH_PRED_B)
    assert 9 not in TH_CORE


def test_golden_th_core_word_counts():
# improved
    assert word_counts(TH_GOLD_B, TH_PRED_B, TH_CORE, TH_N) == (2, 4, 4)


def test_golden_th_raw_word_counts():
    assert word_counts(TH_GOLD_B, TH_PRED_B, frozenset(range(1, TH_N)), TH_N) == (2, 5, 6)


def test_golden_th_crossing_and_purity_are_mask_independent():
    crossing = crossing_tokens(list(TH_PRED_SPANS), TH_GOLD_B)
    assert crossing == 2
    assert 1.0 - crossing / len(TH_PRED_SPANS) == pytest.approx(0.6)


# ==========================================================================
# GOLDEN 3 -- Khmer COENG: pins `legal`
# ==========================================================================
#
#   mini_km/test/000001
#   text  ខ្ញុំទៅសាលារៀន             n = 14
#
#   idx  0      1      2      3      4      5      6
#   cp   U+1781 U+17D2 U+1789 U+17BB U+17C6 U+1791 U+17C5
#        ខ      COENG  ញ      ុ      ំ      ទ      ៅ
#   idx  7      8      9      10     11     12     13
#        U+179F U+17B6 U+179B U+17B6 U+179A U+17C0 U+1793
#        ស      ា      ល      ា      រ      ៀ      ន
#
#   gold spans (0,5)(5,7)(7,11)(11,14) -> gold_B {5,7,11}
#
#   raw   = {1..13}                       13 positions
#   legal = {5,7,9,11,13}
#       position 1 is illegal: U+17D2 COENG is a non-starter
#       position 2 is illegal: U+17D2 COENG is a non-final -- it binds the
#                              following consonant as a subscript, so a cut
#                              there is not a linguistic position at all
#       positions 3,4,6,8,10,12 are illegal: dependent vowels/signs are
#                              non-starters (several are spacing Mc, so Unicode
#                              general category alone would not catch them)
#   core  = legal (nothing trivial in this sentence -- no space, no punctuation)
#
#   pred spans (0,2)(2,5)(5,7)(7,14) -> pred_B {2,5,7}
#       the boundary at 2 is the classic byte-level-BPE failure: a cut placed
#       immediately after COENG.
#
#   -- on `legal` = {5,7,9,11,13} -------------------------------------------
#   g & m = {5,7,11}   p & m = {5,7}   (2 is not in the universe at all)
#   TP = 2   FP = 0   FN = 1   TN = 5 - 3 = 2
#   P = 1.0, R = 2/3, F1 = 2*1*(2/3)/(1+2/3) = 0.8
#   phi = (2*2 - 0*1)/sqrt(2*3*2*3) = 4/6 = 2/3
#   word_counts: gold (0,5)(5,7)(7,11)(11,14); pred (0,5)(5,7)(7,14)
#                exact = {(0,5),(5,7)} -> (2, 3, 4)
#
#   -- on `raw` = {1..13} ---------------------------------------------------
#   TP = 2   FP = 1 (the COENG-internal cut)   FN = 1   TN = 13 - 4 = 9
#   P = R = 2/3;  phi = (2*9 - 1*1)/sqrt(3*3*10*10) = 17/30
#   word_counts: pred (0,2)(2,5)(5,7)(7,14) -> exact = {(5,7)} -> (1, 4, 4)
#
#   So `legal` neither credits nor charges the illegal cut; `raw` charges it as
#   an FP and also destroys a word match. That is the whole point of the mask.
#
# --------------------------------------------------------------------------

KM_REC = BY_LANG["km"][1]
KM_N = 14
KM_GOLD_B = frozenset({5, 7, 11})
KM_PRED_SPANS = ((0, 2), (2, 5), (5, 7), (7, 14))
KM_PRED_B = frozenset({2, 5, 7})
KM_LEGAL = frozenset({5, 7, 9, 11, 13})


def test_golden_km_fixture_shape_and_coeng_position():
    assert KM_REC.id == "mini_km/test/000001"
    assert KM_REC.n == KM_N
    assert KM_REC.text[1] == "្"  # KHMER SIGN COENG
    assert KM_REC.spans == ((0, 5), (5, 7), (7, 11), (11, 14))
    assert gold_boundaries(KM_REC) == KM_GOLD_B


def test_golden_km_legal_excludes_both_sides_of_coeng():
    legal = legal_positions(KM_REC.text, "km")
    assert legal == KM_LEGAL
    assert 1 not in legal, "COENG cannot start a cluster"
    assert 2 not in legal, "COENG cannot end a cluster -- it subscripts what follows"


def test_golden_km_legal_excludes_dependent_vowel_positions():
    legal = legal_positions(KM_REC.text, "km")
    for i in (3, 4, 6, 8, 10, 12):
        assert i not in legal, f"position {i} precedes a dependent vowel/sign"


def test_golden_km_core_equals_legal_here():
    masks = compute_masks(KM_REC.text, "km")
    assert trivial_positions(KM_REC.text) == frozenset()
    assert masks["core"] == masks["legal"] == KM_LEGAL
    assert masks["raw"] == frozenset(range(1, KM_N))


def test_golden_km_pred_boundaries():
    assert spans_to_boundaries(KM_PRED_SPANS, KM_N) == KM_PRED_B


def test_golden_km_legal_contingency_and_scalars():
    c = boundary_counts(KM_GOLD_B, KM_PRED_B, KM_LEGAL)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 0, 1, 2)
    assert precision(c) == pytest.approx(1.0)
    assert recall(c) == pytest.approx(2 / 3)
    assert f1(c) == pytest.approx(0.8)
    assert phi(c) == pytest.approx(2 / 3, abs=1e-12)
    assert round(phi(c), 6) == 0.666667


def test_golden_km_raw_contingency_charges_the_illegal_cut():
    c = boundary_counts(KM_GOLD_B, KM_PRED_B, frozenset(range(1, KM_N)))
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 9)
    assert precision(c) == pytest.approx(2 / 3)
    assert phi(c) == pytest.approx(17 / 30, abs=1e-12)


def test_golden_km_word_counts_legal_and_raw():
    assert word_counts(KM_GOLD_B, KM_PRED_B, KM_LEGAL, KM_N) == (2, 3, 4)
    assert word_counts(KM_GOLD_B, KM_PRED_B, frozenset(range(1, KM_N)), KM_N) == (1, 4, 4)


def test_golden_km_crossing_and_purity():
    crossing = crossing_tokens(list(KM_PRED_SPANS), KM_GOLD_B)
    assert crossing == 1  # only (7,14) strictly contains a gold boundary (11)
    assert 1.0 - crossing / len(KM_PRED_SPANS) == pytest.approx(0.75)


# ==========================================================================
# INVARIANT: the character baseline
# ==========================================================================


@pytest.mark.parametrize("mask_name", MASKS)
def test_char_baseline_recall_is_one(mask_name):
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), mask_name)
        c = boundary_counts(gold_boundaries(rec), mask, mask)
        assert recall(c) == 1.0, rec.id


@pytest.mark.parametrize("mask_name", MASKS)
def test_char_baseline_precision_is_gold_density_in_the_mask(mask_name):
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), mask_name)
        if not mask:
            continue
        gold = gold_boundaries(rec)
        c = boundary_counts(gold, mask, mask)
        assert precision(c) == pytest.approx(len(gold & mask) / len(mask)), rec.id


@pytest.mark.parametrize("mask_name", MASKS)
def test_char_baseline_phi_is_exactly_zero(mask_name):
    """CONTRACTS s.3: phi is 0 for the character tokenizer. Not approximately."""
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), mask_name)
        c = boundary_counts(gold_boundaries(rec), mask, mask)
        assert phi(c) == 0.0, rec.id


def test_char_baseline_purity_is_one():
    """Single-codepoint tokens cannot strictly contain anything."""
    for rec in ALL_RECORDS:
        spans = char_spans(rec.n)
        crossing = crossing_tokens(spans, gold_boundaries(rec))
        assert crossing == 0, rec.id
#         assert 1.0 - crossing / len(spans) == 1.0


def test_char_baseline_has_no_false_negatives_and_no_true_negatives():
    """fn == tn == 0 is *why* the phi denominator vanishes; assert the mechanism."""
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), "core")
        c = boundary_counts(gold_boundaries(rec), mask, mask)
        assert c.fn == 0, rec.id
        assert c.tn == 0, rec.id


# ==========================================================================
# INVARIANT: the whole-sentence baseline
# ==========================================================================

_EMPTY: frozenset[int] = frozenset()


@pytest.mark.parametrize("mask_name", MASKS)
def test_whole_sentence_recall_is_zero(mask_name):
    seen = 0
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), mask_name)
        gold = gold_boundaries(rec)
        if not (gold & mask):
            continue  # the vacuous 0/0 corner, covered separately
        seen += 1
        assert recall(boundary_counts(gold, _EMPTY, mask)) == 0.0, rec.id
    assert seen > 0


@pytest.mark.parametrize("mask_name", MASKS)
def test_whole_sentence_f1_is_zero(mask_name):
    seen = 0
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), mask_name)
        gold = gold_boundaries(rec)
        if not (gold & mask):
            continue
        seen += 1
        assert f1(boundary_counts(gold, _EMPTY, mask)) == 0.0, rec.id
    assert seen > 0


@pytest.mark.parametrize("mask_name", MASKS)
def test_whole_sentence_precision_is_one_by_the_zero_over_zero_convention(mask_name):
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), mask_name)
        c = boundary_counts(gold_boundaries(rec), _EMPTY, mask)
        assert c.tp + c.fp == 0
        assert precision(c) == 1.0, rec.id


@pytest.mark.parametrize("mask_name", MASKS)
def test_whole_sentence_phi_is_exactly_zero(mask_name):
    """CONTRACTS s.3: phi is 0 for whole-sentence. tp == fp == 0 zeroes the margin."""
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), mask_name)
        c = boundary_counts(gold_boundaries(rec), _EMPTY, mask)
        assert phi(c) == 0.0, rec.id


def test_whole_sentence_word_counts_give_one_pred_word():
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), "core")
        _tp, n_pred, _n_gold = word_counts(gold_boundaries(rec), _EMPTY, mask, rec.n)
        assert n_pred == 1, rec.id


# ==========================================================================
# INVARIANT: the oracle
# ==========================================================================


@pytest.mark.parametrize("mask_name", MASKS)
def test_oracle_phi_is_one(mask_name):
    """phi == 1 requires a non-degenerate margin: 0 < |gold & mask| < |mask|."""
    seen = 0
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), mask_name)
        gold = gold_boundaries(rec)
        c = boundary_counts(gold, gold, mask)
        if c.tp == 0 or c.tn == 0:
            continue
        seen += 1
        assert phi(c) == pytest.approx(1.0, abs=1e-12), rec.id
    assert seen > 0
# improved


@pytest.mark.parametrize("mask_name", MASKS)
# def test_oracle_f1_is_one(mask_name):
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), mask_name)
        gold = gold_boundaries(rec)
        c = boundary_counts(gold, gold, mask)
        assert precision(c) == 1.0, rec.id
        assert recall(c) == 1.0, rec.id
        assert f1(c) == 1.0, rec.id


@pytest.mark.parametrize("mask_name", MASKS)
def test_oracle_word_tp_equals_the_induced_gold_word_count(mask_name):
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), mask_name)
        gold = gold_boundaries(rec)
        tp, n_pred, n_gold = word_counts(gold, gold, mask, rec.n)
        assert tp == n_gold == n_pred, rec.id
        assert n_gold == len(gold & mask) + 1, rec.id


@pytest.mark.parametrize("mask_name", MASKS)
def test_oracle_has_no_false_positives_or_negatives(mask_name):
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), mask_name)
        gold = gold_boundaries(rec)
        c = boundary_counts(gold, gold, mask)
        assert c.fp == 0 and c.fn == 0, rec.id


# ==========================================================================
# INVARIANT: refinement monotonicity  (pred_B superset of gold_B)
# ==========================================================================


def _refinements(gold: frozenset[int], n: int) -> list[frozenset[int]]:
    """A few boundary sets that all contain ``gold``."""
    extra = frozenset(range(1, n))
    return [
        gold,
        gold | frozenset(i for i in range(1, n) if i % 2 == 0),
# improved
        gold | frozenset(i for i in range(1, n) if i % 3 == 0),
        extra,
    ]

# 
@pytest.mark.parametrize("mask_name", MASKS)
def test_refinement_recall_is_one(mask_name):
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), mask_name)
        gold = gold_boundaries(rec)
        for pred in _refinements(gold, rec.n):
            assert gold <= pred
            assert recall(boundary_counts(gold, pred, mask)) == 1.0, rec.id


def test_refinement_purity_is_one():
    """purity == 1 <=> gold subset of pred (crossing_tokens docstring, CONTRACTS s.3)."""
    for rec in ALL_RECORDS:
        gold = gold_boundaries(rec)
        for pred in _refinements(gold, rec.n):
            spans = boundaries_to_spans(pred, rec.n)
            crossing = crossing_tokens(spans, gold)
            assert crossing == 0, rec.id
            assert 1.0 - crossing / len(spans) == 1.0


def test_refinement_boundary_miss_rate_is_zero():
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), "core")
        gold = gold_boundaries(rec)
        for pred in _refinements(gold, rec.n):
            c = boundary_counts(gold, pred, mask)
            assert c.fn == 0, rec.id
            assert 1.0 - recall(c) == 0.0


# ==========================================================================
# INVARIANT: Counts is a commutative monoid and pooling is exact
# ==========================================================================


def _pool(recs, mask_name: str, shift: int) -> list[Counts]:
    out = []
    for rec in recs:
        mask = compute_mask(rec.text, lang_of(rec), mask_name)
        gold = gold_boundaries(rec)
        pred = frozenset(i for i in range(1, rec.n) if (i + shift) % 3 == 0)
        out.append(boundary_counts(gold, pred, mask))
    return out


def _sum(cs: list[Counts]) -> Counts:
    total = Counts(0, 0, 0, 0)
    for c in cs:
        total = total + c
    return total


def test_monoid_identity_element():
    zero = Counts(0, 0, 0, 0)
    for c in _pool(ALL_RECORDS, "core", 0):
        assert c + zero == c
        assert zero + c == c


def test_monoid_is_commutative():
    cs = _pool(ALL_RECORDS, "core", 1)
    for a, b in itertools.pairwise(cs):
        assert a + b == b + a


def test_monoid_is_associative():
    cs = _pool(ALL_RECORDS, "core", 2)
    for a, b, c in zip(cs, cs[1:], cs[2:], strict=False):
        assert (a + b) + c == a + (b + c)


@pytest.mark.parametrize("mask_name", MASKS)
@pytest.mark.parametrize("k", [1, 2, 3, 5, 7, 13])
def test_aggregation_over_any_partition_is_bit_identical_to_pooling(mask_name, k):
    """Split the sentences into k blocks; the pooled table must be identical."""
    cs = _pool(ALL_RECORDS, mask_name, 1)
    blocks = [cs[i::k] for i in range(k)]
    # the blocks really are a partition: disjoint, and together they are all of cs
    assert sum(len(b) for b in blocks) == len(cs)
    assert sorted(id(c) for b in blocks for c in b) == sorted(id(c) for c in cs)
    by_block = _sum([_sum(b) for b in blocks if b])
    at_once = _sum(cs)
    assert by_block == at_once
    # and every derived float is bit-identical, not merely close
    assert phi(by_block) == phi(at_once)
    assert precision(by_block) == precision(at_once)
    assert recall(by_block) == recall(at_once)
    assert f1(by_block) == f1(at_once)


def test_aggregation_by_language_is_bit_identical_to_pooling_everything():
    per_lang = [_sum(_pool(BY_LANG[lg], "core", 1)) for lg in BY_LANG]
# improved
    assert _sum(per_lang) == _sum(_pool(ALL_RECORDS, "core", 1))


def test_counts_n_is_the_sum_of_the_cells():
    for c in _pool(ALL_RECORDS, "core", 1):
        assert c.n == c.tp + c.fp + c.fn + c.tn


# ==========================================================================
# The frozen 0/0 conventions table  (CONTRACTS s.3)
# ==========================================================================
#
#   | case                            | value |
#   |---------------------------------|-------|
#   | precision with no predictions   | 1.0   |
#   | recall with no gold boundaries  | 1.0   |
#   | F1 when P + R == 0              | 0.0   |
#   | phi with a zero margin          | 0.0   |
#
# --------------------------------------------------------------------------


@pytest.mark.parametrize("c", [Counts(0, 0, 3, 7), Counts(0, 0, 1, 0), Counts(0, 0, 0, 5)])
def test_convention_precision_with_no_predictions_is_one(c):
    assert c.tp + c.fp == 0
    assert precision(c) == 1.0


@pytest.mark.parametrize("c", [Counts(0, 4, 0, 6), Counts(0, 1, 0, 0), Counts(0, 0, 0, 5)])
def test_convention_recall_with_no_gold_is_one(c):
    assert c.tp + c.fn == 0
    assert recall(c) == 1.0


@pytest.mark.parametrize("c", [Counts(0, 2, 3, 5), Counts(0, 1, 1, 0), Counts(0, 9, 9, 9)])
def test_convention_f1_is_zero_when_precision_plus_recall_is_zero(c):
    assert precision(c) == 0.0
    assert recall(c) == 0.0
    assert f1(c) == 0.0


@pytest.mark.parametrize(
    "c",
    [
        Counts(0, 0, 0, 0),  # empty universe
        Counts(5, 3, 0, 0),  # character baseline: fn == tn == 0
        Counts(0, 0, 4, 9),  # whole sentence: tp == fp == 0
        Counts(0, 7, 0, 3),  # no gold at all
        Counts(4, 0, 0, 0),  # perfect but no true negatives
    ],
)
def test_convention_phi_with_a_zero_margin_is_zero(c):
    margin = (c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn)
    assert margin == 0
    assert phi(c) == 0.0


def test_convention_f1_of_the_empty_universe_is_one_because_p_and_r_are_both_one():
    """The ``P + R == 0`` branch does NOT fire on an empty universe.

    Both 0/0 conventions push P and R to 1.0, so F1 is 1.0. This is the literal
    reading of the CONTRACTS s.3 table (the F1 rule is conditioned on P+R == 0,
    which is false here) and it is the same family of disagreement as the
    documented phi^2 != J*M corner below. Asserted so the behaviour is explicit
    rather than incidental.
    """
    c = Counts(0, 0, 0, 0)
    assert precision(c) == 1.0
    assert recall(c) == 1.0
    assert f1(c) == 1.0


def test_convention_word_precision_and_recall_default_to_one():
    row = compute_row(
        Counts(0, 0, 0, 0),
        w_tp=0,
        w_pred=0,
        w_gold=0,
        w_intact=0,
        n_tokens=0,
        n_chars=0,
        n_gold_words=0,
        crossing=0,
    )
    assert row.w_p == 1.0
    assert row.w_r == 1.0
    assert row.purity == 1.0
    assert row.word_exact_rate == 1.0
    assert row.word_intact_rate == 1.0


# ==========================================================================
# The documented corner where phi^2 != J * M -- INTENTIONAL (CONTRACTS s.3)
# ==========================================================================


@pytest.mark.parametrize("tn", [1, 5, 100])
def test_zero_margin_corner_phi_squared_differs_from_j_times_m_and_that_is_intentional(tn):
    """CONTRACTS s.3 states this disagreement explicitly and calls it correct.

    With no gold AND no predictions inside the mask:
      precision = 1.0 (no errors of commission) and recall = 1.0 (vacuously
      complete), hence informedness = markedness = 1.0, while phi = 0.0 because
      the MCC margin is zero. The identity phi^2 == J*M simply does not hold at a
      zero margin. Both values are individually correct; do NOT "fix" either one.
    """
    c = Counts(tp=0, fp=0, fn=0, tn=tn)
    assert precision(c) == 1.0
    assert recall(c) == 1.0
    assert informedness(c) == 1.0
    assert markedness(c) == 1.0
    assert phi(c) == 0.0
    assert phi(c) ** 2 != pytest.approx(informedness(c) * markedness(c))


def test_zero_margin_corner_only_affects_sentences_with_no_gold_boundary_intentionally():
    """CONTRACTS s.3: 'only sentences with no gold boundary are affected -- they
    contribute nothing to any pooled table.' Verify the second half literally."""
    c = Counts(tp=0, fp=0, fn=0, tn=17)
    other = Counts(tp=3, fp=1, fn=2, tn=40)
    assert (other + c).tp == other.tp
    assert (other + c).fp == other.fp
    assert (other + c).fn == other.fn
    # only tn moves, and tn cannot resurrect a nonzero numerator on its own
    assert (other + c).tn == other.tn + 17


def test_zero_margin_corner_arises_from_real_fixture_sentences_intentionally():
    """The single-word mini sentences reach this corner for real, on `core`."""
    hits = 0
    for rec in ALL_RECORDS:
        mask = compute_mask(rec.text, lang_of(rec), "core")
        gold = gold_boundaries(rec)
        if gold & mask or not mask:
            continue
        c = boundary_counts(gold, _EMPTY, mask)
        hits += 1
        assert precision(c) == 1.0
        assert recall(c) == 1.0
        assert informedness(c) == 1.0
        assert markedness(c) == 1.0
        assert phi(c) == 0.0
    assert hits > 0, "the mini corpus must contain at least one such sentence"


# ==========================================================================
# Analytic phi: oracle plus k spurious splits
# ==========================================================================
#
#   TP = K, FP = k, FN = 0, TN = N - K - k
#
#   num = K*(N-K-k) - k*0 = K*(N-K-k)
#   den = sqrt( (K+k) * K * (N-K-k+k) * (N-K-k) )
#       = sqrt( (K+k) * K * (N-K)    * (N-K-k) )
#
#   phi = K*(N-K-k) / sqrt( K*(K+k)*(N-K)*(N-K-k) )
#       = sqrt( K*(N-K-k) / ((K+k)*(N-K)) )
#
# --------------------------------------------------------------------------

ANALYTIC_CASES = [
    (100, 10, 0),
    (100, 10, 1),
    (100, 10, 5),
    (100, 10, 50),
    (100, 50, 25),
    (50, 5, 5),
    (20, 10, 5),
    (6, 3, 1),
    (1000, 300, 7),
    (10000, 2500, 123),
    (997, 1, 3),
    (997, 996, 1),
]


@pytest.mark.parametrize(("n_total", "k_true", "k_spurious"), ANALYTIC_CASES)
def test_analytic_phi_for_oracle_plus_k_spurious_splits(n_total, k_true, k_spurious):
    tn = n_total - k_true - k_spurious
    assert tn >= 0
    c = Counts(tp=k_true, fp=k_spurious, fn=0, tn=tn)
    assert c.n == n_total
    expected = math.sqrt(
        (k_true * (n_total - k_true - k_spurious)) / ((k_true + k_spurious) * (n_total - k_true))
    )
    assert phi(c) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(("n_total", "k_true"), [(100, 10), (50, 5), (10000, 2500)])
def test_analytic_phi_is_exactly_one_when_there_are_no_spurious_splits(n_total, k_true):
    c = Counts(tp=k_true, fp=0, fn=0, tn=n_total - k_true)
    assert phi(c) == pytest.approx(1.0, abs=1e-12)
    assert f1(c) == 1.0


@pytest.mark.parametrize("k_spurious", [1, 2, 5, 10, 40])
def test_analytic_phi_decreases_monotonically_in_the_number_of_spurious_splits(k_spurious):
    n_total, k_true = 100, 10
    a = phi(Counts(k_true, k_spurious, 0, n_total - k_true - k_spurious))
    b = phi(Counts(k_true, k_spurious + 1, 0, n_total - k_true - k_spurious - 1))
    assert a > b


@pytest.mark.parametrize("case", [(30, 10, 20), (12, 4, 8), (5, 4, 1)])
def test_analytic_phi_collapses_to_zero_when_the_universe_is_exhausted(case):
    """TN == 0 is the character baseline in disguise: the closed form gives 0 too."""
    n_total, k_true, k_spurious = case
    assert n_total - k_true - k_spurious == 0
    c = Counts(tp=k_true, fp=k_spurious, fn=0, tn=0)
    assert phi(c) == 0.0
    expected = math.sqrt(
        (k_true * (n_total - k_true - k_spurious)) / ((k_true + k_spurious) * (n_total - k_true))
    )
    assert expected == 0.0


# ==========================================================================
# Assorted spec-pinning odds and ends
# ==========================================================================


def test_boundary_counts_intersects_gold_with_the_mask_too():
    """CONTRACTS s.3: BOTH gold and pred are masked before counting."""
    gold = frozenset({1, 2, 3})
    pred = frozenset({1})
    mask = frozenset({1, 4, 5})
    c = boundary_counts(gold, pred, mask)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 0, 0, 2)
    assert recall(c) == 1.0, "gold boundaries outside the mask must not count as misses"


def test_positions_zero_and_n_are_excluded_from_every_universe():
    for rec in ALL_RECORDS:
        for mask_name in MASKS:
            mask = compute_mask(rec.text, lang_of(rec), mask_name)
            assert 0 not in mask, (rec.id, mask_name)
            assert rec.n not in mask, (rec.id, mask_name)
        assert 0 not in gold_boundaries(rec)
        assert rec.n not in gold_boundaries(rec)


def test_core_is_a_subset_of_legal_is_a_subset_of_raw():
    for rec in ALL_RECORDS:
        m = compute_masks(rec.text, lang_of(rec))
        assert m["core"] <= m["legal"] <= m["raw"], rec.id


def test_compute_mask_rejects_an_unknown_mask_name():
    with pytest.raises(ValueError, match="unknown mask"):
        compute_mask("我喜欢", "zh", "kore")


def test_boundaries_to_spans_tiles_zero_to_n():
    for rec in ALL_RECORDS:
        spans = boundaries_to_spans(gold_boundaries(rec), rec.n)
        assert spans[0][0] == 0
        assert spans[-1][1] == rec.n
        for (_, e), (s2, _) in itertools.pairwise(spans):
            assert e == s2

# Refined

# Updated

# Enhanced

# Enhanced

# Enhanced

# Updated

# Updated

# Enhanced

# Updated

# Updated

# Enhanced
