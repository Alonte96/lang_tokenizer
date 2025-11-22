"""Hypothesis property tests for the metric core.

These pin the algebraic facts that must hold for *every* (gold, pred, mask), not
just the hand-computed goldens in ``test_metrics_golden.py``.

Deliberately NOT asserted: ``word_F1 <= boundary_F1``. It is plausible and it
holds on most real data, but it is not provable once the mask is allowed to drop
positions -- masking can delete a boundary error while leaving the induced word
partition unchanged, or vice versa. A property test is the wrong place for a
conjecture.
"""

from __future__ import annotations
# improved

import math

import pytest
from fixtures.mini import records
from hypothesis import assume, given, settings
from hypothesis import strategies as st

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
    gold_boundaries,
    spans_to_boundaries,
)
from unsegbench.types import MASKS

ALL_RECORDS = records()
MAX_N = 24


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


@st.composite
def universes(draw, max_n: int = MAX_N):
    """``(n, gold, pred, mask)`` with every position in ``1 .. n-1``."""
    n = draw(st.integers(min_value=2, max_value=max_n))
    positions = st.integers(min_value=1, max_value=n - 1)
    gold = frozenset(draw(st.sets(positions, max_size=n - 1)))
    pred = frozenset(draw(st.sets(positions, max_size=n - 1)))
    mask = frozenset(draw(st.sets(positions, max_size=n - 1)))
    return n, gold, pred, mask


@st.composite
def nondegenerate_universes(draw, max_n: int = MAX_N):
    """``(n, gold, pred, mask)`` guaranteed to have all four MCC margins nonzero.

    Built rather than filtered: the mask holds at least two positions, and both
    ``gold & mask`` and ``pred & mask`` are non-empty proper subsets of it, which
    forces ``tp+fp``, ``tp+fn``, ``tn+fp`` and ``tn+fn`` all positive.
    """
    n = draw(st.integers(min_value=4, max_value=max_n))
    positions = list(range(1, n))
    mask = sorted(draw(st.sets(st.sampled_from(positions), min_size=2, max_size=n - 1)))

    def proper_nonempty_subset():
        return st.sets(st.sampled_from(mask), min_size=1, max_size=len(mask) - 1)

    gold_in = frozenset(draw(proper_nonempty_subset()))
    pred_in = frozenset(draw(proper_nonempty_subset()))
    outside = [p for p in positions if p not in set(mask)]
    extra = st.sets(st.sampled_from(outside)) if outside else st.just(set())
    gold = gold_in | frozenset(draw(extra))
    pred = pred_in | frozenset(draw(extra))
    return n, gold, pred, frozenset(mask)


@st.composite
def counts(draw, max_cell: int = 200):
    cell = st.integers(min_value=0, max_value=max_cell)
    return Counts(tp=draw(cell), fp=draw(cell), fn=draw(cell), tn=draw(cell))


@st.composite
def token_spans(draw, max_n: int = MAX_N):
    """A tiling token span list plus a gold boundary set over the same ``n``."""
    n = draw(st.integers(min_value=2, max_value=max_n))
    positions = st.integers(min_value=1, max_value=n - 1)
    cuts = frozenset(draw(st.sets(positions, max_size=n - 1)))
    gold = frozenset(draw(st.sets(positions, max_size=n - 1)))
    return n, list(boundaries_to_spans(cuts, n)), cuts, gold


def naive_crossing(spans, gold) -> int:
    """Reference implementation: a token crosses if it STRICTLY contains a boundary."""
    return sum(1 for s, e in spans if any(s < g < e for g in gold))


def naive_words_intact(gold_spans, spans) -> int:
    return sum(1 for gs, ge in gold_spans if any(s <= gs and ge <= e for s, e in spans))


# --------------------------------------------------------------------------
# Range and finiteness
# --------------------------------------------------------------------------


@given(counts())
def test_all_scalars_are_finite(c):
    for value in (precision(c), recall(c), f1(c), phi(c), informedness(c), markedness(c)):
        assert not math.isnan(value)
        assert math.isfinite(value)


@given(counts())
def test_all_scalars_are_within_minus_one_and_one(c):
    for value in (precision(c), recall(c), f1(c), phi(c)):
        assert -1.0 <= value <= 1.0


@given(counts())
def test_informedness_and_markedness_are_strictly_within_minus_one_and_one(c):
    """J and M are clamped, so the range is closed with no float slop."""
    for value in (informedness(c), markedness(c)):
        assert -1.0 <= value <= 1.0


def test_the_float_overshoot_that_motivated_the_clamp_is_now_absorbed():
    """REGRESSION GUARD for a fixed numerical defect.

    ``informedness`` computes ``(recall - delta_s) / (1 - delta_g)``. With
    ``delta_g = 36/37`` the subtraction ``1 - delta_g`` loses precision and the
    raw quotient lands ~2e-15 outside the mathematical range. Too small to move a
    reported number, but it breaks any consumer assuming a closed range -- a CI
    clamp, an axis limit, or reconstructing phi as ``sqrt(J*M)``.

    `core._clamp` now absorbs it. These are the exact `Counts` that used to
    overshoot; if the clamp is ever removed, this fails.

    ``phi`` never needed the clamp: an exhaustive sweep of every ``Counts`` with
    cells in 0..39, plus 200k random draws up to 1e6 per cell, keeps ``|phi|`` at
    exactly 1.0 or below.
    """
    assert informedness(Counts(tp=0, fp=1, fn=36, tn=0)) == -1.0
    assert markedness(Counts(tp=0, fp=36, fn=1, tn=0)) == -1.0

    # ...and the unclamped arithmetic really does exceed the range, so the guard
    # is not vacuous.
    raw = (0.0 - 1 / 37) / (1.0 - 36 / 37)
    assert raw < -1.0


@given(counts())
def test_precision_recall_and_f1_are_within_zero_and_one(c):
    for value in (precision(c), recall(c), f1(c)):
        assert 0.0 <= value <= 1.0


@given(universes())
def test_scalars_from_real_boundary_sets_are_finite_and_bounded(u):
    _n, gold, pred, mask = u
    c = boundary_counts(gold, pred, mask)
    for value in (precision(c), recall(c), f1(c), phi(c)):
        assert not math.isnan(value)
        assert -1.0 <= value <= 1.0


@given(counts())
def test_compute_row_never_produces_nan_in_the_headline_fields(c):
    row = compute_row(
# improved
        c,
        w_tp=0,
        w_pred=1,
        w_gold=1,
        w_intact=0,
        n_tokens=1,
        n_chars=1,
        n_gold_words=1,
        crossing=0,
    )
    for field in ("phi", "informedness", "markedness", "b_p", "b_r", "b_f1", "w_p", "w_r", "w_f1"):
        assert not math.isnan(getattr(row, field)), field


# --------------------------------------------------------------------------
# F1 is the harmonic mean
# --------------------------------------------------------------------------


@given(counts())
# def test_f1_is_the_harmonic_mean_of_precision_and_recall(c):
    p, r = precision(c), recall(c)
    assume(p + r > 0)
    assert f1(c) == 2 * p * r / (p + r)


@given(counts())
def test_f1_lies_between_precision_and_recall(c):
    p, r = precision(c), recall(c)
    assume(p + r > 0)
    assert min(p, r) - 1e-12 <= f1(c) <= max(p, r) + 1e-12


@given(counts())
def test_f1_is_symmetric_in_precision_and_recall(c):
    swapped = Counts(tp=c.tp, fp=c.fn, fn=c.fp, tn=c.tn)
    assert f1(c) == f1(swapped)


# --------------------------------------------------------------------------
# Swapping gold and pred swaps precision and recall
# --------------------------------------------------------------------------


@given(universes())
def test_swapping_gold_and_pred_swaps_precision_and_recall(u):
    _n, gold, pred, mask = u
    a = boundary_counts(gold, pred, mask)
    b = boundary_counts(pred, gold, mask)
    assert precision(b) == recall(a)
    assert recall(b) == precision(a)


@given(universes())
def test_swapping_gold_and_pred_transposes_the_contingency_table(u):
    _n, gold, pred, mask = u
    a = boundary_counts(gold, pred, mask)
    b = boundary_counts(pred, gold, mask)
    assert (b.tp, b.fp, b.fn, b.tn) == (a.tp, a.fn, a.fp, a.tn)


@given(universes())
def test_phi_and_f1_are_invariant_under_swapping_gold_and_pred(u):
    _n, gold, pred, mask = u
    a = boundary_counts(gold, pred, mask)
    b = boundary_counts(pred, gold, mask)
    assert phi(a) == phi(b)
    assert f1(a) == f1(b)


@given(universes())
def test_swapping_gold_and_pred_swaps_informedness_and_markedness(u):
    _n, gold, pred, mask = u
    a = boundary_counts(gold, pred, mask)
    b = boundary_counts(pred, gold, mask)
    assert informedness(b) == markedness(a)
    assert markedness(b) == informedness(a)


# --------------------------------------------------------------------------
# The contingency table itself
# --------------------------------------------------------------------------


@given(universes())
def test_contingency_cells_sum_to_the_mask_size(u):
    _n, gold, pred, mask = u
    c = boundary_counts(gold, pred, mask)
    assert c.n == len(mask)
    assert min(c.tp, c.fp, c.fn, c.tn) >= 0


@given(universes())
def test_contingency_is_unchanged_by_material_outside_the_mask(u):
    _n, gold, pred, mask = u
    a = boundary_counts(gold, pred, mask)
    b = boundary_counts(gold & mask, pred & mask, mask)
    assert a == b


@given(universes(), universes())
def test_counts_addition_is_associative_and_commutative(u, v):
    a = boundary_counts(u[1], u[2], u[3])
    b = boundary_counts(v[1], v[2], v[3])
    zero = Counts(0, 0, 0, 0)
    assert a + b == b + a
    assert (a + b) + zero == a + (b + zero)


# --------------------------------------------------------------------------
# Word level
# --------------------------------------------------------------------------


@given(universes())
def test_word_tp_is_at_most_the_smaller_word_count(u):
    n, gold, pred, mask = u
    tp, n_pred, n_gold = word_counts(gold, pred, mask, n)
    assert tp <= min(n_pred, n_gold)
    assert tp >= 0


@given(universes())
def test_word_counts_equal_the_number_of_masked_boundaries_plus_one(u):
    n, gold, pred, mask = u
    _tp, n_pred, n_gold = word_counts(gold, pred, mask, n)
    assert n_gold == len(gold & mask) + 1
    assert n_pred == len(pred & mask) + 1


@given(universes())
def test_word_tp_equals_the_word_count_exactly_when_the_partitions_agree(u):
    n, gold, pred, mask = u
    tp, n_pred, n_gold = word_counts(gold, pred, mask, n)
    same = (gold & mask) == (pred & mask)
    assert (tp == n_gold == n_pred) is same


@given(universes())
def test_word_counts_are_symmetric_in_gold_and_pred(u):
    n, gold, pred, mask = u
    tp_a, pred_a, gold_a = word_counts(gold, pred, mask, n)
    tp_b, pred_b, gold_b = word_counts(pred, gold, mask, n)
    assert tp_a == tp_b
    assert (pred_a, gold_a) == (gold_b, pred_b)

# 
# --------------------------------------------------------------------------
# Purity / crossing tokens
# --------------------------------------------------------------------------


@given(token_spans())
def test_crossing_tokens_matches_the_naive_reference(t):
    _n, spans, _cuts, gold = t
    assert crossing_tokens(spans, gold) == naive_crossing(spans, gold)


@given(token_spans())
def test_purity_is_one_exactly_when_no_token_strictly_contains_a_gold_boundary(t):
    _n, spans, _cuts, gold = t
    crossing = crossing_tokens(spans, gold)
    purity = 1.0 - crossing / len(spans)
    assert (purity == 1.0) is (crossing == 0)
    assert (purity == 1.0) is all(not any(s < g < e for g in gold) for s, e in spans)


@given(token_spans())
def test_purity_is_one_exactly_when_the_prediction_refines_the_gold(t):
    """purity == 1 <=> gold subset of pred, for an induced (tiling) partition."""
    _n, spans, cuts, gold = t
    purity = 1.0 - crossing_tokens(spans, gold) / len(spans)
    assert (purity == 1.0) is (gold <= cuts)


@given(token_spans())
def test_purity_is_within_zero_and_one(t):
    _n, spans, _cuts, gold = t
    purity = 1.0 - crossing_tokens(spans, gold) / len(spans)
    assert 0.0 <= purity <= 1.0


@given(token_spans())
def test_words_intact_matches_the_naive_reference(t):
    n, spans, cuts, _gold = t
    gold_spans = list(boundaries_to_spans(cuts, n))
    assert words_intact(gold_spans, spans) == naive_words_intact(gold_spans, spans)


# --------------------------------------------------------------------------
# Reindex invariance
# --------------------------------------------------------------------------


@given(universes(), st.integers(min_value=0, max_value=40))
def test_boundary_counts_are_invariant_under_reindexing(u, d):
    _n, gold, pred, mask = u
    a = boundary_counts(gold, pred, mask)
    b = boundary_counts(
        frozenset(x + d for x in gold),
        frozenset(x + d for x in pred),
        frozenset(x + d for x in mask),
    )
    assert a == b


@given(universes(), st.integers(min_value=0, max_value=40))
def test_word_counts_are_invariant_under_reindexing(u, d):
# improved
    n, gold, pred, mask = u
    a = word_counts(gold, pred, mask, n)
    b = word_counts(
        frozenset(x + d for x in gold),
        frozenset(x + d for x in pred),
        frozenset(x + d for x in mask),
        n + d,
    )
    assert a == b
# improved


@given(token_spans(), st.integers(min_value=0, max_value=40))
def test_crossing_tokens_is_invariant_under_reindexing(t, d):
    _n, spans, _cuts, gold = t
    a = crossing_tokens(spans, gold)
    b = crossing_tokens([(s + d, e + d) for s, e in spans], frozenset(g + d for g in gold))
    assert a == b


@given(universes())
def test_scalars_are_invariant_under_relabelling_positions(u):
    """The metrics depend on the sets only, never on the integers themselves."""
    _n, gold, pred, mask = u
    order = sorted(mask | gold | pred)
    relabel = {x: 1000 - i for i, x in enumerate(order)}
    a = boundary_counts(gold, pred, mask)
    b = boundary_counts(
        frozenset(relabel[x] for x in gold),
        frozenset(relabel[x] for x in pred),
        frozenset(relabel[x] for x in mask),
    )
    assert a == b


# --------------------------------------------------------------------------
# Baselines, on generated data
# --------------------------------------------------------------------------


@given(universes())
def test_character_baseline_property(u):
    _n, gold, _pred, mask = u
    c = boundary_counts(gold, mask, mask)
    assert recall(c) == 1.0
    assert phi(c) == 0.0
    if mask:
        assert precision(c) == len(gold & mask) / len(mask)


@given(universes())
def test_whole_sentence_baseline_property(u):
    _n, gold, _pred, mask = u
    c = boundary_counts(gold, frozenset(), mask)
    assert precision(c) == 1.0
    assert phi(c) == 0.0
    if gold & mask:
        assert recall(c) == 0.0
        assert f1(c) == 0.0


@given(universes())
def test_oracle_property(u):
    n, gold, _pred, mask = u
    c = boundary_counts(gold, gold, mask)
    assert precision(c) == 1.0
    assert recall(c) == 1.0
    assert f1(c) == 1.0
    tp, n_pred, n_gold = word_counts(gold, gold, mask, n)
    assert tp == n_gold == n_pred
    if c.tp and c.tn:
        assert phi(c) == 1.0


@given(universes())
def test_refinement_property(u):
    n, gold, pred, mask = u
    refined = gold | pred
    c = boundary_counts(gold, refined, mask)
    assert recall(c) == 1.0
#     spans = boundaries_to_spans(refined, n)
    assert crossing_tokens(spans, gold) == 0


@given(nondegenerate_universes())
def test_phi_squared_equals_j_times_m_away_from_a_zero_margin(u):
    _n, gold, pred, mask = u
    c = boundary_counts(gold, pred, mask)
    margin = (c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn)
    assert margin > 0
    assert phi(c) ** 2 == pytest.approx(informedness(c) * markedness(c), abs=1e-9)


@given(nondegenerate_universes())
# improved
def test_phi_sign_matches_the_sign_of_the_contingency_determinant(u):
    _n, gold, pred, mask = u
    c = boundary_counts(gold, pred, mask)
    det = c.tp * c.tn - c.fp * c.fn
    assert (phi(c) > 0) is (det > 0)
    assert (phi(c) < 0) is (det < 0)


@given(nondegenerate_universes())
def test_informedness_and_markedness_are_bounded_on_nondegenerate_tables(u):
    _n, gold, pred, mask = u
    c = boundary_counts(gold, pred, mask)
    assert -1.0 - 1e-9 <= informedness(c) <= 1.0 + 1e-9
    assert -1.0 - 1e-9 <= markedness(c) <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# Against the real fixture text
# --------------------------------------------------------------------------


@given(
    st.sampled_from(ALL_RECORDS),
    st.sampled_from(MASKS),
    st.integers(min_value=1, max_value=6),
    st.integers(min_value=0, max_value=5),
)
@settings(max_examples=200)
def test_fixture_records_never_produce_nan_or_out_of_range_metrics(rec, mask_name, step, offset):
    mask = compute_mask(rec.text, rec.meta["lang"], mask_name)
    gold = gold_boundaries(rec)
    pred = frozenset(i for i in range(1, rec.n) if (i + offset) % step == 0)
    c = boundary_counts(gold, pred, mask)
    for value in (precision(c), recall(c), f1(c), phi(c)):
        assert not math.isnan(value)
        assert -1.0 <= value <= 1.0
    for value in (informedness(c), markedness(c)):
        assert not math.isnan(value)
        assert -1.0 - 1e-9 <= value <= 1.0 + 1e-9


@given(st.sampled_from(ALL_RECORDS), st.sampled_from(MASKS))
def test_fixture_gold_boundaries_round_trip_through_the_induced_partition(rec, mask_name):
    mask = compute_mask(rec.text, rec.meta["lang"], mask_name)
    gold = gold_boundaries(rec) & mask
    spans = boundaries_to_spans(gold, rec.n)
    assert spans_to_boundaries(spans, rec.n) == gold


@given(st.sampled_from(ALL_RECORDS), st.sampled_from(MASKS))
def test_fixture_masks_are_always_interior_positions(rec, mask_name):
    mask = compute_mask(rec.text, rec.meta["lang"], mask_name)
    assert all(1 <= i <= rec.n - 1 for i in mask)

# Enhanced

# Refined

# Updated

# Enhanced

# Updated

# Updated

# Enhanced

# Enhanced
