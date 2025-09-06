"""Density controls: PREREGISTRATION.md section 4, release blocker #2.

The blocker, verbatim: *"N0 and N1 nulls score phi_B ~ 0 at **every observed
density**"*, failing if ``|phi_B| >= 0.02`` anywhere on the sweep. The
consequence of a failure is pre-committed -- we would have to report lift over
N1 instead of raw phi_B -- so the sweep in `test_n0_scores_zero_phi_at_every_density`
and `test_n1_scores_zero_phi_at_every_density` is the single most load-bearing
assertion in this file. Both print the whole per-density table on failure so the
diagnosis is immediate: a monotone drift means the chance correction is leaking
fertility, a spike at one end means a small-margin artefact.

Everything else here supports those two: the closed form is checked against a
Monte-Carlo of the hypergeometric it claims to be, and `E[recall]` vs
`E[precision]` are asserted to behave in the two DIFFERENT ways that make the
density confound concrete -- recall rises linearly with how finely you chop,
precision does not move at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from unsegbench.metrics.core import phi
from unsegbench.metrics.nulls import (
    NullSummary,
    empirical_length_dist,
    lift,
    lift_ci,
    n0_expectation,
    n0_micro_precision,
    n0_null,
    n1_counts,
    n1_null,
    n1_sample,
    snap_to_mask,
)
from unsegbench.metrics.stats import pooled

#: The pre-registered failure threshold. Do not loosen this without amending
#: PREREGISTRATION.md section 4.
BLOCKER_TOL = 0.02

#: Predicted-boundary density sweep: 0.05 to 0.95 in steps of 0.10.
DENSITY_SWEEP = [round(0.05 + 0.10 * i, 2) for i in range(10)]

#: N1 replicates per density. The replicate mean is what the blocker is about;
#: 40 replicates on ~80k positions puts its MC error near 5e-4, i.e. 40x below
#: the 0.02 threshold, so a failure would be signal and not noise.
N1_REPS = 40


# --------------------------------------------------------------------------
# Synthetic corpora
# --------------------------------------------------------------------------


def _margins(
    rng: np.random.Generator,
    n_sent: int = 4000,
    len_lo: int = 12,
    len_hi: int = 48,
    dg_lo: float = 0.10,
    dg_hi: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sentence ``(N, K)`` margins, deliberately heterogeneous.

    Sentence length AND gold density both vary, because a null that is only
    zero on a homogeneous corpus is not a null: pooling unequal sentences is
    exactly where Simpson's-paradox residue would show up.
    """
    n = rng.integers(len_lo, len_hi, size=n_sent).astype(np.int64)
    dg = rng.uniform(dg_lo, dg_hi, size=n_sent)
    k = np.clip(np.rint(n * dg).astype(np.int64), 1, n - 1)
    return n, k


def _n0_draw(rng: np.random.Generator, n: np.ndarray, k: np.ndarray, density: float) -> np.ndarray:
    """One realisation of N0 at ``density``: scatter ``m = hN`` uniformly.

    Draws ``TP`` straight from the hypergeometric, which IS the null (uniform
    placement without replacement), so this is a genuine Monte-Carlo of N0 and
    not a re-implementation of the closed form under test.
    """
    m = np.clip(np.rint(n * density).astype(np.int64), 0, n)
    tp = rng.hypergeometric(k, n - k, m)
    return np.column_stack([tp, m - tp, k - tp, n - m - k + tp]).astype(np.int64)


def _null_corpus(
    rng: np.random.Generator,
    n_sent: int = 400,
    len_lo: int = 120,
    len_hi: int = 280,
    dg_lo: float = 0.15,
    dg_hi: float = 0.55,
    mask_step: int = 0,
) -> tuple[list[frozenset[int]], list[frozenset[int]], list[int]]:
    """Gold sets, masks and text lengths with gold placed UNIFORMLY at random.

    Uniform gold is the point: it makes the true null phi exactly zero, so any
    departure the sweep finds belongs to the null model rather than to a
    conspiracy between gold word lengths and the sampled token lengths.

    ``mask_step`` punches out every ``mask_step``-th position, giving the
    illegal-position case that `snap_to_mask` exists for.
    """
    gold: list[frozenset[int]] = []
    masks: list[frozenset[int]] = []
    lengths: list[int] = []
    for _ in range(n_sent):
        n = int(rng.integers(len_lo, len_hi))
        positions = np.arange(1, n)
        if mask_step:
            positions = positions[positions % mask_step != 0]
        n_gold = max(1, round(positions.size * float(rng.uniform(dg_lo, dg_hi))))
        chosen = rng.choice(positions, size=n_gold, replace=False)
        gold.append(frozenset(int(x) for x in chosen))
        masks.append(frozenset(int(x) for x in positions))
        lengths.append(n)
    return gold, masks, lengths


def bimodal_length_dist(density: float) -> dict[int, float]:
    """A deliberately NON-geometric token-length pmf with mean ``1/density``.

    Two well-separated point masses -- lots of single-character tokens and lots
    of very long ones -- rather than the geometric shape a memoryless null would
    have. That matters: N1's whole claim is that matching a tokenizer's length
    SHAPE does not by itself manufacture alignment, and a geometric test
    distribution would quietly dodge the question.
    """
    mean_len = 1.0 / density
    long = max(2, round(2.0 * mean_len))
    w_short = (long - mean_len) / (long - 1.0)
    return {1: w_short, long: 1.0 - w_short}


def _table(rows: list[tuple[float, float]]) -> str:
    return "\n".join(f"    density={h:.2f}   phi={v:+.6f}" for h, v in rows)


# --------------------------------------------------------------------------
# BLOCKER #2 -- N0 at every density
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def n0_sweep() -> list[tuple[float, float, float, float]]:
    """``(density, closed-form phi, mean MC phi, max |MC phi| over replicates)``."""
    rng = np.random.default_rng(20240921)
    n, k = _margins(rng)
    out: list[tuple[float, float, float, float]] = []
    for h in DENSITY_SWEEP:
        mats = [_n0_draw(rng, n, k, h) for _ in range(12)]
        mc = np.array([phi(pooled(m)) for m in mats], dtype=float)
        out.append((h, n0_null(mats[0]).e_phi, float(mc.mean()), float(np.abs(mc).max())))
    return out


def test_n0_scores_zero_phi_at_every_density(n0_sweep):
    """BLOCKER #2, closed-form arm. |phi| < 0.02 at EVERY density, not on average."""
    rows = [(h, cf) for h, cf, _, _ in n0_sweep]
    worst_h, worst = max(rows, key=lambda r: abs(r[1]))
    assert abs(worst) < BLOCKER_TOL, (
        "PREREGISTRATION.md S4 blocker #2 FAILED for N0 (closed form): the chance "
        f"correction does not hold at every density.\n"
        f"  worst |phi| = {abs(worst):.6f} at density {worst_h:.2f} "
        f"(threshold {BLOCKER_TOL})\n" + _table(rows)
    )


def test_n0_monte_carlo_agrees_with_the_closed_form_at_every_density(n0_sweep):
    """BLOCKER #2, Monte-Carlo arm: the closed form is not lying about the sweep."""
    rows = [(h, mc) for h, _, mc, _ in n0_sweep]
    worst_h, worst = max(rows, key=lambda r: abs(r[1]))
    assert abs(worst) < BLOCKER_TOL, (
        "PREREGISTRATION.md S4 blocker #2 FAILED for N0 (simulated): uniform "
        "matched-count placement does not score zero at every density.\n"
        f"  worst |mean phi| = {abs(worst):.6f} at density {worst_h:.2f} "
        f"(threshold {BLOCKER_TOL})\n" + _table(rows)
    )


def test_n0_every_single_monte_carlo_replicate_is_within_tolerance(n0_sweep):
    """Not just the replicate mean: no INDIVIDUAL N0 realisation crosses 0.02."""
    rows = [(h, mx) for h, _, _, mx in n0_sweep]
    worst_h, worst = max(rows, key=lambda r: r[1])
    assert worst < BLOCKER_TOL, (
        f"a single N0 replicate reached |phi| = {worst:.6f} at density {worst_h:.2f}\n"
        + _table(rows)
    )


def test_n0_closed_form_and_simulation_agree_pointwise(n0_sweep):
    for h, closed, mc, _ in n0_sweep:
        assert abs(closed - mc) < 0.01, f"closed form {closed} vs MC {mc} at density {h}"


# --------------------------------------------------------------------------
# N0 closed form: hypergeometric E and Var
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hypergeom_mc() -> tuple[np.ndarray, float, float, np.ndarray]:
    """20k Monte-Carlo draws of ``sum_j TP_j`` against the closed form."""
    rng = np.random.default_rng(7)
    n, k = _margins(rng, n_sent=150, len_lo=14, len_hi=60)
    mat = _n0_draw(rng, n, k, 0.35)
    e_tp, var_tp = n0_expectation(mat)
    m = mat[:, 0] + mat[:, 1]
    draws = rng.hypergeometric(k, n - k, m, size=(20_000, n.size)).sum(axis=1)
    return draws, e_tp, var_tp, mat


def test_n0_expectation_matches_monte_carlo_mean(hypergeom_mc):
    """``E[TP] = sum_j m_j K_j / N_j``, to within the MC standard error."""
    draws, e_tp, var_tp, _ = hypergeom_mc
    se = np.sqrt(var_tp / draws.size)
    assert abs(draws.mean() - e_tp) < 4.0 * se, (
        f"E[TP] closed form {e_tp:.3f} vs MC {draws.mean():.3f} (MC se {se:.3f})"
    )


def test_n0_expectation_matches_monte_carlo_variance(hypergeom_mc):
    """``Var[TP] = sum_j m (K/N)((N-K)/N)((N-m)/(N-1))``, to within MC error."""
    draws, _, var_tp, _ = hypergeom_mc
    mc_var = float(draws.var(ddof=1))
    assert abs(mc_var - var_tp) / var_tp < 0.05, f"Var[TP] closed form {var_tp} vs MC {mc_var}"


def test_n0_sd_is_the_root_of_the_variance(hypergeom_mc):
    _, _, var_tp, mat = hypergeom_mc
    assert n0_null(mat).sd_tp == pytest.approx(var_tp**0.5)


def test_n0_z_for_scales_in_null_standard_deviations(hypergeom_mc):
    _, e_tp, _, mat = hypergeom_mc
    null = n0_null(mat)
    assert null.z_for(round(e_tp)) == pytest.approx(0.0, abs=0.05)
    assert null.z_for(round(e_tp + 2 * null.sd_tp)) == pytest.approx(2.0, abs=0.05)


def test_n0_z_for_is_nan_without_variance():
    # One sentence, everything predicted: TP is deterministic, so z is undefined.
    null = n0_null([(4, 0, 0, 0)])
    assert np.isnan(null.z_for(4))


# --------------------------------------------------------------------------
# The density confound, made explicit
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def n0_metric_sweep() -> tuple[list[tuple[float, float, float]], float]:
    rng = np.random.default_rng(4242)
    n, k = _margins(rng, n_sent=1500)
    rows = []
    for h in DENSITY_SWEEP:
        null = n0_null(_n0_draw(rng, n, k, h))
        rows.append((h, null.e_recall, null.e_precision))
    delta_g = float(k.sum() / n.sum())
    return rows, delta_g


def test_n0_expected_recall_tracks_density_linearly(n0_metric_sweep):
    """``E[recall] = h``: recall IS fertility under the null, which is the whole
    reason boundary-F1 cannot be the headline metric."""
    rows, _ = n0_metric_sweep
    for h, e_recall, _ in rows:
        assert e_recall == pytest.approx(h, abs=0.01), f"E[recall]={e_recall} at density {h}"


def test_n0_expected_recall_has_unit_slope_in_density(n0_metric_sweep):
    rows, _ = n0_metric_sweep
    slope, intercept = np.polyfit([h for h, _, _ in rows], [r for _, r, _ in rows], 1)
    assert slope == pytest.approx(1.0, abs=0.02)
    assert intercept == pytest.approx(0.0, abs=0.01)


def test_n0_expected_precision_is_flat_across_densities(n0_metric_sweep):
    """``E[precision] = delta_g`` at every density -- no fertility signal at all."""
    rows, _ = n0_metric_sweep
    precisions = [p for _, _, p in rows]
    spread = max(precisions) - min(precisions)
    assert spread < 0.005, f"E[precision] moved by {spread} across the sweep: {precisions}"


def test_n0_expected_precision_equals_gold_density(n0_metric_sweep):
    rows, delta_g = n0_metric_sweep
    for h, _, e_precision in rows:
        assert e_precision == pytest.approx(delta_g, abs=0.005), f"at density {h}"


def test_n0_recall_and_precision_disagree_about_density(n0_metric_sweep):
    """The confound in one assertion: recall spans the sweep, precision does not."""
    rows, _ = n0_metric_sweep
    recalls = [r for _, r, _ in rows]
    precisions = [p for _, _, p in rows]
    assert (max(recalls) - min(recalls)) > 0.85
    assert (max(precisions) - min(precisions)) < 0.005


# --------------------------------------------------------------------------
# N0 plumbing
# --------------------------------------------------------------------------


def test_n0_expectation_of_a_single_sentence_is_the_textbook_formula():
    # N=10, K=4, m=5.
    e_tp, var_tp = n0_expectation([(2, 3, 2, 3)])
    assert e_tp == pytest.approx(5 * 4 / 10)
    assert var_tp == pytest.approx(5 * (4 / 10) * (6 / 10) * (5 / 9))


def test_n0_expectation_sums_over_independent_sentences():
    one = n0_expectation([(2, 3, 2, 3)])
    two = n0_expectation([(2, 3, 2, 3), (2, 3, 2, 3)])
    assert two[0] == pytest.approx(2 * one[0])
    assert two[1] == pytest.approx(2 * one[1])


def test_n0_expectation_ignores_sentences_with_one_position():
    """With a single scored position there is no choice to make."""
    assert n0_expectation([(1, 0, 0, 0)]) == (0.0, 0.0)
    with_stub = n0_expectation([(2, 3, 2, 3), (1, 0, 0, 0)])
    assert with_stub == n0_expectation([(2, 3, 2, 3)])


def test_n0_expectation_of_an_empty_corpus_is_zero():
    assert n0_expectation(np.zeros((0, 4), dtype=np.int64)) == (0.0, 0.0)


def test_n0_micro_precision_equals_e_tp_over_predictions():
    mat = np.array([[2, 3, 2, 3], [4, 6, 5, 15], [1, 1, 3, 5]], dtype=np.int64)
    e_tp, _ = n0_expectation(mat)
    n_pred = float((mat[:, 0] + mat[:, 1]).sum())
    assert n0_micro_precision(mat) == pytest.approx(e_tp / n_pred)


def test_n0_micro_precision_with_no_predictions_is_one():
    """The frozen no-predictions convention, mirrored from `core.precision`."""
    assert n0_micro_precision([(0, 0, 4, 6)]) == 1.0


def test_n0_micro_precision_of_a_short_only_corpus_is_one():
    assert n0_micro_precision([(1, 0, 0, 0)]) == 1.0


def test_n0_null_reports_the_observed_margins():
    mat = np.array([[2, 3, 2, 3], [4, 6, 5, 15]], dtype=np.int64)
    null = n0_null(mat)
    assert null.n_positions == int(mat.sum())
    assert null.n_gold == int((mat[:, 0] + mat[:, 2]).sum())
    assert null.n_pred == int((mat[:, 0] + mat[:, 1]).sum())


def test_n0_null_is_blind_to_whether_the_tokenizer_was_right():
    """N0 depends on the MARGINS only, so a perfect and a hopeless tokenizer with
    the same fertility get the same null -- which is what makes it a null."""
    perfect = np.array([[10, 0, 0, 20], [8, 0, 0, 22]], dtype=np.int64)
    hopeless = np.array([[0, 10, 10, 10], [0, 8, 8, 14]], dtype=np.int64)
    assert n0_null(perfect).e_tp == pytest.approx(n0_null(hopeless).e_tp)
    assert abs(n0_null(perfect).e_phi) < BLOCKER_TOL
    assert phi(pooled(perfect)) == pytest.approx(1.0)


def test_n0_null_e_f1_is_the_harmonic_mean_of_its_own_terms():
    null = n0_null(np.array([[2, 3, 2, 3], [4, 6, 5, 15]], dtype=np.int64))
    expect = 2 * null.e_precision * null.e_recall / (null.e_precision + null.e_recall)
    assert null.e_f1 == pytest.approx(expect)


# --------------------------------------------------------------------------
# BLOCKER #2 -- N1 at every density
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def n1_sweep() -> list[tuple[float, NullSummary]]:
    rng = np.random.default_rng(31337)
    gold, masks, lengths = _null_corpus(rng)
    return [
        (
            h,
            n1_null(
                gold, masks, lengths, bimodal_length_dist(h), statistic=phi, R=N1_REPS, seed=17
            ),
        )
        for h in DENSITY_SWEEP
    ]


@pytest.mark.slow
def test_n1_scores_zero_phi_at_every_density(n1_sweep):
    """BLOCKER #2 for the length-matched null -- the one that actually binds.

    N1 matches the tokenizer on fertility, on length SHAPE and on legality, and
    the length shape here is bimodal on purpose. If chance correction leaked, a
    null with realistic token lengths is where it would show.
    """
    rows = [(h, s.mean) for h, s in n1_sweep]
    worst_h, worst = max(rows, key=lambda r: abs(r[1]))
    detail = "\n".join(
        f"    density: target={h:.2f} achieved={s.density:.4f}   "
        f"phi={s.mean:+.6f}  sd_over_reps={s.sd:.6f}  worst_rep={np.abs(s.values).max():+.6f}"
        for h, s in n1_sweep
    )
    assert abs(worst) < BLOCKER_TOL, (
        "PREREGISTRATION.md S4 blocker #2 FAILED for N1: a length-matched null "
        "scores non-zero phi somewhere on the density sweep, so the chance "
        "correction does not work empirically and the pre-committed consequence "
        "is to report lift over N1 rather than raw phi_B.\n"
        f"  worst |phi| = {abs(worst):.6f} at density {worst_h:.2f} "
        f"(threshold {BLOCKER_TOL}, {N1_REPS} replicates per density)\n" + detail
    )


@pytest.mark.slow
def test_n1_achieved_density_tracks_the_target(n1_sweep):
    """The sweep really did visit 0.05..0.95, so the blocker covers that range."""
    for h, summary in n1_sweep:
        assert summary.density == pytest.approx(h, abs=0.02), (
            f"asked for density {h}, N1 achieved {summary.density}"
        )


@pytest.mark.slow
def test_n1_replicate_spread_is_non_degenerate(n1_sweep):
    """Guard against a vacuous pass: the replicates must actually vary."""
    for h, summary in n1_sweep:
        assert summary.sd > 0.0, f"N1 produced identical replicates at density {h}"
        assert summary.n_reps == N1_REPS
        assert summary.values.shape == (N1_REPS,)


@pytest.mark.slow
def test_n1_scores_zero_phi_with_illegal_positions_masked_out():
    """Snapping to a holey mask must not manufacture alignment either."""
    rng = np.random.default_rng(99)
    gold, masks, lengths = _null_corpus(rng, n_sent=250, mask_step=7)
    rows = []
    for h in (0.05, 0.35, 0.65, 0.95):
        summary = n1_null(gold, masks, lengths, bimodal_length_dist(h), statistic=phi, R=24, seed=5)
        rows.append((h, summary.mean))
    worst_h, worst = max(rows, key=lambda r: abs(r[1]))
    assert abs(worst) < BLOCKER_TOL, (
        f"N1 on a masked corpus reached |phi| = {abs(worst):.6f} at density {worst_h:.2f}\n"
        + _table(rows)
    )


def test_the_sweep_distribution_is_bimodal_and_not_geometric():
    """Documents that the N1 sweep is a real test and not a soft one.

    A geometric pmf with the same mean puts ``1/mean`` on length 1; ours puts
    about half its mass there and the rest on a mode an order of magnitude
    further out, which is the shape a real BPE vocabulary has and a memoryless
    null does not.
    """
    for h in DENSITY_SWEEP[:5]:
        dist = bimodal_length_dist(h)
        values, probs = empirical_length_dist(dist)
        assert values.size == 2, f"not two modes at density {h}"
        assert probs.min() > 0.15, f"second mode is vestigial at density {h}: {probs}"
        assert values[1] >= 2 * values[0]
        geometric_p1 = h  # geometric with mean 1/h
        assert probs[0] > 2.0 * geometric_p1 or h > 0.3


def test_the_sweep_distribution_has_the_requested_mean():
    for h in DENSITY_SWEEP:
        values, probs = empirical_length_dist(bimodal_length_dist(h))
        assert float((values * probs).sum()) == pytest.approx(1.0 / h, rel=1e-9)


# --------------------------------------------------------------------------
# empirical_length_dist
# --------------------------------------------------------------------------


def test_empirical_length_dist_from_observed_lengths():
    values, probs = empirical_length_dist([1, 1, 2, 3, 3, 3])
    assert values.tolist() == [1, 2, 3]
    assert probs.tolist() == pytest.approx([2 / 6, 1 / 6, 3 / 6])


def test_empirical_length_dist_from_a_mapping():
    values, probs = empirical_length_dist({3: 1.0, 1: 3.0})
    assert values.tolist() == [1, 3]
    assert probs.tolist() == pytest.approx([0.75, 0.25])


def test_empirical_length_dist_from_a_pmf_pair():
    values, probs = empirical_length_dist((np.array([2, 5]), np.array([1.0, 3.0])))
    assert values.tolist() == [2, 5]
    assert probs.tolist() == pytest.approx([0.25, 0.75])


def test_empirical_length_dist_probabilities_sum_to_one():
    _, probs = empirical_length_dist([1, 2, 2, 7, 9, 9, 9])
    assert probs.sum() == pytest.approx(1.0)


def test_empirical_length_dist_drops_non_positive_lengths():
    values, _ = empirical_length_dist([0, -3, 2, 4])
    assert values.tolist() == [2, 4]


def test_empirical_length_dist_drops_zero_weight_entries():
    values, _ = empirical_length_dist({1: 0.0, 4: 2.0})
    assert values.tolist() == [4]


def test_empirical_length_dist_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        empirical_length_dist({1: -1.0, 2: 2.0})


def test_empirical_length_dist_rejects_an_empty_distribution():
    with pytest.raises(ValueError, match="empty"):
        empirical_length_dist([0, -1])


# --------------------------------------------------------------------------
# snap_to_mask
# --------------------------------------------------------------------------


def test_snap_to_mask_takes_the_nearest_legal_position():
    mask = np.array([2, 6, 11], dtype=np.int64)
    assert snap_to_mask(np.array([3, 7, 10]), mask).tolist() == [2, 6, 11]


def test_snap_to_mask_breaks_ties_to_the_left():
    mask = np.array([2, 6], dtype=np.int64)
    assert snap_to_mask(np.array([4]), mask).tolist() == [2]


def test_snap_to_mask_deduplicates_collisions():
    """Two offsets can snap together; the achieved density drops accordingly."""
    mask = np.array([5, 50], dtype=np.int64)
    assert snap_to_mask(np.array([4, 5, 6]), mask).tolist() == [5]


def test_snap_to_mask_clamps_beyond_both_ends():
    mask = np.array([4, 9], dtype=np.int64)
    assert snap_to_mask(np.array([1, 99]), mask).tolist() == [4, 9]


def test_snap_to_mask_returns_sorted_output():
    mask = np.array([1, 5, 9, 13], dtype=np.int64)
    assert snap_to_mask(np.array([12, 2, 8]), mask).tolist() == [1, 9, 13]


def test_snap_to_mask_with_an_empty_mask_is_empty():
    assert snap_to_mask(np.array([1, 2]), np.empty(0, dtype=np.int64)).size == 0


def test_snap_to_mask_with_no_offsets_is_empty():
    assert snap_to_mask(np.empty(0), np.array([1, 2], dtype=np.int64)).size == 0


# --------------------------------------------------------------------------
# n1_sample / n1_counts
# --------------------------------------------------------------------------


def test_n1_sample_shape_and_legality():
    rng = np.random.default_rng(0)
    _, masks, lengths = _null_corpus(rng, n_sent=8, len_lo=40, len_hi=60)
    reps = n1_sample(lengths, masks, {1: 0.5, 6: 0.5}, R=5, seed=1)
    assert len(reps) == 5
    for rep in reps:
        assert len(rep) == len(masks)
        for pred, mask, n in zip(rep, masks, lengths, strict=True):
            assert pred <= mask
            assert 0 not in pred
            assert all(0 < b < n for b in pred)


def test_n1_sample_is_deterministic_under_a_seed():
    rng = np.random.default_rng(0)
    _, masks, lengths = _null_corpus(rng, n_sent=6, len_lo=40, len_hi=60)
    a = n1_sample(lengths, masks, {1: 0.5, 6: 0.5}, R=3, seed=11)
    b = n1_sample(lengths, masks, {1: 0.5, 6: 0.5}, R=3, seed=11)
    c = n1_sample(lengths, masks, {1: 0.5, 6: 0.5}, R=3, seed=12)
    assert a == b
    assert a != c


def test_n1_sample_density_follows_the_length_distribution():
    """Short tokens -> many boundaries; long tokens -> few. Fertility matching
    is the property the whole null rests on."""
    rng = np.random.default_rng(3)
    _, masks, lengths = _null_corpus(rng, n_sent=40, len_lo=100, len_hi=140)
    fine = n1_sample(lengths, masks, {2: 1.0}, R=4, seed=2)
    coarse = n1_sample(lengths, masks, {20: 1.0}, R=4, seed=2)
    n_fine = sum(len(p) for rep in fine for p in rep)
    n_coarse = sum(len(p) for rep in coarse for p in rep)
    assert n_fine > 5 * n_coarse


def test_n1_sample_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="masks"):
        n1_sample([10, 10], [frozenset({1, 2})], {2: 1.0}, R=1, seed=0)


def test_n1_sample_handles_sentences_with_no_legal_positions():
    reps = n1_sample([10, 1], [frozenset(), frozenset()], {2: 1.0}, R=2, seed=0)
    assert reps == [(frozenset(), frozenset()), (frozenset(), frozenset())]


def test_n1_counts_shape_and_row_totals():
    rng = np.random.default_rng(5)
    gold, masks, lengths = _null_corpus(rng, n_sent=7, len_lo=30, len_hi=50)
    reps = n1_sample(lengths, masks, {1: 0.5, 5: 0.5}, R=4, seed=1)
    counts = n1_counts(gold, masks, reps)
    assert counts.shape == (4, 7, 4)
    for j, mask in enumerate(masks):
        assert (counts[:, j, :].sum(axis=1) == len(mask)).all()


def test_n1_counts_restricts_gold_to_the_mask():
    gold = [frozenset({1, 2, 3})]
    masks = [frozenset({1, 2})]
    reps = [[frozenset({1})]]
    tp, fp, fn, tn = n1_counts(gold, masks, reps)[0, 0].tolist()
    assert (tp, fp, fn, tn) == (1, 0, 1, 0)


def test_n1_counts_ignores_predictions_outside_the_mask():
    gold = [frozenset({1})]
    masks = [frozenset({1, 2})]
    reps = [[frozenset({1, 99})]]
    tp, fp, fn, tn = n1_counts(gold, masks, reps)[0, 0].tolist()
    assert (tp, fp, fn, tn) == (1, 0, 0, 1)


def test_n1_null_summary_fields_are_consistent():
    rng = np.random.default_rng(8)
    gold, masks, lengths = _null_corpus(rng, n_sent=30, len_lo=60, len_hi=90)
    summary = n1_null(gold, masks, lengths, {1: 0.4, 6: 0.6}, statistic=phi, R=16, seed=4)
    assert summary.mean == pytest.approx(float(summary.values.mean()))
    assert summary.ci == (summary.p025, summary.p975)
    assert summary.p025 <= summary.mean <= summary.p975
    assert 0.0 < summary.density < 1.0


def test_n1_null_with_one_replicate_has_zero_sd():
    rng = np.random.default_rng(8)
    gold, masks, lengths = _null_corpus(rng, n_sent=10, len_lo=40, len_hi=60)
    summary = n1_null(gold, masks, lengths, {3: 1.0}, statistic=phi, R=1, seed=0)
    assert summary.sd == 0.0
    assert summary.n_reps == 1


def test_n1_null_is_reproducible_under_a_seed():
    rng = np.random.default_rng(8)
    gold, masks, lengths = _null_corpus(rng, n_sent=12, len_lo=40, len_hi=60)
    kwargs = {"statistic": phi, "R": 8}
    a = n1_null(gold, masks, lengths, {1: 0.5, 5: 0.5}, seed=3, **kwargs)
    b = n1_null(gold, masks, lengths, {1: 0.5, 5: 0.5}, seed=3, **kwargs)
    assert a.values.tolist() == b.values.tolist()


# --------------------------------------------------------------------------
# lift
# --------------------------------------------------------------------------


def test_lift_is_a_difference_not_a_ratio():
    assert lift(0.42, 0.01) == pytest.approx(0.41)
    assert lift(0.42, 0.0) == pytest.approx(0.42)  # a ratio would have exploded
    assert lift(0.1, 0.3) == pytest.approx(-0.2)


def test_lift_ci_labels_its_arms_and_recovers_the_lift():
    rng = np.random.default_rng(12)
    n, k = _margins(rng, n_sent=120, len_lo=20, len_hi=40)
    null_mat = _n0_draw(rng, n, k, 0.35)
    # An observed tokenizer that genuinely beats the null: keep its margins but
    # move most of the misses onto gold positions.
    obs = null_mat.copy()
    gain = (obs[:, 2] * 0.8).astype(np.int64)
    obs[:, 0] += gain
    obs[:, 1] -= gain
    obs[:, 2] -= gain
    obs[:, 3] += gain
    result = lift_ci(obs, null_mat, B=500, seed=0)
    assert (result.name_a, result.name_b) == ("observed", "null")
    assert result.diff == pytest.approx(phi(pooled(obs)) - phi(pooled(null_mat)))
    assert result.lo > 0.0 and result.significant


def test_lift_ci_of_a_null_against_itself_is_zero():
    rng = np.random.default_rng(13)
    n, k = _margins(rng, n_sent=80, len_lo=20, len_hi=40)
    mat = _n0_draw(rng, n, k, 0.35)
    result = lift_ci(mat, mat, B=200, seed=0)
    assert result.diff == 0.0
    assert not result.significant

# Refined

# Updated
