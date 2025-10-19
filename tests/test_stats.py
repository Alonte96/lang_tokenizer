"""Uncertainty, multiplicity and rank stability -- the inference layer.

Four claims are load-bearing here and each gets a test that could actually fail:

1. The cluster bootstrap has its nominal coverage. Asserted by simulation
   against a KNOWN true phi rather than by inspecting interval widths.
2. `paired_bootstrap` really does resample both arms on identical sentence
   indices. Asserted as an exact identity against two marginal bootstraps at the
   same seed, and then in the form that matters: a case where the two marginal
   intervals OVERLAP -- which a naive reader would call "no difference" -- while
   the paired interval is decisive.
3. `bh_fdr` is BH, not something that merely looks like it. Asserted against a
   hand-computed six-p-value example including the step-up rescue, plus an FDR
   control simulation.
4. `split_half_noise_floor` measures noise. Asserted in BOTH directions: tau ~ 1
   on a genuinely separated leaderboard, tau ~ 0 on tokenizers that differ by
   nothing but sampling.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kendalltau

from unsegbench.metrics.core import Counts, f1, phi
from unsegbench.metrics.stats import (
    as_counts_matrix,
    bh_fdr,
    bootstrap,
    design_effect,
    kendall_tau_b,
    leaderboard,
    max_t_band,
    paired_bootstrap,
    pooled,
    rank_intervals,
    ranking,
    split_half_noise_floor,
)

# --------------------------------------------------------------------------
# Generators with a KNOWN population value
# --------------------------------------------------------------------------


def cell_probs(p_tp: float, p_fp: float, p_fn: float) -> np.ndarray:
    return np.array([p_tp, p_fp, p_fn, 1.0 - p_tp - p_fp - p_fn], dtype=float)


def population_phi(probs: np.ndarray) -> float:
    """The phi the pooled plug-in converges to under these cell probabilities."""
    scaled = np.rint(probs * 1e9).astype(np.int64)
    return phi(Counts(*(int(x) for x in scaled)))


def iid_corpus(
    rng: np.random.Generator, probs: np.ndarray, n_sent: int = 60, n_pos: int = 20
) -> np.ndarray:
    """Sentences of i.i.d. positions: no within-sentence dependence at all."""
    return rng.multinomial(n_pos, probs, size=n_sent).astype(np.int64)


def heterogeneous_corpus(rng: np.random.Generator, n_sent: int = 200) -> np.ndarray:
    """Sentences whose quality varies enormously, so the MARGINAL SE is large."""
    rows = []
    for _ in range(n_sent):
        n = int(rng.integers(20, 60))
        n_gold = max(2, round(n * rng.uniform(0.15, 0.55)))
        quality = float(rng.uniform(0.05, 0.98))
        tp = round(n_gold * quality)
        n_pred = min(n, max(tp, round(n_gold * rng.uniform(0.7, 1.4))))
        fp, fn = n_pred - tp, n_gold - tp
        rows.append((tp, fp, fn, n - tp - fp - fn))
    return np.array(rows, dtype=np.int64)


def degraded(mat: np.ndarray) -> np.ndarray:
    """The SAME corpus with one true positive turned into a false negative in
    every sentence that has one.

    A small, systematic, sentence-by-sentence effect on top of huge
    between-sentence variance: precisely the regime the paired bootstrap exists
    for, and precisely the regime in which comparing marginal error bars fails.
    """
    lost = (mat[:, 0] > 0).astype(np.int64)
    return np.column_stack([mat[:, 0] - lost, mat[:, 1] + lost, mat[:, 2] + lost, mat[:, 3] - lost])


def tokenizer_of_quality(
    rng: np.random.Generator, quality: float, n_sent: int = 300, n_pos: int = 30, n_gold: int = 10
) -> np.ndarray:
    """A tokenizer that recovers each gold boundary with probability ``quality``."""
    tp = rng.binomial(n_gold, quality, size=n_sent)
    n_pred = np.full(n_sent, n_gold, dtype=np.int64)
    return np.column_stack([tp, n_pred - tp, n_gold - tp, n_pos - n_pred - n_gold + tp]).astype(
        np.int64
    )


# --------------------------------------------------------------------------
# bootstrap: mechanics
# --------------------------------------------------------------------------


def test_bootstrap_estimate_is_the_pooled_plug_in():
    mat = heterogeneous_corpus(np.random.default_rng(0), n_sent=40)
    assert bootstrap(mat, B=200, seed=0).estimate == pytest.approx(phi(pooled(mat)))


def test_bootstrap_is_reproducible_under_a_fixed_seed():
    mat = heterogeneous_corpus(np.random.default_rng(1), n_sent=40)
    a = bootstrap(mat, B=500, seed=42)
    b = bootstrap(mat, B=500, seed=42)
    assert a.draws.tolist() == b.draws.tolist()
    assert (a.lo, a.hi, a.se) == (b.lo, b.hi, b.se)


def test_bootstrap_with_a_different_seed_gives_different_draws():
    mat = heterogeneous_corpus(np.random.default_rng(1), n_sent=40)
    a = bootstrap(mat, B=500, seed=42)
    b = bootstrap(mat, B=500, seed=43)
    assert a.draws.tolist() != b.draws.tolist()
    assert a.estimate == b.estimate


def test_bootstrap_interval_is_ordered_and_brackets_the_estimate():
    mat = heterogeneous_corpus(np.random.default_rng(2), n_sent=80)
    ci = bootstrap(mat, B=1000, seed=0)
    assert ci.lo < ci.estimate < ci.hi
    assert ci.width == pytest.approx(ci.hi - ci.lo)
    assert ci.se > 0.0


def test_bootstrap_narrows_as_sentences_accumulate():
    rng = np.random.default_rng(3)
    small = bootstrap(heterogeneous_corpus(rng, n_sent=50), B=1000, seed=0)
    large = bootstrap(heterogeneous_corpus(rng, n_sent=800), B=1000, seed=0)
    assert large.width < 0.6 * small.width


def test_bootstrap_records_the_number_of_clusters_not_positions():
    mat = heterogeneous_corpus(np.random.default_rng(4), n_sent=37)
    assert bootstrap(mat, B=100, seed=0).n_clusters == 37


def test_bootstrap_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        bootstrap([(1, 1, 1, 1)], B=10, method="jackknife")


def test_bootstrap_percentile_and_bca_are_both_available():
    mat = heterogeneous_corpus(np.random.default_rng(5), n_sent=60)
    pct = bootstrap(mat, B=1000, seed=0, method="percentile")
    bca = bootstrap(mat, B=1000, seed=0, method="bca")
    assert pct.method == "percentile" and bca.method == "bca"
    assert pct.draws.tolist() == bca.draws.tolist()  # same indices, different endpoints
    assert (pct.lo, pct.hi) != (bca.lo, bca.hi)


def test_bootstrap_accepts_counts_objects_and_tuples_alike():
    rows = [(3, 1, 2, 10), (5, 0, 1, 9)]
    from_tuples = bootstrap(rows, B=200, seed=0)
    from_counts = bootstrap([Counts(*r) for r in rows], B=200, seed=0)
    assert from_tuples.estimate == from_counts.estimate


def test_bootstrap_works_for_any_statistic():
    mat = heterogeneous_corpus(np.random.default_rng(6), n_sent=40)
    ci = bootstrap(mat, statistic=f1, B=500, seed=0)
    assert ci.estimate == pytest.approx(f1(pooled(mat)))
    assert 0.0 <= ci.lo <= ci.hi <= 1.0


def test_bootstrap_ci_covers_the_true_phi_at_the_nominal_rate():
    """THE coverage test: 200 fresh corpora with a known population phi."""
    probs = cell_probs(0.18, 0.10, 0.12)
    truth = population_phi(probs)
    rng = np.random.default_rng(11)
    covered = 0
    trials = 200
    for t in range(trials):
        ci = bootstrap(iid_corpus(rng, probs), B=1000, seed=t)
        covered += ci.lo <= truth <= ci.hi
    rate = covered / trials
    assert 0.90 <= rate <= 0.99, f"95% BCa interval covered phi={truth:.4f} {rate:.1%} of the time"


def test_percentile_bootstrap_also_reaches_nominal_coverage():
    probs = cell_probs(0.18, 0.10, 0.12)
    truth = population_phi(probs)
    rng = np.random.default_rng(11)
    covered = 0
    trials = 200
    for t in range(trials):
        ci = bootstrap(iid_corpus(rng, probs), B=1000, seed=t, method="percentile")
        covered += ci.lo <= truth <= ci.hi
    rate = covered / trials
    assert 0.90 <= rate <= 0.99, f"95% percentile interval covered {rate:.1%} of the time"


# --------------------------------------------------------------------------
# paired_bootstrap: the shared resample indices
# --------------------------------------------------------------------------


def test_paired_bootstrap_uses_identical_resample_indices_for_both_arms():
    """The exact identity: paired draws == arm-A draws minus arm-B draws.

    Two marginal bootstraps at the same seed see the same sentence indices, so
    if the paired difference draws did NOT match this element-wise, the two arms
    would have been resampled independently and the pairing would be a fiction.
    """
    rng = np.random.default_rng(7)
    a = heterogeneous_corpus(rng, n_sent=150)
    b = degraded(a)
    marginal_a = bootstrap(a, B=800, seed=7)
    marginal_b = bootstrap(b, B=800, seed=7)
    paired = paired_bootstrap(a, b, B=800, seed=7)
    assert np.allclose(paired.draws, marginal_a.draws - marginal_b.draws)


def test_paired_bootstrap_se_is_far_below_variances_add():
    """Correlated arms: the paired SE must be a fraction of sqrt(se_a^2+se_b^2)."""
    rng = np.random.default_rng(5)
    a = heterogeneous_corpus(rng, n_sent=200)
    b = degraded(a)
    ca, cb = bootstrap(a, B=2000, seed=7), bootstrap(b, B=2000, seed=7)
    paired = paired_bootstrap(a, b, B=2000, seed=7)
    unpaired_se = float(np.hypot(ca.se, cb.se))
    assert paired.se < 0.25 * unpaired_se, (
        f"paired se {paired.se:.5f} vs variances-add {unpaired_se:.5f}"
    )


def test_paired_bootstrap_is_decisive_where_the_marginal_cis_overlap():
    """The case a naive analysis gets wrong.

    The two marginal 95% intervals overlap, which reads as "not distinguishable"
    to anyone comparing error bars by eye. The paired interval on the difference
    excludes zero by a mile, because the sentence-level variation the marginal
    bars are dominated by cancels exactly.
    """
    rng = np.random.default_rng(5)
    a = heterogeneous_corpus(rng, n_sent=200)
    b = degraded(a)
    ca, cb = bootstrap(a, B=2000, seed=7), bootstrap(b, B=2000, seed=7)
    paired = paired_bootstrap(a, b, B=2000, seed=7)
    assert ca.lo <= cb.hi and cb.lo <= ca.hi, "marginal intervals must overlap for this test"
    assert paired.significant
    assert paired.lo > 0.0
    assert paired.p_value < 0.01


def test_paired_bootstrap_diff_matches_the_two_pooled_estimates():
    rng = np.random.default_rng(8)
    a = heterogeneous_corpus(rng, n_sent=60)
    b = degraded(a)
    paired = paired_bootstrap(a, b, B=300, seed=0)
    assert paired.estimate_a == pytest.approx(phi(pooled(a)))
    assert paired.estimate_b == pytest.approx(phi(pooled(b)))
    assert paired.diff == pytest.approx(paired.estimate_a - paired.estimate_b)
    assert paired.point == paired.diff
    assert paired.pvalue == paired.p_value


def test_paired_bootstrap_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(9)
    a = heterogeneous_corpus(rng, n_sent=50)
    b = degraded(a)
    first = paired_bootstrap(a, b, B=500, seed=3)
    second = paired_bootstrap(a, b, B=500, seed=3)
    other = paired_bootstrap(a, b, B=500, seed=4)
    assert first.draws.tolist() == second.draws.tolist()
    assert (first.lo, first.hi, first.p_value) == (second.lo, second.hi, second.p_value)
    assert first.draws.tolist() != other.draws.tolist()


def test_paired_bootstrap_of_a_tokenizer_against_itself_is_exactly_zero():
    mat = heterogeneous_corpus(np.random.default_rng(10), n_sent=50)
    paired = paired_bootstrap(mat, mat, B=300, seed=0)
    assert paired.diff == 0.0
    assert not np.any(paired.draws)
    assert not paired.significant


def test_paired_bootstrap_carries_its_arm_labels():
    mat = heterogeneous_corpus(np.random.default_rng(11), n_sent=30)
    paired = paired_bootstrap(mat, mat, B=100, seed=0, name_a="bert", name_b="gpt")
    assert (paired.name_a, paired.name_b) == ("bert", "gpt")


def test_paired_bootstrap_requires_matched_sentences():
    a = heterogeneous_corpus(np.random.default_rng(12), n_sent=20)
    b = heterogeneous_corpus(np.random.default_rng(13), n_sent=21)
    with pytest.raises(ValueError, match="matched sentences"):
        paired_bootstrap(a, b, B=10)


def test_paired_bootstrap_rejects_an_unknown_method():
    mat = heterogeneous_corpus(np.random.default_rng(14), n_sent=20)
    with pytest.raises(ValueError, match="unknown method"):
        paired_bootstrap(mat, mat, B=10, method="studentised")


def test_paired_bootstrap_ci_covers_the_true_difference():
    probs_a = cell_probs(0.18, 0.10, 0.12)
    probs_b = cell_probs(0.14, 0.14, 0.16)
    truth = population_phi(probs_a) - population_phi(probs_b)
    rng = np.random.default_rng(21)
    covered = 0
    trials = 200
    for t in range(trials):
        paired = paired_bootstrap(
            iid_corpus(rng, probs_a, n_sent=80), iid_corpus(rng, probs_b, n_sent=80), B=1000, seed=t
        )
        covered += paired.lo <= truth <= paired.hi
    rate = covered / trials
    assert 0.90 <= rate <= 0.99, f"paired interval covered diff={truth:.4f} {rate:.1%} of the time"


# --------------------------------------------------------------------------
# rank_intervals
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def separated_leaderboard() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    return {f"tok{i}": tokenizer_of_quality(rng, q) for i, q in enumerate([0.9, 0.75, 0.6, 0.45])}


@pytest.fixture(scope="module")
def tied_leaderboard() -> dict[str, np.ndarray]:
    """Five tokenizers that are permutations of one another over the sentences.

    Their pooled tables are IDENTICAL, so the leaderboard is exactly tied and
    every apparent ordering on a subset of the sentences is sampling noise. That
    is the honest hard case for both `rank_intervals` and the noise floor.
    """
    rng = np.random.default_rng(4)
    base = tokenizer_of_quality(rng, 0.6)
    return {f"tok{i}": base[rng.permutation(base.shape[0])] for i in range(5)}


@pytest.fixture(scope="module")
def noise_leaderboard() -> dict[str, np.ndarray]:
    """Six tokenizers drawn INDEPENDENTLY from identical parameters.

    They differ only by sampling, so whatever ordering the data shows is noise
    and the split-half tau between two disjoint halves has expectation zero.
    """
    rng = np.random.default_rng(9)
    return {f"tok{i}": tokenizer_of_quality(rng, 0.6) for i in range(6)}


def test_rank_intervals_are_ordered_and_in_range(separated_leaderboard):
    result = rank_intervals(separated_leaderboard, B=800, seed=0)
    n_tok = len(separated_leaderboard)
    for name, interval in result.items():
        assert interval.name == name
        assert 1 <= interval.lo <= interval.hi <= n_tok
        assert interval.lo <= interval.point <= interval.hi
        assert 0.0 <= interval.p_best <= 1.0
        assert interval.lo <= interval.median <= interval.hi


def test_rank_intervals_cover_every_input_in_order(separated_leaderboard):
    result = rank_intervals(separated_leaderboard, B=400, seed=0)
    assert list(result) == list(separated_leaderboard)


def test_a_separated_ordering_gives_tight_intervals(separated_leaderboard):
    result = rank_intervals(separated_leaderboard, B=800, seed=0)
    for i, name in enumerate(separated_leaderboard, start=1):
        interval = result[name]
        assert (interval.lo, interval.hi) == (i, i), f"{name} should be pinned to rank {i}"
    assert result["tok0"].p_best == 1.0
    assert str(result["tok0"]) == "Rank 1"


def test_a_tied_ordering_gives_wide_overlapping_intervals(tied_leaderboard):
    """The honest presentation of a leaderboard that is not distinguishable."""
    result = rank_intervals(tied_leaderboard, B=800, seed=0)
    widths = [interval.hi - interval.lo for interval in result.values()]
    assert min(widths) >= 2, f"tied tokenizers got suspiciously tight ranks: {widths}"
    # Every interval overlaps every other one: no claim survives.
    intervals = list(result.values())
    for a in intervals:
        for b in intervals:
            assert a.lo <= b.hi and b.lo <= a.hi
    assert max(i.p_best for i in result.values()) < 0.95
    assert "-" in str(intervals[0])


def test_tied_intervals_are_wider_than_separated_ones(separated_leaderboard, tied_leaderboard):
    sep = rank_intervals(separated_leaderboard, B=800, seed=0)
    tied = rank_intervals(tied_leaderboard, B=800, seed=0)
    assert max(i.hi - i.lo for i in sep.values()) < min(i.hi - i.lo for i in tied.values())


def test_rank_intervals_point_ranks_agree_with_the_ranking(separated_leaderboard):
    result = rank_intervals(separated_leaderboard, B=400, seed=0)
    order = ranking(separated_leaderboard)
    assert [result[n].point for n in order] == sorted(result[n].point for n in order)
    assert result[order[0]].point == 1


def test_rank_intervals_respect_lower_is_better(separated_leaderboard):
    ascending = rank_intervals(separated_leaderboard, B=400, seed=0, higher_is_better=False)
    descending = rank_intervals(separated_leaderboard, B=400, seed=0)
    names = list(separated_leaderboard)
    assert ascending[names[0]].point == len(names)
    assert descending[names[0]].point == 1


def test_rank_intervals_reject_mismatched_sentence_counts():
    rng = np.random.default_rng(15)
    with pytest.raises(ValueError, match="same sentences"):
        rank_intervals(
            {"a": heterogeneous_corpus(rng, n_sent=20), "b": heterogeneous_corpus(rng, n_sent=19)},
            B=10,
        )


def test_leaderboard_rejects_an_empty_mapping():
    with pytest.raises(ValueError, match="empty"):
        leaderboard({})


def test_max_t_band_is_wider_than_a_marginal_interval(separated_leaderboard):
    """Sanity on the simultaneous band that rank intervals are read beside."""
    band = max_t_band(separated_leaderboard, B=800, seed=0)
    assert band.c_star > 1.96
    name = next(iter(separated_leaderboard))
    lo, hi = band.interval(name)
    assert lo < band.estimates[name] < hi


# --------------------------------------------------------------------------
# bh_fdr
# --------------------------------------------------------------------------


def test_bh_fdr_matches_a_hand_computed_example():
    """m=6, q=0.05. Thresholds k/m*q are .00833 .01667 .025 .03333 .04167 .05.

    Sorted p = .001 .008 .039 .041 .042 .060 clears the first two and nothing
    after, so BH rejects exactly the two smallest. Adjusted p-values are the
    running minimum of m/k * p_(k) from the top down:
    m/k*p = .006 .024 .078 .0615 .0504 .060 -> running min .006 .024 .0504 .0504 .0504 .060.
    """
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060]
    result = bh_fdr(p, q=0.05)
    assert result.rejected.tolist() == [True, True, False, False, False, False]
# improved
    assert result.n_rejected == 2
    assert result.threshold == pytest.approx(0.008)
    assert result.qvalues.tolist() == pytest.approx(
        [0.006, 0.024, 0.0504, 0.0504, 0.0504, 0.060], abs=1e-9
    )


def test_bh_fdr_step_up_rescues_p_values_that_fail_their_own_threshold():
    """The step-up property: the largest k that clears drags everything below it
    in, even though p_(1)=.01 exceeds its own .00833 threshold. A step-DOWN
    procedure would reject nothing here."""
    p = [0.010, 0.020, 0.030, 0.040, 0.045, 0.049]
    result = bh_fdr(p, q=0.05)
    assert result.rejected.all()
    assert result.threshold == pytest.approx(0.049)
    assert result.qvalues.tolist() == pytest.approx([0.049] * 6, abs=1e-9)


def test_bh_fdr_is_more_powerful_than_bonferroni():
    p = [0.010, 0.020, 0.030, 0.040, 0.045, 0.049]
    bonferroni = sum(1 for x in p if x <= 0.05 / len(p))
    assert bh_fdr(p, q=0.05).n_rejected > bonferroni == 0


def test_bh_fdr_rejects_nothing_when_no_p_value_clears():
    result = bh_fdr([0.2, 0.4, 0.6, 0.8], q=0.05)
    assert result.n_rejected == 0
    assert result.threshold == 0.0


def test_bh_fdr_preserves_the_input_order():
    p = [0.60, 0.001, 0.30, 0.008]
    result = bh_fdr(p, q=0.05)
    assert result.rejected.tolist() == [False, True, False, True]
    assert result.qvalues[1] < result.qvalues[0]


def test_bh_fdr_qvalues_are_monotone_in_the_p_values():
    rng = np.random.default_rng(17)
    p = rng.uniform(size=40)
    result = bh_fdr(p, q=0.10)
    order = np.argsort(p)
    q_sorted = result.qvalues[order]
    assert np.all(np.diff(q_sorted) >= -1e-12)
    assert np.all(result.qvalues >= p - 1e-12)
    assert np.all(result.qvalues <= 1.0)


def test_bh_fdr_rejection_set_grows_with_q():
    rng = np.random.default_rng(18)
    p = np.concatenate([rng.uniform(size=30), rng.uniform(size=10) * 1e-3])
    counts = [bh_fdr(p, q=q).n_rejected for q in (0.01, 0.05, 0.10, 0.25)]
    assert counts == sorted(counts)


def test_bh_fdr_rejects_exactly_the_p_values_at_or_below_the_threshold():
    rng = np.random.default_rng(19)
    p = np.concatenate([rng.uniform(size=25), rng.uniform(size=8) * 1e-3])
    result = bh_fdr(p, q=0.05)
    assert result.rejected.tolist() == (p <= result.threshold).tolist()


def test_bh_fdr_supports_the_sequence_protocol():
    result = bh_fdr([0.001, 0.9], q=0.05)
    assert len(result) == 2
    assert result[0] is True
    assert list(result) == [True, False]
    assert dict(zip(["a", "b"], result, strict=True)) == {"a": True, "b": False}


def test_bh_fdr_rejects_an_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        bh_fdr([])


@pytest.mark.parametrize("q", [0.0, -0.1, 1.5])
def test_bh_fdr_rejects_an_out_of_range_q(q):
    with pytest.raises(ValueError, match="q must be"):
        bh_fdr([0.1, 0.2], q=q)


@pytest.mark.parametrize("q", [0.05, 0.10])
def test_bh_fdr_controls_the_false_discovery_rate(q):
    """40 true nulls, 10 real effects, 300 trials: mean FDP must not exceed q."""
    rng = np.random.default_rng(31)
    m0, m1 = 40, 10
    fdps = []
    for _ in range(300):
        p = np.concatenate([rng.uniform(size=m0), rng.uniform(size=m1) * 1e-4])
        rejected = bh_fdr(p, q=q).rejected
        n_rejected = int(rejected.sum())
        fdps.append(int(rejected[:m0].sum()) / n_rejected if n_rejected else 0.0)
    assert float(np.mean(fdps)) <= q + 0.01, f"mean FDP {np.mean(fdps):.4f} exceeded q={q}"


# --------------------------------------------------------------------------
# split_half_noise_floor
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def separated_floor(separated_leaderboard):
    return split_half_noise_floor(separated_leaderboard, n_splits=100, seed=0)


@pytest.fixture(scope="module")
def noise_floor(noise_leaderboard):
    return split_half_noise_floor(noise_leaderboard, n_splits=100, seed=0)


@pytest.fixture(scope="module")
def noise_floors() -> list:
    """Noise floors for 16 INDEPENDENT all-tied leaderboards.

    One tied leaderboard is not enough to pin the expected tau: the tokenizers
    are drawn once and every split then re-reads the same realised ordering, so
    a single board's mean tau wanders by +-0.3 for reasons that have nothing to
    do with the estimator. Averaging over boards is the honest way to assert
    "the noise floor sits at zero".
    """
    floors = []
    for board_id in range(16):
        rng = np.random.default_rng(1000 + board_id)
        board = {f"tok{i}": tokenizer_of_quality(rng, 0.6) for i in range(8)}
        floors.append(split_half_noise_floor(board, n_splits=50, seed=board_id))
    return floors


def test_split_half_tau_is_one_on_a_genuinely_separated_ordering(separated_floor):
    """Half the corpus is enough to recover a real ordering: tau ~ 1."""
    assert separated_floor.tau_mean > 0.95
    assert separated_floor.tau_p05 > 0.9
    assert separated_floor.rbo_mean > 0.95


def test_split_half_tau_is_zero_on_pure_noise(noise_floors):
    """The other direction: tokenizers that differ by nothing but sampling agree
    at tau ~ 0. This is the number every cross-convention tau must be read
    against -- an observed 0.85 means one thing beside a floor of 0.98 and quite
    another beside a floor of 0.84."""
    mean_tau = float(np.mean([floor.tau_mean for floor in noise_floors]))
    assert abs(mean_tau) < 0.20, f"mean tau over {len(noise_floors)} tied leaderboards={mean_tau}"


def test_pure_noise_tau_is_widely_spread_not_a_point(noise_floors):
    """The floor is a distribution: on tied data a single split-half tau of 0.5
    is unremarkable, which is exactly why `tau_p05` and not `tau_mean` is what
    `is_above_floor` compares against."""
    spread = float(np.mean([floor.tau_p95 - floor.tau_p05 for floor in noise_floors]))
    assert spread > 0.3, f"mean p05-p95 spread {spread}"


def test_the_noise_floor_separates_the_two_regimes(separated_floor, noise_floors):
    assert separated_floor.tau_mean - max(f.tau_mean for f in noise_floors) > 0.4
    for floor in noise_floors:
        assert floor.tau_mean < separated_floor.tau_p05


def test_split_half_rbo_also_drops_on_noise(separated_floor, noise_floor):
    """The top-weighted view agrees with tau: a tied board's podium is unstable."""
    assert noise_floor.rbo_mean < separated_floor.rbo_mean
    assert 0.0 <= noise_floor.rbo_mean <= 1.0


def test_is_above_floor_reads_in_the_direction_that_matters(separated_floor):
    """A tau below the 5th percentile of the floor is real disagreement."""
    assert separated_floor.is_above_floor(0.1)
    assert not separated_floor.is_above_floor(1.0)


def test_split_half_reports_its_shape(separated_floor, separated_leaderboard):
    assert separated_floor.n_splits == 100
    assert separated_floor.taus.shape == (100,)
    assert separated_floor.rbos.shape == (100,)
    assert separated_floor.n_tokenizers == len(separated_leaderboard)
    assert separated_floor.tau_p05 <= separated_floor.tau_median <= separated_floor.tau_p95


def test_split_half_is_reproducible_under_a_seed(tied_leaderboard):
    a = split_half_noise_floor(tied_leaderboard, n_splits=30, seed=5)
    b = split_half_noise_floor(tied_leaderboard, n_splits=30, seed=5)
    c = split_half_noise_floor(tied_leaderboard, n_splits=30, seed=6)
    assert a.taus.tolist() == b.taus.tolist()
    assert a.taus.tolist() != c.taus.tolist()


def test_split_half_needs_at_least_two_tokenizers():
    mat = heterogeneous_corpus(np.random.default_rng(20), n_sent=20)
    with pytest.raises(ValueError, match="2 tokenizers"):
        split_half_noise_floor({"only": mat}, n_splits=5)


def test_split_half_needs_at_least_two_sentences():
    row = [(2, 1, 1, 6)]
    with pytest.raises(ValueError, match="2 sentences"):
        split_half_noise_floor({"a": row, "b": row}, n_splits=5)


# --------------------------------------------------------------------------
# design_effect
# --------------------------------------------------------------------------


def test_design_effect_is_about_one_on_independent_positions():
    """With i.i.d. positions the clustering costs nothing, and the measured deff
    says so instead of assuming the folklore ICC of 0.10."""
    rng = np.random.default_rng(100)
    mat = iid_corpus(rng, cell_probs(0.18, 0.10, 0.12), n_sent=500, n_pos=24)
    result = design_effect(mat, B=8000, seed=0)
    assert 0.85 < result.deff < 1.20, f"deff={result.deff}"
    assert result.se_cluster == pytest.approx(result.var_cluster**0.5)
    assert result.se_iid == pytest.approx(result.var_iid**0.5)


def test_design_effect_is_large_on_all_or_nothing_sentences():
    """Every position in a sentence right or every one wrong: the extreme of
    within-sentence dependence, and a position-level analysis would report
    standard errors several times too small."""
    rng = np.random.default_rng(4)
    rows = []
    for _ in range(400):
        n, n_gold = 24, 8
        if rng.random() < 0.5:
            rows.append((n_gold, 0, 0, n - n_gold))
        else:
            rows.append((0, n_gold, n_gold, n - 2 * n_gold))
    result = design_effect(np.array(rows, dtype=np.int64), B=4000, seed=0)
    assert result.deff > 5.0, f"deff={result.deff}"
    assert result.var_cluster > result.var_iid


def test_design_effect_reports_effective_sample_size():
    rng = np.random.default_rng(101)
    mat = iid_corpus(rng, cell_probs(0.18, 0.10, 0.12), n_sent=200, n_pos=24)
    result = design_effect(mat, B=2000, seed=0)
    assert result.n_positions == int(mat.sum())
    assert result.n_clusters == 200
    assert result.effective_n == pytest.approx(result.n_positions / result.deff)


def test_design_effect_of_an_empty_corpus_is_undefined_not_one():
    result = design_effect(np.zeros((3, 4), dtype=np.int64), B=100, seed=0)
    assert np.isnan(result.deff)
    assert result.n_positions == 0


def test_design_effect_is_undefined_on_a_saturated_table():
    """A table with no iid variance gives nan -- a real 'undefined', not a 1.0."""
    result = design_effect([(5, 0, 0, 0), (7, 0, 0, 0)], B=200, seed=0)
    assert np.isnan(result.deff)


# --------------------------------------------------------------------------
# kendall_tau_b
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_kendall_tau_b_matches_scipy_on_random_vectors(seed):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=12), rng.normal(size=12)
    assert kendall_tau_b(a.tolist(), b.tolist()) == pytest.approx(
        kendalltau(a, b, variant="b").statistic
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_kendall_tau_b_matches_scipy_with_heavy_ties(seed):
    """tau-b, not tau-a: leaderboards tie and the denominator has to know."""
    rng = np.random.default_rng(100 + seed)
    a = rng.integers(0, 3, size=15).astype(float)
    b = rng.integers(0, 3, size=15).astype(float)
    expected = kendalltau(a, b, variant="b").statistic
    got = kendall_tau_b(a.tolist(), b.tolist())
    if np.isfinite(expected):
        assert got == pytest.approx(expected)
    else:
        assert got == 0.0


def test_kendall_tau_b_on_a_correlated_pair_matches_scipy():
    rng = np.random.default_rng(9)
    a = rng.normal(size=25)
    b = a + 0.2 * rng.normal(size=25)
    assert kendall_tau_b(a, b) == pytest.approx(kendalltau(a, b, variant="b").statistic)
    assert kendall_tau_b(a, b) > 0.7


def test_kendall_tau_b_of_identical_and_reversed_orderings():
    scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert kendall_tau_b(scores, scores) == pytest.approx(1.0)
    assert kendall_tau_b(scores, scores[::-1]) == pytest.approx(-1.0)


def test_kendall_tau_b_of_a_constant_vector_is_zero_not_nan():
    """A constant leaderboard carries no ordering information, which is what
    tau 0 means; scipy returns nan here."""
    assert kendall_tau_b([1.0, 2.0, 3.0], [7.0, 7.0, 7.0]) == 0.0
    assert np.isnan(kendalltau([1.0, 2.0, 3.0], [7.0, 7.0, 7.0], variant="b").statistic)


def test_kendall_tau_b_of_a_degenerate_vector_is_zero():
    assert kendall_tau_b([1.0], [2.0]) == 0.0
    assert kendall_tau_b([], []) == 0.0


def test_kendall_tau_b_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal-length"):
        kendall_tau_b([1.0, 2.0], [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------
# counts plumbing the rest of the module rests on
# --------------------------------------------------------------------------


def test_pooled_is_the_sum_of_the_counts_matrix():
    mat = heterogeneous_corpus(np.random.default_rng(21), n_sent=25)
    total = pooled(mat)
    assert (total.tp, total.fp, total.fn, total.tn) == tuple(mat.sum(axis=0).tolist())


def test_as_counts_matrix_accepts_tuples_counts_and_arrays():
    rows = [(1, 2, 3, 4), (5, 6, 7, 8)]
    from_tuples = as_counts_matrix(rows)
    from_counts = as_counts_matrix([Counts(*r) for r in rows])
#     from_array = as_counts_matrix(np.array(rows))
    assert from_tuples.tolist() == from_counts.tolist() == from_array.tolist()
    assert from_tuples.dtype == np.int64

# Refined

# Enhanced

# Enhanced

# Enhanced
