"""Density controls: the defences against "the headline metric is fertility".

PURE MODULE. No I/O, no tokenizers, no corpora.

THE THREAT. A tokenizer that chops more finely finds more gold boundaries for
free: under uniform random placement at hypothesis density ``h`` we have
``E[recall] = h``. So any leaderboard ranked on recall, boundary-F1 or purity
is partly a leaderboard of fertility, and a reader is entitled to assume ours is
too until we show otherwise. `phi` is claimed to be density-invariant; these are
the null models that make the claim falsifiable, and one of them goes on every
row of every table.

TWO NULLS, because one is not enough:

* **N0 -- positional null, closed form.** Hold each sentence's predicted count
  ``m`` fixed and scatter those boundaries uniformly over the scored positions.
  Then ``TP`` is exactly Hypergeometric and we need no Monte Carlo at all:
  ``E[TP] = mK/N`` per sentence, summed over independent sentences. It is free,
  so it costs nothing to print beside every observed number.

* **N1 -- length-distribution-matched.** N0 is too weak to be the primary
  control. A real tokenizer's token-length distribution is not geometric, and
  matching only the length distribution already buys some chance alignment,
  because token lengths correlate with word lengths. N1 resamples lengths from
  the tokenizer's OWN empirical distribution, lays boundaries at the cumulative
  offsets, and snaps them to the nearest legal position -- giving a null that
  matches the tokenizer on fertility, on length shape and on legality, and
  differs from it only in knowing where the words are. That is the null line on
  the plots.

`lift` is the number that survives: observed minus null, with a paired CI in
which both terms are recomputed on the same resampled sentences.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from unsegbench.metrics.core import Counts, phi
from unsegbench.metrics.stats import PairedDiff, Statistic, as_counts_matrix, paired_bootstrap

__all__ = [
    "Null0",
    "NullSummary",
    "empirical_length_dist",
    "lift",
    "lift_ci",
    "n0_expectation",
    "n0_micro_precision",
    "n0_null",
    "n1_counts",
    "n1_null",
    "n1_sample",
    "snap_to_mask",
]


# --------------------------------------------------------------------------
# N0: the closed-form positional null
# --------------------------------------------------------------------------


def n0_expectation(counts_per_sentence: Any) -> tuple[float, float]:
    """``(E[TP], Var[TP])`` under the hypergeometric positional null.

    For one sentence with ``N`` scored positions, ``K`` gold boundaries among
    them and ``m`` predicted boundaries scattered uniformly and without
    replacement, the number of hits is Hypergeometric:

    .. math::
        E[TP] = \\frac{mK}{N}, \\qquad
        Var[TP] = m\\,\\frac{K}{N}\\,\\frac{N-K}{N}\\,\\frac{N-m}{N-1}

    Sentences are independent, so both quantities simply sum. There is no Monte
    Carlo here and there does not need to be -- which is why the N0 column is
    free enough to print on every leaderboard row rather than in an appendix.

    Args:
        counts_per_sentence: per-sentence ``[TP, FP, FN, TN]``; the null reads
            only the margins ``N = TP+FP+FN+TN``, ``K = TP+FN``, ``m = TP+FP``.

    Returns:
        ``(E[TP], Var[TP])`` summed over sentences. Sentences with ``N < 2``
        contribute nothing: with one scored position there is no choice to make.
    """
    mat = as_counts_matrix(counts_per_sentence).astype(np.float64)
    n = mat.sum(axis=1)
    k = mat[:, 0] + mat[:, 2]
    m = mat[:, 0] + mat[:, 1]
    keep = n >= 2
    n, k, m = n[keep], k[keep], m[keep]
    if n.size == 0:
        return 0.0, 0.0
    e_tp = float(np.sum(m * k / n))
    var_tp = float(np.sum(m * (k / n) * ((n - k) / n) * ((n - m) / (n - 1.0))))
    return e_tp, var_tp


def n0_micro_precision(counts_per_sentence: Any) -> float:
    """Analytic micro-averaged null precision ``sum_j m_j * delta_g,j / sum_j m_j``.

    The gold-density weighted mean, weighted by how many boundaries each
    sentence's tokenizer actually proposed. This is the number a random
    tokenizer's precision converges to, and it is exactly ``E[TP] / sum_j m_j``
    -- the point being that it does NOT depend on ``m`` except through the
    weighting, which is the density-invariance of precision made explicit.

    Args:
        counts_per_sentence: per-sentence ``[TP, FP, FN, TN]``.

    Returns:
        The expected micro precision, or ``1.0`` if nothing was predicted
        anywhere (the frozen no-predictions convention).
    """
    mat = as_counts_matrix(counts_per_sentence).astype(np.float64)
    n = mat.sum(axis=1)
    keep = n >= 2
    if not keep.any():
        return 1.0
    n = n[keep]
    k = (mat[:, 0] + mat[:, 2])[keep]
    m = (mat[:, 0] + mat[:, 1])[keep]
    denom = float(np.sum(m))
    return float(np.sum(m * k / n) / denom) if denom > 0 else 1.0


@dataclass(frozen=True, slots=True)
class Null0:
    """The closed-form N0 null for one (tokenizer, corpus, mask)."""

    e_tp: float
    var_tp: float
    sd_tp: float
    e_precision: float
    e_recall: float
    e_f1: float
    e_phi: float
    n_positions: int
    n_gold: int
    n_pred: int

    def z_for(self, observed_tp: int) -> float:
        """How many null standard deviations the observed TP sits above N0.

        Not a p-value: the hypergeometric sum is only asymptotically normal, and
        the positions are clustered by sentence anyway. It is a magnitude check
        -- if this is 2 the tokenizer is barely beating dice.
        """
        return (observed_tp - self.e_tp) / self.sd_tp if self.sd_tp > 0 else float("nan")


def _phi_float(tp: float, fp: float, fn: float, tn: float) -> float:
    """`core.phi` on real-valued cells, for expected tables. Same 0/0 convention."""
    num = tp * tn - fp * fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (num / den) if den > 0 else 0.0


def n0_null(counts_per_sentence: Any) -> Null0:
    """Full N0 summary: expected TP and the metrics that follow from it.

    The expected metrics are plug-ins at ``E[TP]`` rather than expectations of
    the metrics, which is the right object here: our estimator is itself a
    plug-in at the pooled table, so the null it must be compared against is the
    plug-in at the pooled null table.

    ``e_phi`` is ~0 by construction and is NOT a free parameter: per sentence,
    ``E[TP]*E[TN] - E[FP]*E[FN]`` vanishes identically at ``E[TP] = mK/N``, so
    any departure from zero after pooling is pure Simpson's-paradox residue from
    sentences of different length and density. Its smallness is the empirical
    version of the density-invariance claim, and it is swept across densities in
    ``tests/test_nulls.py``.

    Args:
        counts_per_sentence: per-sentence ``[TP, FP, FN, TN]``.

    Returns:
        A `Null0`.
    """
    mat = as_counts_matrix(counts_per_sentence)
    total = mat.sum(axis=0)
    n_pos = int(total.sum())
    n_gold = int(total[0] + total[2])
    n_pred = int(total[0] + total[1])
    e_tp, var_tp = n0_expectation(mat)
    e_fp = n_pred - e_tp
    e_fn = n_gold - e_tp
    e_tn = n_pos - n_pred - n_gold + e_tp
    e_p = (e_tp / n_pred) if n_pred else 1.0
    e_r = (e_tp / n_gold) if n_gold else 1.0
    return Null0(
        e_tp=e_tp,
        var_tp=var_tp,
        sd_tp=math.sqrt(var_tp),
        e_precision=e_p,
        e_recall=e_r,
        e_f1=(2 * e_p * e_r / (e_p + e_r)) if (e_p + e_r) else 0.0,
        e_phi=_phi_float(e_tp, e_fp, e_fn, e_tn),
        n_positions=n_pos,
        n_gold=n_gold,
        n_pred=n_pred,
    )


# --------------------------------------------------------------------------
# N1: the length-distribution-matched null
# --------------------------------------------------------------------------


def empirical_length_dist(lengths: Any) -> tuple[np.ndarray, np.ndarray]:
    """Build a token-length pmf from observed lengths or a length histogram.

    Args:
        lengths: one of
            * an iterable of observed token lengths in CHARACTERS,
            * a mapping ``{length: weight}``,
            * a pair ``(values, probs)`` already in pmf form.

    Returns:
        ``(values, probs)`` with ``values`` a sorted int64 array of positive
        lengths and ``probs`` summing to 1.

    Raises:
        ValueError: if no positive length survives, or a weight is negative.
    """
    if isinstance(lengths, Mapping):
        values = np.asarray(list(lengths.keys()), dtype=np.int64)
        weights = np.asarray(list(lengths.values()), dtype=np.float64)
    elif (
        isinstance(lengths, tuple)
        and len(lengths) == 2
        and np.ndim(lengths[0]) == 1
        and np.ndim(lengths[1]) == 1
        and len(lengths[0]) == len(lengths[1])
    ):
        values = np.asarray(lengths[0], dtype=np.int64)
        weights = np.asarray(lengths[1], dtype=np.float64)
    else:
        obs = np.asarray(list(lengths), dtype=np.int64)
        values, counts = np.unique(obs[obs > 0], return_counts=True)
        weights = counts.astype(np.float64)
    if np.any(weights < 0):
        raise ValueError("length distribution weights must be non-negative")
    keep = (values > 0) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if values.size == 0:
        raise ValueError("length distribution is empty after dropping non-positive lengths")
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return values, weights / weights.sum()


def snap_to_mask(offsets: np.ndarray, sorted_mask: np.ndarray) -> np.ndarray:
    """Snap character offsets to the NEAREST legal position, ties going left.

    Snapping is what makes N1 a fair null rather than a straw man: a real
    tokenizer never places a boundary inside a Khmer COENG sequence either, so a
    null that does would be beaten trivially and would prove nothing. The cost
    is that two offsets can snap together, which lowers the achieved density
    slightly below the nominal one -- callers should report the density they got.

    Args:
        offsets: raw cumulative character offsets, any order.
        sorted_mask: ascending array of legal positions. May be empty.

    Returns:
        The sorted unique snapped positions.
    """
    if sorted_mask.size == 0 or offsets.size == 0:
        return np.empty(0, dtype=np.int64)
    idx = np.searchsorted(sorted_mask, offsets)
    left = np.clip(idx - 1, 0, sorted_mask.size - 1)
    right = np.clip(idx, 0, sorted_mask.size - 1)
    take_right = np.abs(sorted_mask[right] - offsets) < np.abs(offsets - sorted_mask[left])
    return np.unique(np.where(take_right, sorted_mask[right], sorted_mask[left]))


def n1_sample(
    text_lengths: Sequence[int],
    masks: Sequence[frozenset[int] | set[int] | Sequence[int]],
    length_dist: Any,
    R: int = 200,
    seed: int = 0,
) -> list[tuple[frozenset[int], ...]]:
    """Draw ``R`` length-matched null segmentations for a whole corpus.

    Per sentence and per replicate: draw token lengths i.i.d. from
    ``length_dist`` until they span the sentence, put boundaries at the
    cumulative offsets, drop the sentence edges, and snap what remains to the
    nearest position in that sentence's mask.

    The result matches the real tokenizer on fertility, on the SHAPE of its
    length distribution and on legality, and differs from it only in not knowing
    where the words are. That is a far more honest null than N0, which a
    tokenizer can beat just by having plausible token lengths.

    Args:
        text_lengths: ``n`` in codepoints for each sentence.
        masks: the scored universe for each sentence, same order and length as
            ``text_lengths``.
        length_dist: the tokenizer's empirical token-length distribution; see
            `empirical_length_dist`.
        R: number of replicates. 200 is enough for a plot line: the replicate SE
            of a corpus-level statistic falls as ``1/sqrt(R)`` on top of an
            already-pooled quantity.
        seed: PRNG seed.

    Returns:
        ``R`` replicates, each a tuple of ``J`` predicted boundary sets.

    Raises:
        ValueError: if ``text_lengths`` and ``masks`` differ in length.
    """
    if len(text_lengths) != len(masks):
        raise ValueError(f"text_lengths has {len(text_lengths)} entries but masks has {len(masks)}")
    values, probs = empirical_length_dist(length_dist)
    min_len = int(values[0])
    rng = np.random.default_rng(seed)
    sorted_masks = [np.asarray(sorted(m), dtype=np.int64) for m in masks]
    # Enough lengths per sentence to overshoot n even if every draw is minimal.
    quota = [
        (int(n) // min_len + 4) if (n >= 2 and mk.size) else 0
        for n, mk in zip(text_lengths, sorted_masks, strict=True)
    ]
    edges = np.cumsum([0, *quota])
    total = int(edges[-1])

    reps: list[tuple[frozenset[int], ...]] = []
    for _ in range(R):
        pool = rng.choice(values, size=total, p=probs) if total else np.empty(0, dtype=np.int64)
        rep: list[frozenset[int]] = []
        for j, (n, mk) in enumerate(zip(text_lengths, sorted_masks, strict=True)):
            if quota[j] == 0:
                rep.append(frozenset())
                continue
            offs = np.cumsum(pool[edges[j] : edges[j + 1]])
            offs = offs[(offs > 0) & (offs < n)]
            rep.append(frozenset(snap_to_mask(offs, mk).tolist()))
        reps.append(tuple(rep))
    return reps


def n1_counts(
    gold: Sequence[frozenset[int] | set[int]],
    masks: Sequence[frozenset[int] | set[int] | Sequence[int]],
    reps: Sequence[Sequence[frozenset[int]]],
) -> np.ndarray:
    """Score sampled null segmentations into an ``(R, J, 4)`` counts array.

    Uses the same masked contingency table as everything else -- both gold and
    prediction intersected with the mask -- so a null row is interchangeable
    with an observed row everywhere downstream, including in `lift_ci`.

    Args:
        gold: gold boundary set per sentence.
        masks: scored universe per sentence.
        reps: output of `n1_sample`.

    Returns:
        ``(R, J, 4)`` int64 array of ``[TP, FP, FN, TN]``.
    """
    prepared = [
        (frozenset(g) & frozenset(m), frozenset(m)) for g, m in zip(gold, masks, strict=True)
    ]
    out = np.empty((len(reps), len(prepared), 4), dtype=np.int64)
    for r, rep in enumerate(reps):
        for j, ((g, m), p_raw) in enumerate(zip(prepared, rep, strict=True)):
            p = frozenset(p_raw) & m
            tp = len(g & p)
            out[r, j] = (tp, len(p) - tp, len(g) - tp, len(m) - len(g | p))
    return out


@dataclass(frozen=True, slots=True)
class NullSummary:
    """Distribution of a statistic across N1 replicates."""

    mean: float
    sd: float
    p025: float
    p975: float
    density: float
    n_reps: int
    values: np.ndarray

    @property
    def ci(self) -> tuple[float, float]:
        return (self.p025, self.p975)


def n1_null(
    gold: Sequence[frozenset[int] | set[int]],
    masks: Sequence[frozenset[int] | set[int] | Sequence[int]],
    text_lengths: Sequence[int],
    length_dist: Any,
    statistic: Statistic = phi,
    R: int = 200,
    seed: int = 0,
) -> NullSummary:
    """`n1_sample` + `n1_counts` + pool + summarise, in one call.

    Args:
        gold: gold boundary set per sentence.
        masks: scored universe per sentence.
        text_lengths: ``n`` per sentence.
        length_dist: the tokenizer's token-length distribution.
        statistic: pure function of the pooled `Counts`.
        R: replicates.
        seed: PRNG seed.

    Returns:
        A `NullSummary`, including the ACHIEVED predicted-boundary density,
        which is reported rather than assumed because snapping collapses
        colliding offsets and so lands slightly below the nominal density.
    """
    reps = n1_sample(text_lengths, masks, length_dist, R=R, seed=seed)
    counts = n1_counts(gold, masks, reps)
    totals = counts.sum(axis=1)  # (R, 4)
    values = np.array(
        [statistic(Counts(a, b, c, d)) for a, b, c, d in totals.tolist()], dtype=float
    )
    n_pos = totals[0].sum()
    density = float(np.mean((totals[:, 0] + totals[:, 1]) / n_pos)) if n_pos else 0.0
    return NullSummary(
        mean=float(np.mean(values)),
        sd=float(np.std(values, ddof=1)) if R > 1 else 0.0,
        p025=float(np.percentile(values, 2.5)),
        p975=float(np.percentile(values, 97.5)),
        density=density,
        n_reps=R,
        values=values,
    )


# --------------------------------------------------------------------------
# Lift
# --------------------------------------------------------------------------


def lift(observed_phi: float, null_phi: float) -> float:
    """Observed minus null: the part of the score the null does not explain.

    A difference, not a ratio. A ratio explodes as the null approaches zero --
    which is exactly what a well-behaved `phi` null does -- turning the headline
    number into an artefact of the denominator.

    Args:
        observed_phi: the real tokenizer's statistic.
        null_phi: the matched null's statistic (N0 or N1).

    Returns:
        ``observed_phi - null_phi``.
    """
    return observed_phi - null_phi


def lift_ci(
    counts_observed: Any,
    counts_null: Any,
    statistic: Statistic = phi,
    B: int = 10_000,
    seed: int = 0,
    method: str = "bca",
    level: float = 0.95,
) -> PairedDiff:
    """Paired cluster-bootstrap CI on the lift.

    Both terms are recomputed on the SAME resampled sentences, which is the only
    way the interval means anything: the null is a function of the very
    sentences it is being subtracted from, so the two arms share almost all of
    their sampling variation and an unpaired interval would be far too wide.

    Args:
        counts_observed: per-sentence counts for the real tokenizer.
        counts_null: per-sentence counts for the null, sentence-aligned. For N1
            use one replicate (or the replicate-mean rounded to integers) from
            `n1_counts`.
        statistic: pure function of the pooled `Counts`.
        B: replicates.
        seed: PRNG seed.
        method: ``"bca"`` or ``"percentile"``.
        level: nominal coverage.

    Returns:
        A `PairedDiff` whose ``diff`` is the lift.
    """
    return paired_bootstrap(
        counts_observed,
        counts_null,
        statistic,
        B=B,
        seed=seed,
        method=method,
        level=level,
        name_a="observed",
        name_b="null",
    )
