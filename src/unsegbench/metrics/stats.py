"""Uncertainty, multiplicity and rank stability over the per-sentence counters.

PURE MODULE. Integers in, floats out. Imports only `metrics.core`, numpy and
scipy -- no I/O, no tokenizers, no corpora. Every function here is a groupby
over the ``(J, 4)`` matrix of per-sentence ``[TP, FP, FN, TN]`` that the
expensive stage persists, which is why the whole inference layer costs ~2s.

WHY each of these exists, since none of them is the default thing to do:

* **Sentence-level cluster bootstrap.** Gap positions inside one sentence are
  emphatically not independent: a tokenizer that mis-segments a compound gets
  every position in it wrong together. Resampling positions would understate
  the standard error by the design effect, which `design_effect` MEASURES
  rather than assuming the folklore ICC of 0.10.
* **Paired everything.** Two tokenizers are evaluated on the *same* sentences,
  so their errors are strongly positively correlated. Comparing two marginal
  CIs for overlap is both invalid (it is not a test) and far less powerful than
  a CI on the paired difference under identical resample indices. Hence
  `paired_bootstrap` and the fact that nothing here returns a lone marginal CI
  as the primary comparison object.
* **max-t band, not 25 marginal CIs.** A leaderboard invites the reader to make
  every pairwise comparison at once. `max_t_band` gives error bars that are
  valid *simultaneously* across all tokenizers, which is what the picture is
  actually claiming.
* **Rank intervals, not ranks.** "Rank 2" is a fiction when the bootstrap puts
  a tokenizer anywhere in 1--4. `rank_intervals` reports the interval.
* **BH, not Bonferroni.** Our tests all share one reference corpus and are
  heavily positively dependent, so they satisfy PRDS; BH controls FDR there and
  is dramatically more powerful than a Bonferroni correction that assumes the
  worst-case dependence we demonstrably do not have.
* **A noise floor for tau.** A between-convention Kendall tau of 0.85 means
  nothing until you know what tau *pure sampling noise* produces on this much
  data. `split_half_noise_floor` measures exactly that, and it is the single
  most load-bearing function in this module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import kendalltau, norm, rankdata

from unsegbench.metrics.core import (
    Counts,
    f1,
    informedness,
    markedness,
    phi,
    precision,
    recall,
)

__all__ = [
    "BHResult",
    "BootstrapCI",
    "DesignEffect",
    "MaxTBand",
    "NoiseFloor",
    "PairedDiff",
    "RankInterval",
    "Statistic",
    "as_counts_matrix",
    "bh_fdr",
    "bootstrap",
    "design_effect",
    "f1",
    "informedness",
    "kendall_tau_b",
    "leaderboard",
    "markedness",
    "max_t_band",
    "paired_bootstrap",
    "phi",
    "pooled",
    "precision",
    "rank_intervals",
    "ranking",
    "rbo",
    "recall",
    "split_half_noise_floor",
    "top_k_jaccard",
]

#: A statistic is any pure function of a pooled contingency table. `phi`, `f1`,
#: `informedness`, `markedness`, `precision` and `recall` are re-exported from
#: `metrics.core` so callers never have to write an adapter.
Statistic = Callable[[Counts], float]

#: Cap on the number of ``(draw, sentence)`` gather cells held at once. The
#: bootstrap is chunked over draws to keep peak memory flat in ``B * J``.
_CHUNK_CELLS = 500_000


# --------------------------------------------------------------------------
# The counts matrix
# --------------------------------------------------------------------------


def as_counts_matrix(counts: Any) -> np.ndarray:
    """Coerce per-sentence counts to a ``(J, 4)`` int64 array of ``[TP,FP,FN,TN]``.

    Args:
        counts: a ``(J, 4)`` array-like, an iterable of `Counts`, or an iterable
            of 4-tuples. Objects exposing ``b_tp``/``b_fp``/``b_fn``/``b_tn``
            (i.e. `types.RowStats`) are accepted too, so the runner's rows can
            be handed over untouched.

    Returns:
        A ``(J, 4)`` C-contiguous int64 array.

    Raises:
        ValueError: if the result is not two-dimensional with four columns.
    """
    if isinstance(counts, np.ndarray) and counts.ndim == 2 and counts.shape[1] == 4:
        return np.ascontiguousarray(counts, dtype=np.int64)

    rows: list[tuple[int, int, int, int]] = []
    for row in counts:
        if isinstance(row, Counts):
            rows.append((row.tp, row.fp, row.fn, row.tn))
        elif hasattr(row, "b_tp"):
            rows.append((row.b_tp, row.b_fp, row.b_fn, row.b_tn))
        else:
            a, b, c, d = row
            rows.append((int(a), int(b), int(c), int(d)))
    arr = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"expected a (J, 4) counts matrix, got shape {arr.shape}")
    return arr


def pooled(counts: Any) -> Counts:
    """Micro-aggregate a counts matrix into one `Counts`.

    Summation is the whole aggregation story: `Counts` is a commutative monoid
    under ``+``, so micro-averaging over any partition of the sentences gives a
    bit-identical result. That is asserted in ``tests/test_metrics_golden.py``.
    """
    total = as_counts_matrix(counts).sum(axis=0)
    return Counts(int(total[0]), int(total[1]), int(total[2]), int(total[3]))


def _stat(statistic: Statistic, totals: np.ndarray) -> np.ndarray:
    """Apply a scalar statistic to every row of a ``(B, 4)`` totals array."""
    return np.array([statistic(Counts(a, b, c, d)) for a, b, c, d in totals.tolist()], dtype=float)


def _resample_totals(
    mats: Sequence[np.ndarray], n_draws: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Cluster-bootstrap totals for several count matrices under SHARED indices.

    One draw is: sample ``J`` sentence indices with replacement, then SUM those
    rows. It is a sum over resampled rows, never a Python loop over sentences.
    Every matrix in ``mats`` is resampled with the *same* indices, which is what
    makes `paired_bootstrap`, `max_t_band` and `rank_intervals` paired.

    Args:
        mats: count matrices, all with the same number of rows ``J``.
        n_draws: number of bootstrap replicates ``B``.
        rng: the generator to draw indices from.

    Returns:
        One ``(B, 4)`` int64 totals array per input matrix.
    """
    n_sent = mats[0].shape[0]
    for m in mats:
        if m.shape[0] != n_sent:
            raise ValueError("paired resampling requires the same sentences in every matrix")
    out = [np.empty((n_draws, 4), dtype=np.int64) for _ in mats]
    if n_sent == 0:
        for arr in out:
            arr[:] = 0
        return out
    step = max(1, _CHUNK_CELLS // n_sent)
    for lo in range(0, n_draws, step):
        hi = min(n_draws, lo + step)
        idx = rng.integers(0, n_sent, size=(hi - lo, n_sent))
        for arr, mat in zip(out, mats, strict=True):
            arr[lo:hi] = mat[idx].sum(axis=1)
    return out


# --------------------------------------------------------------------------
# Confidence intervals
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """A cluster-bootstrap interval for one statistic on one tokenizer."""

    estimate: float
    lo: float
    hi: float
    se: float
    level: float
    method: str
    n_clusters: int
    draws: np.ndarray = field(repr=False)

    @property
    def width(self) -> float:
        return self.hi - self.lo


def _percentile_ci(draws: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    return (
        float(np.percentile(draws, 100.0 * alpha)),
        float(np.percentile(draws, 100.0 * (1.0 - alpha))),
    )


def _jackknife(statistic: Statistic, mat: np.ndarray) -> np.ndarray:
    """Leave-one-sentence-out values. Cheap: the total minus one row."""
    total = mat.sum(axis=0)
    return _stat(statistic, total[None, :] - mat)


def _acceleration(jack: np.ndarray) -> float:
    """BCa acceleration from the jackknife (Efron & Tibshirani eq. 14.15)."""
    dev = jack.mean() - jack
    denom = 6.0 * float(np.sum(dev**2)) ** 1.5
    if denom == 0.0 or not np.isfinite(denom):
        return 0.0
    return float(np.sum(dev**3) / denom)


def _bca_ci(
    draws: np.ndarray, estimate: float, jack: np.ndarray, level: float
) -> tuple[float, float]:
    """Bias-corrected and accelerated interval, degrading to percentile.

    The degradation is deliberate and not a silent repair: when every draw falls
    on one side of the estimate the bias correction ``z0`` is infinite and BCa
    is undefined, which happens routinely for saturated tables (phi == 1). The
    caller sees ``method="bca"`` either way because the *requested* method is
    what documents the analysis; the fallback is a numerical fact about that
    particular table, not a different estimator choice.
    """
    prop = float(np.mean(draws < estimate)) + 0.5 * float(np.mean(draws == estimate))
    if not (0.0 < prop < 1.0):
        return _percentile_ci(draws, level)
    z0 = float(norm.ppf(prop))
    acc = _acceleration(jack)
    alpha = (1.0 - level) / 2.0
    out: list[float] = []
    for a in (alpha, 1.0 - alpha):
        z = z0 + float(norm.ppf(a))
        denom = 1.0 - acc * z
        if denom <= 0.0:
            return _percentile_ci(draws, level)
        adj = float(norm.cdf(z0 + z / denom))
        out.append(float(np.percentile(draws, 100.0 * min(max(adj, 0.0), 1.0))))
    lo, hi = out
    if lo > hi:  # pathological acceleration; do not silently invert the interval
        return _percentile_ci(draws, level)
    return lo, hi


def bootstrap(
    counts: Any,
    statistic: Statistic = phi,
    B: int = 10_000,
    seed: int = 0,
    method: str = "bca",
    level: float = 0.95,
) -> BootstrapCI:
    """Sentence-level cluster bootstrap CI for one statistic.

    Resamples SENTENCES with replacement -- never positions. Positions within a
    sentence are correlated, so a position-level bootstrap would understate the
    standard error by the design effect; `design_effect` measures how much.

    Args:
        counts: per-sentence ``[TP, FP, FN, TN]``; see `as_counts_matrix`.
        statistic: a pure function of the pooled `Counts`. Defaults to `phi`.
        B: number of bootstrap replicates.
        seed: PRNG seed. Two calls with the same seed and the same number of
            sentences use identical resample indices.
        method: ``"bca"`` (default) or ``"percentile"``.
        level: nominal coverage, e.g. ``0.95``.

    Returns:
        A `BootstrapCI`, carrying the raw draws so callers can re-use them.

    Raises:
        ValueError: on an unknown ``method``.
    """
    if method not in ("bca", "percentile"):
        raise ValueError(f"unknown method {method!r}; expected 'bca' or 'percentile'")
    mat = as_counts_matrix(counts)
    est = statistic(pooled(mat))
    rng = np.random.default_rng(seed)
    (totals,) = _resample_totals([mat], B, rng)
    draws = _stat(statistic, totals)
    if method == "percentile" or mat.shape[0] < 3:
        lo, hi = _percentile_ci(draws, level)
    else:
        lo, hi = _bca_ci(draws, est, _jackknife(statistic, mat), level)
    return BootstrapCI(
        estimate=est,
        lo=lo,
        hi=hi,
        se=float(np.std(draws, ddof=1)) if B > 1 else 0.0,
        level=level,
        method=method,
        n_clusters=int(mat.shape[0]),
        draws=draws,
    )


@dataclass(frozen=True, slots=True)
class PairedDiff:
    """A paired cluster-bootstrap comparison of two tokenizers.

    ``diff = statistic(A) - statistic(B)`` on the pooled tables, with the CI and
    p-value computed from resamples that use IDENTICAL sentence indices for both
    arms. This is the only comparison object this module offers on purpose:
    checking two marginal CIs for overlap is not a test, and it throws away the
    positive correlation that makes the paired comparison powerful.
    """

    name_a: str
    name_b: str
    estimate_a: float
    estimate_b: float
    diff: float
    lo: float
    hi: float
    se: float
    p_value: float
    level: float
    method: str
    n_clusters: int
    draws: np.ndarray = field(repr=False)

    @property
    def significant(self) -> bool:
        """True if the interval on the difference excludes zero."""
        return self.lo > 0.0 or self.hi < 0.0

    @property
    def point(self) -> float:
        """Alias for `diff`: the point estimate of the paired difference."""
        return self.diff

    @property
    def pvalue(self) -> float:
        """Alias for `p_value`, for callers that prefer scipy's spelling."""
        return self.p_value


def paired_bootstrap(
    counts_a: Any,
    counts_b: Any,
    statistic: Statistic = phi,
    B: int = 10_000,
    seed: int = 0,
    method: str = "bca",
    level: float = 0.95,
    name_a: str = "a",
    name_b: str = "b",
) -> PairedDiff:
    """Paired cluster bootstrap on ``statistic(A) - statistic(B)``.

    MANDATORY for every comparison. Both arms are resampled with the same
    sentence indices, so the shared sentence-level variation cancels in the
    difference; the resulting interval is typically several times narrower than
    what two marginal intervals would suggest, and unlike overlap-checking it is
    an actual test.

    Args:
        counts_a: per-sentence counts for arm A. Row ``j`` must be the same
            sentence as row ``j`` of ``counts_b``.
        counts_b: per-sentence counts for arm B.
        statistic: pure function of the pooled `Counts`.
        B: number of replicates.
        seed: PRNG seed.
# improved
        method: ``"bca"`` or ``"percentile"``, applied to the difference.
        level: nominal coverage.
        name_a: label for arm A, carried into the result.
        name_b: label for arm B.

    Returns:
        A `PairedDiff`.

    Raises:
        ValueError: if the two matrices do not have the same number of
            sentences, or on an unknown ``method``.
    """
    if method not in ("bca", "percentile"):
        raise ValueError(f"unknown method {method!r}; expected 'bca' or 'percentile'")
    mat_a, mat_b = as_counts_matrix(counts_a), as_counts_matrix(counts_b)
    if mat_a.shape[0] != mat_b.shape[0]:
        raise ValueError(
            f"paired bootstrap needs matched sentences: got {mat_a.shape[0]} vs {mat_b.shape[0]}"
        )
    est_a, est_b = statistic(pooled(mat_a)), statistic(pooled(mat_b))
    est = est_a - est_b
    rng = np.random.default_rng(seed)
    tot_a, tot_b = _resample_totals([mat_a, mat_b], B, rng)
    draws = _stat(statistic, tot_a) - _stat(statistic, tot_b)
    if method == "percentile" or mat_a.shape[0] < 3:
        lo, hi = _percentile_ci(draws, level)
    else:
        jack = _jackknife(statistic, mat_a) - _jackknife(statistic, mat_b)
        lo, hi = _bca_ci(draws, est, jack, level)
    # Two-sided bootstrap p-value: the achieved significance level of zero.
    p_lo = float(np.mean(draws <= 0.0))
    p_hi = float(np.mean(draws >= 0.0))
    p_value = min(1.0, 2.0 * min(p_lo, p_hi))
    return PairedDiff(
        name_a=name_a,
        name_b=name_b,
        estimate_a=est_a,
        estimate_b=est_b,
        diff=est,
        lo=lo,
        hi=hi,
        se=float(np.std(draws, ddof=1)) if B > 1 else 0.0,
        p_value=p_value,
        level=level,
        method=method,
        n_clusters=int(mat_a.shape[0]),
        draws=draws,
    )


# --------------------------------------------------------------------------
# Leaderboard-wide objects
# --------------------------------------------------------------------------


def _ordered(counts_by_tokenizer: Mapping[str, Any]) -> tuple[list[str], list[np.ndarray]]:
    names = list(counts_by_tokenizer)
    if not names:
        raise ValueError("counts_by_tokenizer is empty")
    mats = [as_counts_matrix(counts_by_tokenizer[k]) for k in names]
    n_sent = mats[0].shape[0]
    for name, m in zip(names, mats, strict=True):
        if m.shape[0] != n_sent:
            raise ValueError(
                f"{name!r} has {m.shape[0]} sentences but the first tokenizer has {n_sent}; "
                "every tokenizer must be scored on the same sentences in the same order"
            )
    return names, mats


def leaderboard(
    counts_by_tokenizer: Mapping[str, Any], statistic: Statistic = phi
) -> dict[str, float]:
    """Point estimate per tokenizer from the pooled tables."""
    names, mats = _ordered(counts_by_tokenizer)
    return {n: statistic(pooled(m)) for n, m in zip(names, mats, strict=True)}


def ranking(
    counts_by_tokenizer: Mapping[str, Any],
    statistic: Statistic = phi,
    higher_is_better: bool = True,
) -> list[str]:
    """Tokenizer names best-first. Ties broken by name for determinism."""
# improved
    scores = leaderboard(counts_by_tokenizer, statistic)
    sign = -1.0 if higher_is_better else 1.0
    return sorted(scores, key=lambda k: (sign * scores[k], k))


@dataclass(frozen=True, slots=True)
class MaxTBand:
    """Simultaneous confidence band across a whole leaderboard.

    ``c_star`` is the ``level`` quantile of ``max_X |stat*(X) - stat(X)| / SE(X)``
    over bootstrap draws, so ``stat(X) +- c_star * SE(X)`` covers every
    tokenizer's true value at once with probability ``level``. Marginal 95%
    bars do not: with 25 tokenizers the family-wise error of reading all the
    pairwise comparisons off marginal bars is nowhere near 5%.
    """

    c_star: float
    level: float
    estimates: dict[str, float]
    se: dict[str, float]
    lo: dict[str, float]
    hi: dict[str, float]
    n_clusters: int

    def interval(self, name: str) -> tuple[float, float]:
        return (self.lo[name], self.hi[name])


def max_t_band(
    counts_by_tokenizer: Mapping[str, Any],
    statistic: Statistic = phi,
    B: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> MaxTBand:
    """Simultaneous (max-t) confidence band over all tokenizers.

    Args:
        counts_by_tokenizer: name -> per-sentence counts. Every value must cover
            the SAME sentences in the SAME order; the shared resample indices are
            what make the band simultaneous rather than 25 marginal intervals.
        statistic: pure function of the pooled `Counts`.
        B: number of replicates.
        seed: PRNG seed.
        level: simultaneous coverage.

    Returns:
        A `MaxTBand`.
    """
    names, mats = _ordered(counts_by_tokenizer)
    est = np.array([statistic(pooled(m)) for m in mats], dtype=float)
    rng = np.random.default_rng(seed)
    totals = _resample_totals(mats, B, rng)
    draws = np.column_stack([_stat(statistic, t) for t in totals])  # (B, T)
    se = np.std(draws, axis=0, ddof=1) if B > 1 else np.zeros(len(names))
    # A zero-SE column is a degenerate tokenizer (identical on every resample).
    # Give it no say in the maximum rather than an infinite one.
    safe = np.where(se > 0.0, se, np.inf)
    t_stats = np.abs(draws - est[None, :]) / safe[None, :]
    c_star = float(np.percentile(np.max(t_stats, axis=1), 100.0 * level))
    lo = est - c_star * se
    hi = est + c_star * se
    return MaxTBand(
        c_star=c_star,
        level=level,
        estimates=dict(zip(names, est.tolist(), strict=True)),
        se=dict(zip(names, se.tolist(), strict=True)),
        lo=dict(zip(names, lo.tolist(), strict=True)),
        hi=dict(zip(names, hi.tolist(), strict=True)),
        n_clusters=int(mats[0].shape[0]),
    )


@dataclass(frozen=True, slots=True)
class RankInterval:
    """A tokenizer's bootstrap rank interval. Rank 1 is best."""

    name: str
    point: int
    lo: int
    hi: int
    median: float
    p_best: float

    def __str__(self) -> str:
        return f"Rank {self.lo}" if self.lo == self.hi else f"Rank {self.lo}-{self.hi}"


def rank_intervals(
    counts_by_tokenizer: Mapping[str, Any],
    statistic: Statistic = phi,
    B: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
    higher_is_better: bool = True,
) -> dict[str, RankInterval]:
    """Bootstrap rank intervals: what the leaderboard is entitled to claim.

    Each draw re-ranks every tokenizer under identical resample indices; the
    returned interval is the ``level`` percentile range of those ranks, widened
    outward to whole ranks. A tokenizer reported as "Rank 1-4" is one whose
    apparent position is not distinguishable from three others, and saying so is
    the honest presentation.

    Args:
        counts_by_tokenizer: name -> per-sentence counts, matched sentences.
        statistic: pure function of the pooled `Counts`.
        B: number of replicates.
        seed: PRNG seed.
        level: interval coverage, e.g. ``0.95`` -> the 2.5-97.5 percentile ranks.
        higher_is_better: rank descending by the statistic (the default; true
            for `phi`, `f1` and friends).

    Returns:
        name -> `RankInterval`, in the input order.
    """
    names, mats = _ordered(counts_by_tokenizer)
    sign = -1.0 if higher_is_better else 1.0
    est = np.array([statistic(pooled(m)) for m in mats], dtype=float)
    point = rankdata(sign * est, method="min").astype(int)
    rng = np.random.default_rng(seed)
    totals = _resample_totals(mats, B, rng)
    draws = np.column_stack([_stat(statistic, t) for t in totals])  # (B, T)
    ranks = np.apply_along_axis(lambda r: rankdata(sign * r, method="min"), 1, draws)
    alpha = (1.0 - level) / 2.0
    lo = np.floor(np.percentile(ranks, 100.0 * alpha, axis=0)).astype(int)
    hi = np.ceil(np.percentile(ranks, 100.0 * (1.0 - alpha), axis=0)).astype(int)
    med = np.median(ranks, axis=0)
    p_best = np.mean(ranks == 1, axis=0)
    return {
        name: RankInterval(
            name=name,
            point=int(point[i]),
            lo=int(lo[i]),
            hi=int(hi[i]),
            median=float(med[i]),
            p_best=float(p_best[i]),
        )
        for i, name in enumerate(names)
    }


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BHResult:
    """Benjamini-Hochberg outcome, in the input order."""

    rejected: np.ndarray
    qvalues: np.ndarray
    threshold: float
#     q: float

    @property
    def n_rejected(self) -> int:
        return int(self.rejected.sum())

    def __iter__(self) -> Iterator[bool]:
        """Iterate the rejection flags, so ``zip(keys, bh_fdr(p))`` just works."""
        return iter(self.rejected.tolist())

    def __len__(self) -> int:
        return int(self.rejected.size)

    def __getitem__(self, i: int) -> bool:
        return bool(self.rejected[i])


def bh_fdr(pvalues: Iterable[float], q: float = 0.05) -> BHResult:
    """Benjamini-Hochberg step-up FDR control.

    Not Bonferroni. Our comparisons all share one reference corpus and one set
    of gold boundaries, so they are heavily positively dependent -- they satisfy
    PRDS, under which BH controls FDR exactly. Bonferroni instead pays for a
    worst-case dependence structure we can rule out, and at 25 tokenizers that
    costs most of the power we have.

    Args:
        pvalues: p-values, e.g. `PairedDiff.p_value` from every comparison.
        q: target false discovery rate.

    Returns:
        A `BHResult` whose arrays are aligned with the input order.
        ``threshold`` is the largest p-value rejected, or ``0.0`` if none is.

    Raises:
        ValueError: if ``pvalues`` is empty or ``q`` is not in ``(0, 1]``.
    """
    p = np.asarray(list(pvalues), dtype=float)
    if p.size == 0:
        raise ValueError("bh_fdr needs at least one p-value")
    if not (0.0 < q <= 1.0):
        raise ValueError(f"q must be in (0, 1], got {q}")
    m = p.size
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, m + 1, dtype=float)
    p_sorted = p[order]
    # Step-up: largest k with p_(k) <= k/m * q; reject everything at or below it.
    below = p_sorted <= (ranks / m) * q
    k = int(np.max(np.nonzero(below)[0]) + 1) if below.any() else 0
    rejected_sorted = np.zeros(m, dtype=bool)
    rejected_sorted[:k] = True
    # Adjusted p-values: running minimum of m/k * p_(k) from the largest down.
    q_sorted = np.minimum.accumulate((m / ranks * p_sorted)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    rejected = np.empty(m, dtype=bool)
    qvalues = np.empty(m, dtype=float)
    rejected[order] = rejected_sorted
    qvalues[order] = q_sorted
    return BHResult(
        rejected=rejected,
        qvalues=qvalues,
        threshold=float(p_sorted[k - 1]) if k else 0.0,
        q=q,
    )


# --------------------------------------------------------------------------
# Design effect
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DesignEffect:
    """Measured cost of within-sentence dependence.

    ``deff = Var(cluster bootstrap) / Var(iid-over-positions bootstrap)``. A
    deff of 4 means a position-level analysis reports standard errors half the
    size of the honest ones and an effective sample size a quarter of the
    nominal one. We MEASURE this rather than assuming the usual ICC ~ 0.10.
    """

    deff: float
    var_cluster: float
    var_iid: float
    se_cluster: float
    se_iid: float
    n_clusters: int
    n_positions: int

    @property
    def effective_n(self) -> float:
        """Positions' worth of information the clustered data actually carries."""
        return self.n_positions / self.deff if self.deff > 0 else float("nan")


def design_effect(
    counts: Any,
    statistic: Statistic = phi,
    B: int = 10_000,
    seed: int = 0,
) -> DesignEffect:
    """Empirical design effect of the sentence clustering.

    The iid arm resamples POSITIONS: it redraws the pooled ``N`` positions from
    a multinomial over the four cells, which is exactly what a naive analysis
    that treats gap positions as independent trials assumes. The ratio of
    variances is the price of that assumption, measured on the data in hand.

    Args:
        counts: per-sentence counts.
        statistic: pure function of the pooled `Counts`.
        B: number of replicates for both arms.
        seed: PRNG seed.

    Returns:
        A `DesignEffect`. ``deff`` is ``nan`` when the iid arm has zero variance
        (a saturated table), which is a real "undefined", not a 1.0.
    """
    mat = as_counts_matrix(counts)
    total = mat.sum(axis=0)
    n_pos = int(total.sum())
    rng = np.random.default_rng(seed)
    (tot_cluster,) = _resample_totals([mat], B, rng)
    var_cluster = float(np.var(_stat(statistic, tot_cluster), ddof=1)) if B > 1 else 0.0
    if n_pos == 0:
        return DesignEffect(float("nan"), var_cluster, 0.0, 0.0, 0.0, int(mat.shape[0]), 0)
    probs = total / n_pos
    tot_iid = rng.multinomial(n_pos, probs, size=B).astype(np.int64)
    var_iid = float(np.var(_stat(statistic, tot_iid), ddof=1)) if B > 1 else 0.0
    deff = (var_cluster / var_iid) if var_iid > 0.0 else float("nan")
    return DesignEffect(
        deff=deff,
        var_cluster=var_cluster,
        var_iid=var_iid,
        se_cluster=var_cluster**0.5,
        se_iid=var_iid**0.5,
        n_clusters=int(mat.shape[0]),
        n_positions=n_pos,
    )


# --------------------------------------------------------------------------
# Rank agreement
# --------------------------------------------------------------------------


def kendall_tau_b(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall's tau-b between two score vectors, aligned element-wise.

    tau-b, not tau-a: leaderboards tie, and tau-a's denominator pretends they do
    not. Returns ``0.0`` rather than ``nan`` when a vector is constant -- a
    constant leaderboard carries no ordering information, which is precisely
    what tau 0 means.

    Args:
        a: scores for each tokenizer under one condition.
        b: scores for the SAME tokenizers, same order, under the other.

    Returns:
        tau-b in ``[-1, 1]``.

    Raises:
        ValueError: if the vectors differ in length.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"kendall_tau_b needs equal-length vectors, got {x.shape} and {y.shape}")
    if x.size < 2:
        return 0.0
    tau = kendalltau(x, y, variant="b").statistic
    return 0.0 if not np.isfinite(tau) else float(tau)


def rbo(a: Sequence[Any], b: Sequence[Any], p: float = 0.9) -> float:
    """Rank-biased overlap between two RANKED LISTS (best first).

    Top-weighted by construction: the weight on depth ``d`` decays as ``p^d``,
    so disagreement about who is 1st costs far more than disagreement about who
    is 20th. That matches how a leaderboard is actually read, which Kendall tau
    -- uniform over all pairs -- does not.

    This is the extrapolated RBO of Webber et al. (2010) truncated at the depth
    of the shorter list, with the tail term assuming the observed overlap
    continues; for two orderings of the same set it returns exactly 1.0 when
    they agree and 0.0 when they are reversed disjoint prefixes.

    Args:
        a: ranked items, best first.
        b: ranked items, best first.
        p: persistence in ``(0, 1)``. Higher = more weight deeper down; 0.9
            puts roughly 86% of the weight on the top 10.

    Returns:
        RBO in ``[0, 1]``.

    Raises:
        ValueError: if ``p`` is not in ``(0, 1)``.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"rbo persistence p must be in (0, 1), got {p}")
    depth = min(len(a), len(b))
    if depth == 0:
        return 1.0 if len(a) == len(b) else 0.0
    seen_a: set[Any] = set()
    seen_b: set[Any] = set()
    total = 0.0
    overlap = 0
    for d in range(1, depth + 1):
        seen_a.add(a[d - 1])
        seen_b.add(b[d - 1])
        overlap = len(seen_a & seen_b)
        total += (p ** (d - 1)) * (overlap / d)
    return (1.0 - p) * total + (p**depth) * (overlap / depth)


def top_k_jaccard(a: Sequence[Any], b: Sequence[Any], k: int = 5) -> float:
    """Jaccard overlap of the top ``k`` of two ranked lists.

    The blunt version of `rbo`, and the one a reader can check by eye: "do the
    two conventions agree about the podium at all?"

    Args:
        a: ranked items, best first.
        b: ranked items, best first.
        k: prefix depth.

    Returns:
        ``|top_k(a) & top_k(b)| / |top_k(a) | top_k(b)|``, or ``1.0`` if both
        prefixes are empty.

    Raises:
        ValueError: if ``k < 1``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    sa, sb = set(a[:k]), set(b[:k])
    union = sa | sb
    return (len(sa & sb) / len(union)) if union else 1.0


@dataclass(frozen=True, slots=True)
class NoiseFloor:
    """What rank agreement pure sampling noise produces on this much data.

    THE reference point for every cross-convention claim. A between-convention
    tau of 0.85 is a large effect if two halves of the SAME corpus under the
    SAME convention agree at tau 0.98, and is indistinguishable from noise if
    they agree at 0.84. Reporting a between-convention tau without this number
    beside it is reporting an effect size with no scale.
    """

    tau_mean: float
    tau_median: float
    tau_p05: float
    tau_p95: float
    rbo_mean: float
    n_splits: int
    n_tokenizers: int
    n_clusters: int
    taus: np.ndarray = field(repr=False)
    rbos: np.ndarray = field(repr=False)

    def is_above_floor(self, tau: float) -> bool:
        """True if an observed tau is below the 5th percentile of the floor.

        Deliberately named for the direction that matters: a between-condition
        tau *below* the floor is real disagreement, and one inside the floor's
        range is a null result however far it is from 1.0.
        """
        return tau < self.tau_p05


def split_half_noise_floor(
    counts_by_tokenizer: Mapping[str, Any],
    statistic: Statistic = phi,
    n_splits: int = 200,
    seed: int = 0,
    higher_is_better: bool = True,
) -> NoiseFloor:
    """Rank agreement between two random halves of the SAME data.

    Same corpus, same convention, same tokenizers -- so every disagreement
    between the two halves is sampling noise and nothing else. The resulting
    distribution of tau is the ceiling any cross-convention comparison could
    ever reach, and the yardstick every reported tau must be read against.

    Args:
        counts_by_tokenizer: name -> per-sentence counts, matched sentences.
        statistic: pure function of the pooled `Counts`.
        n_splits: number of random half-splits.
        seed: PRNG seed.
        higher_is_better: ordering direction, for the RBO lists.

    Returns:
        A `NoiseFloor`.

    Raises:
        ValueError: if there are fewer than 2 tokenizers or 2 sentences.
    """
    names, mats = _ordered(counts_by_tokenizer)
    n_tok, n_sent = len(names), mats[0].shape[0]
    if n_tok < 2:
        raise ValueError("split_half_noise_floor needs at least 2 tokenizers")
    if n_sent < 2:
        raise ValueError("split_half_noise_floor needs at least 2 sentences")
    sign = -1.0 if higher_is_better else 1.0
    rng = np.random.default_rng(seed)
    taus = np.empty(n_splits, dtype=float)
    rbos = np.empty(n_splits, dtype=float)
    half = n_sent // 2
    for s in range(n_splits):
        perm = rng.permutation(n_sent)
        idx_a, idx_b = perm[:half], perm[half:]
        scores_a = np.array([statistic(pooled(m[idx_a])) for m in mats], dtype=float)
        scores_b = np.array([statistic(pooled(m[idx_b])) for m in mats], dtype=float)
        taus[s] = kendall_tau_b(scores_a.tolist(), scores_b.tolist())
        order_a = [names[i] for i in np.argsort(sign * scores_a, kind="stable")]
        order_b = [names[i] for i in np.argsort(sign * scores_b, kind="stable")]
        rbos[s] = rbo(order_a, order_b)
    return NoiseFloor(
        tau_mean=float(np.mean(taus)),
        tau_median=float(np.median(taus)),
        tau_p05=float(np.percentile(taus, 5.0)),
        tau_p95=float(np.percentile(taus, 95.0)),
        rbo_mean=float(np.mean(rbos)),
        n_splits=n_splits,
        n_tokenizers=n_tok,
        n_clusters=int(n_sent),
        taus=taus,
        rbos=rbos,
    )

# Enhanced

# Enhanced

# Enhanced

# Enhanced

# Updated

# Updated

# Enhanced

# Updated

# Refined

# Refined

# Enhanced

# Enhanced
