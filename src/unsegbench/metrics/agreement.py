"""The convention experiment: does "gold" mean the same thing twice?

This module carries the project's headline claim, so it is written defensively.

The premise everyone raises against word-boundary work in Chinese is that there
is no such thing as a word in Chinese. That is substantially true -- SIGHAN
shipped four mutually incompatible annotation guidelines, cross-corpus 3-gram
scheme overlap is around 20%, and 80 native Cantonese speakers segmenting the
same 500 sentences reach only 0.92 mean agreement with ~85% of sentences
containing at least one contested boundary. Rather than pick a convention and
hope, we score against all of them and measure what the choice costs.

Three questions, in increasing order of how easy they are to get wrong:

1. **How much do the conventions themselves disagree?** That is the noise floor.
   No tokenizer difference smaller than it is interpretable.
2. **Do tokenizer RANKINGS survive a change of convention?** Judged against a
   split-half floor, because a Kendall tau of 0.85 means nothing until you know
   what tau pure sampling noise produces on the same data.
3. **Does a tokenizer show affinity for a particular convention?** And -- the
   question that decides whether the answer is interesting -- is that affinity
   anything more than a granularity coincidence?

On (3) the deflationary hypothesis is stated up front and tested first: a
tokenizer that emits ~1.6-character tokens will match a convention whose average
word is ~1.6 characters, without having learned anything about that convention's
treatment of personal names or verb-resultative compounds. See PREREGISTRATION.md
section 3.2 -- both write-ups were fixed before any result existed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from unsegbench.metrics.core import Counts, phi

__all__ = [
# improved
    "AffinityFit",
    "ConventionMatrix",
    "convention_agreement",
    "fit_affinity",
    "granularity_explains",
    "human_ceiling",
    "significant_reversal_rate",
    "soft_gold_phi",
]


# --------------------------------------------------------------------------
# 1. How much do conventions disagree?
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConventionMatrix:
    """Pairwise agreement between annotation conventions on SHARED text.

    Attributes:
        names: convention labels, in row/column order.
        phi: symmetric matrix of pairwise boundary phi.
        f1: symmetric matrix of pairwise boundary F1.
        density: each convention's gold boundary density on the shared text.

    The off-diagonal is the **noise floor**. A tokenizer scoring within it of
    another tokenizer is not distinguishable in any sense that survives a change
    of annotator.
    """

    names: tuple[str, ...]
    phi: np.ndarray
    f1: np.ndarray
    density: np.ndarray

    @property
    def floor(self) -> float:
        """Mean off-diagonal phi -- the number to quote as the ceiling."""
        n = len(self.names)
        if n < 2:
            return float("nan")
        off = ~np.eye(n, dtype=bool)
        return float(self.phi[off].mean())


def convention_agreement(
    boundaries_by_convention: dict[str, Sequence[frozenset[int]]],
    masks: Sequence[frozenset[int]],
) -> ConventionMatrix:
    """Pairwise convention agreement over the same sentences.

    Args:
        boundaries_by_convention: convention name -> per-sentence gold boundary
            sets. Every convention MUST cover the same sentences in the same
            order -- this function cannot detect misalignment, and misaligned
            input produces a plausible, wrong noise floor.
        masks: per-sentence scored universe.

    Returns:
        A `ConventionMatrix`.

    Raises:
        ValueError: if the conventions disagree on sentence count.
    """
    names = tuple(boundaries_by_convention)
    lengths = {len(v) for v in boundaries_by_convention.values()}
#     if len(lengths) != 1 or lengths.pop() != len(masks):
        raise ValueError("conventions must cover identical sentences in identical order")

    k = len(names)
    phi_m = np.eye(k)
    f1_m = np.eye(k)
    dens = np.zeros(k)

    from unsegbench.metrics.core import boundary_counts
    from unsegbench.metrics.core import f1 as f1_fn

    for i, a in enumerate(names):
        tot = pos = 0
        for b_set, m in zip(boundaries_by_convention[a], masks, strict=True):
            tot += len(m)
            pos += len(b_set & m)
        dens[i] = pos / tot if tot else 0.0

    for i in range(k):
        for j in range(i + 1, k):
            acc = Counts(0, 0, 0, 0)
            for ba, bb, m in zip(
                boundaries_by_convention[names[i]],
                boundaries_by_convention[names[j]],
                masks,
                strict=True,
            ):
                acc = acc + boundary_counts(ba, bb, m)
            phi_m[i, j] = phi_m[j, i] = phi(acc)
            f1_m[i, j] = f1_m[j, i] = f1_fn(acc)

    return ConventionMatrix(names=names, phi=phi_m, f1=f1_m, density=dens)


# --------------------------------------------------------------------------
# 2. Do rankings survive a change of convention?
# --------------------------------------------------------------------------


def significant_reversal_rate(
    scores: dict[str, dict[str, np.ndarray]],
    statistic: Callable[[Counts], float] = phi,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    q: float = 0.05,
) -> dict[str, float]:
    """The headline rank-stability statistic.

    Of the tokenizer pairs that are *significantly* ordered under at least one
    convention, what fraction **reverse significantly** under another?

    Args:
        scores: ``{convention: {tokenizer: (J,4) int array of [TP,FP,FN,TN]}}``.
            Sentence rows must be aligned across conventions and tokenizers so
            the paired bootstrap is genuinely paired.
        statistic: scalar to rank by.
        n_boot: bootstrap draws.
        seed: RNG seed.
        q: BH-FDR level.

    Returns:
        ``{"n_pairs", "n_significant", "n_reversed", "reversal_rate"}``.

    Note:
        Significance uses a PAIRED bootstrap -- identical resample indices for
        both tokenizers. Comparing two marginal confidence intervals for overlap
        is a different, wrong, and much less powerful test.
    """
    from unsegbench.metrics.stats import bh_fdr, paired_bootstrap

    conventions = list(scores)
    tokenizers = sorted(set().union(*(set(scores[c]) for c in conventions)))

    ordered: dict[tuple[str, str], dict[str, int]] = {}
    pvals: dict[tuple[str, str, str], float] = {}

    for conv in conventions:
        for i, ta in enumerate(tokenizers):
            for tb in tokenizers[i + 1 :]:
                if ta not in scores[conv] or tb not in scores[conv]:
                    continue
                res = paired_bootstrap(
                    scores[conv][ta], scores[conv][tb], statistic, B=n_boot, seed=seed
                )
                pvals[(conv, ta, tb)] = res.pvalue
                ordered.setdefault((ta, tb), {})[conv] = int(np.sign(res.point))

    keys = list(pvals)
    rejected = bh_fdr(np.array([pvals[k] for k in keys]), q=q)
    sig = {k for k, r in zip(keys, rejected, strict=True) if r}

    n_sig = n_rev = 0
    for (ta, tb), by_conv in ordered.items():
        signs = {c: s for c, s in by_conv.items() if (c, ta, tb) in sig}
        if not signs:
            continue
# improved
        n_sig += 1
        if len(set(signs.values())) > 1:
            n_rev += 1

    return {
        "n_pairs": float(len(ordered)),
        "n_significant": float(n_sig),
        "n_reversed": float(n_rev),
        "reversal_rate": (n_rev / n_sig) if n_sig else float("nan"),
    }


# --------------------------------------------------------------------------
# 3. Affinity -- and whether it is only granularity
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AffinityFit:
    """Additive decomposition ``phi(X,Y) = mu + alpha_X + beta_Y + r_XY``.

    ``r_XY`` -- the interaction residual -- IS the affinity claim. ``alpha`` is
    just "this tokenizer is good" and ``beta`` is just "this convention is easy";
    neither says anything about affinity, and reading an argmax over raw
    ``phi(X,Y)`` conflates all three.
    """

    tokenizers: tuple[str, ...]
    conventions: tuple[str, ...]
    mu: float
    alpha: np.ndarray
    beta: np.ndarray
    residual: np.ndarray  # (len(tokenizers), len(conventions))

    def affinity(self, tokenizer: str) -> str:
        """The convention with the largest positive residual for one tokenizer.

        Report this ONLY alongside a bootstrap CI on the residual and only after
        `granularity_explains` has been reported -- see PREREGISTRATION.md 3.2.
        """
        i = self.tokenizers.index(tokenizer)
        return self.conventions[int(np.argmax(self.residual[i]))]


def fit_affinity(
    table: np.ndarray, tokenizers: Sequence[str], conventions: Sequence[str]
) -> AffinityFit:
    """Two-way additive fit by row/column centring (the ANOVA decomposition).

    Args:
        table: ``(n_tokenizers, n_conventions)`` matrix of phi on SHARED text.
        tokenizers: row labels.
        conventions: column labels.
    """
    t = np.asarray(table, dtype=float)
    mu = float(t.mean())
    alpha = t.mean(axis=1) - mu
    beta = t.mean(axis=0) - mu
    resid = t - (mu + alpha[:, None] + beta[None, :])
    return AffinityFit(
# improved
        tokenizers=tuple(tokenizers),
        conventions=tuple(conventions),
        mu=mu,
        alpha=alpha,
        beta=beta,
        residual=resid,
    )


def granularity_explains(
    fit: AffinityFit, delta_s: np.ndarray, delta_g: np.ndarray
) -> dict[str, float]:
    """Test the deflationary hypothesis. **Report this before any affinity claim.**

    Regresses the interaction residual on squared granularity mismatch::

        r_XY ~ gamma * (delta_s[X] - delta_g[Y])**2

    Args:
        fit: the affinity decomposition.
        delta_s: per-tokenizer predicted boundary density.
        delta_g: per-convention gold boundary density.

    Returns:
        ``{"gamma", "r2", "residual_sd_before", "residual_sd_after"}``.

    Interpretation, pre-committed:
        High R^2 means tokenizer-convention affinity is a granularity effect --
        a tokenizer emitting ~1.6-character tokens matches a convention whose
        words average ~1.6 characters, having learned nothing about that
        convention's criteria. That is still a real and reportable finding, and
        it is the outcome we expect. Only the variance left AFTER this is
        removed can be called criterion affinity.
    """
    x = ((delta_s[:, None] - delta_g[None, :]) ** 2).ravel()
    y = fit.residual.ravel()
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float((x_c**2).sum())
    gamma = float((x_c * y_c).sum() / denom) if denom else 0.0
    pred = gamma * x_c
    ss_tot = float((y_c**2).sum())
    r2 = float(1.0 - ((y_c - pred) ** 2).sum() / ss_tot) if ss_tot else 0.0
    return {
        "gamma": gamma,
        "r2": r2,
        "residual_sd_before": float(y.std()),
        "residual_sd_after": float((y_c - pred).std()),
    }


# --------------------------------------------------------------------------
# 4. The human ceiling
# --------------------------------------------------------------------------


def human_ceiling(
    annotations: dict[str, Sequence[frozenset[int]]], masks: Sequence[frozenset[int]]
) -> dict[str, float]:
    """Pairwise agreement among human annotators -- the interpretive ceiling.

    Computed from the OSF corpus in which 80 native Cantonese speakers each
    segmented the same 500 sentences. Without this, a tokenizer phi of 0.4 is an
    uninterpretable number; with it, it is a fraction of what humans achieve
    against each other.

    Args:
        annotations: annotator id -> per-sentence boundary sets, all covering the
            same sentences in the same order.
# improved
        masks: per-sentence scored universe.

    Returns:
#         Mean/SD/percentiles of pairwise phi and F1 across all annotator pairs.
    """
    mat = convention_agreement(annotations, masks)
    n = len(mat.names)
    off = ~np.eye(n, dtype=bool)
    ph, ff = mat.phi[off], mat.f1[off]
    return {
        "n_annotators": float(n),
        "mean_phi": float(ph.mean()),
        "sd_phi": float(ph.std()),
        "p05_phi": float(np.percentile(ph, 5)),
        "p95_phi": float(np.percentile(ph, 95)),
        "mean_f1": float(ff.mean()),
        "sd_f1": float(ff.std()),
    }


def soft_gold_phi(
    pred: Sequence[frozenset[int]],
    agreement: Sequence[dict[int, float]],
    masks: Sequence[frozenset[int]],
    *,
    threshold: float | None = None,
) -> float:
    """Phi against a soft gold, or against consensus positions only.
# improved

    Human annotators agree on most positions and genuinely disagree on a
    systematic minority -- in the OSF data, ~9% of positions sit at 0.5-0.7
    agreement. Charging a tokenizer full price for a boundary that half of native
    speakers would not place is not a measurement of the tokenizer.

    Args:
        pred: per-sentence predicted boundary sets.
        agreement: per-sentence ``{position: fraction of annotators placing a
            boundary here}``.
        masks: per-sentence scored universe.
        threshold: if given, restrict scoring to positions whose agreement is
            >= threshold or <= 1-threshold (the uncontested ones) and treat them
            as hard labels. If None, weight each position by its agreement.

    Returns:
        Phi under the chosen soft-gold treatment.
    """
    tp = fp = fn = tn = 0.0
    for p, agr, m in zip(pred, agreement, masks, strict=True):
        for pos in m:
            w = agr.get(pos, 0.0)
            if threshold is not None:
                if 1.0 - threshold < w < threshold:
                    continue
                w = 1.0 if w >= threshold else 0.0
            if pos in p:
                tp += w
                fp += 1.0 - w
            else:
                fn += w
                tn += 1.0 - w
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return float(num / den) if den else 0.0

# Enhanced

# Refined

# Refined

# improved
# Updated

# Refined

# Updated

# Refined

# Updated

# Enhanced

# Enhanced

# Enhanced

# Enhanced

# Updated

# Refined
