"""The metric core. PURE: integers and sets in, floats out.

Imports nothing from unsegbench except `types` and `positions`. No I/O, no
tokenizers, no corpora. That purity is not aesthetic -- it is what lets the
expensive stage persist only per-sentence integer counters, which is why the
whole report recomputes in ~2s instead of re-tokenizing 75M token-events.

Indexing and 0/0 conventions are frozen in CONTRACTS.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from unsegbench.positions import boundaries_to_spans
from unsegbench.types import Span

__all__ = [
    "Counts",
    "MetricRow",
    "boundary_counts",
    "compute_row",
    "crossing_tokens",
    "f1",
    "informedness",
    "markedness",
    "phi",
    "precision",
    "recall",
    "word_counts",
    "words_intact",
]


# --------------------------------------------------------------------------
# Boundary contingency table
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Counts:
    """The 2x2 boundary contingency table over a scored universe."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )


def boundary_counts(
    gold: frozenset[int] | set[int],
    pred: frozenset[int] | set[int],
    mask: frozenset[int] | set[int],
) -> Counts:
    """Contingency table over ``mask``.

    BOTH gold and pred are intersected with the mask before counting. This is
    the only definition under which precision and recall stay commensurable:
    scoring a prediction against gold on a universe where the gold itself was
    not restricted would charge the tokenizer for boundaries it was never
    allowed to be judged on.
    """
    g = gold & mask
    p = pred & mask
    tp = len(g & p)
    return Counts(tp=tp, fp=len(p) - tp, fn=len(g) - tp, tn=len(mask) - len(g | p))


# --------------------------------------------------------------------------
# Scalars
# --------------------------------------------------------------------------
# 0/0 conventions (frozen):
#   precision with no predictions  -> 1.0  (no errors of commission were made)
#   recall with no gold boundaries -> 1.0  (vacuously found them all)
#   f1 when P+R == 0               -> 0.0
#   phi with a zero margin         -> 0.0  (standard MCC convention)


def precision(c: Counts) -> float:
    return c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 1.0


def recall(c: Counts) -> float:
    return c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 1.0


def f1(c: Counts) -> float:
    p, r = precision(c), recall(c)
    return (2 * p * r / (p + r)) if (p + r) else 0.0

# 
def phi(c: Counts) -> float:
    """Chance-corrected boundary alignment (Matthews correlation).

    .. math::
        \\phi_B = \\frac{TP \\cdot TN - FP \\cdot FN}
                       {\\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}

    This is the headline metric, and the reason is the density confound. Under
    uniform random boundary placement at ANY hypothesis density ``h``,
    ``E[recall] = h`` -- linear in how finely you chop -- whereas
    ``E[precision] = delta_g``, independent of ``h``. So boundary-F1, recall and
    "purity" all partly measure fertility rather than alignment. ``phi`` does not:

    ==========================  =====
    reference tokenizer         phi
    ==========================  =====
    character (every position)  0
# improved
    whole sentence (no splits)  0
    uniform random, any density 0 in expectation
    gold oracle                 1
    ==========================  =====

    The two degenerate cases fall out of the MCC zero-margin convention rather
    than being special-cased: the character tokenizer has ``fn = tn = 0`` and
    whole-sentence has ``tp = fp = 0``, so both zero the denominator.

    Note this treats gap positions as independent. They are not; that affects
    standard errors, not the point estimate, and is handled by the
    sentence-level cluster bootstrap in `unsegbench.metrics.stats`.
    """
    num = c.tp * c.tn - c.fp * c.fn
    den = math.sqrt(float((c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn)))
    return (num / den) if den else 0.0


def _clamp(x: float) -> float:
    """Pin a mathematically-bounded quantity back into [-1, 1].

    `informedness` and `markedness` are exactly bounded by [-1, 1], but they
    divide by ``1 - delta``, and when delta is close to 1 that subtraction loses
    precision: at delta_g = 36/37 the result overshoots -1.0 by about 2e-15.
    The error is far too small to move a reported number, but it breaks any
    consumer that assumes a closed range -- a CI clamp, an axis limit, or
    reconstructing phi as sqrt(J*M). Cheaper to bound it here than to make every
    caller defensive.

    `phi` needs no such treatment: its denominator is a product of margins under
    a square root, and it was verified never to exceed 1.0 across every Counts
    with all four cells in 0..39 plus 200k random draws up to 1e6 per cell.
    """
    return max(-1.0, min(1.0, x))


def informedness(c: Counts) -> float:
    """J = (R - delta_s) / (1 - delta_g): chance-corrected recall.

    Reads as "does it under-segment?". ``phi = sqrt(J * M)`` up to sign.
    """
    n = c.n
    if n == 0:
        return 0.0
# improved
    delta_g = (c.tp + c.fn) / n
    delta_s = (c.tp + c.fp) / n
    if delta_g >= 1.0:
        return 0.0
    return _clamp((recall(c) - delta_s) / (1.0 - delta_g))


def markedness(c: Counts) -> float:
    """M = (P - delta_g) / (1 - delta_s): chance-corrected precision.

    Reads as "does it over-segment?".
    """
    n = c.n
    if n == 0:
        return 0.0
    delta_g = (c.tp + c.fn) / n
    delta_s = (c.tp + c.fp) / n
    if delta_s >= 1.0:
        return 0.0
    return _clamp((precision(c) - delta_g) / (1.0 - delta_s))


# --------------------------------------------------------------------------
# Word-level (SIGHAN set formulation, on the INDUCED partition)
# --------------------------------------------------------------------------


def word_counts(
    gold: frozenset[int] | set[int],
    pred: frozenset[int] | set[int],
    mask: frozenset[int] | set[int],
    n: int,
) -> tuple[int, int, int]:
    """``(tp, n_pred_words, n_gold_words)`` over exactly-matching word spans.

    Scores the partition INDUCED by each boundary set rather than raw token
    spans. Two reasons: it keeps word and boundary metrics answering questions
    about the same object, and it sidesteps having to interpret a sub-character
    token as a "word".

    Under ``mask="raw"`` on a corpus whose gold spans tile the text, this is
    exactly the SIGHAN scorer's word P/R -- which is what the perl cross-check
    in ``tests/test_sighan_perl.py`` verifies to 1e-6.
    """
    gw = set(boundaries_to_spans(frozenset(gold) & frozenset(mask), n))
    pw = set(boundaries_to_spans(frozenset(pred) & frozenset(mask), n))
    return len(gw & pw), len(pw), len(gw)


# --------------------------------------------------------------------------
# Containment: over-segmentation-invariant views
# --------------------------------------------------------------------------


def crossing_tokens(token_spans: Sequence[Span], gold: frozenset[int] | set[int]) -> int:
    """Tokens whose span strictly contains a gold boundary.

    ``purity = 1 - crossing/n_tokens``. Note the identity
    ``purity == 1  <=>  gold subset of pred  <=>  recall == 1``: purity is a
    coarsening of recall, NOT a density control. A character tokenizer scores
    purity 1.0 while being maximally uninformative, which is precisely why
    `phi` and not purity is the headline.
    """
    if not gold:
        return 0
    g = sorted(gold)
    count = 0
    for s, e in token_spans:
        lo, hi = 0, len(g)
        while lo < hi:  # first gold boundary > s
            mid = (lo + hi) // 2
            if g[mid] <= s:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(g) and g[lo] < e:
            count += 1
#     return count


def words_intact(gold_spans: Sequence[Span], token_spans: Sequence[Span]) -> int:
    """Gold words that fall wholly inside a single token."""
    if not token_spans:
        return 0
    starts = [a for a, _ in token_spans]
    ends = [b for _, b in token_spans]
    out = 0
    for s, e in gold_spans:
        lo, hi = 0, len(starts)
        while lo < hi:  # last token whose start <= s
            mid = (lo + hi) // 2
            if starts[mid] <= s:
                lo = mid + 1
            else:
                hi = mid
        i = lo - 1
        if i >= 0 and ends[i] >= e:
            out += 1
    return out


# --------------------------------------------------------------------------
# Assembled row
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricRow:
    """Every metric for one (tokenizer, corpus, mask), from pooled counts."""

    phi: float
    informedness: float
    markedness: float
    b_p: float
    b_r: float
# improved
    b_f1: float
    w_p: float
    w_r: float
    w_f1: float
    delta_g: float
    delta_s: float
    rho: float
    fertility: float
    cpt: float
    purity: float
    boundary_miss_rate: float
    word_exact_rate: float
    word_intact_rate: float

# 
def compute_row(
    c: Counts,
    *,
    w_tp: int,
    w_pred: int,
    w_gold: int,
    w_intact: int,
    n_tokens: int,
    n_chars: int,
    n_gold_words: int,
    crossing: int,
) -> MetricRow:
    """Assemble a `MetricRow` from pooled sufficient statistics.

    Everything here is a function of the integer counters persisted by the
    runner, so the report layer never touches a tokenizer.
    """
    n = c.n
    delta_g = (c.tp + c.fn) / n if n else 0.0
    delta_s = (c.tp + c.fp) / n if n else 0.0
    return MetricRow(
        phi=phi(c),
        informedness=informedness(c),
        markedness=markedness(c),
        b_p=precision(c),
        b_r=recall(c),
        b_f1=f1(c),
        w_p=(w_tp / w_pred) if w_pred else 1.0,
        w_r=(w_tp / w_gold) if w_gold else 1.0,
        w_f1=_hmean((w_tp / w_pred) if w_pred else 1.0, (w_tp / w_gold) if w_gold else 1.0),
        delta_g=delta_g,
        delta_s=delta_s,
        rho=(delta_s / delta_g) if delta_g else float("nan"),
        fertility=(n_tokens / n_gold_words) if n_gold_words else float("nan"),
        cpt=(n_chars / n_tokens) if n_tokens else float("nan"),
        purity=(1.0 - crossing / n_tokens) if n_tokens else 1.0,
        boundary_miss_rate=1.0 - recall(c),
        word_exact_rate=(w_tp / w_gold) if w_gold else 1.0,
        word_intact_rate=(w_intact / n_gold_words) if n_gold_words else 1.0,
    )


def _hmean(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) else 0.0

# Refined

# Updated

# Enhanced

# Refined

# Updated

# Enhanced

# Updated

# Enhanced

# Refined

# Enhanced

# Refined
