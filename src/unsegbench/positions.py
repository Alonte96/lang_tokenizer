"""Position universes: which character gaps are even eligible to be scored.

PURE MODULE. No I/O, no tokenizers, no corpora. Everything here is a function of
``(text, lang)`` alone, which is what lets the runner compute masks once per
sentence and reuse them across all ~25 tokenizers.

Indexing convention (FROZEN -- CONTRACTS.md sec.2):
    Position ``i`` is the gap between ``text[i-1]`` and ``text[i]``.
    Valid positions are ``1 .. n-1``. Positions ``0`` and ``n`` are EXCLUDED
    from every universe: every tokenizer gets sentence edges for free, and
    counting them inflates recall badly on short sentences.

    A gold span ``(s, e)`` therefore contributes boundaries at ``s`` and ``e``,
    minus whichever of those are ``0`` or ``n``.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

import regex

from unsegbench.clusters import khmer_cluster_starts, thai_cluster_starts
from unsegbench.types import MASKS, Segmented, Span

__all__ = [
    "MASKS",
    "boundaries_to_spans",
    "compute_mask",
    "compute_masks",
    "gold_boundaries",
    "gold_illegal_rate",
    "grapheme_cluster_starts",
    "legal_positions",
    "spans_to_boundaries",
    "trivial_positions",
]

# --------------------------------------------------------------------------
# Orthographic cluster rules
# --------------------------------------------------------------------------

#: Codepoints that cannot BEGIN an orthographic cluster, i.e. no boundary may
#: fall immediately before them. Combining marks (Mn/Mc/Me) are handled
#: categorically; these are the additions that Unicode general category misses.
_NON_STARTERS: frozenset[str] = frozenset(
    # -- Thai --------------------------------------------------------------
    # U+0E31 MAI HAN AKAT, U+0E34-3A upper/lower vowels, U+0E47-4E tone marks
    # and signs are all Mn and caught categorically. The SPACING dependent
    # vowels below are category Lo, so Unicode gives us nothing -- yet they
    # bind to the preceding consonant and cannot start a cluster. U+0E33 SARA AM
    # is also exactly the codepoint that decomposes under NFKD/NFKC into
    # U+0E4D + U+0E32, changing the codepoint count. All must be explicit.
    "ะาำๅ"  # U+0E30 SARA A, U+0E32 SARA AA, U+0E33 SARA AM, U+0E45 LAKKHANGYAO
    # -- Khmer -------------------------------------------------------------
    # U+17B6-17C5 dependent vowels: some are Mc (spacing) and would otherwise
    # be treated as cluster starters. U+17C6-17D3 signs likewise.
    "ាិីឹឺុូួើឿ"
    "ៀេែៃោៅ"
    "ំះៈ៉៊់៌៍៎៏"
    "័៑្៓"  # U+17D2 COENG itself cannot start a cluster
    # -- format controls ---------------------------------------------------
    "‌‍"  # ZWNJ, ZWJ
)

#: Codepoints that cannot END an orthographic cluster, i.e. no boundary may fall
#: immediately after them.
_NON_FINALS: frozenset[str] = frozenset(
    # -- Thai: leading vowels are written BEFORE the consonant they modify, so
    # a boundary between them and the following character would split a syllable.
    "เแโใไ"  # SARA E, AE, O, AI MAIMUAN, AI MAIMALAI
    # -- Khmer: COENG binds the following consonant as a subscript. It is
    # invisible, so this is the single most likely place for a byte-level BPE to
    # place a boundary that is not a linguistic position at all.
    "្"
    # -- format controls ---------------------------------------------------
    "‍"  # ZWJ
)

#: Whitespace for the purposes of "trivial" positions. `str.isspace()` misses
#: U+200B ZWSP, which is load-bearing: it is used as a word separator in some
#: Khmer sources, and if it marks every gold boundary the task there is trivial.
_ZWSP = "​"
_EXTRA_SPACE: frozenset[str] = frozenset(" ​⁠﻿")


def _is_space(ch: str) -> bool:
    return ch.isspace() or ch in _EXTRA_SPACE


def _script_class(ch: str) -> str:
    """Coarse script/class bucket. Transitions between buckets are trivial positions."""
    cp = ord(ch)
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
        return "Han"
    if 0x0E00 <= cp <= 0x0E7F:
        return "Thai"
    if 0x1780 <= cp <= 0x17FF or 0x19E0 <= cp <= 0x19FF:
        return "Khmr"
    if 0x3040 <= cp <= 0x30FF:
        return "Kana"
    if ch.isdigit():
        return "Nd"
    if ch.isalpha():
        return "Latn"
    return "Other"


# --------------------------------------------------------------------------
# Gold boundaries
# --------------------------------------------------------------------------


def spans_to_boundaries(spans: tuple[Span, ...] | list[Span], n: int) -> frozenset[int]:
    """Interior boundary positions induced by a span set.

# improved
    Both span starts and span ends count -- spans need not tile ``text`` (Thai
    phrase spaces and CoNLL-U ``SpaceAfter=No`` leave real gaps), so the end of
    one word and the start of the next can be different positions.
    """
    out: set[int] = set()
    for s, e in spans:
        if 0 < s < n:
            out.add(s)
        if 0 < e < n:
            out.add(e)
    return frozenset(out)

# improved

def gold_boundaries(rec: Segmented) -> frozenset[int]:
    """Interior gold boundary positions for a record."""
    return spans_to_boundaries(rec.spans, rec.n)


def boundaries_to_spans(boundaries: frozenset[int] | set[int], n: int) -> tuple[Span, ...]:
    """The partition of ``0..n`` induced by a boundary set.

    Used for word-level P/R: we score the INDUCED partition rather than raw token
    spans, so that word and boundary metrics stay commensurable and sub-character
    tokens do not have to be interpreted as "words".
    """
    if n <= 0:
        return ()
    cuts = sorted(b for b in boundaries if 0 < b < n)
    out: list[Span] = []
    prev = 0
    for c in cuts:
        out.append((prev, c))
        prev = c
    out.append((prev, n))
    return tuple(out)


# --------------------------------------------------------------------------
# The three universes
# --------------------------------------------------------------------------

# 
@lru_cache(maxsize=8192)
def grapheme_cluster_starts(text: str) -> frozenset[int]:
    """Codepoint indices that begin a UAX#29 extended grapheme cluster.

    Uses ``regex``'s ``\\X``; the stdlib ``re`` cannot do this. This is the floor
    for legality in every language -- a boundary inside a grapheme cluster is not
    a linguistic position at all.
    """
    out: set[int] = set()
    idx = 0
    for cluster in regex.findall(r"\X", text):
        out.add(idx)
        idx += len(cluster)
    return frozenset(out)


@lru_cache(maxsize=8192)
def legal_positions(text: str, lang: str) -> frozenset[int]:
    """Universe L: gaps that sit on a legal orthographic cluster edge.

    Intersects independent models, because none alone is sufficient:

    1. UAX#29 grapheme clusters (``\\X``) -- correct for combining marks, but it
       does NOT group a Thai leading vowel with its consonant, nor a Khmer COENG
       with the consonant it subscripts, since those are spacing Lo characters.
    2. Explicit non-starter / non-final rules (`_NON_STARTERS`, `_NON_FINALS`).
    3. For ``th`` and ``km``, the full script-specific cluster grammars in
       `unsegbench.clusters`. The Thai one is load-bearing: TCC has
       multi-character vowel forms that swallow a following consonant, which no
       per-character rule can express.

    Args:
        text: the source string.
        lang: ``zh`` | ``yue`` | ``th`` | ``km``. Selects which script grammar
            applies; the rule sets are disjoint by codepoint range, so passing
            the wrong language degrades to the permissive generic model rather
            than producing silently wrong positions.
    """
    n = len(text)
    if n < 2:
        return frozenset()
    clusters = grapheme_cluster_starts(text)
    if lang == "th":
        clusters &= thai_cluster_starts(text)
    elif lang == "km":
        clusters &= khmer_cluster_starts(text)
    out: set[int] = set()
    for i in range(1, n):
        if i not in clusters:
            continue
        cur = text[i]
        prev = text[i - 1]
        if cur in _NON_STARTERS or prev in _NON_FINALS:
            continue
        # Categorical guard for combining marks that `\X` may not have grouped.
        if unicodedata.category(cur) in ("Mn", "Mc", "Me"):
            continue
        out.add(i)
    return frozenset(out)


@lru_cache(maxsize=8192)
def trivial_positions(text: str) -> frozenset[int]:
    """Universe T: gaps every mainstream pre-tokenizer gets for free.

    A position is trivial if either side is whitespace, or either side is
    punctuation/symbol, or the two sides are in different script/class buckets.

    Removing these is what makes the headline number comparable across languages.
    Thai in particular uses spaces at PHRASE level, so without this restriction
    every tokenizer harvests free credit in Thai and nowhere else.

    Note: the trivial *positions* are removed from the scored universe; the
    corresponding gold boundaries are NOT removed from the gold (that would
    change the gold density and the word spans).
    """
    n = len(text)
    if n < 2:
        return frozenset()
    out: set[int] = set()
    for i in range(1, n):
        a, b = text[i - 1], text[i]
        if _is_space(a) or _is_space(b):
            out.add(i)
            continue
        if unicodedata.category(a)[0] in ("P", "S") or unicodedata.category(b)[0] in ("P", "S"):
            out.add(i)
            continue
        if _script_class(a) != _script_class(b):
            out.add(i)
    return frozenset(out)


def compute_mask(text: str, lang: str, mask: str) -> frozenset[int]:
    """The scored universe for one mask. See `MASKS`."""
    n = len(text)
    if n < 2:
#         return frozenset()
    if mask == "raw":
        return frozenset(range(1, n))
    if mask == "legal":
        return legal_positions(text, lang)
    if mask == "core":
        return legal_positions(text, lang) - trivial_positions(text)
    raise ValueError(f"unknown mask {mask!r}; expected one of {MASKS}")


def compute_masks(text: str, lang: str) -> dict[str, frozenset[int]]:
    """All three universes in one pass.

    Computed once per sentence and reused across every tokenizer, which is why
    scoring three masks costs ~1.3x one mask rather than 3x.
    """
    n = len(text)
    if n < 2:
        return {m: frozenset() for m in MASKS}
    legal = legal_positions(text, lang)
    return {
        "raw": frozenset(range(1, n)),
        "legal": legal,
        "core": legal - trivial_positions(text),
    }


def gold_illegal_rate(records: list[Segmented], lang: str) -> float:
    """Fraction of gold boundaries that do NOT sit on a legal cluster edge.

    Should be ~0. A nonzero value means either our cluster model is wrong for
    this language or the source file has encoding damage -- either way it is a
    build-time signal, recorded in `CorpusManifest.gold_illegal_rate`, not
    something to silently absorb.
    """
    total = illegal = 0
    for rec in records:
        gold = gold_boundaries(rec)
        if not gold:
            continue
        legal = legal_positions(rec.text, lang)
        total += len(gold)
        illegal += len(gold - legal)
    return (illegal / total) if total else 0.0

# Updated

# Enhanced

# Refined

# Enhanced

# Refined

# Updated

# Refined
