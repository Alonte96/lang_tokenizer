"""Offset integrity: turning a tokenizer's idea of "where" into codepoint spans.

PURE MODULE. No tokenizer imports, no I/O. Everything here is a function of
``(text, offsets)`` alone, which is what makes it unit-testable without a network.

This is the highest-risk module in the package, and the risk is not that it
crashes -- it is that it silently succeeds. A boundary placed one codepoint off,
or a boundary invented where the tokenizer never put one, produces a number that
looks completely plausible and is simply wrong, and every table downstream
inherits the error with no symptom.

Two independent hazards live here:

1. **The mid-codepoint collapse.** Thai (U+0E00-0E7F) and Khmer (U+1780-17FF)
   are 3 bytes in UTF-8, so a byte-level BPE splits them routinely. HuggingFace's
   ``return_offsets_mapping`` reports CHARACTER offsets, and it COLLAPSES every
   byte of a multi-byte codepoint onto that codepoint. Three tokens covering the
   three bytes of one Thai consonant therefore come back as three IDENTICAL char
   spans. Reading those naively invents two boundaries that do not exist and
   would score as a spectacular over-segmentation defect that never happened.
   The guard accepts ``start_i >= end_{i-1}`` and rejects only true overlaps.
   A forward gap is legitimate -- Metaspace tokenizers drop the delimiter space
   from their offsets -- and its skipped codepoints are recorded in
   ``dropped_chars`` rather than treated as a failed boundary. See `accept_spans`
   for the full table, and CONTRACTS.md section 4 for why the stricter
   equals-only reading is wrong.

2. **Byte accounting.** Where per-token byte lengths are obtainable (tiktoken via
   ``decode_single_token_bytes``, and any encoding whose tokens are byte strings)
   we do not need the collapse guard at all -- `spans_from_byte_ends` computes
   the answer EXACTLY and gets the mid-codepoint statistic for free, because a
   token edge either does or does not land on a UTF-8 lead byte. Prefer that
   path wherever it is available; `accept_spans` is the lossy fallback.

NO SILENT REPAIR. Every boundary this module cannot verify is dropped and
counted in `flags`, never guessed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from unsegbench.positions import legal_positions
from unsegbench.types import Span

__all__ = [
    "accept_spans",
    "byte_to_char_map",
    "count_cluster_splits",
    "dropped_chars",
    "spans_from_byte_ends",
]


# --------------------------------------------------------------------------
# UTF-8 byte <-> codepoint accounting
# --------------------------------------------------------------------------


def byte_to_char_map(text: str) -> tuple[list[int], list[bool]]:
    """Map every UTF-8 byte offset of ``text`` to the codepoint it belongs to.

    A byte offset ``p`` is a codepoint boundary iff the byte at ``p`` is not a
    UTF-8 continuation byte, i.e. ``(b[p] & 0xC0) != 0x80``. That single bit test
    is the whole of the mid-codepoint question, and it is why the byte path is
    exact where the character path can only guess.

    Args:
        text: the source string.

    Returns:
        ``(char_of_byte, is_char_boundary)``, both of length
        ``len(text.encode("utf-8")) + 1``. The final entry describes the
        end-of-string position: it is always a boundary and maps to ``len(text)``.
        For a non-boundary offset, ``char_of_byte[p]`` is the index of the
        codepoint that offset falls INSIDE.
    """
    char_of_byte: list[int] = []
    is_boundary: list[bool] = []
    for i, ch in enumerate(text):
        width = len(ch.encode("utf-8"))
        char_of_byte.extend([i] * width)
        is_boundary.append(True)
        if width > 1:
            is_boundary.extend([False] * (width - 1))
    char_of_byte.append(len(text))
    is_boundary.append(True)
    return char_of_byte, is_boundary


def _is_multibyte_region(text: str, start: int, end: int) -> bool:
    """True if any codepoint in ``text[start:end]`` needs more than one UTF-8 byte.
# improved

    Used only to distinguish an honest mid-codepoint collapse from a genuinely
    disordered offset list: an overlap inside pure ASCII cannot be a UTF-8
    artefact, so it must be reported as the different defect it is.
    """
    return any(ord(ch) > 0x7F for ch in text[start:end])


def _classify_overlap(text: str, start: int, end: int, prev_end: int) -> str:
    """Name the defect behind ``start < prev_end``.

    The question is whether the reported start can be explained by HuggingFace
    collapsing a multi-byte codepoint that the previous token already entered. So
    the region to inspect is the OVERLAP, ``text[start:min(end, prev_end)]``, not
    the whole span -- a token may legitimately straddle, beginning inside one
    codepoint and ending inside the next. bloom-560m does exactly this on Khmer:
    ``("áŀ¶áŀ", (3,5))`` followed by ``("ļáŀ", (4,6))``, where the second token
    holds the last byte of codepoint 3 plus the first two bytes of codepoint 4.
    Judging that by its end offset alone would file a routine UTF-8 artefact as a
    disordered offset list and quietly deflate the headline mid-codepoint rate.
    """
    if _is_multibyte_region(text, start, min(end, prev_end)):
        return "midcodepoint_split"
    return "overlap_rejected"


# --------------------------------------------------------------------------
# The exact path: per-token byte lengths
# --------------------------------------------------------------------------


def spans_from_byte_ends(
    text: str, byte_lengths: Sequence[int]
) -> tuple[tuple[Span, ...], Counter[str]]:
    """Exact codepoint spans from per-token UTF-8 byte lengths.

    This is the preferred path. Given the byte length of each token in order, the
    token edges are exact byte offsets, and a boundary is legitimate iff its byte
    offset is a UTF-8 lead byte. No heuristic is involved and the
    ``midcodepoint_split`` count is a direct observation rather than an inference.

    Token edges that fall inside a codepoint yield no boundary: the adjacent
    tokens are merged into one accepted span, so the returned spans still tile
    whatever prefix of the text the tokenizer accounted for.

    Args:
        text: the source string.
        byte_lengths: UTF-8 byte length of each token, in token order.

    Returns:
        ``(spans, flags)``. ``flags`` carries ``midcodepoint_split`` (token edges
        strictly inside a codepoint) and ``dropped_chars`` (codepoints the token
        stream did not account for, which is nonzero only when the tokenizer's
        byte total disagrees with the text -- a lossy or normalising encoding).
    """
    flags: Counter[str] = Counter()
    n = len(text)
    if n == 0:
        return (), flags

    char_of_byte, is_boundary = byte_to_char_map(text)
    n_bytes = len(char_of_byte) - 1

    cuts: list[int] = []
    pos = 0
    truncated = False
    for length in byte_lengths:
        pos += length
        if pos > n_bytes:
            # The token stream ran past the text: the encoding is not a faithful
            # byte-for-byte round trip (normalisation, or a lossy decode). Stop
            # here rather than fabricating offsets.
            truncated = True
            break
        if pos == n_bytes:
            break
        if is_boundary[pos]:
            cuts.append(char_of_byte[pos])
        else:
            flags["midcodepoint_split"] += 1

    end = char_of_byte[min(pos, n_bytes)] if truncated else n
    spans: list[Span] = []
    prev = 0
    for c in cuts:
        if c <= prev or c >= end:
            continue
        spans.append((prev, c))
        prev = c
    if prev < end:
        spans.append((prev, end))

    flags["dropped_chars"] += n - sum(e - s for s, e in spans)
    return tuple(spans), flags


# --------------------------------------------------------------------------
# The lossy path: character offsets from a tokenizer that already collapsed them
# --------------------------------------------------------------------------


def accept_spans(
    raw_spans: Iterable[Span], text: str, *, allow_gaps: bool = True
) -> tuple[tuple[Span, ...], Counter[str]]:
    """Filter a raw character-offset list down to boundaries we can actually verify.

    Walks the list in order keeping ``prev_end``, the end of the last ACCEPTED
    span, and applies the frozen guard from CONTRACTS.md sec.4:

    * ``start_i == prev_end`` -- the tokens meet exactly. Accept.
    * ``start_i > prev_end``  -- the tokenizer accounted for no token over the
      intervening codepoints. Accept the span; the gap is counted in
      ``dropped_chars``. See the note below on why this is not a rejection.
    * ``start_i < prev_end``  -- the spans overlap, so the reported boundary is
      not a codepoint edge. REJECT and count it. If the overlapping region
      contains a multi-byte codepoint this is HuggingFace's mid-codepoint
      collapse (``midcodepoint_split``); inside pure ASCII no UTF-8 artefact can
      explain it, so it is a genuinely disordered offset list
      (``overlap_rejected``).

    **On forward gaps.** CONTRACTS.md sec.4 states the guard only as
    ``start_i == end_{i-1}``, which read literally rejects forward gaps too. That
    reading is wrong and would be catastrophic: every Metaspace/SentencePiece
    tokenizer drops the delimiter space from its offsets, so XLM-R on
    ``"ฉัน ทำ งาน ที่ บ้าน"`` returns ``[(0,3),(4,6),(7,10),(11,14),(15,19)]`` --
    perfectly correct segmentation with four one-character gaps. Rejecting on the
    gap would discard four of its five tokens and report a near-perfect Thai
    tokenizer as a disaster. A gap is not an unverifiable boundary; it is a
    verified boundary plus unaccounted-for codepoints, and those codepoints are
    reported honestly in ``dropped_chars``. ``allow_gaps=False`` restores the
    literal reading for testing.

    Args:
        raw_spans: ``(start, end)`` codepoint offsets in token order, possibly
            containing duplicates, overlaps, zero-width entries and collapse.
        text: the ORIGINAL source string -- never a normalised copy.
        allow_gaps: accept a span whose start is beyond ``prev_end``. Default
            True; see above.

    Returns:
        ``(spans, flags)`` where ``spans`` are strictly ascending, non-overlapping
        and each ``text[s:e]`` is non-empty.
    """
    flags: Counter[str] = Counter()
    n = len(text)
    accepted: list[Span] = []
    prev_end = 0

    for s, e in raw_spans:
        if s == e:
            # Zero-width: a special token, or a prefix-space marker the adapter
            # already neutralised. It induces no boundary and is not a defect.
            continue
        if not (0 <= s < e <= n):
            flags["overlap_rejected"] += 1
            continue
        if s < prev_end:
            flags[_classify_overlap(text, s, e, prev_end)] += 1
            continue
        if s > prev_end and not allow_gaps:
            flags["overlap_rejected"] += 1
            continue
        accepted.append((s, e))
        prev_end = e

    flags["dropped_chars"] += n - sum(e - s for s, e in accepted)
    return tuple(accepted), flags


def dropped_chars(spans: Sequence[Span], text: str) -> int:
    """Codepoints of ``text`` covered by no span. Assumes ``spans`` are disjoint."""
    return len(text) - sum(e - s for s, e in spans)


# --------------------------------------------------------------------------
# Tier-0 defect: boundaries inside an orthographic cluster
# --------------------------------------------------------------------------


def count_cluster_splits(spans: Sequence[Span], text: str, lang: str) -> int:
    """Accepted token boundaries that are not on a legal orthographic cluster edge.

    A boundary between COENG and the consonant it subscripts, or between a Thai
    leading vowel and its consonant, is not a segmentation error -- it is not a
    linguistic position at all. Counting it as an ordinary false positive would
    hide a categorically different defect, so it feeds ``flags["cluster_split"]``
    and the Tier-0 ``cluster_violation_rate``.

    Uses the same boundary definition as `types.EncodeResult.boundaries`: the
    start of every span after the first, excluding the sentence edges, which are
    outside every scored universe.

    Args:
        spans: accepted token spans, ascending and non-overlapping.
        text: the original source string.
        lang: ``zh`` | ``yue`` | ``th`` | ``km``; selects the cluster grammar.
    """
    n = len(text)
    if n < 2 or len(spans) < 2:
        return 0
    legal = legal_positions(text, lang)
    return sum(1 for s, _ in spans[1:] if 0 < s < n and s not in legal)

# Enhanced

# Updated

# Enhanced

# Refined
