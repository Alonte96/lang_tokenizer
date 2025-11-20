"""Canonical intermediate representation for unsegbench.

FROZEN CONTRACT. Every corpus loader emits `Segmented`; every metric consumes
derivations of it. Do not add fields without updating CONTRACTS.md and the
# improved
validator together.

Design rule: store the MINIMAL FAITHFUL encoding and nothing derivable.
`text` + `spans` is the whole truth. `words`, `boundaries`, `n` and every mask
are computed by pure functions in `positions.py` and are never serialised --
storing them would create three encodings of one fact with no guardian, and
`"".join(words) == text` is false for most of our corpora anyway (Thai carries
real phrase spaces, CoNLL-U has SpaceAfter=No).
"""

from __future__ import annotations

import gzip
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from unsegbench.errors import BuildValidationError

__all__ = [
    "MASKS",
    "STATS_COLUMNS",
    "CorpusManifest",
    "EncodeResult",
    "RowStats",
    "Segmented",
    "Span",
    "read_jsonl",
    "validate_corpus",
    "validate_record",
    "write_jsonl",
]

Span = tuple[int, int]

#: The three evaluation universes. See CONTRACTS.md sec.2.
#:   RAW   -- every interior gap position 1..n-1
#:   LEGAL -- gaps that sit on a legal orthographic cluster edge
#:   CORE  -- LEGAL minus trivial (whitespace/punct/script-transition) positions.  HEADLINE.
# improved
MASKS: tuple[str, ...] = ("raw", "legal", "core")


# --------------------------------------------------------------------------
# Gold data
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Segmented:
    """One gold-segmented sentence.

    Attributes:
        id: stable identifier, ``"<corpus_id>/<split>/<zero-padded index>"``.
        text: the EXACT source string a tokenizer will see. Never normalised at
            build time -- normalisation is a tokenizer-side concern. This is what
            protects Thai U+0E33 (SARA AM), which changes codepoint count under
            NFKD/NFKC, and Khmer marks, which NFKC reorders.
        spans: gold word spans as ``(start, end)`` CODEPOINT offsets into ``text``.
            Sorted, non-overlapping, ``0 <= start < end <= len(text)``. Need not
            cover ``text``: uncovered codepoints are inter-word gaps (spaces,
            punctuation) and must all be in the corpus's declared ``gap_charset``.
        meta: per-record provenance only. Corpus-level constants belong in
            `CorpusManifest`, not repeated here.
    """

    id: str
    text: str
    spans: tuple[Span, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Length of ``text`` in codepoints."""
        return len(self.text)

    @property
    def words(self) -> tuple[str, ...]:
        """Gold words. Derived, never stored."""
        return tuple(self.text[s:e] for s, e in self.spans)

    def to_json(self) -> bytes:
        payload: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "spans": [list(sp) for sp in self.spans],
        }
        if self.meta:
            payload["meta"] = self.meta
        return orjson.dumps(payload)

    @classmethod
    def from_json(cls, raw: bytes | str) -> Segmented:
        d = orjson.loads(raw)
        return cls(
            id=d["id"],
            text=d["text"],
            spans=tuple((int(a), int(b)) for a, b in d["spans"]),
            meta=d.get("meta", {}),
        )


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Corpus-level constants. Written once per corpus, never per record."""

    corpus_id: str
    lang: str  # ISO 639: zh | yue | th | km
    script: str  # ISO 15924: Hans | Hant | Thai | Khmr
    convention: str  # the annotation standard, e.g. "pku", "msr", "ud", "hkcancor-multi-d"
    license: str
    redistributable: bool
    source_url: str
    version: str
    splits: dict[str, str]  # split name -> sha256 of the .jsonl.gz
    n_sents: int
    n_words: int
    n_chars: int
    #: Codepoints permitted to be uncovered by any gold span. A build fails if a
    #: character outside this set is left uncovered -- that means the loader lost data.
    gap_charset: str
    #: Fraction of gold boundaries NOT on a legal cluster edge. Should be ~0; a
    #: nonzero value means our cluster model or the source encoding is wrong.
    gold_illegal_rate: float
    #: True if U+200B ZWSP appears in the text. For Khmer this is load-bearing:
    #: if ZWSP marks every gold boundary the task is trivial and the corpus is
    #: NOT usable for Tier-1 claims. See CONTRACTS.md sec.6.
    zwsp_present: bool
    builder_version: str
    notes: str = ""

    def to_json(self) -> bytes:
        from dataclasses import asdict

        return orjson.dumps(asdict(self), option=orjson.OPT_INDENT_2)

    @classmethod
    def from_json(cls, raw: bytes | str) -> CorpusManifest:
        return cls(**orjson.loads(raw))


# --------------------------------------------------------------------------
# Tokenizer output
# --------------------------------------------------------------------------

#: Vocabulary for `EncodeResult.flags`. Every flag becomes a results column.
#: Anything the tokenizer layer cannot express honestly goes here. NO SILENT REPAIR.
FLAG_KEYS: tuple[str, ...] = (
    "midcodepoint_split",  # token boundary fell strictly inside a UTF-8 codepoint
    "cluster_split",  # boundary inside a grapheme cluster / Khmer COENG sequence
    "overlap_rejected",  # offsets went backwards or overlapped; boundary discarded
    "dropped_chars",  # codepoints not covered by any accepted token span
    "prefix_space_trim",  # Metaspace/add_prefix_space leading-space attribution fixed
    "normaliser_mutated",  # tokenizer's normaliser changed the string; offsets suspect
)


@dataclass(frozen=True, slots=True)
class EncodeResult:
    """What a tokenizer adapter returns for one sentence.

    Attributes:
        spans: accepted token spans as codepoint offsets, strictly ascending and
            non-overlapping. Boundaries that failed the integrity guard are absent
            here and counted in ``flags`` instead.
        n_tokens: RAW token count, including tokens whose boundary was rejected.
            Fertility and chars-per-token are computed from this -- the count must
            not lie just because we could not place a boundary.
        flags: counters keyed by `FLAG_KEYS`.
    """

    spans: tuple[Span, ...]
    n_tokens: int
    flags: Counter[str] = field(default_factory=Counter)

    @property
    def boundaries(self) -> frozenset[int]:
        """Interior boundary positions induced by the accepted spans.

        The start of the first span and the end of the last are NOT boundaries --
        sentence edges are excluded from every universe (CONTRACTS.md sec.2).
        """
        if not self.spans:
            return frozenset()
        return frozenset(s for s, _ in self.spans[1:])


# --------------------------------------------------------------------------
# # Sufficient statistics -- the ONLY thing the expensive stage persists
# --------------------------------------------------------------------------

#: Column order for the per-sentence stats parquet. Every table, plot, bootstrap,
#: Kendall tau and rank interval is a groupby over these integers, so the whole
#: report costs ~2s with zero re-tokenization. Keep them integers.
STATS_COLUMNS: tuple[str, ...] = (
    "sent_id",
    "n_chars",
# improved
    "n_mask",  # |P|, the size of the scored universe for this sentence+mask
    "n_gold_words",
    "n_tokens",  # raw
#     "n_tokens_accepted",
    "b_tp",
    "b_fp",
    "b_fn",
    "b_tn",
    "w_tp",  # exactly-matching word spans
    "w_pred",  # |W_s| on the INDUCED partition
    "w_gold",  # |W_g|
    "w_intact",  # gold words wholly inside a single token
    "crossing_tokens",  # tokens whose span strictly contains a gold boundary
    "f_midcodepoint",
    "f_cluster_split",
    "f_overlap_rejected",
# improved
    "f_dropped_chars",
    "f_prefix_space_trim",
)


@dataclass(frozen=True, slots=True)
class RowStats:
    """Per-sentence sufficient statistics. Mirrors `STATS_COLUMNS` exactly."""

    sent_id: str
    n_chars: int
    n_mask: int
    n_gold_words: int
    n_tokens: int
    n_tokens_accepted: int
    b_tp: int
    b_fp: int
    b_fn: int
    b_tn: int
    w_tp: int
    w_pred: int
    w_gold: int
    w_intact: int
    crossing_tokens: int
    f_midcodepoint: int = 0
    f_cluster_split: int = 0
    f_overlap_rejected: int = 0
    f_dropped_chars: int = 0
    f_prefix_space_trim: int = 0

    def as_tuple(self) -> tuple[Any, ...]:
        return tuple(getattr(self, c) for c in STATS_COLUMNS)


# --------------------------------------------------------------------------
# JSONL codec
# --------------------------------------------------------------------------


def write_jsonl(path: Path, records: Iterable[Segmented]) -> int:
    """Write gzipped JSONL atomically. Returns the record count.

    Writes to ``<path>.part`` and renames on success, so a reader never sees a
    half-written corpus. If the iterable raises part-way -- which is the normal
    outcome when a loader hits an invalid record, since validation runs during
    iteration -- the staging file is removed rather than left behind. Without
    that cleanup a failed build leaves a plausible-looking ``.part`` next to the
    real data, and the next failure silently overwrites it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    count = 0
    try:
        with gzip.open(tmp, "wb", compresslevel=6) as fh:
            for rec in records:
                fh.write(rec.to_json())
                fh.write(b"\n")
                count += 1
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return count


def read_jsonl(path: Path) -> Iterator[Segmented]:
    """Stream gzipped (or plain) JSONL of `Segmented`."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Segmented.from_json(line)


# --------------------------------------------------------------------------
# Validation -- fail closed
# --------------------------------------------------------------------------


def validate_record(rec: Segmented, gap_charset: str | frozenset[str] | None = None) -> None:
    """Raise `BuildValidationError` if the record violates the IR contract.

    Checks: non-empty text; spans sorted, non-overlapping, in range, non-empty;
    and (when ``gap_charset`` is given) that every uncovered codepoint is a
    declared gap character. That last check is what catches a loader silently
    dropping content.
    """
    n = len(rec.text)
    if n == 0:
        raise BuildValidationError(f"{rec.id}: empty text")

    prev_end = 0
    for i, (s, e) in enumerate(rec.spans):
        if not (0 <= s < e <= n):
            raise BuildValidationError(f"{rec.id}: span {i} = ({s},{e}) out of range for n={n}")
        if s < prev_end:
            raise BuildValidationError(
                f"{rec.id}: span {i} = ({s},{e}) overlaps or precedes previous end {prev_end}"
            )
        prev_end = e

    if gap_charset is not None:
        allowed = frozenset(gap_charset)
        covered = bytearray(n)
        for s, e in rec.spans:
            for i in range(s, e):
                covered[i] = 1
        for i, flag in enumerate(covered):
            if not flag and rec.text[i] not in allowed:
                raise BuildValidationError(
                    f"{rec.id}: uncovered codepoint {rec.text[i]!r} (U+{ord(rec.text[i]):04X}) "
                    f"at {i} is not in gap_charset -- the loader lost data"
                )


def validate_corpus(
    records: Iterable[Segmented], gap_charset: str | frozenset[str] | None = None
) -> tuple[int, int, int]:
    """Validate every record. Returns ``(n_sents, n_words, n_chars)``."""
    n_sents = n_words = n_chars = 0
    for rec in records:
        validate_record(rec, gap_charset)
        n_sents += 1
        n_words += len(rec.spans)
        n_chars += len(rec.text)
    if n_sents == 0:
        raise BuildValidationError("corpus is empty")
    return n_sents, n_words, n_chars

# Updated

# Enhanced

# Updated

# Updated

# Refined

# Refined

# Updated

# Refined

# Refined

# Refined

# Updated

# Updated

# Enhanced

# Refined

# Enhanced
