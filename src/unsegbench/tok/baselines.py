"""Reference tokenizers that bracket every real one. No network, no vocabulary.

These are not competitors -- they are the calibration of the scale, and their
job is to make a metric's failure modes visible rather than to score well.

`CharBaseline` predicts EVERY legal position and `WholeBaseline` predicts NONE.
Both must score ``phi == 0`` exactly (CONTRACTS.md sec.3): the first has
``fn == tn == 0`` and the second ``tp == fp == 0``, so each zeroes the MCC
margin. That is the property that makes ``phi`` the headline metric, because
boundary-F1 rates the character tokenizer near the top and recall rates it at a
perfect 1.0 while it carries no information whatsoever. If either of these ever
scores nonzero ``phi``, the metric layer is broken, not the baseline.

`CharBaseline` splits on legal orthographic CLUSTERS, not on codepoints. A
per-codepoint split would cut between a Khmer COENG and the consonant it
subscripts, which is not a linguistic position at all -- it would make the
"character tokenizer" a systematic Tier-0 defect generator and destroy the
identity ``recall == 1``.
"""

from __future__ import annotations

import hashlib
from collections import Counter

from unsegbench.positions import boundaries_to_spans, compute_mask
from unsegbench.tok.base import ADAPTER_VERSION, TokenizerAdapter, TokenizerSpec
from unsegbench.types import EncodeResult, Span

__all__ = [
    "BUILTINS",
    "CharBaseline",
    "WhitespaceBaseline",
    "WholeBaseline",
    "make_builtin",
]

#: Zero-width separators that `str.isspace()` misses. U+200B ZWSP is load-bearing:
#: it is a genuine word separator in several Khmer sources, and whether the
#: whitespace baseline picks it up is exactly the `zwsp_present` triviality signal
#: from CONTRACTS.md sec.6 -- a Khmer corpus where this baseline scores highly is
#: not usable for Tier-1 claims. Mirrors `positions._EXTRA_SPACE`.
_ZERO_WIDTH_SPACES: frozenset[str] = frozenset("​⁠﻿ ")


def _is_space(ch: str) -> bool:
    return ch.isspace() or ch in _ZERO_WIDTH_SPACES


class _Baseline(TokenizerAdapter):
    """Shared plumbing: builtins need no loading and have a trivial fingerprint."""

    def __init__(self, spec: TokenizerSpec, lang: str) -> None:
        self.spec = spec
        self.lang = lang

    def load(self) -> _Baseline:
        """No-op. Builtins have nothing to download and cannot fail."""
        return self

    def fingerprint(self) -> str:
        """Content hash of the baseline's definition.

        Includes ``lang``, because `CharBaseline` is a genuinely different
        tokenizer per language -- the cluster grammar is part of its definition.
        """
        payload = f"builtin/{self.spec.ref}/{self.lang}/v{ADAPTER_VERSION}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CharBaseline(_Baseline):
    """One token per legal orthographic cluster. Predicts the whole `legal` mask.

    Its boundary set is exactly ``compute_mask(text, lang, "legal")``, hence
    ``recall == 1.0`` on the ``legal`` and ``core`` universes (both are subsets
    of what it predicts) and ``phi == 0.0`` exactly.
    """

    def encode(self, text: str) -> EncodeResult:
        n = len(text)
        if n == 0:
            return EncodeResult(spans=(), n_tokens=0, flags=Counter())
        spans = boundaries_to_spans(compute_mask(text, self.lang, "legal"), n)
        return EncodeResult(spans=spans, n_tokens=len(spans), flags=Counter())


class WholeBaseline(_Baseline):
    """One token for the entire sentence. Predicts no interior boundary at all."""

    def encode(self, text: str) -> EncodeResult:
        n = len(text)
        if n == 0:
            return EncodeResult(spans=(), n_tokens=0, flags=Counter())
        return EncodeResult(spans=((0, n),), n_tokens=1, flags=Counter())


class WhitespaceBaseline(_Baseline):
    """Split on whitespace runs only; the delimiters themselves are not tokens.

    The floor for "how much of this task is free". In Thai, spaces are placed at
    PHRASE level, so this baseline picks up real gold boundaries without knowing
    anything -- which is precisely why the `core` mask strips those positions
    before any headline number is computed.
    """

    def encode(self, text: str) -> EncodeResult:
        n = len(text)
        spans: list[Span] = []
        start: int | None = None
        for i, ch in enumerate(text):
            if _is_space(ch):
                if start is not None:
                    spans.append((start, i))
                    start = None
            elif start is None:
                start = i
        if start is not None:
            spans.append((start, n))
        flags: Counter[str] = Counter()
        dropped = n - sum(e - s for s, e in spans)
        if dropped:
            flags["dropped_chars"] = dropped
        return EncodeResult(spans=tuple(spans), n_tokens=len(spans), flags=flags)


#: Builtin ``ref`` -> implementation. Keys match `registry.ENTRIES`.
BUILTINS: dict[str, type[_Baseline]] = {
    "char": CharBaseline,
    "whole": WholeBaseline,
    "whitespace": WhitespaceBaseline,
}


def make_builtin(spec: TokenizerSpec, lang: str) -> _Baseline:
    """Instantiate a builtin baseline. Raises `KeyError` on an unknown ``ref``."""
    return BUILTINS[spec.ref](spec, lang)

# Updated

# Updated

# Enhanced
