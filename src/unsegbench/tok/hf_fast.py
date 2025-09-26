"""HuggingFace fast tokenizers, with every offset treated as a claim to verify.

``return_offsets_mapping`` looks like the answer to this whole problem and is
not. Three separate ways it lies about text in these scripts:

1. **The mid-codepoint collapse.** Offsets are CHARACTER offsets, and every byte
   of a multi-byte codepoint is collapsed onto that codepoint. Thai and Khmer are
   3 bytes per character, so a byte-level BPE that tears one Khmer consonant into
   two tokens reports two IDENTICAL char spans. Read naively that invents a
   boundary. Everything goes through `offsets.accept_spans`.

2. **Prefix-space attribution.** Metaspace tokenizers emit a bare ``"▁"``
   token for a word-initial space, and where no space exists in the source (the
   start of the string) HuggingFace still has to give it an offset -- so it
   assigns the FIRST CHARACTER OF THE FOLLOWING WORD. XLM-R on
   ``"สวัสดี ..."`` returns ``("▁", (0,1))``
   overlapping ``("สว", (0,2))``. Left alone, the guard rejects the
   real token and keeps the fake one, inventing a boundary one character into the
   first word. Such tokens are zero-width in the source and are trimmed, counted
   in ``flags["prefix_space_trim"]``.

3. **Normaliser mutation.** Several configs apply NFC/NFKC, which reorders Khmer
   marks and changes the codepoint count for Thai U+0E33. When the normaliser
   changes the string, the offsets are indices into a string we are not scoring.
   We flag it and still score against the ORIGINAL text (CONTRACTS.md sec.4) --
   never against the normalised copy.

Also handled: ``pre_tokenizer: null`` repos (Yi, MiniCPM4), where a token may
span anything at all including across whitespace, so no structural assumption
about token shape is safe.

Downloads tokenizer files ONLY. Never weights.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import orjson

from unsegbench.errors import TokenizerUnavailable
from unsegbench.tok.base import TokenizerAdapter, TokenizerSpec
from unsegbench.tok.offsets import accept_spans, count_cluster_splits
from unsegbench.types import EncodeResult, Span

__all__ = ["HFAdapter"]

#: Characters a token may be built from and still be "just a space marker":
#: U+2581 LOWER ONE EIGHTH BLOCK (Metaspace), U+0120 / U+010A / U+0109 (ByteLevel's
#: printable stand-ins for space, LF and TAB) and literal whitespace.
_SPACE_MARKERS: frozenset[str] = frozenset("▁ĠĊĉ \t\r\n")


class HFAdapter(TokenizerAdapter):
    """A `transformers` fast tokenizer, with its offsets independently verified."""

    def __init__(self, spec: TokenizerSpec, lang: str) -> None:
        self.spec = spec
        self.lang = lang
        self._tok: Any | None = None
        self._fp: str | None = None
        self._normalizer: Any | None = None

    # ------------------------------------------------------------------
    def load(self) -> HFAdapter:
        """Materialise the fast tokenizer. Raises `TokenizerUnavailable` on failure.

        A slow-only tokenizer is unusable here rather than merely slow: without a
        Rust backend there is no ``offset_mapping`` at all, and guessing offsets
        by re-searching decoded pieces is exactly the silent repair this layer
        exists to prevent.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise TokenizerUnavailable(
                f"{self.spec.tokenizer_id}: transformers not installed"
            ) from exc
        try:
            tok = AutoTokenizer.from_pretrained(
                self.spec.ref,
                use_fast=True,
                trust_remote_code=self.spec.trust_remote_code,
            )
        except Exception as exc:
            hint = " (needs sentencepiece)" if self.spec.needs_sentencepiece else ""
            raise TokenizerUnavailable(
                f"{self.spec.tokenizer_id}: could not load {self.spec.ref!r}{hint}: {exc}"
            ) from exc
        if not getattr(tok, "is_fast", False) or not hasattr(tok, "backend_tokenizer"):
            raise TokenizerUnavailable(
                f"{self.spec.tokenizer_id}: {self.spec.ref!r} has no fast backend, "
# improved
                "so no offset mapping is available"
            )
        self._tok = tok
        self._normalizer = getattr(tok.backend_tokenizer, "normalizer", None)
        return self

    # ------------------------------------------------------------------
    def _trim_prefix_space(
        self, tokens: list[str], offsets: list[tuple[int, int]]
    ) -> tuple[list[Span], int]:
        """Zero out space-marker tokens whose span overlaps a neighbour.

        A bare space marker whose reported span overlaps the adjacent token is not
        describing any codepoint of the source -- it is HuggingFace attributing a
        zero-width prefix space to a character that belongs to the next word. A
        space marker that overlaps nothing IS a real space in the source (gemma's
        ``("▁", (4,5))`` on a genuine space) and is left alone.
        """
        out: list[Span] = []
        trimmed = 0
        for i, (s, e) in enumerate(offsets):
            if s < e and set(tokens[i]) <= _SPACE_MARKERS:
                prev_end = offsets[i - 1][1] if i else None
                next_start = offsets[i + 1][0] if i + 1 < len(offsets) else None
                overlaps_next = next_start is not None and next_start < e
                overlaps_prev = prev_end is not None and s < prev_end
                if overlaps_next or overlaps_prev:
                    out.append((s, s))
                    trimmed += 1
                    continue
            out.append((s, e))
        return out, trimmed

    def encode(self, text: str) -> EncodeResult:
        """Tokenize and return verified codepoint spans plus integrity flags."""
        if self._tok is None:
            self.load()
        assert self._tok is not None
        if not text:
            return EncodeResult(spans=(), n_tokens=0, flags=Counter())

        flags: Counter[str] = Counter()
        if self._normalizer is not None:
            try:
                if self._normalizer.normalize_str(text) != text:
                    flags["normaliser_mutated"] += 1
            except Exception:  # pragma: no cover - some normalisers are not exposed
                pass

        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        tokens = self._tok.convert_ids_to_tokens(ids)

        offsets, trimmed = self._trim_prefix_space(tokens, offsets)
        if trimmed:
            flags["prefix_space_trim"] += trimmed

        spans, span_flags = accept_spans(offsets, text)
        flags += span_flags

        splits = count_cluster_splits(spans, text, self.lang)
        if splits:
            flags["cluster_split"] += splits
        return EncodeResult(spans=spans, n_tokens=len(ids), flags=flags)

    # ------------------------------------------------------------------
    def fingerprint(self) -> str:
        """sha256 over ``model.vocab`` and ``model.merges`` only.

        Deliberately NOT a hash of ``tokenizer.json``. Mirrors of the same
        tokenizer differ in added tokens, chat templates and padding config while
        sharing a byte-identical vocabulary -- hashing the whole file would report
        them as different tokenizers and we would benchmark the same thing twice.

        Merges are also normalised across the two serialisation forms the
        ``tokenizers`` library emits: older files store ``"a b"`` strings, newer
        ones ``["a","b"]`` pairs. unsloth and NousResearch mirrors of Llama-3.1
        differ in exactly this way and in nothing else. Pairs are joined rather
        than strings split, because splitting is ambiguous if a piece can contain
        a space, whereas joining never is.

        Merge ORDER is preserved -- position in the list is the rank, so it is
        content. Only the vocabulary is sorted.
        """
        if self._fp is not None:
            return self._fp
        if self._tok is None:
            self.load()
        assert self._tok is not None
        try:
            payload = orjson.loads(self._tok.backend_tokenizer.to_str())
        except Exception as exc:
            raise TokenizerUnavailable(
                f"{self.spec.tokenizer_id}: cannot serialise backend tokenizer: {exc}"
            ) from exc

        model = payload.get("model") or {}
        canonical = {
            "type": model.get("type", ""),
            "vocab": _canonical_vocab(model.get("vocab")),
            "merges": [_canonical_merge(m) for m in (model.get("merges") or [])],
        }
        self._fp = hashlib.sha256(orjson.dumps(canonical, option=orjson.OPT_SORT_KEYS)).hexdigest()
        return self._fp


def _canonical_merge(merge: Any) -> str:
    """Normalise one merge rule to the ``"a b"`` string form."""
    if isinstance(merge, str):
        return merge
    if isinstance(merge, (list, tuple)):
        return " ".join(str(part) for part in merge)
    return str(merge)  # pragma: no cover - no third form exists in the wild


def _canonical_vocab(vocab: Any) -> list[Any]:
    """Sort a vocabulary into a form independent of serialisation order.

    BPE/WordPiece store ``{token: id}``; Unigram stores ``[[piece, score], ...]``.
    Both are reduced to a sorted list of ``[token, value]``.
    """
    if isinstance(vocab, dict):
        return sorted(([str(k), v] for k, v in vocab.items()), key=lambda kv: kv[0])
    if isinstance(vocab, list):
        return sorted(
            ([str(e[0]), e[1]] if isinstance(e, (list, tuple)) and len(e) >= 2 else [str(e), 0])
            for e in vocab
        )
    return []  # pragma: no cover - every model type has one of the two forms

# Enhanced

# Refined

# Refined
