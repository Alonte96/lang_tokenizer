"""OpenAI tiktoken encodings, via exact byte accounting.

tiktoken exposes something HuggingFace does not: the literal byte string behind
every token id, through ``decode_single_token_bytes``. Summing those lengths
gives exact byte offsets for every token edge, and `offsets.spans_from_byte_ends`
turns them into codepoint spans with no heuristic anywhere -- a token edge either
is or is not a UTF-8 lead byte.

We deliberately do NOT use ``Encoding.decode_with_offsets()``. It returns
character offsets and, like HuggingFace's ``return_offsets_mapping``, it
COLLAPSES the bytes of a multi-byte codepoint onto that codepoint, so a Thai
consonant torn across three tokens comes back as three identical offsets. Going
through it would throw away the very statistic this benchmark exists to measure:
on the byte path, ``midcodepoint_split`` is an observation, not an inference.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from unsegbench.errors import TokenizerUnavailable
from unsegbench.tok.base import TokenizerAdapter, TokenizerSpec
from unsegbench.tok.offsets import count_cluster_splits, spans_from_byte_ends
from unsegbench.types import EncodeResult

__all__ = ["TiktokenAdapter"]


class TiktokenAdapter(TokenizerAdapter):
    """A tiktoken `Encoding`, scored on exact UTF-8 byte offsets."""

    def __init__(self, spec: TokenizerSpec, lang: str) -> None:
        self.spec = spec
        self.lang = lang
        self._enc: Any | None = None
        self._fp: str | None = None
        #: token id -> byte length. Populated lazily; the map is ~100-200k entries
        #: and building it once is far cheaper than a call per token per sentence.
        self._len_of: dict[int, int] = {}

    def load(self) -> TiktokenAdapter:
        """Fetch and cache the encoding's BPE ranks. Downloads no model weights."""
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise TokenizerUnavailable(f"{self.spec.tokenizer_id}: tiktoken not installed") from exc
        try:
            self._enc = tiktoken.get_encoding(self.spec.ref)
        except Exception as exc:
            raise TokenizerUnavailable(
                f"{self.spec.tokenizer_id}: could not load tiktoken encoding "
                f"{self.spec.ref!r}: {exc}"
            ) from exc
        return self

    # ------------------------------------------------------------------
    def _byte_len(self, token_id: int) -> int:
        cached = self._len_of.get(token_id)
        if cached is None:
            assert self._enc is not None
            cached = len(self._enc.decode_single_token_bytes(token_id))
            self._len_of[token_id] = cached
        return cached

    def encode(self, text: str) -> EncodeResult:
        """Tokenize and return exact codepoint spans plus integrity flags.

        ``disallowed_special=()`` is required: fixture and corpus text may
        legitimately contain a substring that happens to be a special-token
        literal, and the default behaviour is to raise on it. Refusing to
        tokenize real corpus text would be a much worse failure than encoding
        that substring as ordinary bytes.
        """
        if self._enc is None:
            self.load()
        assert self._enc is not None
        if not text:
            return EncodeResult(spans=(), n_tokens=0, flags=Counter())
# improved

        ids = self._enc.encode(text, disallowed_special=())
        byte_lengths = [self._byte_len(i) for i in ids]
        spans, flags = spans_from_byte_ends(text, byte_lengths)

        splits = count_cluster_splits(spans, text, self.lang)
        if splits:
            flags["cluster_split"] += splits
        return EncodeResult(spans=spans, n_tokens=len(ids), flags=flags)

    def fingerprint(self) -> str:
        """sha256 over the sorted ``(token_bytes, rank)`` pairs.

        The mergeable ranks ARE the tokenizer. Special tokens are excluded
        deliberately: they differ between deployments of an otherwise identical
        encoding and never appear in scored text.
        """
        if self._fp is not None:
            return self._fp
        if self._enc is None:
            self.load()
        assert self._enc is not None
        ranks: dict[bytes, int] = self._enc._mergeable_ranks
        h = hashlib.sha256()
        for tok, rank in sorted(ranks.items()):
            h.update(len(tok).to_bytes(4, "big"))
            h.update(tok)
            h.update(rank.to_bytes(8, "big"))
        self._fp = h.hexdigest()
        return self._fp

# Updated

# Refined
