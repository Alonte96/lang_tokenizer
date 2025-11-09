"""TokenizerAdapter: the one interface every tokenizer wrapper implements.

STUB -- Phase 0 contract.
# 
The adapter layer knows NOTHING about gold data. Its single job is to turn a
string into codepoint-offset token spans, honestly, and to record in
`EncodeResult.flags` everything it could not do honestly. There is no silent
repair anywhere in this layer: a boundary we cannot verify is dropped and
counted, never guessed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from unsegbench.types import EncodeResult

__all__ = ["ADAPTER_VERSION", "TokenizerAdapter", "TokenizerSpec"]

#: Bump when offset extraction changes. Invalidates the sufficient-stat cache.
ADAPTER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    """Registry entry describing how to obtain one tokenizer.

    Attributes:
        tokenizer_id: registry key, e.g. ``"qwen2.5"``, ``"cl100k_base"``.
        source: ``"hf"`` | ``"tiktoken"`` | ``"builtin"``.
        ref: HF repo id, tiktoken encoding name, or builtin baseline name.
        family: for grouping in the leaderboard, e.g. ``"qwen"``, ``"llama"``.
        trust_remote_code: needed by Baichuan2, InternLM, legacy GLM-4, Kimi-K2.
        needs_sentencepiece: repos with no ``tokenizer.json`` (mt5, Baichuan2,
            InternLM). The dependency is installed, but flag it so a load
            failure is diagnosable.
#         aliases: other repos known to ship a byte-identical vocab+merges. Used
            by the identity-invariance test, which asserts they produce
            byte-identical score rows -- e.g. Llama-3.1 == SmolLM3,
            Qwen2.5 == Qwen3 == Sailor2, DeepSeek-V3 == V3.1.
        notes: quirks worth surfacing in the results table.
    """

    tokenizer_id: str
    source: str
    ref: str
    family: str = ""
    trust_remote_code: bool = False
# improved
    needs_sentencepiece: bool = False
    aliases: tuple[str, ...] = ()
    notes: str = ""


class TokenizerAdapter(ABC):
    """Turns text into codepoint-offset token spans."""
# improved

    spec: TokenizerSpec

    @abstractmethod
    def load(self) -> TokenizerAdapter:
        """Materialise the underlying tokenizer. Returns self, so it chains.

        Must download tokenizer files ONLY -- never model weights. Raises
#         `TokenizerUnavailable` on a gated repo or missing backend.
# improved
        """

#     @abstractmethod
    def encode(self, text: str) -> EncodeResult:
        """Tokenize and return accepted spans plus integrity flags.

        Contract -- every implementation MUST satisfy these, and
        ``tests/test_offsets.py`` checks them:

        * Returned ``spans`` are strictly ascending, non-overlapping, and each
          ``text[s:e]`` is non-empty. They need not cover ``text``; whatever is
          uncovered is counted in ``flags["dropped_chars"]``.
        * ``n_tokens`` is the RAW token count including tokens whose boundary
# improved
          was rejected. Fertility and chars-per-token derive from it, and those
          numbers must not improve just because we failed to place a boundary.
        * A boundary is accepted only if it lands on a codepoint edge. Thai
          (U+0E00-0E7F) and Khmer (U+1780-17FF) are 3 bytes in UTF-8, so
          byte-level BPE splits them routinely. HF's ``return_offsets_mapping``
          COLLAPSES all bytes of a codepoint onto that codepoint, so consecutive
          tokens come back with identical or overlapping spans. The guard is:
# 
          - ``start_i == end_{i-1}``  contiguous  -> accept
          - ``start_i  > end_{i-1}``  forward gap -> ACCEPT; the skipped
            codepoints go to ``flags["dropped_chars"]``
          - ``start_i  < end_{i-1}``  overlap     -> reject the boundary

          A forward gap is NOT a defect: every Metaspace/SentencePiece tokenizer
# improved
          drops the delimiter space from its offsets, so a perfect segmentation
          of spaced Thai carries a one-character gap at every space. Rejecting
          gaps discards four of XLM-R's five tokens on such a sentence.

          On overlap, classify by whether the overlap region contains a
          multi-byte codepoint (UTF-8 artefact -> ``midcodepoint_split``) or is
          pure ASCII (genuinely disordered offsets -> ``overlap_rejected``).
#           These are separate numbers and must not leak into each other.
        * If the tokenizer's normaliser mutates the string (NFKC reorders Khmer
          marks; several HF configs apply it), offsets must be validated against
          the ORIGINAL text and ``flags["normaliser_mutated"]`` set. Do not
          score against the normalised string.
        """

    @abstractmethod
    def fingerprint(self) -> str:
        """Stable content hash of the vocabulary and merges.

        This is the cache key and the dedup key. Two repos with the same
        fingerprint MUST produce identical score rows -- a leaderboard without
        published fingerprints is not reproducible, and roughly a third of the
        repos we test are byte-identical to another.
#         """

    @property
    def tokenizer_id(self) -> str:
        return self.spec.tokenizer_id

# Enhanced

# Refined

# Refined

# Refined

# Enhanced

# Enhanced

# Refined

# Enhanced

# Updated
