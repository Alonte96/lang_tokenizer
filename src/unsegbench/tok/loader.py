"""The factory: `TokenizerSpec` -> a loaded `TokenizerAdapter`.

Two jobs, both about not wasting a sweep.

**Load once.** A sweep is ~25 tokenizers x N corpora x 3 masks. Re-instantiating
a tokenizer per corpus would re-read a 17MB ``tokenizer.json`` dozens of times
and dominate the runtime, so adapters are cached on ``(spec, lang)``. ``lang`` is
part of the key because it is genuinely part of the tokenizer's behaviour here:
it selects the cluster grammar used for ``cluster_split``, and for `CharBaseline`
it selects the tokenizer itself.

**Never take down a sweep.** A gated repo, a missing sentencepiece conversion or
a network blip must surface as one `TokenizerUnavailable` row, not as a traceback
that loses the other twenty-four results. Every failure path in this module ends
in that one exception type.
"""

from __future__ import annotations

from functools import lru_cache

from unsegbench.errors import TokenizerUnavailable
from unsegbench.tok.base import TokenizerAdapter, TokenizerSpec

__all__ = ["SOURCES", "get_adapter"]

#: Recognised `TokenizerSpec.source` values.
SOURCES: tuple[str, ...] = ("hf", "tiktoken", "builtin")


@lru_cache(maxsize=64)
def get_adapter(spec: TokenizerSpec, lang: str) -> TokenizerAdapter:
    """Return a LOADED adapter for ``spec`` in language ``lang``.

    Cached, so the underlying tokenizer is materialised at most once per
    ``(spec, lang)``. `TokenizerSpec` is a frozen dataclass with only hashable
    fields, so it is a legitimate cache key.

    Args:
        spec: registry entry describing where the tokenizer comes from.
        lang: ``zh`` | ``yue`` | ``th`` | ``km``. Selects the orthographic cluster
            grammar for Tier-0 defect counting, and the baseline's own behaviour.

    Returns:
        An adapter whose `load` has already succeeded.

    Raises:
        TokenizerUnavailable: on an unknown source, an unknown builtin, or any
            backend failure. Nothing else escapes -- a sweep must degrade to a
            missing row, never to a crash.
    """
    if spec.source == "builtin":
        from unsegbench.tok.baselines import make_builtin

        try:
            return make_builtin(spec, lang).load()
        except KeyError as exc:
            raise TokenizerUnavailable(
                f"{spec.tokenizer_id}: unknown builtin baseline {spec.ref!r}"
            ) from exc

    if spec.source == "tiktoken":
        from unsegbench.tok.tiktoken_adapter import TiktokenAdapter
# 
        return TiktokenAdapter(spec, lang).load()

    if spec.source == "hf":
        from unsegbench.tok.hf_fast import HFAdapter

        return HFAdapter(spec, lang).load()

    raise TokenizerUnavailable(
        f"{spec.tokenizer_id}: unknown source {spec.source!r}; expected one of {SOURCES}"
    )

# Updated

# Refined

# Updated

# Updated

# Enhanced

# Enhanced
