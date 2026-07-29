"""Corpus registry.

MERGE DISCIPLINE: this file is shared, so it must stay tiny and stable. Each
loader module exposes a module-level ``ENTRIES: tuple[CorpusSpec, ...]`` and is
listed in `_MODULES` below. Nobody edits anyone else's entries, so eight agents
can add loaders in parallel without touching the same lines.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

from unsegbench.corpora.base import CorpusSpec

__all__ = ["SELECTORS", "all_corpora", "get_corpus", "resolve"]

_MODULES: tuple[str, ...] = (
    "unsegbench.corpora.sighan",
    "unsegbench.corpora.ud",
    "unsegbench.corpora.hkcancor",
    "unsegbench.corpora.wisesight",
    "unsegbench.corpora.khpos",
    "unsegbench.corpora.vistec",
    "unsegbench.corpora.alt_km",
    "unsegbench.corpora.osf_yue",
)

#: ``@``-prefixed names usable on the CLI.
SELECTORS: dict[str, str] = {
    "@all": "every registered corpus",
    "@zh": "Mandarin only",
    "@yue": "Cantonese only",
    "@th": "Thai only",
    "@km": "Khmer only",
    "@permissive": "only redistributable corpora -- the zero-licence-friction subset",
}


@lru_cache(maxsize=1)
def all_corpora() -> tuple[CorpusSpec, ...]:
    """Every registered corpus. Modules that do not import yet are skipped."""
    out: list[CorpusSpec] = []
    for mod_name in _MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue  # loader not built yet (Phase 1 in flight)
        out.extend(getattr(mod, "ENTRIES", ()))
    return tuple(out)


def get_corpus(corpus_id: str) -> CorpusSpec:
    for spec in all_corpora():
        if spec.corpus_id == corpus_id:
            return spec
    known = ", ".join(sorted(s.corpus_id for s in all_corpora())) or "(none registered)"
    raise KeyError(f"unknown corpus {corpus_id!r}; known: {known}")


def resolve(names: list[str] | tuple[str, ...]) -> tuple[CorpusSpec, ...]:
    """Expand a mix of corpus ids and ``@`` selectors."""
    out: list[CorpusSpec] = []
    seen: set[str] = set()
    for name in names:
        if name == "@all":
            batch: tuple[CorpusSpec, ...] = all_corpora()
        elif name == "@permissive":
            batch = tuple(s for s in all_corpora() if s.redistributable)
        elif name.startswith("@"):
            lang = name[1:]
            batch = tuple(s for s in all_corpora() if s.lang == lang)
            if not batch:
                raise KeyError(f"selector {name!r} matched nothing")
        else:
            batch = (get_corpus(name),)
        for spec in batch:
            if spec.corpus_id not in seen:
                seen.add(spec.corpus_id)
                out.append(spec)
    return tuple(out)
