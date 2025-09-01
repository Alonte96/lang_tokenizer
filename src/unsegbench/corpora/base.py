"""CorpusSpec: the one interface every gold corpus loader implements.

STUB -- Phase 0 contract. Implementors fill in `artifacts` and `parse`; the rest
is provided. Do not change these signatures without updating CONTRACTS.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from unsegbench.types import CorpusManifest, Segmented

__all__ = ["BUILDER_VERSION", "Artifact", "CorpusSpec"]

#: Bump when a loader change alters canonical output. Invalidates built corpora.
BUILDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class Artifact:
    """One downloadable file a corpus needs.

    Attributes:
        name: filename under ``cache/raw/<corpus_id>/<version>/``.
        url: where to get it.
        sha256: expected digest. ``None`` only during initial development --
            record the real value into ``corpora.lock.json`` before shipping,
            because on the TLS-downgrade path (SIGHAN) the hash is the ONLY
            integrity guarantee.
        extract: unpack a zip/tar into ``extracted/`` after download.
    """

    name: str
    url: str
    sha256: str | None = None
    extract: bool = False


class CorpusSpec(ABC):
    """A gold-segmented corpus under one annotation convention.

    One `CorpusSpec` == one (text, convention) pair. A source that ships several
    conventions over the same text -- SIGHAN's four standards, hkcancor-multi's
    tiers -- registers one spec PER CONVENTION. That is what makes the
    convention experiment expressible: the registry can hand back several specs
    that differ only in `convention`.
    """

    #: Registry key, e.g. ``"sighan_pku"``, ``"ud_zh_gsdsimp"``, ``"hkcancor_multi_d"``.
    corpus_id: str
    lang: str  # zh | yue | th | km
    script: str  # Hans | Hant | Thai | Khmr
    convention: str
    license: str
    #: False for SIGHAN (research-use only) and the CC BY-NC-SA Khmer corpora.
    #: False means: never write derived text outside the cache, and require an
    #: explicit --accept-license before fetching.
    redistributable: bool
    source_url: str
    version: str
    #: Codepoints allowed to be uncovered by gold spans. Anything else uncovered
    #: is a build error, because it means the loader dropped content.
    gap_charset: str = " \t\n　"
    notes: str = ""

    @property
    @abstractmethod
    def artifacts(self) -> tuple[Artifact, ...]:
        """Files to download before `parse` can run."""

    @abstractmethod
    def parse(self, raw_dir: Path, split: str) -> Iterator[Segmented]:
        """Yield `Segmented` records for one split.

        Args:
            raw_dir: ``cache/raw/<corpus_id>/<version>/``, with ``extracted/``
                populated for any artifact marked ``extract=True``.
            split: ``"test"`` or ``"train"``. A corpus with no split of that
                name yields nothing rather than raising.

        Contract:
            * ``text`` is the source string VERBATIM. Never call
              ``unicodedata.normalize`` here -- Thai U+0E33 changes codepoint
              count under NFKD/NFKC and NFKC reorders Khmer marks. Normalisation
              is a tokenizer-side concern.
            * ``spans`` are codepoint offsets, sorted and non-overlapping.
            * Every uncovered codepoint must be in `gap_charset`.
            * ``id`` is ``f"{corpus_id}/{split}/{i:06d}"``.
        """

    def splits(self) -> tuple[str, ...]:
        """Splits this corpus provides. Override if it has no train split."""
        return ("test", "train")

    def manifest(
        self,
        splits: dict[str, str],
        n_sents: int,
        n_words: int,
        n_chars: int,
        gold_illegal_rate: float,
        zwsp_present: bool,
    ) -> CorpusManifest:
        """Assemble the manifest. Provided -- do not override."""
        return CorpusManifest(
            corpus_id=self.corpus_id,
            lang=self.lang,
            script=self.script,
            convention=self.convention,
            license=self.license,
            redistributable=self.redistributable,
            source_url=self.source_url,
            version=self.version,
            splits=splits,
            n_sents=n_sents,
            n_words=n_words,
            n_chars=n_chars,
            gap_charset=self.gap_charset,
            gold_illegal_rate=gold_illegal_rate,
            zwsp_present=zwsp_present,
            builder_version=BUILDER_VERSION,
            notes=self.notes,
        )
