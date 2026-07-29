"""Universal Dependencies treebanks as gold word segmentation.

WHY UD is in this benchmark at all: it is the only source that gives us the two
CONTROLLED CONTRASTS the study needs, with annotation convention held fixed.

* ``ud_zh_gsd`` vs ``ud_zh_gsdsimp`` are the SAME 4,997 sentences under the same
  guidelines, traditional vs simplified. Everything except SCRIPT is constant,
  so any difference in tokenizer behaviour is a script effect and nothing else.
* ``ud_zh_hk`` vs ``ud_yue_hk`` are 1,004 PARALLEL sentences -- written Standard
  Chinese and written Cantonese of the same content, both in traditional script.
  Everything except LANGUAGE is constant.

Both contrasts are destroyed by reordering or by dropping sentences, so this
loader emits records in strict source order and puts the UD ``sent_id`` in
``meta``. Downstream pairing is then either by index or by ``sent_id``, and a
mismatch is detectable rather than silent.

WHY the text is reconstructed rather than taken from ``# text =``: the gold
spans must be codepoint offsets into the exact string we hand the tokenizer, and
the only string we can prove is consistent with the token layer is the one built
from the forms plus ``SpaceAfter=No``. We still PREFER the ``# text`` line where
it exists (it is the authoritative surface string, and some treebanks record
whitespace there that MISC does not express), but only after re-anchoring every
form inside it; sentences where that fails fall back to the reconstruction and
are counted, never silently accepted.

Multi-word tokens: UD's ``1-2`` range lines carry the SURFACE token, and the
syntactic words under them are an analysis, not orthography. A tokenizer sees the
surface string, so the surface token is the gold segment and the sub-words are
skipped. Empty nodes (``1.1``) have no surface realisation at all and are
skipped everywhere.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import conllu

from unsegbench.corpora.base import Artifact, CorpusSpec
from unsegbench.fetch.gitsrc import fetch_repo, tarball_url
from unsegbench.types import Segmented, Span

__all__ = ["ENTRIES", "ParseStats", "UDTreebank", "parse_conllu_text"]

#: UD file suffix -> our split. ``dev`` folds into ``train`` deliberately: we
#: only ever evaluate on ``test``, and merging dev into train keeps the held-out
#: set exactly what the treebank authors held out. The originating file is kept
#: per record in ``meta["ud_split"]`` so nothing is actually lost.
_SPLIT_MAP: dict[str, str] = {"train": "train", "dev": "train", "test": "test"}

#: Order matters: ``train`` before ``dev`` so that record indices are stable
#: across rebuilds and across the parallel treebanks.
_SPLIT_FILE_ORDER: tuple[str, ...] = ("train", "dev", "test")


@dataclass
class ParseStats:
    """Diagnostics from one parse pass. Reported at build time, never scored.

    Attributes:
        n_sents: sentences emitted.
        n_tokens: surface tokens emitted (== total gold spans).
        n_mwt: multi-word-token range lines seen. Each one replaced two or more
            syntactic words with a single surface token.
        n_empty_nodes: empty nodes (``1.1``) skipped.
        n_text_lines: sentences that carried a ``# text =`` line.
        n_text_mismatch: sentences where that line disagreed with the
            reconstruction from forms + ``SpaceAfter``. A nonzero count is not
            fatal -- it usually means the treebank records whitespace the MISC
            column does not -- but it must be visible.
        n_text_unalignable: sentences where the ``# text`` line could not be
            re-anchored to the forms, so the reconstruction was used instead.
            This is the only case where we discard the authoritative string.
        n_empty_sents: sentences with no usable surface token, skipped.
    """

    n_sents: int = 0
    n_tokens: int = 0
    n_mwt: int = 0
    n_empty_nodes: int = 0
    n_text_lines: int = 0
    n_text_mismatch: int = 0
    n_text_unalignable: int = 0
    n_empty_sents: int = 0

    def merge(self, other: ParseStats) -> None:
        """Accumulate ``other`` into this one, field by field."""
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


# --------------------------------------------------------------------------
# CoNLL-U -> (text, spans)
# --------------------------------------------------------------------------


def _space_after(token: dict[str, Any]) -> bool:
    """True if a space follows this token's surface form.

    UD marks the absence of a space with ``SpaceAfter=No`` in MISC; anything
    else (including no MISC at all) means a single space. ``SpacesAfter`` is
    ignored on purpose: it encodes exotic whitespace runs that only appear in
    treebanks we do not use, and guessing at it would put characters into
    ``text`` that no annotator vouched for.
    """
    misc = token.get("misc")
    return not (misc and misc.get("SpaceAfter") == "No")


def _surface_tokens(sentence: Any, stats: ParseStats) -> list[tuple[str, bool]]:
    """The surface token layer of one sentence, as ``(form, space_after)``.

    Range lines (``1-2``) win over the syntactic words they cover, because the
    range line is what is actually written. Empty nodes (``1.1``) are dropped:
    they have no surface realisation, so they cannot own codepoints.
    """
    out: list[tuple[str, bool]] = []
    covered_through = 0
    for token in sentence:
        tid = token["id"]
        if isinstance(tid, tuple):
            if tid[1] == ".":  # empty node, e.g. 1.1
                stats.n_empty_nodes += 1
                continue
            stats.n_mwt += 1
            covered_through = int(tid[2])
            form = token["form"]
            if form:
                out.append((form, _space_after(token)))
            continue
        if int(tid) <= covered_through:  # a word inside an MWT range
            continue
        form = token["form"]
        if form:
            out.append((form, _space_after(token)))
    return out


def _reconstruct(tokens: list[tuple[str, bool]]) -> tuple[str, list[Span]]:
    """Build the surface string from forms + ``SpaceAfter``, with gold spans.

    No trailing gap is emitted after the final token: ``SpaceAfter`` on the last
    token describes the join to the next sentence, not content of this one.
    """
    parts: list[str] = []
    spans: list[Span] = []
    pos = 0
    last = len(tokens) - 1
    for i, (form, space_after) in enumerate(tokens):
        spans.append((pos, pos + len(form)))
        parts.append(form)
        pos += len(form)
        if space_after and i != last:
            parts.append(" ")
            pos += 1
    return "".join(parts), spans


def _align(text: str, tokens: list[tuple[str, bool]]) -> list[Span] | None:
    """Re-anchor the forms inside an authoritative ``# text`` string.

    A forward scan, each form found at or after the previous form's end. This is
    exact whenever the disagreement with the reconstruction is whitespace-only,
    which is the realistic case.

    Returns:
        Codepoint spans into ``text``, or ``None`` if any form is missing --
        which means the ``# text`` line and the token layer genuinely disagree
        about content, and the caller must fall back to the reconstruction.
    """
    spans: list[Span] = []
    cursor = 0
    for form, _ in tokens:
        found = text.find(form, cursor)
        if found < 0:
            return None
        spans.append((found, found + len(form)))
        cursor = found + len(form)
    return spans


def _sentence_to_record(
    sentence: Any, rec_id: str, meta: dict[str, Any], stats: ParseStats
) -> Segmented | None:
    """One CoNLL-U sentence -> one `Segmented`, or ``None`` if it has no surface.

    Args:
        sentence: a ``conllu.TokenList``.
        rec_id: the ``<corpus_id>/<split>/<index>`` identifier to stamp on it.
        meta: base metadata; ``sent_id`` is added from the sentence's own
            ``# sent_id`` line when present.
        stats: mutated in place with the diagnostics for this sentence.
    """
    tokens = _surface_tokens(sentence, stats)
    if not tokens:
        stats.n_empty_sents += 1
        return None

    text, spans = _reconstruct(tokens)
    declared = (sentence.metadata or {}).get("text")
    if declared:
        stats.n_text_lines += 1
        if declared != text:
            stats.n_text_mismatch += 1
        aligned = _align(declared, tokens)
        if aligned is None:
            stats.n_text_unalignable += 1
        else:
            text, spans = declared, aligned

    if not text:
        stats.n_empty_sents += 1
        return None

    rec_meta = dict(meta)
    sent_id = (sentence.metadata or {}).get("sent_id")
    if sent_id:
        rec_meta["sent_id"] = sent_id

    stats.n_sents += 1
    stats.n_tokens += len(spans)
    return Segmented(id=rec_id, text=text, spans=tuple(spans), meta=rec_meta)


def parse_conllu_text(
    data: str,
    *,
    corpus_id: str = "ud",
    split: str = "test",
    start_index: int = 0,
    meta: dict[str, Any] | None = None,
    stats: ParseStats | None = None,
) -> list[Segmented]:
    """Parse a CoNLL-U string into records, in source order.

    Args:
        data: the CoNLL-U text.
        corpus_id: used to build record ids.
        split: used to build record ids.
        start_index: first record index, so several files can feed one split.
        meta: base metadata copied onto every record.
        stats: accumulator; created internally when omitted.

    Returns:
        The records, in the order they appear in ``data``.
    """
    stats = stats if stats is not None else ParseStats()
    base = meta or {}
    out: list[Segmented] = []
    i = start_index
    for sentence in conllu.parse(data):
        rec = _sentence_to_record(sentence, f"{corpus_id}/{split}/{i:06d}", base, stats)
        if rec is not None:
            out.append(rec)
            i += 1
    return out


# --------------------------------------------------------------------------
# The corpus spec
# --------------------------------------------------------------------------


@dataclass(eq=False)
class UDTreebank(CorpusSpec):
    """One UD treebank, fetched as a GitHub branch snapshot.

    Attributes:
        repo: the ``UniversalDependencies/<repo>`` repository name.
        branch: preferred branch; `fetch_repo` falls back across master/main.
        test_only: treebanks like PUD that ship a test file only.
        stats: diagnostics accumulated by the most recent `parse` calls.
    """

    corpus_id: str
    lang: str
    script: str
    convention: str
    repo: str
    branch: str = "master"
    owner: str = "UniversalDependencies"
    test_only: bool = False
    license: str = "CC BY-SA 4.0"
    redistributable: bool = True
    version: str = "master"
    # SpaceAfter gaps are plain U+0020. The rest are kept from the base default
    # so that a stray ideographic space in a source file is a build error only if
    # it is really unaccounted for.
    gap_charset: str = " \t\n　"
    notes: str = ""
    stats: ParseStats = field(default_factory=ParseStats)

    @property
    def source_url(self) -> str:  # type: ignore[override]
        """The treebank's GitHub page."""
        return f"https://github.com/{self.owner}/{self.repo}"

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        """The single branch tarball. Unpacked into ``<raw_dir>/extracted/``."""
        return (
            Artifact(
                name=f"{self.repo}-{self.branch}.tar.gz",
                url=tarball_url(self.owner, self.repo, self.branch),
                sha256=None,  # branch snapshots are not byte-stable; see gitsrc
                extract=True,
            ),
        )

    def splits(self) -> tuple[str, ...]:
        """``("test",)`` for the PUD-style treebanks, which have no train file."""
        return ("test",) if self.test_only else ("test", "train")

    def fetch(self, raw_dir: Path) -> Path:
        """Ensure the repo snapshot is present under ``raw_dir/extracted``.

        Idempotent, and safe to call from `parse` -- a second call is a directory
        stat, not a download.

        Args:
            raw_dir: ``cache/raw/<corpus_id>/<version>/``.

        Returns:
            The extracted repository root.
        """
        return fetch_repo(self.owner, self.repo, self.branch, Path(raw_dir) / "extracted")

    def conllu_files(self, raw_dir: Path) -> list[tuple[str, Path]]:
        """``(ud_split, path)`` for every CoNLL-U file in the snapshot.

        Ordered ``train``, ``dev``, ``test`` so that record indices are stable.
        """
        found: dict[str, Path] = {}
        for path in sorted(Path(raw_dir).rglob("*.conllu")):
            stem = path.stem.rsplit("-", 1)[-1]
            if stem in _SPLIT_MAP:
                found[stem] = path
        return [(s, found[s]) for s in _SPLIT_FILE_ORDER if s in found]

    def parse(self, raw_dir: Path, split: str) -> Iterator[Segmented]:
        """Yield the records of one split, in source order.

        Fetches the snapshot if it is not already extracted, so a caller that
        only has a cache directory does not need to know about `fetch`.
        """
        raw_dir = Path(raw_dir)
        if not self.conllu_files(raw_dir):
            self.fetch(raw_dir)

        index = 0
        for ud_split, path in self.conllu_files(raw_dir):
            if _SPLIT_MAP[ud_split] != split:
                continue
            records = parse_conllu_text(
                path.read_text(encoding="utf-8"),
                corpus_id=self.corpus_id,
                split=split,
                start_index=index,
                meta={"ud_split": ud_split},
                stats=self.stats,
            )
            index += len(records)
            yield from records


#: Registered treebanks. ORDER IS LOAD-BEARING: the two adjacent pairs are the
#: script control (gsd/gsdsimp) and the language control (zh_hk/yue_hk).
ENTRIES: tuple[CorpusSpec, ...] = (
    UDTreebank(
        corpus_id="ud_zh_gsd",
        lang="zh",
        script="Hant",
        convention="ud-gsd",
        repo="UD_Chinese-GSD",
        notes="Traditional half of the script control; same 4,997 sentences as ud_zh_gsdsimp.",
    ),
    UDTreebank(
        corpus_id="ud_zh_gsdsimp",
        lang="zh",
        script="Hans",
        convention="ud-gsd",
        repo="UD_Chinese-GSDSimp",
        notes="Simplified half of the script control; same 4,997 sentences as ud_zh_gsd.",
    ),
    UDTreebank(
        corpus_id="ud_zh_hk",
        lang="zh",
        script="Hant",
        convention="ud-hk",
        repo="UD_Chinese-HK",
        test_only=True,
        notes="Standard Chinese half of the language control; parallel to ud_yue_hk.",
    ),
    UDTreebank(
        corpus_id="ud_yue_hk",
        lang="yue",
        script="Hant",
        convention="ud-hk",
        repo="UD_Cantonese-HK",
        test_only=True,
        notes="Cantonese half of the language control; parallel to ud_zh_hk.",
    ),
    UDTreebank(
        corpus_id="ud_th_pud",
        lang="th",
        script="Thai",
        convention="ud-pud",
        repo="UD_Thai-PUD",
        test_only=True,
        notes="PUD is test-only by construction.",
    ),
)
