"""Khmer ALT (Asian Language Treebank), NOVA-tokenised.

20,106 sentences translated from English Wikinews and manually word-segmented
and POS-tagged by NICT and NIPTICT. CC BY-NC-SA 4.0, so ``redistributable=False``
and fetching is gated behind the ``"alt"`` licence key (`cache.LICENSE_GATED`).

THE ZWSP QUESTION -- resolved, and the answer is good
----------------------------------------------------
Khmer has no obligatory inter-word space, but some digital Khmer uses U+200B
ZERO WIDTH SPACE as a word separator. Where that happens the segmentation task
collapses: a tokenizer only has to split on an invisible character it can
already see, and any Tier-1 claim built on the corpus measures nothing. That is
why `CorpusManifest.zwsp_present` exists (CONTRACTS.md sec.6).

Measured over the whole of ``data_km.km-tok.nova``: **U+200B occurs exactly
zero times.** So does U+00A0 and U+200C. Every separator in the file is a plain
U+0020 that the NOVA annotation introduced, and which this loader removes.
``zwsp_present=False``, no gold boundary is marked by an invisible character,
and Khmer ALT is usable for Tier-1 claims. `zwsp_stats` computes the two rates
on demand for any record list -- it is kept general because the sibling khPOS
corpus does use ZWSP and needs the same audit.

Format::

    SNT.80188.1<TAB>អ៊ីតាលី បាន ឈ្នះ លើ ព័រទុយហ្គាល់ 31-5 ក្នុង ប៉ូល C ។

Delimiter spaces are removed per CONTRACTS.md sec.1, exactly as for SIGHAN, so
``text`` is the unsegmented string a tokenizer would face. One exception, and it
is reconstruction rather than repair: when two adjacent tokens both face each
other with ASCII alphanumerics -- ``Comando|Vermelho``, ``The|Holy``, ``1,000|IU``
-- the space between them is genuine source orthography for the embedded Latin,
not an annotation artifact. Joining those would produce ``ComandoVermelho``,
a string no tokenizer would ever see. This affects 2,890 of 694,907 adjacent
pairs (0.416%); the restored space is uncovered and declared in `gap_charset`.

ALT ships no train/test division inside the Khmer archive, and unsegbench never
trains on gold, so everything is exposed as a single ``test`` split.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path

from unsegbench.corpora.base import Artifact, CorpusSpec
from unsegbench.positions import gold_boundaries
from unsegbench.types import Segmented, Span

__all__ = ["ENTRIES", "ZWSP", "AltKm", "ZwspStats", "join_tokens", "zwsp_stats"]

#: U+200B ZERO WIDTH SPACE. ``str.isspace()`` returns False for it, which is
#: precisely why it can hide in a corpus and trivialise the task unnoticed.
ZWSP = "​"

_ASCII_ALNUM = re.compile(r"[A-Za-z0-9]")

_MEMBER = "km-nova-181101/data_km.km-tok.nova"


def join_tokens(tokens: Sequence[str]) -> tuple[str, tuple[Span, ...]]:
    """Concatenate NOVA tokens into source text plus gold spans.

    Annotation-delimiter spaces are dropped. A space is re-inserted only between
    two tokens that face each other with ASCII alphanumerics, where it is real
    orthography of an embedded Latin phrase rather than a delimiter.

    Args:
        tokens: whitespace-split tokens of one NOVA line, in order.

    Returns:
        ``(text, spans)`` with codepoint offsets. The re-inserted spaces are the
        only uncovered codepoints.
    """
    parts: list[str] = []
    spans: list[Span] = []
    pos = 0
    prev: str | None = None
    for tok in tokens:
        if not tok:
            continue
        if prev is not None and _ASCII_ALNUM.fullmatch(prev[-1]) and _ASCII_ALNUM.fullmatch(tok[0]):
            parts.append(" ")
            pos += 1
        parts.append(tok)
        spans.append((pos, pos + len(tok)))
        pos += len(tok)
        prev = tok
    return "".join(parts), tuple(spans)


class ZwspStats:
    """The two rates that decide whether a Khmer corpus is scientifically usable.

    Attributes:
        n_zwsp: U+200B occurrences across all records.
        n_gold: interior gold boundary positions across all records.
        n_gold_at_zwsp: gold boundaries adjacent to a U+200B.
        gold_at_zwsp_rate: share of gold boundaries a ZWSP already marks. Near
            1.0 means the gold IS the ZWSPs and the task is trivial.
        zwsp_is_gold_rate: share of ZWSPs that are gold boundaries. Near 1.0
            means splitting on ZWSP is a near-perfect free tokenizer.
    """

    __slots__ = ("n_gold", "n_gold_at_zwsp", "n_zwsp", "n_zwsp_is_gold")

    def __init__(self, n_zwsp: int, n_gold: int, n_gold_at_zwsp: int, n_zwsp_is_gold: int) -> None:
        self.n_zwsp = n_zwsp
        self.n_gold = n_gold
        self.n_gold_at_zwsp = n_gold_at_zwsp
        self.n_zwsp_is_gold = n_zwsp_is_gold

    @property
    def gold_at_zwsp_rate(self) -> float:
        """Fraction of gold boundaries that a ZWSP already marks."""
        return self.n_gold_at_zwsp / self.n_gold if self.n_gold else 0.0

    @property
    def zwsp_is_gold_rate(self) -> float:
        """Fraction of ZWSPs that coincide with a gold boundary."""
        return self.n_zwsp_is_gold / self.n_zwsp if self.n_zwsp else 0.0

    @property
    def trivial(self) -> bool:
        """True if ZWSP effectively gives the segmentation away."""
        return self.n_zwsp > 0 and self.gold_at_zwsp_rate > 0.9 and self.zwsp_is_gold_rate > 0.9

    def __repr__(self) -> str:
        return (
            f"ZwspStats(n_zwsp={self.n_zwsp}, n_gold={self.n_gold}, "
            f"gold_at_zwsp_rate={self.gold_at_zwsp_rate:.4f}, "
            f"zwsp_is_gold_rate={self.zwsp_is_gold_rate:.4f}, trivial={self.trivial})"
        )


def zwsp_stats(records: Sequence[Segmented]) -> ZwspStats:
    """Measure how much of the gold segmentation U+200B gives away.

    A position ``i`` counts as sitting at a ZWSP when either neighbouring
    codepoint is U+200B, since the separator may be written on either side of
    the boundary the annotator recorded.

    Args:
        records: any list of `Segmented` -- ALT, khPOS, or the mini fixture.

    Returns:
        A `ZwspStats`. ``n_zwsp == 0`` makes both rates 0.0 by convention.
    """
    n_zwsp = n_gold = n_gold_at = n_zwsp_gold = 0
    for rec in records:
        text = rec.text
        gold = gold_boundaries(rec)
        n_gold += len(gold)
        at_zwsp = {i for i in range(1, len(text)) if ZWSP in (text[i - 1], text[i])}
        n_gold_at += len(gold & at_zwsp)
        for j, ch in enumerate(text):
            if ch != ZWSP:
                continue
            n_zwsp += 1
            # A separator sits BETWEEN two positions, j and j+1; the annotator's
            # boundary may have been recorded on either side of it. Counting the
            # positions instead would double-count every ZWSP.
            if gold & {j, j + 1}:
                n_zwsp_gold += 1
    return ZwspStats(n_zwsp, n_gold, n_gold_at, n_zwsp_gold)


class AltKm(CorpusSpec):
    """Khmer ALT, NOVA manual word segmentation."""

    corpus_id = "alt_km"
    lang = "km"
    script = "Khmr"
    convention = "alt"
    license = "CC BY-NC-SA 4.0"
    redistributable = False
    source_url = "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/"
    version = "181101"
    #: U+0020 only, and only where re-inserted inside an embedded Latin phrase.
    gap_charset = " "
    #: Key into `unsegbench.fetch.cache.LICENSE_GATED`.
    license_key = "alt"
    notes = (
        "NOVA delimiter spaces removed. U+200B ZWSP occurs zero times in the source, so "
        "zwsp_present=False and the segmentation task is not trivialised. Space restored "
        "between adjacent ASCII-alphanumeric tokens (0.416% of pairs) where it is genuine "
        "Latin orthography rather than an annotation delimiter. Single 'test' split."
    )

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        return (
            Artifact(
                name="km-nova-181101.zip",
                url="https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/km-nova-181101.zip",
            ),
        )

    def splits(self) -> tuple[str, ...]:
        """ALT ships no split for Khmer, and unsegbench never trains on gold."""
        return ("test",)

    def parse(self, raw_dir: Path, split: str) -> Iterator[Segmented]:
        """Yield gold records.

        Args:
            raw_dir: directory holding ``km-nova-181101.zip`` (or its extraction).
            split: only ``"test"`` yields anything.
        """
        if split != "test":
            return
        for i, (sid, line) in enumerate(self._lines(raw_dir)):
            text, spans = join_tokens(line.split(" "))
            if not text:
                continue
            yield Segmented(
                id=f"{self.corpus_id}/{split}/{i:06d}",
                text=text,
                spans=spans,
                meta={"alt_id": sid},
            )

    @staticmethod
    def _lines(raw_dir: Path) -> Iterator[tuple[str, str]]:
        """``(SNT id, tokenised sentence)`` from the zip or the extracted tree."""
        extracted = raw_dir / "extracted" / _MEMBER
        if extracted.exists():
            body = extracted.read_text(encoding="utf-8")
        else:
            archive = raw_dir / "km-nova-181101.zip"
            if not archive.exists():
                return
            with zipfile.ZipFile(archive) as zf:
                body = zf.read(_MEMBER).decode("utf-8")
        for raw_line in body.split("\n"):
            if "\t" not in raw_line:
                continue
            sid, _, sent = raw_line.partition("\t")
            if sent.strip():
                yield sid, sent.strip()


ENTRIES: tuple[CorpusSpec, ...] = (AltKm(),)
