"""CWSA: the human ceiling. Per-position segmentation agreement over 500 sentences.

OSF project https://osf.io/m3rcf/ -- "A corpus of Chinese word segmentation
agreement" (CWSA). CC BY 4.0, redistributable.

This is not a normal corpus. It is the reference against which every tokenizer
score becomes interpretable: if trained native readers only agree 92% of the
time about where a word ends, a tokenizer scoring 0.92 is not "92% correct", it
is at the human ceiling.

TWO CORRECTIONS TO THE BRIEF, both verified against the data
------------------------------------------------------------
1. **It is Chinese, not Cantonese, and there are 20 raters, not 80.** The OSF
   node title is "Chinese Word Segmentation Agreement Corpus". The sentences are
   Standard Written Chinese in Traditional script, drawn from the Beijing
   Sentence Corpus, GECO-CN, MECO and Hong Kong newspaper material -- not
   vernacular Cantonese, and they contain none of the Cantonese-specific
   characters. Every published agreement score is an exact multiple of 1/20
   across all 9,813 positions, which fixes the rater pool at 20. So this
   registers as ``lang="zh"``, ``script="Hant"``.

2. **The per-annotator segmentations are NOT published.** The repository holds
   only aggregated per-position agreement. ``CWSA2.xlsx`` gives one score per
   character position; ``all_word.xlsx`` gives per-word aggregates; the two
   ``segdata_*.csv`` files are eye-movement reading data whose ``sub_code`` is a
   *reading* participant, not a segmentation rater. There is no file anywhere in
   the node from which an individual rater's boundary set could be recovered.

   Consequently this module does **not** expose per-annotator segmentations, and
   `pairwise_agreement` does not return an empirical distribution over rater
   pairs -- neither is derivable, and fabricating either would be worse than
   admitting the limit. What the marginals *do* support exactly is the expected
   pairwise contingency table, which is what `pairwise_agreement` returns; see
   its docstring for why that is exact rather than an approximation.

The parse is validated by reproducing all three published statistics:

===========================================  ==========  ==========
statistic                                    published   measured
===========================================  ==========  ==========
grand mean agreement (SD)                    0.92 (0.13) 0.9151 (0.1297)
positions with agreement in [0.5, 0.7)       8.96%       8.96%
sentences with >=1 position below 0.7        ~85%        85.0%
===========================================  ==========  ==========

Layout of the ``global`` sheet: columns ``A`` source, ``D`` CWSA code, ``E``
sentence, ``F`` length, ``H..AF`` raw scores ``C1..C25``, ``AI..BG`` the
reverse-scored "converted" agreement (verified to equal ``max(raw, 1-raw)`` at
every one of the 9,813 positions). ``C_i`` is the fraction of raters placing a
boundary *after* character ``i``, which is exactly this project's position ``i``
(CONTRACTS.md sec.2). ``length`` counts characters excluding a sentence-final
period, so ``C_1..C_length`` covers positions ``1..n-1`` for the 497 sentences
that end in one; for the 3 that do not, the final column lands on position ``n``
and is dropped as a sentence edge. Rows 502-505 are summary statistics, not data.

``.xlsx`` is read with `zipfile` and `xml.etree` rather than by adding openpyxl:
it is a zip of XML, the two sheets we need are flat, and a corpus loader should
not drag a spreadsheet library into the dependency set.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path

from unsegbench.corpora.base import Artifact, CorpusSpec
from unsegbench.metrics.core import Counts, f1, phi
from unsegbench.positions import boundaries_to_spans, compute_mask
from unsegbench.types import Segmented, Span

__all__ = [
    "ENTRIES",
    "N_RATERS",
    "AgreementRow",
    "OsfCwsa",
    "agreement",
    "consensus_boundaries",
    "load_rows",
    "pairwise_agreement",
]

#: Rater pool size. Every one of the 9,813 published scores is an exact multiple
#: of 1/20, which pins this down; the brief's "80 annotators" is not what the
#: data shows.
N_RATERS = 20

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_COL = re.compile(r"[A-Z]+")

#: ``C1..C25`` raw-score columns of the ``global`` sheet, in position order.
_RAW_COLS = (
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
    "X", "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF",
)  # fmt: skip

_ARTIFACT = "CWSA2.xlsx"


# --------------------------------------------------------------------------
# Minimal .xlsx reader
# --------------------------------------------------------------------------


def _read_sheet(path: Path, sheet: str = "xl/worksheets/sheet1.xml") -> list[dict[str, str]]:
    """Read one flat worksheet as a list of ``{column letter: value}`` rows.

    Args:
        path: the ``.xlsx`` file.
        sheet: zip member of the worksheet to read.

    Returns:
        One dict per row, in sheet order, with shared strings resolved. Empty
        cells are simply absent.
    """
    with zipfile.ZipFile(path) as zf:
        shared = [
            "".join(t.text or "" for t in si.iter(_NS + "t"))
            for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).iter(_NS + "si")
        ]
        tree = ET.fromstring(zf.read(sheet))
    rows: list[dict[str, str]] = []
    for row in tree.iter(_NS + "row"):
        cells: dict[str, str] = {}
        for cell in row.iter(_NS + "c"):
            value = cell.find(_NS + "v")
            ref = cell.get("r")
            if value is None or value.text is None or ref is None:
                continue
            match = _COL.match(ref)
            if match is None:
                continue
            cells[match.group()] = shared[int(value.text)] if cell.get("t") == "s" else value.text
        rows.append(cells)
    return rows


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


class AgreementRow:
    """One sentence and its per-position rater agreement.

    Attributes:
        code: the CWSA sentence code, e.g. ``"bscsn1"``.
        source: the corpus the sentence was taken from, e.g. ``"BSC"``.
        text: the sentence verbatim.
        scores: ``{position: fraction of raters placing a boundary there}``, over
            interior positions ``1..n-1`` only.
    """

    __slots__ = ("code", "scores", "source", "text")

    def __init__(self, code: str, source: str, text: str, scores: dict[int, float]) -> None:
        self.code = code
        self.source = source
        self.text = text
        self.scores = scores

    @property
    def n(self) -> int:
        """Length of ``text`` in codepoints."""
        return len(self.text)

    def __repr__(self) -> str:
        return f"AgreementRow({self.code!r}, n={self.n}, scored={len(self.scores)})"


def _parse_rows(path: Path) -> list[AgreementRow]:
    """Parse the ``global`` sheet into `AgreementRow` objects.

    Summary rows are rejected structurally: a data row has an integer ``length``
    and a ``code in CWSA``. The trailing "total number of characters" row also
    carries an integer there, so requiring the code is what excludes it.
    """
    out: list[AgreementRow] = []
    for row in _read_sheet(path)[1:]:
        code, text, length = row.get("D"), row.get("E"), row.get("F")
        if not code or not text or not length or not length.isdigit():
            continue
        n_scored = int(length)
        scores: dict[int, float] = {}
        for i in range(n_scored):
            raw = row.get(_RAW_COLS[i]) if i < len(_RAW_COLS) else None
            if raw is None or raw == "NA":
                continue
            pos = i + 1  # C_i marks the gap AFTER character i == our position i
            if 0 < pos < len(text):  # drop sentence edges (CONTRACTS.md sec.2)
                scores[pos] = float(raw)
        out.append(AgreementRow(code, row.get("A", ""), text, scores))
    return out


@lru_cache(maxsize=4)
def load_rows(raw_dir: Path) -> tuple[AgreementRow, ...]:
    """All 500 CWSA sentences with their agreement scores.

    Args:
        raw_dir: directory holding ``CWSA2.xlsx``.

    Returns:
        The rows in sheet order, empty if the artifact is not present.
    """
    path = raw_dir / _ARTIFACT
    if not path.exists():
        return ()
    return tuple(_parse_rows(path))


# --------------------------------------------------------------------------
# The two products
# --------------------------------------------------------------------------


def consensus_boundaries(scores: dict[int, float]) -> frozenset[int]:
    """Majority-vote boundary set.

    A position is a boundary when strictly more than half the raters placed one.
    The 134 positions at exactly 0.5 -- total deadlock -- resolve to *no*
    boundary, the conservative direction: it never invents a word division that
    half the readers rejected.
    """
    return frozenset(pos for pos, val in scores.items() if val > 0.5)


def agreement(text_id: str, raw_dir: Path) -> dict[int, float]:
    """Per-position agreement for one sentence.

    This is the soft gold. Downstream it powers agreement-weighted scoring
    (credit a boundary in proportion to how many humans wanted it) and
    consensus-only scoring (restrict the universe to positions where humans were
    unanimous), neither of which a hard gold can express.

    Args:
        text_id: either a record id ``"osf_cwsa_zh/test/000007"`` or a CWSA code
            such as ``"bscsn1"``.
        raw_dir: directory holding ``CWSA2.xlsx``.

    Returns:
        ``{position: fraction of the 20 raters placing a boundary there}`` for
        interior positions. Empty if the sentence is unknown.

    Raises:
        KeyError: if ``text_id`` looks like a record id whose index is out of range.
    """
    rows = load_rows(raw_dir)
    if "/" in text_id:
        idx = int(text_id.rsplit("/", 1)[1])
        if not 0 <= idx < len(rows):
            raise KeyError(f"no CWSA sentence at index {idx}")
        return dict(rows[idx].scores)
    for row in rows:
        if row.code == text_id:
            return dict(row.scores)
    return {}


def pairwise_agreement(raw_dir: Path, *, mask: str = "core", lang: str = "zh") -> dict[str, float]:
    """The human ceiling: expected boundary agreement between two random raters.

    **This is computed from the published marginals, not from per-rater data,
    because per-rater data is not published** (see the module docstring). That is
    a weaker input than it sounds, for a specific reason: the pooled contingency
    table between two raters drawn uniformly at random is an *exact* function of
    the marginals alone. At a position where ``k`` of ``N`` raters marked a
    boundary, an unordered random pair agrees-on-boundary with probability
    ``k(k-1)/(N(N-1))``, agrees-on-no-boundary with ``(N-k)(N-k-1)/(N(N-1))``,
    and disagrees with ``2k(N-k)/(N(N-1))``. Summing over positions gives the
    expected pooled table exactly, by linearity -- no independence assumption
    between positions is needed, only that raters are exchangeable.

    The table is accumulated pre-multiplied by ``N(N-1)`` so it stays integral;
    ``phi`` and ``f1`` are invariant to that common scaling.

    What this cannot give, and per-rater data could, is the *spread* across
    pairs -- the SD of the ceiling. Reported as a point estimate only.

    Args:
        raw_dir: directory holding ``CWSA2.xlsx``.
        mask: which position universe to score on. ``"core"`` is the headline.
        lang: language for the cluster grammar; CWSA is ``"zh"``.

    Returns:
        ``{"phi": ..., "f1": ..., "mean_agreement": ..., "n_positions": ...}``
        where ``mean_agreement`` is the plain probability that two random raters
        make the same call at a random scored position.
    """
    n = N_RATERS
    tp = fp = tn = 0
    agree = 0
    total = 0
    for row in load_rows(raw_dir):
        universe = compute_mask(row.text, lang, mask)
        for pos, val in row.scores.items():
            if pos not in universe:
                continue
            k = round(val * n)
            tp += k * (k - 1)
            fp += k * (n - k)
            tn += (n - k) * (n - k - 1)
            agree += k * (k - 1) + (n - k) * (n - k - 1)
            total += 1
    counts = Counts(tp=tp, fp=fp, fn=fp, tn=tn)
    denom = total * n * (n - 1)
    return {
        "phi": phi(counts),
        "f1": f1(counts),
        "mean_agreement": (agree / denom) if denom else 0.0,
        "n_positions": float(total),
    }


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------


def _spans(text: str, scores: dict[int, float]) -> tuple[Span, ...]:
    """Consensus word spans: the partition induced by the majority boundaries."""
    return boundaries_to_spans(consensus_boundaries(scores), len(text))


class OsfCwsa(CorpusSpec):
    """CWSA majority-vote consensus segmentation over 500 Chinese sentences.

    Registered as a single spec rather than one per rater: the per-rater
    segmentations do not exist in the published data, and 500 sentences of
    consensus gold plus the `agreement` accessor carries everything the soft-gold
    and consensus-only analyses need without inflating the registry.
    """

    corpus_id = "osf_cwsa_zh"
    lang = "zh"
    script = "Hant"
    convention = "cwsa-consensus"
    license = "CC BY 4.0"
    redistributable = True
    source_url = "https://osf.io/m3rcf/"
    version = "2024"
    #: The consensus partition tiles the text, so nothing is ever uncovered.
    gap_charset = ""
    notes = (
        "Human-ceiling reference, NOT a conventional gold corpus. 500 sentences, 9,813 "
        "scored positions, 20 raters (published scores are exact multiples of 1/20; the "
        "'80 annotators' figure is not supported by the data). Traditional-script Standard "
        "Written Chinese, not Cantonese. Per-rater segmentations are NOT published, so no "
        "per-annotator accessor is provided and pairwise_agreement() is derived from the "
        "marginals. Gold here is the majority vote; ties at 0.5 resolve to no boundary. "
        "Reproduces the published 0.92 mean agreement, 8.96% ambiguous positions and 85% "
        "of sentences containing a sub-0.7 position."
    )

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        return (
            Artifact(
                name=_ARTIFACT,
                url="https://osf.io/download/66f89d83c877fdd1bfa85220/",
            ),
        )

    def splits(self) -> tuple[str, ...]:
        """500 sentences, evaluation only."""
        return ("test",)

    def parse(self, raw_dir: Path, split: str) -> Iterator[Segmented]:
        """Yield the consensus segmentation of each sentence.

        Args:
            raw_dir: directory holding ``CWSA2.xlsx``.
            split: only ``"test"`` yields anything.
        """
        if split != "test":
            return
        for i, row in enumerate(load_rows(raw_dir)):
            yield Segmented(
                id=f"{self.corpus_id}/{split}/{i:06d}",
                text=row.text,
                spans=_spans(row.text, row.scores),
                meta={"cwsa_code": row.code, "source": row.source},
            )


def mean_agreement(rows: Sequence[AgreementRow]) -> tuple[float, float]:
    """Grand mean and SD of the reverse-scored agreement, for validation.

    The published "agreement" is ``max(p, 1-p)``: a position where nobody placed
    a boundary is perfect agreement, not zero. Verified to match the sheet's own
    converted columns at every position.

    Args:
        rows: parsed `AgreementRow` objects.

    Returns:
        ``(mean, population SD)`` over every scored position.
    """
    vals = [max(v, 1.0 - v) for row in rows for v in row.scores.values()]
    if not vals:
        return 0.0, 0.0
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
    return mu, var**0.5


ENTRIES: tuple[CorpusSpec, ...] = (OsfCwsa(),)
