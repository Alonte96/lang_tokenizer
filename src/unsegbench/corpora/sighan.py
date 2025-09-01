"""SIGHAN 2005 Bakeoff (icwb2-data): four Chinese segmentation conventions.

WHY four corpora and not one: the bakeoff distributes four independently
annotated corpora -- Academia Sinica, City University of Hong Kong, Peking
University and Microsoft Research -- that disagree about what a word IS. AS and
CityU are traditional-script (Hant), PKU and MSR simplified (Hans), and the
segmentation standards differ on compounds, names and affixes. Registering one
`CorpusSpec` per convention is what makes the convention experiment expressible:
a tokenizer's score against "Chinese" is meaningless without saying whose
Chinese, and the spread across these four is the honest error bar.

WHY the parsing is fussier than ``line.split()``:

The gold format is "words separated by whitespace, one sentence per line", but
each corpus picked a DIFFERENT separator and none of them is used consistently:

===========  ==================  ==========================================
corpus       delimiter           observed reality in icwb2-data
===========  ==================  ==========================================
``as``       U+3000 IDEOGRAPHIC  no U+0020 anywhere in the file
``cityu``    U+0020              file begins with a UTF-8 BOM
``pku``      U+0020              runs of TWO spaces, occasionally three
``msr``      U+0020              runs of two, but also one, three and five
===========  ==================  ==========================================

So the delimiter is declared per corpus and matched as a RUN (``+``), never as a
fixed width and never as generic "any whitespace". The distinction is
load-bearing rather than pedantic: ``cityu_training.utf8`` contains exactly one
U+3000, and it sits INSIDE a word (``Phang<U+3000>Nga``, a romanised Thai place
name) between two ordinary space delimiters. A generic whitespace split would
silently invent a word boundary there; splitting on U+3000 for AS only keeps it
as the word-internal content it is. Symmetrically, ``msr_training.utf8``
contains one TAB, alone on its own line -- a blank line, not a word.

Per CONTRACTS.md sec.1 the separators are DELIMITERS, not content: ``text`` is
the words concatenated without them and the spans tile ``text`` exactly, so
``gap_charset`` is empty. Whitespace genuinely inside a word (the Phang Nga
case) survives inside its span and therefore needs no gap declaration.

Licence: research use only, per the original competition terms. Never
redistributed by unsegbench -- see ``licenses/SIGHAN2005.txt``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from unsegbench.corpora.base import Artifact, CorpusSpec
from unsegbench.types import Segmented, Span

__all__ = [
    "CONVENTIONS",
    "ENTRIES",
    "ICWB2_SHA256",
    "ICWB2_URL",
    "SighanCorpus",
    "sighan_score_script",
    "training_wordlist",
]

#: The bakeoff archive. HTTP, deliberately: the host's certificate has been
#: broken for years, so `fetch.http.download` performs a host-scoped, hash-gated
#: downgrade. The digest below is the only integrity guarantee on that path.
ICWB2_URL = "http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip"

#: sha256 of icwb2-data.zip (52,640,127 bytes). MUST equal the value in
#: ``corpora.lock.json`` -- ``tests/test_sighan_loader.py`` asserts it, so the
#: two cannot drift. Duplicated only because an installed wheel has no repo root
#: from which to read the lock file.
ICWB2_SHA256 = "296c54adf4abc3702eca0cbce70b55fe22a344ea02d33175a4d1183e074072b4"

VERSION = "icwb2"

CONVENTIONS: tuple[str, ...] = ("as", "cityu", "pku", "msr")

#: Per-corpus word delimiter. See the module docstring -- this is NOT
#: interchangeable with a generic whitespace split.
_DELIMITER: dict[str, str] = {
    "as": "　",  # IDEOGRAPHIC SPACE
    "cityu": " ",
    "pku": " ",
    "msr": " ",
}

#: Gold test files. Note the naming inconsistency in the archive itself: AS is
#: ``as_testing_gold``, everyone else is ``<conv>_test_gold``.
_GOLD_TEST: dict[str, str] = {
    "as": "gold/as_testing_gold.utf8",
    "cityu": "gold/cityu_test_gold.utf8",
    "pku": "gold/pku_test_gold.utf8",
    "msr": "gold/msr_test_gold.utf8",
}

_TRAINING: dict[str, str] = {conv: f"training/{conv}_training.utf8" for conv in CONVENTIONS}

#: The official vocabulary lists, used by the perl cross-check to classify
#: out-of-vocabulary recall. Same AS naming quirk as above.
_TRAINING_WORDS: dict[str, str] = {
    "as": "gold/as_training_words.utf8",
    "cityu": "gold/cityu_training_words.utf8",
    "pku": "gold/pku_training_words.utf8",
    "msr": "gold/msr_training_words.utf8",
}

#: Directory the zip unpacks into.
_ARCHIVE_ROOT = "icwb2-data"


def data_root(raw_dir: Path) -> Path:
    """Locate the unpacked ``icwb2-data`` tree under ``raw_dir``.

    Args:
        raw_dir: ``cache/raw/<corpus_id>/<version>/``, with ``extracted/``
            populated by the fetch layer.

    Returns:
        The directory directly containing ``gold/``, ``training/`` and
        ``scripts/``.

    Raises:
        FileNotFoundError: the archive was never extracted.
    """
    for candidate in (
        raw_dir / "extracted" / _ARCHIVE_ROOT,
        raw_dir / "extracted",
        raw_dir / _ARCHIVE_ROOT,
    ):
        if (candidate / "gold").is_dir():
            return candidate
    raise FileNotFoundError(
        f"icwb2-data not extracted under {raw_dir}; expected {raw_dir}/extracted/{_ARCHIVE_ROOT}"
    )


def sighan_score_script(raw_dir: Path) -> Path:
    """Path to the official perl scorer shipped in the archive.

    The cross-check runs OUR word P/R against this script rather than
    reimplementing it, because "matches the official scorer to 1e-6" is a
    stronger claim than "matches our reading of the official scorer".

    Args:
        raw_dir: the corpus raw directory.

    Returns:
        Path to ``scripts/score``.

    Raises:
        FileNotFoundError: the archive was never extracted, or is truncated.
    """
    path = data_root(raw_dir) / "scripts" / "score"
    if not path.is_file():
        raise FileNotFoundError(f"official SIGHAN scorer missing at {path}")
    return path


def training_wordlist(raw_dir: Path, conv: str) -> Path:
    """Path to the official training vocabulary for one convention.

    Args:
        raw_dir: the corpus raw directory.
        conv: one of `CONVENTIONS`.

    Returns:
        Path to the UTF-8 word list, one word per line.

    Raises:
        KeyError: unknown convention.
        FileNotFoundError: the archive was never extracted, or is truncated.
    """
    if conv not in _TRAINING_WORDS:
        raise KeyError(f"unknown SIGHAN convention {conv!r}; expected one of {CONVENTIONS}")
    path = data_root(raw_dir) / _TRAINING_WORDS[conv]
    if not path.is_file():
        raise FileNotFoundError(f"training wordlist missing at {path}")
    return path


def _segment_line(line: str, splitter: re.Pattern[str]) -> tuple[str, tuple[Span, ...]]:
    """Turn one delimited gold line into ``(text, spans)``.

    The delimiters are dropped and the spans tile the result exactly. Leading
    and trailing whitespace (including the CRLF terminator) is stripped first:
    it is layout, not content.

    Args:
        line: one raw line, delimiters intact.
        splitter: compiled ``<delimiter>+`` pattern for this corpus.

    Returns:
        ``(text, spans)``; ``("", ())`` for a blank line, which the caller skips
        rather than emitting as an empty record.
    """
    words = [w for w in splitter.split(line.strip()) if w]
    if not words:
        return "", ()
    spans: list[Span] = []
    pos = 0
    for word in words:
        spans.append((pos, pos + len(word)))
        pos += len(word)
    return "".join(words), tuple(spans)


class SighanCorpus(CorpusSpec):
    """One SIGHAN 2005 corpus, i.e. one annotation convention."""

    lang = "zh"
    license = "SIGHAN 2005 Bakeoff -- research use only"
    redistributable = False
    source_url = ICWB2_URL
    version = VERSION
    #: The spans tile ``text`` exactly -- the inter-word separators are
    #: delimiters and are removed, so nothing is left uncovered.
    gap_charset = ""

    def __init__(self, corpus_id: str, convention: str, script: str, notes: str = "") -> None:
        self.corpus_id = corpus_id
        self.convention = convention
        self.script = script
        self.notes = notes

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        """The single archive, shared by all four conventions."""
        return (Artifact(name="icwb2-data.zip", url=ICWB2_URL, sha256=ICWB2_SHA256, extract=True),)

    def _source(self, raw_dir: Path, split: str) -> Path | None:
        table = {"test": _GOLD_TEST, "train": _TRAINING}.get(split)
        if table is None:
            return None
        return data_root(raw_dir) / table[self.convention]

    def parse(self, raw_dir: Path, split: str) -> Iterator[Segmented]:
        """Yield one `Segmented` per non-blank gold line.

        Decoded as ``utf-8-sig`` because ``cityu_test_gold.utf8`` opens with a
        BOM, and with universal newlines because every file is CRLF. Neither is
        normalisation: no codepoint of the actual content is touched.

        Args:
            raw_dir: the corpus raw directory, with ``extracted/`` populated.
            split: ``"test"`` or ``"train"``; anything else yields nothing.

        Yields:
            Records whose spans tile ``text`` exactly.
        """
        path = self._source(raw_dir, split)
        if path is None:
            return
        splitter = re.compile(re.escape(_DELIMITER[self.convention]) + "+")
        idx = 0
        with open(path, encoding="utf-8-sig", newline=None) as fh:
            for line in fh:
                text, spans = _segment_line(line, splitter)
                if not text:
                    continue
                yield Segmented(id=f"{self.corpus_id}/{split}/{idx:06d}", text=text, spans=spans)
                idx += 1


ENTRIES: tuple[CorpusSpec, ...] = (
    SighanCorpus(
        corpus_id="sighan_as",
        convention="as",
        script="Hant",
        notes="Academia Sinica, ~5.4M training words; U+3000 delimited.",
    ),
    SighanCorpus(
        corpus_id="sighan_cityu",
        convention="cityu",
        script="Hant",
        notes="City University of Hong Kong, ~1.46M training words; gold test file carries a BOM.",
    ),
    SighanCorpus(
        corpus_id="sighan_pku",
        convention="pku",
        script="Hans",
        notes="Peking University, ~1.11M training words; double-space delimited.",
    ),
    SighanCorpus(
        corpus_id="sighan_msr",
        convention="msr",
        script="Hans",
        notes="Microsoft Research, ~2.37M training words; double-space delimited.",
    ),
)
