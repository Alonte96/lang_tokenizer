"""VISTEC-TP-TH-2021: the largest human-annotated Thai word-segmentation corpus.

49,997 Twitter sentences / 3.39M words, annotated by linguists at VISTEC and
Chiang Mai University under the LST20 guideline. CC BY-SA 3.0, so redistributable.

WHY this loader is more than a ``split("|")``
---------------------------------------------
The distributed files are *inline-markup* text, not a token stream. One line
carries four annotation layers at once::

    ขาย|/|เทรด|<msp value="ค่ะ">ค่า</msp>| |*|หาก|จะ|ซื้อ

  * ``|``                       word delimiter
  * ``<ne>..</ne>``             named-entity boundary
  * ``<compound>..</compound>`` compound spanning several words
  * ``<msp value="X">Y</msp>``  misspelling: ``Y`` is what was actually written,
                                ``X`` is the annotator's correction

The corrections are the trap. ``value="X"`` is *not* in the source text -- only
the inner ``Y`` is -- so a naive tag stripper that keeps attributes, or one that
substitutes the correction, silently rewrites 65,638 tokens of the biggest Thai
corpus we have and no downstream check would notice. Equally, leaving any ``|``
or ``<ne>`` in ``text`` would hand every tokenizer a free boundary marker.

So this module does not trust its own regex. VISTEC ships a parallel
``*_raw.txt`` holding the same sentences with no markup at all, and
`verify_against_raw` asserts that stripping the markup reproduces it byte for
byte. It does: 50,001/50,001 lines across both splits, which is the strongest
available evidence that ``text`` is exactly what a tokenizer would see.

Two further decisions, both measured rather than assumed:

  * A ``|``-delimited piece that is entirely whitespace is a GAP, not a word.
    Thai uses space at phrase level, so these are real source characters that
    belong to no word (631,019 of them). They stay in ``text`` and are declared
    in `gap_charset`; leading/trailing space on a content piece is treated the
    same way.
  * A piece may legitimately contain an interior space -- ``the market``,
    ``AP Honda``, ``จริง ๆ``. Those are single annotated units (Latin
    multi-word NEs, and Thai ``ๆ`` repetition with the spacing required by the
    Royal Institute), so the span covers the space. It is not a delimiter and
    must not be split on.

The lone annotation defect in the source is a stray self-closing ``<compound/>``
at the end of test line 7494, which the tag pattern absorbs; that is why the
pattern allows ``/?>``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from unsegbench.corpora.base import Artifact, CorpusSpec
from unsegbench.types import Segmented, Span

__all__ = ["ENTRIES", "VistecTh", "parse_line", "strip_markup", "verify_against_raw"]

#: Every markup construct VISTEC uses, including the ``value="..."`` attribute of
#: ``<msp>`` and the single malformed self-closing ``<compound/>``. The attribute
#: MUST be inside the pattern: it holds the annotator's spelling correction, which
#: is not part of the source text and must never leak into ``text``.
_TAG = re.compile(r"</?(?:ne|compound|msp)(?:\s+value=\"[^\"]*\")?\s*/?>")

#: The word delimiter. Verified absent from the raw files, so it is unambiguous.
_DELIM = "|"


def strip_markup(line: str) -> str:
    """Remove every annotation tag, keeping the inner text.

    Args:
        line: one line of a ``*_proprocessed.txt`` file.

    Returns:
        The line with tags gone but ``|`` delimiters still present.
    """
    return _TAG.sub("", line)


def parse_line(line: str) -> tuple[str, tuple[Span, ...]]:
    """Turn one annotated line into ``(text, spans)``.

    ``text`` is the source string verbatim: markup removed, delimiters removed,
    every other codepoint -- including phrase spaces -- preserved in order.

    Args:
        line: one line of a ``*_proprocessed.txt`` file.

    Returns:
        ``(text, spans)`` where ``spans`` are codepoint offsets of the gold
        words. Whitespace-only pieces, and whitespace padding a content piece,
        are left uncovered as inter-word gaps.
    """
    parts: list[str] = []
    spans: list[Span] = []
    pos = 0
    for piece in strip_markup(line).split(_DELIM):
        if not piece:
            continue
        core = piece.strip()
        if not core:
            # A whitespace-only token: a real Thai phrase space, belonging to no
            # word. Keep the characters, cover them with no span.
            parts.append(piece)
            pos += len(piece)
            continue
        lead = len(piece) - len(piece.lstrip())
        parts.append(piece)
        spans.append((pos + lead, pos + lead + len(core)))
        pos += len(piece)
    return "".join(parts), tuple(spans)


def _read_lines(path: Path) -> list[str]:
    """Non-empty lines of a VISTEC text file, in order."""
    return [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]


def verify_against_raw(raw_dir: Path, split: str) -> tuple[int, int]:
    """Check that markup stripping reproduces VISTEC's own unannotated text.

    This is the integrity guard for the whole loader. VISTEC distributes
    ``*_raw.txt`` alongside ``*_proprocessed.txt``; the former is the same
    sentences with no markup. If removing tags and delimiters does not
    regenerate it exactly, our parser is rewriting the corpus.

    Args:
        raw_dir: the artifact directory holding both files for this split.
        split: ``"test"`` or ``"train"``.

    Returns:
        ``(n_matching, n_total)``. Anything but equality is a build error.
    """
    pre = _read_lines(raw_dir / f"VISTEC-TP-TH-2021_{split}_proprocessed.txt")
    raw = _read_lines(raw_dir / f"VISTEC-TP-TH-2021_{split}_raw.txt")
    n = min(len(pre), len(raw))
    ok = sum(1 for i in range(n) if strip_markup(pre[i]).replace(_DELIM, "") == raw[i])
    return ok, max(len(pre), len(raw))


class VistecTh(CorpusSpec):
    """VISTEC-TP-TH-2021 Thai word segmentation, both official splits."""

    corpus_id = "vistec_th"
    lang = "th"
    script = "Thai"
    convention = "vistec"
    license = "CC BY-SA 3.0"
    redistributable = True
    source_url = "https://github.com/mrpeerat/OSKut/tree/main/VISTEC-TP-TH-2021"
    version = "2021"
    #: U+0020 only. Measured over both splits: no tab, no NBSP, no ZWSP.
    gap_charset = " "
    notes = (
        "Inline markup (| delimiter, <ne>, <compound>, <msp value=...>) is stripped from "
        "text and encoded as spans. Verified byte-exact against the distributed *_raw.txt. "
        "Whitespace-only pieces are Thai phrase spaces and are left uncovered; interior "
        "spaces inside a single annotated unit (Latin NEs, Thai 'ๆ') stay inside the span."
    )

    _BASE = "https://raw.githubusercontent.com/mrpeerat/OSKut/main/VISTEC-TP-TH-2021"

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        """Both the annotated and the unannotated file for each split.

        The ``*_raw.txt`` files are not needed to parse anything -- they exist so
        `verify_against_raw` can prove the markup stripping is lossless.
        """
        return tuple(
            Artifact(
                name=f"VISTEC-TP-TH-2021_{split}_{kind}.txt",
                url=f"{self._BASE}/{split}/VISTEC-TP-TH-2021_{split}_{kind}.txt",
            )
            for split in ("test", "train")
            for kind in ("proprocessed", "raw")
        )

    def parse(self, raw_dir: Path, split: str) -> Iterator[Segmented]:
        """Yield gold records for one split.

        Args:
            raw_dir: directory holding the downloaded artifacts.
            split: ``"test"`` or ``"train"``; anything else yields nothing.
        """
        if split not in ("test", "train"):
            return
        path = raw_dir / f"VISTEC-TP-TH-2021_{split}_proprocessed.txt"
        if not path.exists():
            return
        for i, line in enumerate(_read_lines(path)):
            text, spans = parse_line(line)
            if not text:
                continue
            yield Segmented(id=f"{self.corpus_id}/{split}/{i:06d}", text=text, spans=spans)


ENTRIES: tuple[CorpusSpec, ...] = (VistecTh(),)
