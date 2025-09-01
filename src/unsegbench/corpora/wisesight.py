"""wisesight1000: human-tokenised Thai social media text.

WHY THIS CORPUS. Most Thai segmentation evaluation happens on news or Wikipedia,
where the text is clean and the tokenizer's pre-tokenizer has already done half
the job. wisesight1000 is 993 real social-media messages -- emoji, Latin brand
names, elongated vowels ("มั่ยยยยยยยย"), stray zero-width spaces -- annotated by
hand. It is the corpus where a subword vocabulary trained on formal text has
nowhere to hide, and its CC0 licence puts it in the ``@permissive`` subset that
a fresh user gets with zero licence friction.

WHAT WE ACTUALLY READ, AND WHY IT IS NOT THE ADVERTISED SCHEMA. The HF dataset
card describes character-level ``char`` / ``char_type`` / ``is_beginning``
columns. Those columns do not exist as data: the repo ships a dataset SCRIPT
plus one file, ``data/wisesight-1000-samples-tokenised.label.gz``, and the
script synthesises the columns at load time. Since ``datasets`` 3.0 removed
script support that path is dead (see `unsegbench.fetch.hfsrc`), so we read the
shipped file directly. It is the upstream artefact -- pipe-separated tokens, one
message per line -- and it carries strictly more information than the
``is_beginning`` view, which is a lossy re-encoding of exactly these pipes.

THREE THINGS THE RAW FILE DOES THAT WOULD OTHERWISE CORRUPT THE BUILD:

* **Whitespace is annotated as its own token** (3,923 lone-space tokens). Those
  are not words. Thai uses space at PHRASE level, so scoring them as gold words
  would hand every tokenizer free credit in Thai and nowhere else -- precisely
  what the ``core`` mask exists to prevent. They stay in ``text`` (they are
  genuine source content, not an annotation delimiter) and are declared in
  `gap_charset` instead.
* **U+200B ZWSP occurs 25 times**, 17 of them in a lone ``"​ "`` token.
  ``str.isspace()`` is False for ZWSP, so a naive whitespace test would promote
  those to gold words. `_is_gap_token` tests the same extended set that
  `positions.trivial_positions` does.
* **Eight empty tokens** appear, from ``"||"`` typos in the annotation
  ("เหลือ||ก็มี"). They contribute no characters, so skipping them loses nothing
  -- but silently indexing past them would shift every later span.

TEST-ONLY. 993 messages is an evaluation set, not a training set, and upstream
ships no split. `splits` returns ``("test",)``.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path

from unsegbench.corpora.base import Artifact, CorpusSpec
from unsegbench.fetch.hfsrc import fetch_dataset_file
from unsegbench.types import Segmented, Span

__all__ = ["ENTRIES", "N_MESSAGES", "REPO_ID", "Wisesight1000"]

REPO_ID = "pythainlp/wisesight1000"
_SOURCE_URL = f"https://huggingface.co/datasets/{REPO_ID}"

#: PINNED. The repo's only data file; see `unsegbench.fetch.hfsrc`.
_FILENAME = "data/wisesight-1000-samples-tokenised.label.gz"

#: Non-empty lines in the shipped file. The name says 1000; 7 were dropped
#: upstream as spam, and the dataset card agrees on 993.
N_MESSAGES = 993

#: Zero-width and non-breaking codepoints that are whitespace in effect but not
#: according to `str.isspace()`. Mirrors `positions._EXTRA_SPACE`.
_ZERO_WIDTH: frozenset[str] = frozenset("​﻿⁠")


def _is_gap_token(tok: str) -> bool:
    """True if a token is pure inter-word material rather than a word.

    Kept in ``text`` (it is real source content) but attributed to no gold span.
    """
    return bool(tok) and all(ch.isspace() or ch in _ZERO_WIDTH for ch in tok)


class Wisesight1000(CorpusSpec):
    """993 hand-tokenised Thai social-media messages."""

    corpus_id = "wisesight1000"
    lang = "th"
    script = "Thai"
    convention = "wisesight"
    license = "CC0-1.0"
    redistributable = True
    source_url = _SOURCE_URL
    version = "1.0.0"
    #: Space and ZWSP only -- the two codepoints that appear as standalone
    #: whitespace tokens. Emoji, Latin and punctuation are all annotated INTO
    #: words here, so anything else turning up uncovered means we lost data.
    gap_charset = " ​"
    notes = (
        "Read from the upstream pipe-separated .label.gz, not the dataset card's "
        "char/is_beginning columns -- those are synthesised by a dataset script that "
        "does not run on datasets>=3. Whitespace-only tokens are gaps, not words; "
        "8 empty tokens from '||' annotation typos are skipped."
    )

    def splits(self) -> tuple[str, ...]:
        """Evaluation-only: upstream ships no train split."""
        return ("test",)

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        """Empty: fetched from the HF hub on first `parse`. See `hkcancor`."""
        return ()

    def parse(self, raw_dir: Path, split: str) -> Iterator[Segmented]:
        """Yield one record per message; spans cover everything but the gaps."""
        if split != "test":
            return
        path = fetch_dataset_file(REPO_ID, _FILENAME, raw_dir)
        with gzip.open(path, "rb") as fh:
            raw = fh.read().decode("utf-8")

        idx = 0
        for line in raw.split("\n"):
            if not line:
                continue
            parts: list[str] = []
            spans: list[Span] = []
            pos = 0
            for tok in line.split("|"):
                if not tok:
                    continue  # '||' annotation typo; contributes no characters
                parts.append(tok)
                if not _is_gap_token(tok):
                    spans.append((pos, pos + len(tok)))
                pos += len(tok)
            text = "".join(parts)
            if not text:
                continue
            yield Segmented(
                id=f"{self.corpus_id}/{split}/{idx:06d}",
                text=text,
                spans=tuple(spans),
            )
            idx += 1


ENTRIES: tuple[CorpusSpec, ...] = (Wisesight1000(),)
