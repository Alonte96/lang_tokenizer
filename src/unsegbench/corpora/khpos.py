"""khPOS: 12,000 manually word-segmented Khmer sentences.

WHY KHMER IS THE HARD CASE, AND WHY THIS CORPUS SETTLES A GO/NO-GO. Khmer is
written without spaces between words, and its orthographic syllable is built
from an invisible joiner (U+17D2 COENG) plus spacing dependent vowels that
Unicode classifies as Lo -- so neither general category nor UAX#29 grapheme
clustering describes where a boundary may legally fall. Every character is 3
bytes in UTF-8, so byte-level BPE cuts through them constantly. That makes khPOS
the corpus that actually exercises `clusters.khmer_cluster_starts`, and its
measured `positions.gold_illegal_rate` is a direct test of our cluster model
against 117k human-placed boundaries.

**THE ZWSP QUESTION -- ANSWERED: THE CORPUS IS USABLE.** Some Khmer sources use
U+200B ZERO WIDTH SPACE as an invisible word separator. Where they do, word
segmentation is not a task at all: split on ZWSP and you are done, and any claim
built on it is vacuous (CONTRACTS.md sec.6). khPOS does not do this. Measured
over all 12,000 train sentences: U+200B occurs **2 times in 602,138 characters**,
**1 of 117,029 gold boundaries (0.00085%)** sits next to one. ZWSP is incidental
noise here, not the annotation. `zwsp_stats` recomputes this at build time
rather than trusting this docstring, and feeds `CorpusManifest.zwsp_present`.

WHERE THE BYTES COME FROM. The HF repo ``seanghay/khPOS`` contains no data --
only a dataset script that downloads from the upstream GitHub repo, and that
script does not run on ``datasets>=3``. So we go to the same GitHub URLs the
script points at, with pinned sha256, and skip HF entirely.

SPLITS. Upstream ships ``train.all2`` (12,000 sentences) plus ``OPEN-TEST`` and
``CLOSE-TEST`` (1,000 each). We use OPEN-TEST as ``test``: it is genuinely held
out (11/1000 incidental duplicates, all short formulaic name sentences), whereas
CLOSE-TEST is a verbatim 1000/1000 subset of train and would silently make any
train-fitted baseline look perfect.

THE ANNOTATION FORMAT AND ITS TWO TRAPS. Each line is space-separated
``word/POS``. Two markers live INSIDE the word field and are annotation, not
text: ``_`` joins a compound ("ស្អប់_ខ្ពើម", 19,027 occurrences) and ``~`` joins
a prefix or suffix ("អ្នក~ភូមិ", 7,118). They are stripped, so the compound
becomes one contiguous gold word -- keeping them would inject two ASCII
codepoints into Khmer text and manufacture script-transition positions that no
tokenizer will ever see. The space delimiters are likewise annotation: khPOS
gives no way to recover which spaces were in the source, so ``text`` is the
words concatenated and the spans TILE it (CONTRACTS.md sec.1).

Also: ``word/POS`` is split from the RIGHT, because a token may itself contain
``/``; and 8 tokens are the truncated fragment ``ស្`` (ending in a bare COENG),
which is genuine upstream annotation damage and the sole source of this corpus's
nonzero `gold_illegal_rate`. We do not repair it -- it is measured and recorded.

LICENCE. CC BY-NC-SA 4.0: `redistributable` is False, the gate key is
``"khpos"`` (`cache.LICENSE_GATED`), and `parse` refuses to fetch until the
licence has been acknowledged. Nothing derived from it may leave the cache.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from unsegbench.corpora.base import Artifact, CorpusSpec
from unsegbench.errors import FetchError, IntegrityError
from unsegbench.fetch import cache
from unsegbench.types import Segmented, Span

__all__ = ["ENTRIES", "LICENSE_KEY", "ZWSP", "KhPOS", "ZwspStats", "zwsp_stats"]

#: U+200B ZERO WIDTH SPACE. `str.isspace()` is False for it, which is exactly
#: why it needs a name of its own.
ZWSP = "​"

#: Key into `cache.LICENSE_GATED`. CC BY-NC-SA 4.0.
LICENSE_KEY = "khpos"

_RAW_BASE = "https://github.com/ye-kyaw-thu/khPOS/raw/master/corpus-draft-ver-1.0/data"
_SOURCE_URL = "https://github.com/ye-kyaw-thu/khPOS"

#: split -> (local filename, upstream url, sha256). PINNED.
_ARTIFACTS: dict[str, tuple[str, str, str]] = {
    "train": (
        "train.all2",
        f"{_RAW_BASE}/after-replace/train.all2",
        "2573de07a60cb1b214bb5d00ac4f4fc3144d13da33bc56e36b7c04d9a4ea0ed7",
    ),
    "test": (
        "OPEN-TEST",
        f"{_RAW_BASE}/OPEN-TEST",
        "55b960e6049e84e8f9244dabe63604cdc392090c92facce9dd43fa490faa3456",
    ),
}

#: Compound / affix join markers. Annotation, not text.
_JOIN_MARKERS = ("_", "~")


# --------------------------------------------------------------------------
# The go/no-go measurement
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ZwspStats:
    """Evidence that Khmer segmentation is or is not trivial on this corpus.

    Attributes:
        n_zwsp: U+200B codepoints in the corpus text.
        n_chars: total codepoints.
        n_gold: total interior gold boundary positions.
        n_zwsp_positions: interior positions with a ZWSP on either side.
        n_gold_at_zwsp: how many of those positions are gold boundaries.
    """

    n_zwsp: int
    n_chars: int
    n_gold: int
    n_zwsp_positions: int
    n_gold_at_zwsp: int

    @property
    def present(self) -> bool:
        """Value for `CorpusManifest.zwsp_present`: does ZWSP occur at all."""
        return self.n_zwsp > 0

    @property
    def gold_at_zwsp_rate(self) -> float:
        """Fraction of gold boundaries that sit at a ZWSP.

        Near 1.0 would mean the annotation is just "split on ZWSP" and the
        corpus is NOT usable for Tier-1 claims.
        """
        return self.n_gold_at_zwsp / self.n_gold if self.n_gold else 0.0

    @property
    def zwsp_is_gold_rate(self) -> float:
        """Fraction of ZWSP positions that are gold boundaries.

        The converse check: near 1.0 means ZWSP is a perfect-precision oracle.
        Only meaningful alongside `gold_at_zwsp_rate`; on a corpus with a handful
        of stray ZWSPs it is a ratio of tiny numbers.
        """
        return self.n_gold_at_zwsp / self.n_zwsp_positions if self.n_zwsp_positions else 0.0


def zwsp_stats(records: Iterable[Segmented]) -> ZwspStats:
    """Measure ZWSP against gold, rather than trusting a docstring.

    A position ``i`` counts as sitting at a ZWSP when ``text[i-1]`` or
    ``text[i]`` is U+200B -- both sides, because a separator produces a boundary
    on each of its flanks.
    """
    from unsegbench.positions import gold_boundaries

    n_zwsp = n_chars = n_gold = n_zpos = n_gold_at = 0
    for rec in records:
        text = rec.text
        n = len(text)
        n_chars += n
        n_zwsp += text.count(ZWSP)
        gold = gold_boundaries(rec)
        n_gold += len(gold)
        if ZWSP not in text:
            continue
        zpos = {i for i in range(1, n) if text[i - 1] == ZWSP or text[i] == ZWSP}
        n_zpos += len(zpos)
        n_gold_at += len(gold & zpos)
    return ZwspStats(
        n_zwsp=n_zwsp,
        n_chars=n_chars,
        n_gold=n_gold,
        n_zwsp_positions=n_zpos,
        n_gold_at_zwsp=n_gold_at,
    )


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------


def _verify(path: Path, sha256: str) -> Path:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != sha256:
        path.unlink(missing_ok=True)
        raise IntegrityError(f"{path.name}: sha256 {got} != expected {sha256}")
    return path


def _ensure_local(raw_dir: Path, name: str, url: str, sha256: str) -> Path:
    """Materialise one upstream file into ``raw_dir``, hash-verified.

    Prefers `unsegbench.fetch.http.download` -- the shared, provenance-recording
    path. That is a Track-B stub today, so this falls back to a minimal direct
    GET. The fallback keeps the mandatory sha256 check; it drops only resume and
    ``_download.json``, and disappears the moment the real downloader lands.
    """
    dest = raw_dir / name
    if dest.exists():
        return _verify(dest, sha256)
    raw_dir.mkdir(parents=True, exist_ok=True)

    from unsegbench.fetch.http import download

    try:
        return download(url, dest, sha256=sha256)
    except NotImplementedError:
        pass

    import httpx

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise FetchError(f"could not fetch khPOS artifact {url!r}: {exc}") from exc
    tmp.replace(dest)
    return _verify(dest, sha256)


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------


class KhPOS(CorpusSpec):
    """Manually segmented and POS-tagged Khmer. Licence-gated, non-redistributable."""

    corpus_id = "khpos"
    lang = "km"
    script = "Khmr"
    convention = "khpos"
    license = "CC BY-NC-SA 4.0"
    redistributable = False
    source_url = _SOURCE_URL
    version = "draft-1.0"
    #: Empty on purpose: khPOS's spaces are annotation delimiters, so the spans
    #: tile ``text`` exactly. Anything uncovered means the parser dropped data.
    gap_charset = ""
    #: Gate key for `cache.require_license`.
    license_key = LICENSE_KEY
    notes = (
        "Fetched from upstream GitHub, not HF: seanghay/khPOS ships only a dataset "
        "script. test = OPEN-TEST (held out); CLOSE-TEST is a verbatim subset of train "
        "and is not used. Compound markers '_' and '~' are annotation and are stripped. "
        "U+200B is incidental (2 occurrences in 602k chars), so Khmer segmentation here "
        "is NOT trivially recoverable from ZWSP -- see zwsp_stats()."
    )

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        """The two upstream files, with pinned digests."""
        return tuple(
            Artifact(name=name, url=url, sha256=sha) for name, url, sha in _ARTIFACTS.values()
        )

    def parse(self, raw_dir: Path, split: str) -> Iterator[Segmented]:
        """Yield one record per line; spans tile ``text``.

        Raises:
            LicenseNotAccepted: the CC BY-NC-SA gate has not been acknowledged.
        """
        entry = _ARTIFACTS.get(split)
        if entry is None:
            return
        cache.require_license(self.license_key)
        name, url, sha = entry
        path = _ensure_local(raw_dir, name, url, sha)
        text_raw = path.read_text(encoding="utf-8")

        idx = 0
        for line in text_raw.split("\n"):
            if not line.strip():
                continue
            parts: list[str] = []
            spans: list[Span] = []
            pos = 0
            for chunk in line.split(" "):
                if not chunk:
                    continue
                # rpartition: a token may itself contain '/', the POS tag never does.
                word, sep, _tag = chunk.rpartition("/")
                if not sep or not word:
                    raise ValueError(f"{self.corpus_id}/{split}/{idx:06d}: bad chunk {chunk!r}")
                for marker in _JOIN_MARKERS:
                    word = word.replace(marker, "")
                if not word:
                    continue  # a token of pure join markers; no text to contribute
                parts.append(word)
                spans.append((pos, pos + len(word)))
                pos += len(word)
            text = "".join(parts)
            if not text:
                continue
            yield Segmented(
                id=f"{self.corpus_id}/{split}/{idx:06d}",
                text=text,
                spans=tuple(spans),
            )
            idx += 1


ENTRIES: tuple[CorpusSpec, ...] = (KhPOS(),)
