"""Bridge to the official SIGHAN 2005 perl ``score`` script.

WHY THIS MODULE EXISTS. Everything downstream of the corpus loader is only as
trustworthy as the offsets it produces, and offsets are exactly the kind of
thing that is wrong by one forever without anybody noticing. The SIGHAN bakeoff
scorer is the field standard for Chinese word segmentation, it has been run on
these four corpora by hundreds of papers, and it is written in a completely
different language against a completely different representation (whitespace-
delimited word strings, not codepoint spans). So if our word-level P/R/F agrees
with it to 1e-6 on real data, then the loader, the offset extraction, the span
logic and `metrics.core.word_counts` are all validated in a single shot. If it
disagrees, nothing downstream is worth reporting.
# improved

HOW THE PERL SCRIPT WORKS, since we have to match it exactly. For each line it
splits on whitespace into words, writes the two word sequences to temp files one
# improved
word per line, runs ``diff -y`` and counts markers in the side-by-side output:
``|`` substitution, ``>`` insertion, ``<`` deletion. Then

    R = matched_gold / total_gold,  P = matched_sys / total_sys,  F = 2PR/(P+R)

plus OOV/IV recall against the training word list passed as ``argv[1]``. Our
`metrics.core.word_counts` uses the equivalent SET formulation over induced
partitions, which is deterministic where diff's LCS is not; the two agree
exactly whenever no two distinct positions carry the same word string in a way
that lets LCS match them out of order.

TWO NON-OBVIOUS THINGS THIS MODULE HAS TO HANDLE.

1. **The script only prints P/R/F rounded to three decimals** (``sprintf
   "%2.3f"``), so a 1e-6 comparison against the printed number is impossible.
   But the script also prints its entire ``diff -y`` output verbatim. So we
   recover the EXACT integer numerators by re-applying the script's own regexes
   to those printed diff lines, and then self-check that our reconstruction
   reproduces its rounded values. That is what makes a 1e-6 assertion meaningful
   rather than a 5e-4 one dressed up.

2. **The four corpora do not share a word delimiter.** PKU and MSR use two
   spaces, AS uses U+3000 IDEOGRAPHIC SPACE, CityU uses a single space. The perl
   script normalises U+3000 to a space and splits on ``\\s+``, so it does not
   care -- but any byte-exactness check against the gold file does, and a test
   that cannot re-emit the gold file byte for byte has not really verified that
   we read it correctly. Hence `to_sighan_format` takes the convention as a
   parameter and `detect_convention` recovers it from a gold file.

Nothing here vendors SIGHAN data. The scorer is fetched to the cache on demand
(it carries its own permissive licence); the corpora are Track B's problem and
stay out of the repo entirely (CONTRACTS.md sec.6).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unsegbench.errors import UnsegbenchError
from unsegbench.metrics.core import word_counts
from unsegbench.positions import boundaries_to_spans, gold_boundaries
from unsegbench.types import Segmented

__all__ = [
    "CONVENTIONS",
    "SCORE_SCRIPT_MIRRORS",
    "PerlScore",
    "SighanConvention",
    "compare_to_perl",
    "detect_convention",
    "ensure_score_script",
    "find_score_script",
    "run_perl_scorer",
    "to_sighan_format",
    "words_from_boundaries",
]


class SighanScorerError(UnsegbenchError):
    """The perl scorer could not be run, or its output could not be parsed."""


# --------------------------------------------------------------------------
# Whitespace conventions
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SighanConvention:
    """One corpus's on-disk whitespace convention.

    Attributes:
        name: the corpus key, ``"pku" | "msr" | "as" | "cityu"``.
        delimiter: the exact string written BETWEEN two words.
        line_ending: the exact line terminator.
        trailing_newline: whether the file ends with ``line_ending``.
    """

    name: str
    delimiter: str
    line_ending: str = "\n"
    trailing_newline: bool = True


#: U+3000 IDEOGRAPHIC SPACE -- the AS (Academia Sinica) delimiter. The perl
#: script rewrites it to U+0020 before splitting (its line 89, matching the raw
#: UTF-8 bytes ``\xe3\x80\x80``), so scoring is unaffected; byte-exactness is not.
IDEOGRAPHIC_SPACE = "　"

#: Per-corpus delimiters. Established by Track B against the real gold files:
#: PKU and MSR write two ASCII spaces, AS writes U+3000, CityU writes one space.
#: Getting this wrong makes any byte-exactness check meaningless.
CONVENTIONS: dict[str, SighanConvention] = {
    "pku": SighanConvention("pku", "  "),
    "msr": SighanConvention("msr", "  "),
    "as": SighanConvention("as", IDEOGRAPHIC_SPACE),
    "cityu": SighanConvention("cityu", " "),
}


def _resolve_convention(convention: str | SighanConvention) -> SighanConvention:
    if isinstance(convention, SighanConvention):
        return convention
    try:
        return CONVENTIONS[convention.lower()]
    except KeyError:
        raise ValueError(
            f"unknown SIGHAN convention {convention!r}; known: {sorted(CONVENTIONS)}. "
            "Pass a SighanConvention explicitly for anything else."
        ) from None


def detect_convention(text: str, *, name: str = "detected") -> SighanConvention:
    """Recover a file's whitespace convention from its own bytes.
# improved

    Used to assert that we re-emit a gold file byte for byte rather than merely
    close to it. Picks the most frequent inter-word run of horizontal whitespace.

    Args:
        text: the decoded contents of a SIGHAN gold or training file.
        name: label for the returned convention.

    Returns:
        The `SighanConvention` whose delimiter, line ending and trailing-newline
        flag reproduce ``text`` most often.

    Raises:
        ValueError: if ``text`` contains no line with an inter-word delimiter.
    """
    line_ending = "\r\n" if "\r\n" in text else "\n"
    body = text[: -len(line_ending)] if text.endswith(line_ending) else text
    trailing = text.endswith(line_ending)
    runs: dict[str, int] = {}
    for line in body.split(line_ending):
        for run in re.findall(r"[ \t　 ]+", line.strip("\r")):
            runs[run] = runs.get(run, 0) + 1
    if not runs:
        raise ValueError("no inter-word whitespace found; is this a SIGHAN segmentation file?")
    delimiter = max(runs.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
    return SighanConvention(name, delimiter, line_ending, trailing)


# --------------------------------------------------------------------------
# Emitting a segmentation
# --------------------------------------------------------------------------

Sentence = Segmented | Sequence[str]


def words_from_boundaries(
    text: str, boundaries: Iterable[int], *, mask: Iterable[int] | None = None
) -> tuple[str, ...]:
    """The word strings of the partition INDUCED by a boundary set.

    This is the same object `metrics.core.word_counts` scores, which is the
    whole point: the perl script must be shown exactly what our metric counted,
    not a differently-derived word list that happens to look similar.

    Args:
        text: the sentence.
        boundaries: interior cut positions.
        mask: if given, boundaries are intersected with it first (the metric
            intersects both gold and pred with the mask before counting).

    Returns:
        The induced words, in order. They concatenate back to ``text``.
    """
    b = frozenset(boundaries)
    if mask is not None:
        b &= frozenset(mask)
    return tuple(text[s:e] for s, e in boundaries_to_spans(b, len(text)))


def _words_of(sentence: Sentence) -> tuple[str, ...]:
    if isinstance(sentence, Segmented):
        return sentence.words
    return tuple(sentence)


def to_sighan_format(
    sentences: Iterable[Sentence],
    convention: str | SighanConvention,
) -> str:
    """Emit a segmentation in SIGHAN's on-disk format.

    Space-delimited words, one sentence per line, encoded UTF-8 by the caller.
    The output reproduces the target corpus's own whitespace convention BYTE
    FOR BYTE, trailing newline included -- see `detect_convention` for how that
    claim is checked against a real gold file.

    Args:
        sentences: `Segmented` records (their gold ``words`` are emitted) or
            plain word sequences, e.g. from `words_from_boundaries`.
        convention: ``"pku" | "msr" | "as" | "cityu"``, or a `SighanConvention`.
            This is a parameter and not a constant because the four corpora
            genuinely differ: PKU/MSR double space, AS U+3000, CityU single
            space.

    Returns:
        The full file contents as a ``str``.

    Raises:
        ValueError: if any word is empty or contains the delimiter -- either
            would make the file unparseable by the scorer and is a bug upstream,
            not something to paper over.
    """
    conv = _resolve_convention(convention)
    lines: list[str] = []
    for i, sentence in enumerate(sentences):
        words = _words_of(sentence)
        for w in words:
            if not w:
                raise ValueError(
                    f"sentence {i}: empty word -- cannot be represented in SIGHAN format"
                )
            if any(ch.isspace() or ch == IDEOGRAPHIC_SPACE for ch in w):
                raise ValueError(
                    f"sentence {i}: word {w!r} contains whitespace, which the scorer would "
                    "split on -- the segmentation would be silently altered"
                )
        lines.append(conv.delimiter.join(words))
    out = conv.line_ending.join(lines)
    if lines and conv.trailing_newline:
        out += conv.line_ending
    return out


def write_sighan_file(
    path: Path,
    sentences: Iterable[Sentence],
    convention: str | SighanConvention,
) -> Path:
    """`to_sighan_format` straight to a UTF-8 file with no newline translation."""
    text = to_sighan_format(sentences, convention)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


# --------------------------------------------------------------------------
# Locating the scorer
# --------------------------------------------------------------------------

#: The scorer ships inside ``icwb2-data.zip`` under ``scripts/score``. It is
#: ~200 lines and carries its own permissive SIGHAN licence, so unlike the
#: corpora it may be cached locally. Mirrors, in preference order.
SCORE_SCRIPT_MIRRORS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/yuikns/icwb2-data/master/scripts/score",
    "https://raw.githubusercontent.com/chantera/icwb2-data/master/scripts/score",
)

_SCORE_SCRIPT_FINGERPRINT = b"TOTAL TRUE WORDS RECALL"


def _cache_root() -> Path:
    env = os.environ.get("UNSEGBENCH_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "unsegbench"


def find_score_script() -> Path | None:
    """Locate an already-present official ``score`` script.

    Search order: ``$UNSEGBENCH_SIGHAN_SCORE``, then the cache root (both the
    standalone copy and the one inside an extracted ``icwb2-data``, which Track
    B's loader unpacks).

    Returns:
        The path, or ``None`` if it has not been fetched yet.
    """
    env = os.environ.get("UNSEGBENCH_SIGHAN_SCORE")
    candidates: list[Path] = [Path(env)] if env else []
    root = _cache_root()
    candidates += [
        root / "sighan" / "score",
        root / "sighan" / "icwb2-data" / "scripts" / "score",
        root / "icwb2-data" / "scripts" / "score",
    ]
    candidates += sorted(root.glob("**/icwb2-data/scripts/score"))
    for p in candidates:
        if p.is_file() and _SCORE_SCRIPT_FINGERPRINT in p.read_bytes():
            return p
    return None


def ensure_score_script(*, allow_download: bool = True, timeout: float = 30.0) -> Path:
    """Return the official ``score`` script, fetching it to the cache if needed.

    Args:
        allow_download: if False, never touch the network -- raise instead.
        timeout: per-mirror HTTP timeout in seconds.

    Returns:
        Path to a verified copy of the script.

    Raises:
        SighanScorerError: if it is absent and cannot be fetched.
# improved
    """
    found = find_score_script()
    if found is not None:
        return found
    if not allow_download:
        raise SighanScorerError(
            "SIGHAN score script not found. Set $UNSEGBENCH_SIGHAN_SCORE or allow the download."
        )
    import httpx

    dest = _cache_root() / "sighan" / "score"
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in SCORE_SCRIPT_MIRRORS:
        try:
            resp = httpx.get(url, timeout=timeout, follow_redirects=True)
            resp.raise_for_status()
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
        if _SCORE_SCRIPT_FINGERPRINT not in resp.content:
            errors.append(f"{url}: content does not look like the SIGHAN scorer")
            continue
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.chmod(0o755)
        tmp.replace(dest)
        return dest
    raise SighanScorerError("could not fetch the SIGHAN score script:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------
# Running and parsing the scorer
# improved
# --------------------------------------------------------------------------

# The script's own regexes, transliterated to byte patterns. Perl runs without
# improved
# `use utf8`, so it matches BYTES -- and so do we, which also makes us immune to
# the mid-codepoint truncation `diff -y` performs at its 61-column gutter.
_RE_TRUTH_WORD = re.compile(rb"^([^\s]+)\s")
_RE_TRUTH_CHANGED = re.compile(rb"^[^\s]+\s.*\s[|><]\s")
_RE_TEST_CHANGED = re.compile(rb"\s[|><]\s.*[^\s]$")
_RE_BLOCK_HEADER = re.compile(rb"^--.*-------.*----(\d+)$")

_PER_LINE_KEYS: dict[bytes, str] = {
    b"INSERTIONS:": "insertions",
    b"DELETIONS:": "deletions",
    b"SUBSTITUTIONS:": "substitutions",
    b"NCHANGE:": "n_change",
    b"NTRUTH:": "n_gold_words",
    b"NTEST:": "n_test_words",
    b"TRUE WORDS RECALL:": "recall_reported",
    b"TEST WORDS PRECISION:": "precision_reported",
}

_SUMMARY_KEYS: dict[bytes, str] = {
    b"TOTAL INSERTIONS:": "insertions",
    b"TOTAL DELETIONS:": "deletions",
    b"TOTAL SUBSTITUTIONS:": "substitutions",
    b"TOTAL NCHANGE:": "n_change",
    b"TOTAL TRUE WORD COUNT:": "n_gold_words",
    b"TOTAL TEST WORD COUNT:": "n_test_words",
    b"TOTAL TRUE WORDS RECALL:": "recall_reported",
    b"TOTAL TEST WORDS PRECISION:": "precision_reported",
    b"F MEASURE:": "f1_reported",
    b"OOV Rate:": "oov_rate_reported",
    b"OOV Recall Rate:": "oov_recall_reported",
    b"IV Recall Rate:": "iv_recall_reported",
}


@dataclass(frozen=True, slots=True)
class PerlScore:
    """What the official scorer said, at full precision.

    ``*_reported`` fields are the script's own 3-decimal ``sprintf`` output.
    ``precision``/``recall``/``f1`` are reconstructed from exact integer
    numerators recovered from the printed ``diff`` blocks, which is the only way
    a 1e-6 comparison means anything.
    """

    precision: float
    recall: float
    f1: float
    matched_gold: int
    matched_test: int
    n_gold_words: int
    n_test_words: int
    insertions: int
    deletions: int
    substitutions: int
    n_change: int
    n_lines: int
    oov_recall: float | None
    iv_recall: float | None
    oov_rate: float | None
    reported: dict[str, float | None] = field(default_factory=dict)
    stdout: bytes = b""

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "oov_recall": self.oov_recall,
            "iv_recall": self.iv_recall,
            "oov_rate": self.oov_rate,
            "matched_gold": self.matched_gold,
            "matched_test": self.matched_test,
            "n_gold_words": self.n_gold_words,
            "n_test_words": self.n_test_words,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "substitutions": self.substitutions,
            "n_change": self.n_change,
            "n_lines": self.n_lines,
            "reported": dict(self.reported),
        }


def _num(raw: bytes) -> float | None:
    """Parse a scorer field. ``--`` means "undefined", not zero."""
    s = raw.strip()
    if s in (b"--", b""):
        return None
    return float(s)


def _parse_report(out: bytes) -> PerlScore:
    """Recover exact counts from the scorer's full report.

    The script prints its ``diff -y`` output verbatim before each per-line
    summary, so we re-apply its own two counting regexes to those lines and get
    the integer numerators it rounded away.
    """
    matched_gold = matched_test = 0
    n_lines = 0
    summary: dict[str, float | None] = {}
    reported: dict[str, float | None] = {}

    diff_lines: list[bytes] = []
    in_block = False
    pending_truth = pending_test = 0

    for raw in out.split(b"\n"):
        line = raw[:-1] if raw.endswith(b"\r") else raw

        if line.startswith(b"=== "):
            in_block = False
            payload = line[4:]
            for prefix, key in _SUMMARY_KEYS.items():
                if payload.startswith(prefix):
                    summary[key] = _num(payload[len(prefix) :])
                    break
            continue
        if line.startswith(b"###\t"):
            in_block = False
            continue

        header = _RE_BLOCK_HEADER.match(line)
        if header is not None and not in_block:
            in_block = True
            diff_lines = []
            pending_truth = pending_test = 0
            n_lines += 1
            continue

        if not in_block:
            continue

        matched_key: str | None = None
        for prefix, key in _PER_LINE_KEYS.items():
            if line.startswith(prefix):
                matched_key = key
                value = _num(line[len(prefix) :])
                if key == "n_gold_words":
                    pending_truth = int(value or 0)
                elif key == "n_test_words":
                    pending_test = int(value or 0)
                break
        if matched_key is None:
            diff_lines.append(line)
            continue
        if matched_key == "precision_reported":
            # End of this sentence's block: fold in its exact numerators.
            gold_err, test_err = _count_diff_errors(diff_lines)
            matched_gold += pending_truth - gold_err
            matched_test += pending_test - test_err
            in_block = False

    if "n_gold_words" not in summary:
        raise SighanScorerError(
            "could not find the scorer's summary block; output was:\n"
            + out.decode("utf-8", "replace")[-2000:]
        )

    reported = dict(summary)
    n_gold = int(summary.get("n_gold_words") or 0)
    n_test = int(summary.get("n_test_words") or 0)
    recall = matched_gold / n_gold if n_gold else 1.0
    precision = matched_test / n_test if n_test else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    _verify_reconstruction(precision, recall, f1, reported)

    return PerlScore(
        precision=precision,
        recall=recall,
        f1=f1,
        matched_gold=matched_gold,
        matched_test=matched_test,
        n_gold_words=n_gold,
        n_test_words=n_test,
        insertions=int(summary.get("insertions") or 0),
        deletions=int(summary.get("deletions") or 0),
        substitutions=int(summary.get("substitutions") or 0),
        n_change=int(summary.get("n_change") or 0),
        n_lines=n_lines,
# improved
        oov_recall=summary.get("oov_recall_reported"),
        iv_recall=summary.get("iv_recall_reported"),
        oov_rate=summary.get("oov_rate_reported"),
        reported=reported,
        stdout=out,
    )


def _count_diff_errors(diff_lines: Sequence[bytes]) -> tuple[int, int]:
    """``(gold_side_errors, test_side_errors)`` using the scorer's own regexes.

    A ``diff -y`` deletion line has nothing after the ``<`` marker, so it is a
    recall error but not a precision error; an insertion line starts with
    whitespace, so it is the reverse. That asymmetry is the script's semantics,
    not an accident, and is reproduced here rather than reasoned about.
    """
    gold_err = test_err = 0
    for line in diff_lines:
        if _RE_TRUTH_WORD.match(line) and _RE_TRUTH_CHANGED.match(line):
            gold_err += 1
        if _RE_TEST_CHANGED.search(line):
            test_err += 1
    return gold_err, test_err


def _verify_reconstruction(
    precision: float, recall: float, f1: float, reported: dict[str, float | None]
) -> None:
    """Self-check: our exact values must round to the scorer's printed ones.

    Without this the reconstruction could drift from the script's semantics and
    the 1e-6 assertion downstream would be comparing us to ourselves.
    """
    for name, exact in (
        ("recall_reported", recall),
        ("precision_reported", precision),
        ("f1_reported", f1),
    ):
        want = reported.get(name)
        if want is None:
            continue
        if abs(round(exact, 3) - want) > 5e-4:
            raise SighanScorerError(
                f"reconstruction disagrees with the scorer's own printed {name}: "
                f"we computed {exact!r} (rounds to {round(exact, 3)}), it printed {want!r}. "
                "The diff-marker counting no longer matches the script's semantics."
            )


def run_perl_scorer(
    score_script: str | Path,
    training_words: str | Path,
    gold_file: str | Path,
    test_file: str | Path,
    *,
    perl: str | None = None,
    timeout: float = 600.0,
) -> PerlScore:
    """Run the official scorer and parse its report.

    Args:
        score_script: path to SIGHAN's ``scripts/score``.
        training_words: ``argv[1]``, the training word list used for OOV/IV.
        gold_file: ``argv[2]``, the gold segmentation.
        test_file: ``argv[3]``, our segmentation.
        perl: interpreter to use; defaults to ``$PERL`` or whatever is on PATH.
        timeout: seconds before giving up.

    Returns:
        A `PerlScore` with exact P/R/F plus the script's own rounded values.

    Raises:
        SighanScorerError: if perl is missing, the script fails, or the report
            cannot be parsed. Never falls back to an approximation -- a silently
            degraded cross-check is worse than none.
    """
    exe = perl or os.environ.get("PERL") or shutil.which("perl")
    if not exe:
        raise SighanScorerError("no perl interpreter found; cannot run the SIGHAN scorer")
    argv = [exe, str(score_script), str(training_words), str(gold_file), str(test_file)]
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise SighanScorerError(f"could not execute {argv[0]!r}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SighanScorerError(f"the SIGHAN scorer timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise SighanScorerError(
            f"scorer exited {proc.returncode}: {proc.stderr.decode('utf-8', 'replace')[:2000]}"
        )
    if b"=== SUMMARY:" not in proc.stdout:
        raise SighanScorerError(
            "scorer produced no summary (usually a usage error):\n"
            + proc.stdout.decode("utf-8", "replace")[:1000]
            + proc.stderr.decode("utf-8", "replace")[:1000]
        )
    return _parse_report(proc.stdout)


# --------------------------------------------------------------------------
# The cross-check itself
# --------------------------------------------------------------------------


def our_word_prf(
    records: Sequence[Segmented],
    predictions: Sequence[frozenset[int] | set[int]],
    masks: Sequence[frozenset[int] | set[int]] | None = None,
) -> dict[str, float | int]:
    """Micro-averaged word P/R/F from `metrics.core.word_counts`.

    Pooled over sentences (sum the numerators, then divide) because that is what
    the perl script does -- it accumulates ``$Truecount`` and ``$RawRecall``
    across all lines and divides once at the end. A macro average over sentences
    would be a different number and would not cross-check anything.

    Args:
        records: gold records, in the same order as the emitted file.
        predictions: predicted interior boundary sets, one per record.
        masks: scored universes, one per record. Defaults to the full
            interior range ``1..n-1`` (``mask="raw"``), which is the mask under
            which our word metrics reduce exactly to SIGHAN's.
    """
    if len(records) != len(predictions):
        raise ValueError(f"{len(records)} records but {len(predictions)} predictions")
    tp = n_pred = n_gold = 0
    for i, rec in enumerate(records):
        mask = masks[i] if masks is not None else frozenset(range(1, rec.n))
        t, p, g = word_counts(gold_boundaries(rec), predictions[i], mask, rec.n)
        tp += t
        n_pred += p
        n_gold += g
    precision = tp / n_pred if n_pred else 1.0
    recall = tp / n_gold if n_gold else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "w_tp": tp,
        "w_pred": n_pred,
        "w_gold": n_gold,
    }


def compare_to_perl(
    records: Sequence[Segmented],
    predictions: Sequence[frozenset[int] | set[int]],
    *,
    workdir: Path,
    convention: str | SighanConvention,
    score_script: str | Path,
    training_words: str | Path | None = None,
    masks: Sequence[frozenset[int] | set[int]] | None = None,
    perl: str | None = None,
) -> dict[str, Any]:
    """Score one prediction both ways and return our numbers, perl's, and the deltas.

    The gold file handed to perl is regenerated from the SAME induced partition
    our metric scores, so a disagreement can only come from the metric, never
    from a formatting difference in what each side was shown.

    Args:
        records: gold records.
        predictions: predicted interior boundary sets, one per record.
        workdir: directory for the three temp files. Not cleaned up, so a
            failing test leaves the exact inputs behind to look at.
        convention: SIGHAN whitespace convention for the emitted files.
        score_script: path to ``scripts/score``.
        training_words: OOV/IV word list. Defaults to an empty file, which makes
            every word OOV -- fine, since P/R/F do not depend on it.
        masks: scored universes, one per record; defaults to ``raw``.
        perl: interpreter override.

    Returns:
        ``{"ours": {...}, "perl": {...}, "delta": {...}, "files": {...}}`` where
        ``delta`` holds signed ``ours - perl`` differences for precision, recall
        and f1.
    """
    if len(records) != len(predictions):
        raise ValueError(f"{len(records)} records but {len(predictions)} predictions")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    gold_lines: list[tuple[str, ...]] = []
    test_lines: list[tuple[str, ...]] = []
    for i, rec in enumerate(records):
        mask = masks[i] if masks is not None else frozenset(range(1, rec.n))
        gold_lines.append(words_from_boundaries(rec.text, gold_boundaries(rec), mask=mask))
        test_lines.append(words_from_boundaries(rec.text, predictions[i], mask=mask))

    gold_path = write_sighan_file(workdir / "gold.utf8", gold_lines, convention)
    test_path = write_sighan_file(workdir / "test.utf8", test_lines, convention)
    if training_words is None:
        training_words = workdir / "training_words.utf8"
        Path(training_words).write_bytes(b"")

    perl_score = run_perl_scorer(score_script, training_words, gold_path, test_path, perl=perl)
    ours = our_word_prf(records, predictions, masks)
    delta = {k: float(ours[k]) - getattr(perl_score, k) for k in ("precision", "recall", "f1")}
    return {
        "ours": ours,
        "perl": perl_score.as_dict(),
        "delta": delta,
        "max_abs_delta": max(abs(v) for v in delta.values()),
        "files": {
            "gold": str(gold_path),
            "test": str(test_path),
            "training_words": str(training_words),
        },
    }

# Enhanced

# Enhanced

# Enhanced

# Enhanced

# Updated

# Updated

# Updated
