"""THE HARD GATE: our word metrics vs. the official SIGHAN 2005 perl scorer.

Everything downstream of the corpus loader is only as trustworthy as the offsets
it produces, and an off-by-one in offsets is exactly the kind of bug that lives
forever unnoticed. The bakeoff ``score`` script is the field standard for Chinese
word segmentation, it has been run on these four corpora by hundreds of papers,
and it is written in a different language against a different representation
(whitespace-delimited word STRINGS, not codepoint spans). So if our
`metrics.core.word_counts` at ``mask="raw"`` reproduces it on real data, then the
loader, the offset extraction, the span logic and the metric are validated in one
shot. If it does not, nothing downstream is worth reporting.

WHAT IS ASSERTED, AND AT WHAT STRENGTH

1. Synthetic perturbations (`test_synthetic_*`): agreement to 1e-6 on P/R/F AND
   exact equality of the integer numerators. The integers are the real check --
   the script only prints ratios to three decimals, so a "matching" 3-decimal
   ratio is worth ~5e-4, whereas ``w_tp == matched_gold == matched_test`` is
   worth nothing less than exact.

2. Real tokenizers (`test_real_*`): exact agreement on sentences of <= 40
   characters, plus a BOUNDED, DOCUMENTED divergence on the long ones. See
   `test_real_tokenizer_long_sentences_diverge_only_by_the_lcs_artifact` for why
   that divergence is methodological and not a bug.

THE ONE PLACE THE TWO FORMULATIONS GENUINELY DIFFER. The perl script runs
``diff -y`` over the two word SEQUENCES, so its LCS is free to pair a word string
in the gold with an EQUAL word string at a DIFFERENT position in the prediction.
Our set formulation requires both edges of a word to align, so it will not count
that pair. LCS >= positional matching always, so perl can only ever score >= us,
never less -- which is exactly the sign we observe and assert below. The effect
is real but tiny (2 words in ~16k for xlm-r on PKU) and it never touches the word
TOTALS, only the intersection.

Because of that, the synthetic check perturbs boundaries under an explicit
# improved
LCS-safety guard (`_lcs_safe`) rather than pretending the ambiguity does not
exist: a perturbation that could be re-paired out of order is redrawn, and the
handful of sentences where no safe draw exists are left unperturbed and counted.
That keeps the 1e-6 assertion honest instead of quietly weakening it.

Perturbations MOVE boundaries rather than deleting them, so induced words stay
short. ``diff -y`` truncates each side of its side-by-side output at the gutter
(~61 columns, i.e. ~20 Han characters), and a truncated word would silently stop
comparing equal -- a deletion-based perturbation would be measuring diff's column
width rather than our metric.
"""

from __future__ import annotations

# import random
import shutil
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from unsegbench.build import is_built, load_corpus
from unsegbench.corpora.registry import get_corpus
from unsegbench.errors import UnsegbenchError
from unsegbench.positions import boundaries_to_spans, gold_boundaries
from unsegbench.sighan_bridge import (
    CONVENTIONS,
    IDEOGRAPHIC_SPACE,
    SighanScorerError,
    compare_to_perl,
    detect_convention,
    ensure_score_script,
    to_sighan_format,
)
from unsegbench.tok.loader import get_adapter
from unsegbench.tok.registry import get_tokenizer_spec
from unsegbench.types import Segmented

pytestmark = [pytest.mark.network, pytest.mark.slow, pytest.mark.perl]

#: corpus id -> the whitespace convention its gold file is written in.
CORPORA: dict[str, str] = {
    "sighan_as": "as",
    "sighan_cityu": "cityu",
    "sighan_pku": "pku",
#     "sighan_msr": "msr",
}

TOKENIZERS: tuple[str, ...] = ("char", "cl100k_base", "xlm-r")

#: Agreement demanded of the synthetic check and of the short-sentence real
#: check. Not 1e-3 dressed up: the integer numerators are compared too.
TOL = 1e-6

#: Sentences per (corpus, k) synthetic run.
SYNTH_SENTENCES = 300
#: Sentences per (tokenizer, corpus) real run.
REAL_SENTENCES = 400

#: Above this length a sentence carries enough repeated word strings for diff's
#: LCS to pair some of them out of order. Chosen as the largest round number at
#: which every tokenizer still agrees with perl EXACTLY on every corpus.
SHORT_MAX_CHARS = 40

#: The observed long-sentence divergence is ~1.2e-4. This bound exists to catch a
#: real regression while documenting the artifact; it is deliberately two orders
#: of magnitude tighter than the 3-decimal precision the script itself prints.
LCS_ARTIFACT_BOUND = 1e-3

#: How many redraws before giving up on finding an LCS-unambiguous perturbation.
PERTURB_TRIES = 32


# --------------------------------------------------------------------------
# Fixtures: everything that can be absent skips, nothing degrades silently
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def score_script() -> Path:
    """The official ``scripts/score``, or a clean skip."""
    if shutil.which("perl") is None:
        pytest.skip("no perl interpreter; the SIGHAN cross-check cannot run")
    try:
        return ensure_score_script()
    except SighanScorerError as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"SIGHAN score script unavailable: {exc}")


@cache
def _records(corpus_id: str) -> tuple[Segmented, ...]:
    """The built test split, cached across tests. Skips if Track B has not built it."""
    if not is_built(corpus_id):
        pytest.skip(f"{corpus_id} is not built in the cache")
    return tuple(load_corpus(corpus_id, split="test"))


@cache
def _lang(corpus_id: str) -> str:
    return get_corpus(corpus_id).lang


def _adapter(tokenizer_id: str, lang: str) -> Any:
    """A loaded adapter, or a skip -- a missing tokenizer is not a metric failure."""
    try:
        return get_adapter(get_tokenizer_spec(tokenizer_id), lang)
    except UnsegbenchError as exc:
        pytest.skip(f"tokenizer {tokenizer_id} unavailable: {exc}")


# --------------------------------------------------------------------------
# Perturbation
# --------------------------------------------------------------------------


def _move_boundaries(
    gold: frozenset[int], n: int, k: int, rng: random.Random
) -> tuple[frozenset[int], int]:
    """Move up to ``k`` gold boundaries one position left or right.

    Moving, not deleting: the induced words stay within one character of their
    gold length, which keeps every line of the scorer's temp files far short of
    the ``diff -y`` gutter. Returns the new boundary set and how many moves
    actually landed (a boundary hemmed in on both sides cannot move).
    """
    b = set(gold)
    order = sorted(b)
    rng.shuffle(order)
    moved = 0
    for pos in order:
        if moved >= k:
            break
        candidates = [pos - 1, pos + 1]
        rng.shuffle(candidates)
        for cand in candidates:
            if 1 <= cand <= n - 1 and cand not in b:
                b.discard(pos)
                b.add(cand)
                moved += 1
                break
    return frozenset(b), moved


def _lcs_safe(text: str, gold: frozenset[int], pred: frozenset[int]) -> bool:
    """True when ``diff``'s LCS cannot out-score the positional matching.

    The two formulations can only disagree when some word string occurs both in
    the unmatched part of the gold and in the unmatched part of the prediction --
    that is the pair LCS is free to align and we are not. If those two string
    sets are disjoint, no such re-pairing exists and the counts must coincide.
    """
    gold_spans = boundaries_to_spans(gold, len(text))
    pred_spans = boundaries_to_spans(pred, len(text))
    matched = set(gold_spans) & set(pred_spans)
    gold_only = {text[s:e] for sp in gold_spans if sp not in matched for s, e in (sp,)}
    pred_only = {text[s:e] for sp in pred_spans if sp not in matched for s, e in (sp,)}
    return not (gold_only & pred_only)


def _perturb(rec: Segmented, k: int, rng: random.Random) -> tuple[frozenset[int], bool]:
    """An LCS-unambiguous ``k``-move perturbation, or the gold unchanged.

    Returns ``(boundaries, perturbed)``. Falling back to the gold is recorded
    rather than hidden, and the caller asserts the fallback stays rare -- a test
    that quietly degraded to ``k=0`` everywhere would pass while checking nothing.
    """
    gold = gold_boundaries(rec)
    if k == 0:
        return gold, True
    for _ in range(PERTURB_TRIES):
        cand, moved = _move_boundaries(gold, rec.n, k, rng)
        if moved == k and _lcs_safe(rec.text, gold, cand):
            return cand, True
    return gold, False


def _assert_agrees(result: dict[str, Any], tol: float, label: str) -> float:
    """Ratios within ``tol`` AND integer numerators equal. Returns max |delta|."""
    ours, perl = result["ours"], result["perl"]
    assert ours["w_gold"] == perl["n_gold_words"], f"{label}: gold word totals differ"
    assert ours["w_pred"] == perl["n_test_words"], f"{label}: predicted word totals differ"
    assert ours["w_tp"] == perl["matched_gold"] == perl["matched_test"], (
        f"{label}: matched-word counts differ -- ours {ours['w_tp']}, "
        f"perl {perl['matched_gold']}/{perl['matched_test']}"
    )
    for key, delta in result["delta"].items():
        assert abs(delta) <= tol, f"{label}: {key} differs by {delta:.3e} (> {tol:.0e})"
    return float(result["max_abs_delta"])


# --------------------------------------------------------------------------
# 1. Synthetic cross-check
# --------------------------------------------------------------------------

# improved

@pytest.mark.parametrize("corpus_id", sorted(CORPORA))
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_synthetic_moved_boundaries_match_perl(
    corpus_id: str, k: int, score_script: Path, tmp_path: Path
) -> None:
    """Move k gold boundaries; our P/R/F must equal the official scorer's.

    ``k=0`` is the oracle and must come back at exactly 1.0 on both sides -- if
# improved
    the loader's offsets were shifted, that is where it shows first. ``k=1..3``
    walk the metric away from the oracle so the check is not just verifying that
    two identical files compare equal.
    """
    records = [r for r in _records(corpus_id) if r.n >= 4][:SYNTH_SENTENCES]
    assert len(records) >= 100, f"{corpus_id}: only {len(records)} usable sentences"

    rng = random.Random(20260729 + k)
    drawn = [_perturb(rec, k, rng) for rec in records]
    predictions = [b for b, _ in drawn]
    fallbacks = sum(1 for _, ok in drawn if not ok)
    assert fallbacks <= len(records) // 5, (
        f"{corpus_id} k={k}: {fallbacks}/{len(records)} sentences admitted no "
        "LCS-unambiguous perturbation; the check has degraded towards k=0"
    )

    result = compare_to_perl(
        records,
        predictions,
        workdir=tmp_path / "sighan",
        convention=CORPORA[corpus_id],
        score_script=score_script,
    )
    max_abs = _assert_agrees(result, TOL, f"{corpus_id} k={k}")

    if k == 0:
        assert result["ours"]["precision"] == 1.0
        assert result["ours"]["recall"] == 1.0
        assert result["perl"]["matched_gold"] == result["perl"]["n_gold_words"]
    else:
        # The perturbation must actually have cost us something, or k=1..3 would
        # be three more copies of the k=0 test.
        assert result["ours"]["f1"] < 1.0, f"{corpus_id} k={k}: perturbation had no effect"
    assert max_abs <= TOL


# --------------------------------------------------------------------------
# 2. Real tokenizer cross-check
# --------------------------------------------------------------------------


def _real_predictions(
    corpus_id: str, tokenizer_id: str, limit: int
) -> tuple[list[Segmented], list[frozenset[int]]]:
    records = list(_records(corpus_id)[:limit])
    adapter = _adapter(tokenizer_id, _lang(corpus_id))
    return records, [adapter.encode(rec.text).boundaries for rec in records]


@pytest.mark.parametrize("corpus_id", sorted(CORPORA))
@pytest.mark.parametrize("tokenizer_id", TOKENIZERS)
def test_real_tokenizer_matches_perl_on_short_sentences(
    corpus_id: str, tokenizer_id: str, score_script: Path, tmp_path: Path
) -> None:
    """Real tokenizer output, sentences <= 40 chars: agreement must be EXACT.

    This is the strongest form of the gate. Real tokenizers place boundaries
    nothing like the gold, so the induced partitions are genuinely different
    objects -- and on sentences short enough that diff's LCS has no room to pair
    equal word strings out of order, our set formulation and the official
    scorer's sequence formulation return the same integers, every time.
    """
    records, predictions = _real_predictions(corpus_id, tokenizer_id, REAL_SENTENCES)
    short = [(r, p) for r, p in zip(records, predictions, strict=True) if r.n <= SHORT_MAX_CHARS]
    assert len(short) >= 50, f"{corpus_id}: only {len(short)} sentences <= {SHORT_MAX_CHARS} chars"

    result = compare_to_perl(
        [r for r, _ in short],
        [p for _, p in short],
        workdir=tmp_path / "sighan",
        convention=CORPORA[corpus_id],
        score_script=score_script,
    )
    max_abs = _assert_agrees(result, TOL, f"{tokenizer_id} on {corpus_id} (<= {SHORT_MAX_CHARS}c)")
    assert max_abs == 0.0, (
        f"{tokenizer_id} on {corpus_id}: expected EXACT agreement on short sentences, "
        f"got {max_abs:.3e}"
    )


@pytest.mark.parametrize("corpus_id", sorted(CORPORA))
@pytest.mark.parametrize("tokenizer_id", TOKENIZERS)
def test_real_tokenizer_long_sentences_diverge_only_by_the_lcs_artifact(
    corpus_id: str, tokenizer_id: str, score_script: Path, tmp_path: Path
) -> None:
    """Long sentences included: bounded, one-signed divergence. NOT papered over.

    With the full length range, ``char`` and ``cl100k_base`` still agree with the
    official scorer to EXACTLY 0.0, and ``xlm-r`` shows ~1e-4 on PKU and MSR.
    That residue is methodological, not a defect:

      * the scorer diffs word SEQUENCES, so its LCS may pair an identical word
        string at two DIFFERENT positions and count it as matched;
      * our set formulation requires both edges of the word to align, so it does
        not count that pair.

# improved
    Two consequences are asserted here, because they are what distinguishes the
    artifact from a genuine bug:

      1. The word TOTALS still agree exactly. Only the intersection moves, so
         nothing about how we cut the text is in question -- 2 words in ~16k get
         re-paired.
      2. The divergence is one-signed: LCS >= positional matching by definition,
         so perl can only score at or above us. A delta of the opposite sign
         would mean we are counting matches the official scorer does not, and
         that WOULD be a bug.
    """
    records, predictions = _real_predictions(corpus_id, tokenizer_id, REAL_SENTENCES)
    result = compare_to_perl(
        records,
        predictions,
        workdir=tmp_path / "sighan",
        convention=CORPORA[corpus_id],
        score_script=score_script,
    )
    ours, perl, delta = result["ours"], result["perl"], result["delta"]
    label = f"{tokenizer_id} on {corpus_id} (all lengths)"

    # (1) the partition itself is not in dispute -- only the pairing of equal strings
    assert ours["w_gold"] == perl["n_gold_words"], f"{label}: gold word totals differ"
    assert ours["w_pred"] == perl["n_test_words"], f"{label}: predicted word totals differ"
    assert perl["matched_gold"] == perl["matched_test"], f"{label}: perl's own numerators differ"

#     # (2) one-signed: the LCS can only find MORE matches than positional alignment
    assert ours["w_tp"] <= perl["matched_gold"], (
        f"{label}: we counted {ours['w_tp']} matched words but the official scorer "
        f"counted only {perl['matched_gold']} -- we are matching words diff will not, "
        "which is a bug, not the LCS artifact"
    )
    for key, value in delta.items():
        assert value <= TOL, f"{label}: {key} delta {value:.3e} has the wrong sign"

# improved
    # (3) bounded
    max_abs = float(result["max_abs_delta"])
    assert max_abs < LCS_ARTIFACT_BOUND, (
        f"{label}: divergence {max_abs:.3e} exceeds the documented LCS-artifact "
        f"bound {LCS_ARTIFACT_BOUND:.0e}"
# improved
    )
    if tokenizer_id in ("char", "cl100k_base"):
#         # These two induce partitions whose words are single characters or short
        # runs that align positionally, so there is nothing for LCS to re-pair.
        assert max_abs == 0.0, f"{label}: expected exactly 0.0, got {max_abs:.3e}"


# --------------------------------------------------------------------------
# 3. Format edge cases
# --------------------------------------------------------------------------


def test_conventions_table_matches_the_corpora_on_disk() -> None:
    """The four corpora genuinely do not share a delimiter. Pin it."""
    assert CONVENTIONS["as"].delimiter == IDEOGRAPHIC_SPACE == "　"
    assert CONVENTIONS["cityu"].delimiter == " "
    assert CONVENTIONS["pku"].delimiter == "  "
    assert CONVENTIONS["msr"].delimiter == "  "


@pytest.mark.parametrize("corpus_id", sorted(CORPORA))
def test_detect_convention_recovers_the_modal_run_and_round_trips(corpus_id: str) -> None:
    """Emit real gold, then recover the convention from the bytes alone.

    A cross-check that cannot re-emit the gold file byte for byte has not really
    verified that we read it correctly, so this asserts the round trip and not
    merely that the delimiter looks plausible.
    """
    records = list(_records(corpus_id)[:200])
    key = CORPORA[corpus_id]
    text = to_sighan_format(records, key)

    detected = detect_convention(text, name=corpus_id)
    assert detected.delimiter == CONVENTIONS[key].delimiter
    assert detected.line_ending == "\n"
    assert detected.trailing_newline is True
    assert to_sighan_format(records, detected) == text


# improved
def test_detect_convention_picks_the_mode_not_the_first_run() -> None:
    """Two lines in three use a double space, so the double space wins."""
    text = "a  b  c\nd e f\ng  h  i\n"
    assert detect_convention(text).delimiter == "  "

    ideographic = f"a{IDEOGRAPHIC_SPACE}b{IDEOGRAPHIC_SPACE}c\nd e f\n"
    assert detect_convention(ideographic).delimiter == IDEOGRAPHIC_SPACE


def test_to_sighan_format_refuses_a_word_containing_whitespace() -> None:
    """The format cannot represent a whitespace-bearing word, so refusing is correct.

    The scorer splits every line on ``\\s+``. A word with a space inside it would
    silently become two words and the segmentation being scored would no longer
    be the one we computed -- a wrong number that looks right. Per CONTRACTS.md
    sec.1 delimiter whitespace is stripped at build time, so hitting this means a
    loader bug upstream, which is exactly what must not be papered over.
    """
    ascii_space = Segmented(id="x/test/000000", text="有 空格", spans=((0, 4),))
    with pytest.raises(ValueError, match="whitespace"):
        to_sighan_format([ascii_space], "pku")

    # U+3000 too: it is AS's delimiter and the scorer rewrites it to U+0020
    # before splitting, so it is every bit as destructive as an ASCII space.
    ideographic = Segmented(id="x/test/000001", text=f"有{IDEOGRAPHIC_SPACE}空", spans=((0, 3),))
    with pytest.raises(ValueError, match="whitespace"):
        to_sighan_format([ideographic], "as")

    # A newline would end the line early and shift every subsequent sentence.
    newline = Segmented(id="x/test/000002", text="有\n空", spans=((0, 3),))
    with pytest.raises(ValueError, match="whitespace"):
        to_sighan_format([newline], "cityu")


def test_to_sighan_format_refuses_an_empty_word() -> None:
    """An empty word vanishes on the way to disk, silently renumbering the line."""
    words: Sequence[str] = ("有", "", "空")
    with pytest.raises(ValueError, match="empty word"):
        to_sighan_format([words], "pku")

# Refined

# Enhanced

# Updated

# Updated

# Enhanced

# Updated

# Enhanced

# Enhanced

# Enhanced
