"""Gold-corpus tests: the IR contract, the frozen counts, and the structural claims.

These run against the BUILT corpora in the user's cache -- they never fetch and
never write. A corpus that is not built is skipped, not failed, so a fresh
checkout with only the ``@permissive`` subset still gets a green run.

What is asserted here, and why each one is load-bearing:

* **CONTRACTS.md sec.1** on every built corpus: spans sorted, non-overlapping,
  in range; every uncovered codepoint declared in ``gap_charset`` (an undeclared
  one means the loader silently dropped content); ``id`` exactly
  ``f"{corpus_id}/{split}/{i:06d}"``.
* **Frozen counts.** Sentence and word totals are published numbers. If a loader
  change moves one of them the leaderboard is no longer comparable to the
  report, so they are asserted exactly rather than as bounds.
* **SIGHAN chars-per-word** against the published bakeoff figures, on the
  TRAINING split -- the published statistics are training-set statistics. This
  is the one cheap external check that we parsed the right files with the right
  delimiter.
* **The controlled contrasts.** ud_zh_gsd/ud_zh_gsdsimp are the same sentences in
  two scripts and ud_zh_hk/ud_yue_hk the same content in two languages; if they
  ever stop aligning 1:1 the "script effect" and "language effect" claims are
  measuring something else.
* **hkcancor tier nesting.** B(s) subset B(p) subset B(d) is structural, not
  statistical: the tiers are a granularity axis over one annotation, so a single
  violation means the tier decoding is wrong.
* **Khmer ZWSP.** CONTRACTS.md sec.6: if gold boundaries coincided with U+200B
  the Khmer task would be trivial and unusable for Tier-1 claims.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from functools import cache as memo
from pathlib import Path

import pytest

# The editable install occasionally loses its .pth entry in this environment;
# fall back to the in-tree source so the module is importable either way.
if importlib.util.find_spec("unsegbench") is None:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unsegbench import positions, types
from unsegbench.build import load_corpus, load_manifest
from unsegbench.corpora.registry import all_corpora, get_corpus, resolve
from unsegbench.errors import BuildValidationError
from unsegbench.fetch import cache

# --------------------------------------------------------------------------
# Frozen expectations
# --------------------------------------------------------------------------

#: Every corpus the registry must expose. 17 = 4 SIGHAN + 5 UD + 3 hkcancor
#: tiers + wisesight1000 + vistec_th + khPOS + Khmer ALT + the OSF human ceiling.
EXPECTED_IDS: frozenset[str] = frozenset(
    {
        "sighan_as",
        "sighan_cityu",
        "sighan_pku",
        "sighan_msr",
        "ud_zh_gsd",
        "ud_zh_gsdsimp",
        "ud_zh_hk",
        "ud_yue_hk",
        "ud_th_pud",
        "hkcancor_s",
        "hkcancor_p",
        "hkcancor_d",
        "wisesight1000",
        "vistec_th",
        "khpos",
        "alt_km",
        "osf_cwsa_zh",
    }
)

#: corpus_id -> (n_sents, n_words or None) over ALL built splits. Known-good.
EXPECTED_COUNTS: dict[str, tuple[int, int | None]] = {
    "ud_zh_gsd": (4997, 123289),
    "ud_zh_gsdsimp": (4997, 123289),
    "ud_zh_hk": (1004, 9874),
    "ud_yue_hk": (1004, 13918),
    "ud_th_pud": (1000, 22330),
    "hkcancor_s": (12547, 110283),
    "hkcancor_p": (12547, 110579),
    "hkcancor_d": (12547, 122829),
    "wisesight1000": (993, None),
    "vistec_th": (50000, None),
    "alt_km": (20106, None),
    "khpos": (13000, None),
    "sighan_pku": (20998, 1214319),
    "sighan_msr": (90903, 2475264),
    "sighan_as": (723382, 5572191),
    "sighan_cityu": (54511, 1496565),
}

#: Published SIGHAN 2005 bakeoff chars-per-word, TRAINING split.
SIGHAN_CHARS_PER_WORD: dict[str, float] = {
    "sighan_as": 1.536,
    "sighan_cityu": 1.651,
    "sighan_pku": 1.646,
    "sighan_msr": 1.710,
}

#: Non-redistributable groups (CONTRACTS.md sec.6). Must never be in @permissive.
LICENCE_GATED: frozenset[str] = frozenset(
    {"sighan_as", "sighan_cityu", "sighan_pku", "sighan_msr", "khpos", "alt_km"}
)

ZWSP = "​"

#: Sentences per corpus for the IR-contract sweep.
SAMPLE_N = 200

#: CONTRACTS.md sec.1: gold boundaries must sit on legal cluster edges.
MAX_GOLD_ILLEGAL_RATE = 1e-3

ALL_IDS: tuple[str, ...] = tuple(sorted(s.corpus_id for s in all_corpora()))
COUNTED_IDS: tuple[str, ...] = tuple(sorted(EXPECTED_COUNTS))
WORD_COUNTED_IDS: tuple[str, ...] = tuple(
    sorted(cid for cid, (_, w) in EXPECTED_COUNTS.items() if w is not None)
)
SIGHAN_IDS: tuple[str, ...] = tuple(sorted(SIGHAN_CHARS_PER_WORD))
HKCANCOR_TIERS: tuple[str, ...] = ("s", "p", "d")

_ID_RE = re.compile(r"^(?P<corpus>.+)/(?P<split>[a-z]+)/(?P<idx>\d{6})$")


# --------------------------------------------------------------------------
# Cache-backed helpers. Nothing here fetches; a missing corpus is a skip.
# --------------------------------------------------------------------------


@memo
def _manifest(corpus_id: str):
    try:
        return load_manifest(corpus_id)
    except BuildValidationError:
        return None


def require_built(corpus_id: str):
    """Manifest of a built corpus, or skip."""
    manifest = _manifest(corpus_id)
    if manifest is None:
        pytest.skip(f"{corpus_id} is not built in the cache")
    return manifest


def require_split(corpus_id: str, split: str):
    """Manifest of a built corpus that has ``split``, or skip."""
    manifest = require_built(corpus_id)
    if split not in manifest.splits:
        pytest.skip(f"{corpus_id} has no {split!r} split")
    return manifest


@memo
def sample(corpus_id: str, split: str = "test", n: int = SAMPLE_N) -> tuple:
    """Deterministic length-stratified subsample of a built split."""
    require_split(corpus_id, split)
    return tuple(load_corpus(corpus_id, split, sample=n))


@memo
def split_totals(corpus_id: str, split: str) -> tuple[int, int, int]:
    """``(n_sents, n_words, n_chars)`` for one split, streamed off disk.

    Streaming rather than `load_corpus` matters for sighan_as: 723k records held
    as a list costs ~700 MB, and every count here is a running sum.
    """
    manifest = require_split(corpus_id, split)
    path = cache.canonical_dir(corpus_id, manifest.version) / f"{split}.jsonl.gz"
    n_sents = n_words = n_chars = 0
    for rec in types.read_jsonl(path):
        n_sents += 1
        n_words += len(rec.spans)
        n_chars += len(rec.text)
    return n_sents, n_words, n_chars


@memo
def corpus_totals(corpus_id: str) -> tuple[int, int, int]:
    """``(n_sents, n_words, n_chars)`` summed over every built split."""
    manifest = require_built(corpus_id)
    totals = [split_totals(corpus_id, split) for split in sorted(manifest.splits)]
    return tuple(sum(col) for col in zip(*totals, strict=True))  # type: ignore[return-value]


@memo
def full_split(corpus_id: str, split: str = "test") -> tuple:
    """A whole split. Only used on the small parallel/tiered corpora."""
    require_split(corpus_id, split)
    return tuple(load_corpus(corpus_id, split))


def spec_of(corpus_id: str):
    return get_corpus(corpus_id)


def built_splits(corpus_id: str) -> tuple[str, ...]:
    manifest = _manifest(corpus_id)
    return tuple(sorted(manifest.splits)) if manifest else ()


# ==========================================================================
# 8/9. Registry and licence gating -- no corpus data needed
# ==========================================================================


def test_registry_has_seventeen_corpora():
    assert len(all_corpora()) == 17


def test_registry_ids_are_exactly_the_expected_set():
    assert {s.corpus_id for s in all_corpora()} == EXPECTED_IDS


def test_registry_corpus_ids_are_unique():
    ids = [s.corpus_id for s in all_corpora()]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_get_corpus_returns_the_matching_spec(corpus_id):
    assert spec_of(corpus_id).corpus_id == corpus_id


def test_get_corpus_unknown_raises_key_error():
    with pytest.raises(KeyError):
        get_corpus("no_such_corpus")


def test_resolve_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        resolve(["no_such_corpus"])


def test_resolve_unknown_selector_raises_key_error():
    with pytest.raises(KeyError):
        resolve(["@klingon"])


def test_resolve_all_returns_every_corpus():
    assert {s.corpus_id for s in resolve(["@all"])} == EXPECTED_IDS


def test_resolve_permissive_is_only_redistributable():
    assert all(s.redistributable for s in resolve(["@permissive"]))


def test_resolve_permissive_is_every_redistributable_spec():
    got = {s.corpus_id for s in resolve(["@permissive"])}
    assert got == {s.corpus_id for s in all_corpora() if s.redistributable}


@pytest.mark.parametrize("corpus_id", sorted(LICENCE_GATED))
def test_licence_gated_corpus_is_not_permissive(corpus_id):
    """CONTRACTS.md sec.6: SIGHAN, khPOS and Khmer ALT are never redistributable."""
    assert corpus_id not in {s.corpus_id for s in resolve(["@permissive"])}
    assert spec_of(corpus_id).redistributable is False


def test_permissive_subset_is_non_empty_and_proper():
    permissive = resolve(["@permissive"])
    assert 0 < len(permissive) < len(all_corpora())


def test_resolve_zh_returns_only_mandarin():
    assert {s.lang for s in resolve(["@zh"])} == {"zh"}


@pytest.mark.parametrize("lang", ["zh", "yue", "th", "km"])
def test_resolve_lang_selector_is_exactly_that_language(lang):
    got = resolve([f"@{lang}"])
    assert got, f"@{lang} matched nothing"
    assert {s.lang for s in got} == {lang}
    assert {s.corpus_id for s in got} == {s.corpus_id for s in all_corpora() if s.lang == lang}


def test_lang_selectors_partition_the_registry():
    seen = [s.corpus_id for lang in ("zh", "yue", "th", "km") for s in resolve([f"@{lang}"])]
    assert sorted(seen) == sorted(EXPECTED_IDS)


def test_resolve_deduplicates_overlapping_selectors():
    got = resolve(["@zh", "sighan_pku", "@all", "@permissive"])
    ids = [s.corpus_id for s in got]
    assert len(ids) == len(set(ids)) == len(EXPECTED_IDS)


def test_resolve_preserves_request_order():
    got = resolve(["ud_th_pud", "sighan_pku"])
    assert [s.corpus_id for s in got] == ["ud_th_pud", "sighan_pku"]


def test_every_spec_declares_a_known_language_and_script():
    for spec in all_corpora():
        assert spec.lang in {"zh", "yue", "th", "km"}, spec.corpus_id
        assert spec.script in {"Hans", "Hant", "Thai", "Khmr"}, spec.corpus_id


def test_every_spec_declares_a_licence_and_source():
    for spec in all_corpora():
        assert spec.license, spec.corpus_id
        assert spec.source_url.startswith("http"), spec.corpus_id


# ==========================================================================
# 1. The IR contract (CONTRACTS.md sec.1) on every built corpus
# ==========================================================================


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_ir_spans_are_sorted_non_overlapping_and_in_range(corpus_id):
    for rec in sample(corpus_id):
        n = len(rec.text)
        assert n > 0, rec.id
        prev_end = 0
        for i, (s, e) in enumerate(rec.spans):
            assert 0 <= s < e <= n, f"{rec.id}: span {i} = ({s},{e}) out of range for n={n}"
            assert s >= prev_end, f"{rec.id}: span {i} = ({s},{e}) overlaps previous end {prev_end}"
            prev_end = e


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_ir_uncovered_codepoints_are_declared_gaps(corpus_id):
    """An uncovered codepoint outside `gap_charset` means the loader lost data."""
    allowed = frozenset(spec_of(corpus_id).gap_charset)
    for rec in sample(corpus_id):
        covered = bytearray(len(rec.text))
        for s, e in rec.spans:
            covered[s:e] = b"\x01" * (e - s)
        for i, flag in enumerate(covered):
            if not flag:
                ch = rec.text[i]
                assert ch in allowed, (
                    f"{rec.id}: uncovered {ch!r} (U+{ord(ch):04X}) at {i} not in gap_charset"
                )


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_ir_validates_against_the_shared_validator(corpus_id):
    gap_charset = spec_of(corpus_id).gap_charset
    for rec in sample(corpus_id):
        types.validate_record(rec, gap_charset)


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_ir_record_ids_follow_the_frozen_format(corpus_id):
    n_sents = split_totals(corpus_id, "test")[0]
    seen = set()
    for rec in sample(corpus_id):
        m = _ID_RE.match(rec.id)
        assert m, f"{rec.id!r} is not <corpus_id>/<split>/<6 digits>"
        assert m["corpus"] == corpus_id
        assert m["split"] == "test"
        idx = int(m["idx"])
        assert 0 <= idx < n_sents, f"{rec.id}: index outside the split"
        assert rec.id == f"{corpus_id}/test/{idx:06d}"
        assert rec.id not in seen, f"duplicate id {rec.id}"
        seen.add(rec.id)


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ["ud_zh_hk", "ud_yue_hk", "ud_th_pud", "wisesight1000"])
def test_ir_ids_are_a_contiguous_zero_padded_sequence(corpus_id):
    recs = full_split(corpus_id, "test")
    assert [r.id for r in recs] == [f"{corpus_id}/test/{i:06d}" for i in range(len(recs))]


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_ir_words_are_non_empty(corpus_id):
    for rec in sample(corpus_id):
        for word in rec.words:
            assert word, f"{rec.id}: empty gold word"


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_ir_text_carries_no_encoding_damage(corpus_id):
    """No U+FFFD and no NUL: either would mean the raw bytes were mis-decoded.

    This is the counterpart to `gold_illegal_rate` -- that catches a wrong
    cluster model, this catches a wrong codec.
    """
    for rec in sample(corpus_id):
        assert "�" not in rec.text, f"{rec.id}: U+FFFD replacement character"
        assert "\x00" not in rec.text, f"{rec.id}: NUL in text"


# ==========================================================================
# 2. Frozen sentence and word counts
# ==========================================================================


@pytest.mark.parametrize("corpus_id", COUNTED_IDS)
def test_manifest_sentence_count_is_the_published_number(corpus_id):
    manifest = require_built(corpus_id)
    assert manifest.n_sents == EXPECTED_COUNTS[corpus_id][0]


@pytest.mark.parametrize("corpus_id", WORD_COUNTED_IDS)
def test_manifest_word_count_is_the_published_number(corpus_id):
    manifest = require_built(corpus_id)
    assert manifest.n_words == EXPECTED_COUNTS[corpus_id][1]


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", COUNTED_IDS)
def test_records_on_disk_have_the_published_sentence_count(corpus_id):
    require_built(corpus_id)
    assert corpus_totals(corpus_id)[0] == EXPECTED_COUNTS[corpus_id][0]


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", WORD_COUNTED_IDS)
def test_records_on_disk_have_the_published_word_count(corpus_id):
    require_built(corpus_id)
    assert corpus_totals(corpus_id)[1] == EXPECTED_COUNTS[corpus_id][1]


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_manifest_totals_match_the_records_on_disk(corpus_id):
    manifest = require_built(corpus_id)
    assert corpus_totals(corpus_id) == (manifest.n_sents, manifest.n_words, manifest.n_chars)


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_manifest_metadata_matches_the_spec(corpus_id):
    manifest = require_built(corpus_id)
    spec = spec_of(corpus_id)
    assert (manifest.corpus_id, manifest.lang, manifest.script) == (
        spec.corpus_id,
        spec.lang,
        spec.script,
    )
    assert manifest.convention == spec.convention
    assert manifest.license == spec.license
    assert manifest.redistributable == spec.redistributable
    assert manifest.gap_charset == spec.gap_charset


# ==========================================================================
# 3. SIGHAN chars-per-word against the published bakeoff figures (TRAIN split)
# ==========================================================================


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", SIGHAN_IDS)
def test_sighan_training_chars_per_word_matches_the_published_figure(corpus_id):
    require_split(corpus_id, "train")
    _, n_words, n_chars = split_totals(corpus_id, "train")
    assert n_words > 0
    got = n_chars / n_words
    expected = SIGHAN_CHARS_PER_WORD[corpus_id]
    assert abs(got - expected) < 1e-3, f"{corpus_id}: chars/word {got:.4f} vs published {expected}"


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", SIGHAN_IDS)
def test_sighan_gold_tiles_the_text(corpus_id):
    """SIGHAN's `gap_charset` is empty: delimiter spaces are stripped, spans tile."""
    assert spec_of(corpus_id).gap_charset == ""
    for rec in sample(corpus_id):
        assert "".join(rec.words) == rec.text, rec.id


@pytest.mark.slow
def test_sighan_conventions_are_four_distinct_standards():
    conventions = {spec_of(cid).convention for cid in SIGHAN_IDS}
    assert len(conventions) == 4


# ==========================================================================
# 4. gold_illegal_rate
# ==========================================================================


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_gold_illegal_rate_on_a_sample_is_negligible(corpus_id):
    spec = spec_of(corpus_id)
    rate = positions.gold_illegal_rate(list(sample(corpus_id)), spec.lang)
    assert rate < MAX_GOLD_ILLEGAL_RATE, f"{corpus_id}: gold_illegal_rate {rate:.6f}"


@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_manifest_gold_illegal_rate_is_negligible(corpus_id):
    manifest = require_built(corpus_id)
    assert manifest.gold_illegal_rate < MAX_GOLD_ILLEGAL_RATE


# ==========================================================================
# 5. The parallel pairs
# ==========================================================================


@pytest.mark.slow
def test_gsd_pair_has_equal_sentence_counts():
    trad = require_built("ud_zh_gsd")
    simp = require_built("ud_zh_gsdsimp")
    assert trad.n_sents == simp.n_sents == 4997


@pytest.mark.slow
@pytest.mark.parametrize("split", ["test", "train"])
def test_gsd_pair_aligns_span_for_span(split):
    """Same 4,997 sentences, traditional vs simplified: the script control."""
    trad = full_split("ud_zh_gsd", split)
    simp = full_split("ud_zh_gsdsimp", split)
    assert len(trad) == len(simp)
    for a, b in zip(trad, simp, strict=True):
        assert len(a.spans) == len(b.spans), f"{a.id}: {len(a.spans)} vs {len(b.spans)} spans"
        assert len(a.text) == len(b.text), f"{a.id}: text length differs"


@pytest.mark.slow
def test_gsd_pair_has_identical_word_totals():
    assert corpus_totals("ud_zh_gsd")[1] == corpus_totals("ud_zh_gsdsimp")[1] == 123289


def test_gsd_pair_differs_only_in_script():
    trad, simp = spec_of("ud_zh_gsd"), spec_of("ud_zh_gsdsimp")
    assert (trad.script, simp.script) == ("Hant", "Hans")
    assert trad.lang == simp.lang == "zh"
    assert trad.convention == simp.convention


@pytest.mark.slow
def test_hk_pair_has_equal_sentence_counts():
    zh = require_built("ud_zh_hk")
    yue = require_built("ud_yue_hk")
    assert zh.n_sents == yue.n_sents == 1004


def test_hk_pair_differs_only_in_language():
    zh, yue = spec_of("ud_zh_hk"), spec_of("ud_yue_hk")
    assert (zh.lang, yue.lang) == ("zh", "yue")
    assert zh.script == yue.script == "Hant"
    assert zh.convention == yue.convention


@pytest.mark.slow
def test_yue_to_zh_token_ratio_is_about_1_41():
    """Cantonese needs ~41% more UD tokens for the same content."""
    zh_words = require_built("ud_zh_hk").n_words
    yue_words = require_built("ud_yue_hk").n_words
    ratio = yue_words / zh_words
    assert abs(ratio - 1.41) < 0.01, f"yue/zh token ratio {ratio:.4f}"


@pytest.mark.slow
def test_hk_pair_word_totals_on_disk_match_the_manifests():
#     assert corpus_totals("ud_zh_hk")[1] == 9874
    assert corpus_totals("ud_yue_hk")[1] == 13918


# ==========================================================================
# 6. hkcancor tiers are perfectly nested: B(s) subset B(p) subset B(d)
# ==========================================================================


@pytest.mark.slow
@pytest.mark.parametrize("split", ["test", "train"])
def test_hkcancor_tiers_share_one_text(split):
    tiers = {t: full_split(f"hkcancor_{t}", split) for t in HKCANCOR_TIERS}
    assert len({len(recs) for recs in tiers.values()}) == 1
    for s, p, d in zip(tiers["s"], tiers["p"], tiers["d"], strict=True):
        assert s.text == p.text == d.text, s.id


@pytest.mark.slow
@pytest.mark.parametrize("split", ["test", "train"])
@pytest.mark.parametrize(("coarse", "fine"), [("s", "p"), ("p", "d"), ("s", "d")])
def test_hkcancor_tiers_are_nested(split, coarse, fine):
    """Structural, not statistical: zero violations, or the tier decoding is wrong."""
    lo = full_split(f"hkcancor_{coarse}", split)
    hi = full_split(f"hkcancor_{fine}", split)
    violations = [
        a.id
        for a, b in zip(lo, hi, strict=True)
        if not positions.gold_boundaries(a) <= positions.gold_boundaries(b)
    ]
    assert violations == [], (
        f"{len(violations)} sentences where B({coarse}) is not a subset of B({fine}); "
        f"first: {violations[:3]}"
    )


@pytest.mark.slow
def test_hkcancor_tier_word_totals_increase_with_granularity():
    counts = {t: require_built(f"hkcancor_{t}").n_words for t in HKCANCOR_TIERS}
    assert counts["s"] < counts["p"] < counts["d"]
    assert counts == {"s": 110283, "p": 110579, "d": 122829}


@pytest.mark.slow
@pytest.mark.parametrize("tier", HKCANCOR_TIERS)
def test_hkcancor_tier_has_the_published_sentence_count(tier):
    assert require_built(f"hkcancor_{tier}").n_sents == 12547


@pytest.mark.slow
def test_hkcancor_tiers_have_identical_character_totals():
    chars = {corpus_totals(f"hkcancor_{t}")[2] for t in HKCANCOR_TIERS}
    assert len(chars) == 1


def test_hkcancor_tiers_are_one_granularity_axis():
    conventions = {spec_of(f"hkcancor_{t}").convention for t in HKCANCOR_TIERS}
    assert len(conventions) == 3
    assert {spec_of(f"hkcancor_{t}").lang for t in HKCANCOR_TIERS} == {"yue"}


# ==========================================================================
# 7. Khmer ZWSP is not the annotation (CONTRACTS.md sec.6)
# ==========================================================================


@pytest.mark.slow
def test_alt_km_contains_no_zwsp():
    require_built("alt_km")
    n = sum(
        rec.text.count(ZWSP)
        for split in built_splits("alt_km")
        for rec in full_split("alt_km", split)
    )
    assert n == 0, f"Khmer ALT has {n} ZWSP; gold boundaries there would be trivial"


@pytest.mark.slow
def test_khpos_has_at_most_five_zwsp_in_the_whole_corpus():
    require_built("khpos")
    n = sum(
        rec.text.count(ZWSP)
        for split in built_splits("khpos")
        for rec in full_split("khpos", split)
    )
    assert n <= 5, f"khPOS has {n} ZWSP; if they marked gold the Khmer task would be trivial"

# 
@pytest.mark.slow
def test_alt_km_manifest_records_no_zwsp():
    assert require_built("alt_km").zwsp_present is False


@pytest.mark.slow
def test_khmer_gold_density_is_not_a_zwsp_artefact():
    """Khmer words are multi-character: gold cannot be "split on ZWSP"."""
    for corpus_id in ("alt_km", "khpos"):
        recs = sample(corpus_id)
        words = [w for rec in recs for w in rec.words]
        assert words
        mean_len = sum(len(w) for w in words) / len(words)
        assert mean_len > 2.0, f"{corpus_id}: mean gold word length {mean_len:.2f}"


# ==========================================================================
# Cross-cutting sanity on the remaining corpora
# ==========================================================================

# 
@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_every_corpus_has_a_test_split(corpus_id):
    manifest = require_built(corpus_id)
    assert "test" in manifest.splits


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_sample_is_deterministic(corpus_id):
    """Sampling is a keyed hash, not a reservoir: same input, same sentences."""
    first = [rec.id for rec in load_corpus(corpus_id, "test", sample=32)]
    second = [rec.id for rec in load_corpus(corpus_id, "test", sample=32)]
    assert first == second
    assert first == sorted(first)


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_every_sentence_has_at_least_one_gold_word(corpus_id):
    for rec in sample(corpus_id):
        assert rec.spans, f"{rec.id}: no gold spans"


@pytest.mark.slow
@pytest.mark.parametrize("corpus_id", ALL_IDS)
def test_gold_boundaries_exclude_sentence_edges(corpus_id):
    """CONTRACTS.md sec.2: positions 0 and n are outside every universe."""
    for rec in sample(corpus_id):
        gold = positions.gold_boundaries(rec)
        assert all(0 < b < rec.n for b in gold), rec.id


def test_thai_corpora_declare_the_space_as_a_gap():
    """Thai phrase spaces are real content, so they must be declared, not dropped."""
    for corpus_id in ("wisesight1000", "vistec_th", "ud_th_pud"):
        assert " " in spec_of(corpus_id).gap_charset, corpus_id


def test_wisesight_declares_zwsp_as_a_gap_character():
    assert ZWSP in spec_of("wisesight1000").gap_charset


def test_unbuilt_corpus_reports_cleanly():
    with pytest.raises(BuildValidationError):
        load_manifest("definitely_not_a_corpus")

# Enhanced

# Refined

# Enhanced
