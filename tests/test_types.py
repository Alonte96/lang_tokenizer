"""Canonical IR contract (CONTRACTS.md sec.1) and the sufficient-stat schema (sec.5).

Everything here is offline and data-free: the records are hand-built in this
file, so nothing in this module depends on a corpus having been fetched.

The two contracts that matter most and that nothing else can catch:

* ``text`` survives a round-trip **verbatim**. Thai U+0E33 SARA AM changes
  codepoint count under NFKD/NFKC and NFKC reorders Khmer marks, so any
  normalisation anywhere in the codec would silently invalidate every span
  offset in the corpus.
* `STATS_COLUMNS` and `RowStats` agree in name *and order*. `RowStats.as_tuple`
  is positional and the runner writes those tuples straight into parquet
  columns, so a drift between the two would write the wrong integer into the
  wrong column -- a corruption with no exception and no obviously wrong number.
"""

from __future__ import annotations

import gzip
from collections import Counter
from dataclasses import fields

import orjson
import pytest

from unsegbench.errors import BuildValidationError
from unsegbench.types import (
    MASKS,
    STATS_COLUMNS,
    CorpusManifest,
    EncodeResult,
    RowStats,
    Segmented,
    read_jsonl,
    validate_corpus,
    validate_record,
    write_jsonl,
)

# --------------------------------------------------------------------------
# Hazard strings. Each one is a codec bug that only shows up in one script.
# --------------------------------------------------------------------------

THAI_SARA_AM = "ฉันทำงานที่บ้าน"  # U+0E33 SARA AM: NFD/NFKD changes the codepoint count
THAI_LEADING_VOWEL = "ไปโรงเรียนทุกวัน"  # U+0E44 written before its consonant
KHMER_COENG = "ភាសាខ្មែរពិរោះណាស់"  # U+17D2 COENG, invisible subscript-former
KHMER_ZWSP = "ខ្ញុំ​ស្រឡាញ់​កម្ពុជា"  # U+200B ZWSP as a word separator
NON_BMP = "𠮷野家𝔘𝔫𝔦😀🇹🇭"  # U+20BB7, math alphanumerics, emoji, a flag sequence


def _seg(text: str, spans=None, **over) -> Segmented:
    """A record whose spans tile ``text`` in width-2 chunks unless given."""
    if spans is None:
        spans = tuple((i, min(i + 2, len(text))) for i in range(0, len(text), 2))
    payload = {"id": "fake/test/000000", "text": text, "spans": tuple(spans)}
    payload.update(over)
    return Segmented(**payload)


def _manifest(**over) -> CorpusManifest:
    base = {
        "corpus_id": "fake_zh",
        "lang": "zh",
        "script": "Hans",
        "convention": "fake",
        "license": "CC0-1.0",
        "redistributable": True,
        "source_url": "https://example.invalid/corpus.zip",
        "version": "v1",
        "splits": {"test": "a" * 64, "train": "b" * 64},
        "n_sents": 3,
        "n_words": 9,
        "n_chars": 27,
        "gap_charset": " ​。！",
        "gold_illegal_rate": 0.0009765625,  # exactly representable: pins float fidelity
        "zwsp_present": True,
        "builder_version": "1",
        "notes": "ノート ខ្មែរ 𠮷",
    }
    base.update(over)
    return CorpusManifest(**base)


# ==========================================================================
# 1. Segmented round-trips losslessly
# ==========================================================================


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("han", "我喜欢吃苹果。"),
        ("thai_sara_am", THAI_SARA_AM),
        ("thai_leading_vowel", THAI_LEADING_VOWEL),
        ("khmer_coeng", KHMER_COENG),
        ("khmer_zwsp", KHMER_ZWSP),
        ("non_bmp", NON_BMP),
        ("mixed_scripts", "我用 Python 写程序 ខ្មែរ ทำงาน 😀"),
    ],
)
def test_segmented_roundtrips_losslessly(name: str, text: str) -> None:
    rec = _seg(text)
    back = Segmented.from_json(rec.to_json())
    assert back == rec
    assert back.text == text
    assert back.spans == rec.spans


def test_roundtrip_preserves_codepoint_count_for_sara_am() -> None:
    """No normalisation anywhere: NFD/NFKD would change U+0E33's length."""
    rec = _seg(THAI_SARA_AM)
    back = Segmented.from_json(rec.to_json())
    assert len(back.text) == len(THAI_SARA_AM)
    assert "ำ" in back.text
    assert back.n == rec.n


def test_roundtrip_preserves_khmer_mark_order() -> None:
    """NFKC reorders Khmer marks; the codec must not touch them."""
    rec = _seg(KHMER_COENG)
    back = Segmented.from_json(rec.to_json())
    assert list(back.text) == list(KHMER_COENG)
    assert "្" in back.text


def test_roundtrip_preserves_non_bmp_astral_codepoints() -> None:
    rec = _seg(NON_BMP)
    back = Segmented.from_json(rec.to_json())
    assert back.text == NON_BMP
    assert max(ord(c) for c in back.text) > 0xFFFF


def test_roundtrip_preserves_meta() -> None:
    rec = _seg("我喜欢", meta={"src": "line 7", "n": 3, "ok": True, "th": "ทำ"})
    back = Segmented.from_json(rec.to_json())
    assert back.meta == rec.meta


def test_to_json_omits_empty_meta() -> None:
    payload = orjson.loads(_seg("我喜欢").to_json())
    assert "meta" not in payload
    assert Segmented.from_json(_seg("我喜欢").to_json()).meta == {}


def test_to_json_stores_nothing_derivable() -> None:
    """`words`, `n`, boundaries and masks are pure functions; never serialised."""
    payload = orjson.loads(_seg("我喜欢吃苹果").to_json())
    assert set(payload) == {"id", "text", "spans"}


def test_from_json_accepts_bytes_and_str() -> None:
    raw = _seg(KHMER_COENG).to_json()
    assert Segmented.from_json(raw) == Segmented.from_json(raw.decode("utf-8"))


def test_from_json_normalises_spans_to_tuples_of_ints() -> None:
    back = Segmented.from_json(
        b'{"id":"a/test/0","text":"\xe6\x88\x91\xe5\x96\x9c","spans":[[0,1],[1,2]]}'
    )
    assert back.spans == ((0, 1), (1, 2))
    assert all(isinstance(x, int) for sp in back.spans for x in sp)


def test_roundtrip_of_record_whose_spans_do_not_tile() -> None:
    rec = _seg("我 喜欢", spans=((0, 1), (2, 4)))
    assert Segmented.from_json(rec.to_json()) == rec


def test_words_property_is_derived_from_text_and_spans() -> None:
    rec = _seg("我喜欢", spans=((0, 1), (1, 3)))
    assert rec.words == ("我", "喜欢")
    assert rec.n == 3


# ==========================================================================
# 2. write_jsonl / read_jsonl: gzipped, atomic, lossless
# ==========================================================================


def _corpus() -> list[Segmented]:
    texts = [THAI_SARA_AM, KHMER_COENG, KHMER_ZWSP, NON_BMP, "我喜欢吃苹果。"]
    return [
        Segmented(
            id=f"fake/test/{i:06d}",
            text=t,
            spans=tuple((s, min(s + 2, len(t))) for s in range(0, len(t), 2)),
            meta={"i": i},
        )
        for i, t in enumerate(texts)
    ]


def test_write_read_jsonl_roundtrip(tmp_path) -> None:
    path = tmp_path / "sub" / "test.jsonl.gz"
    recs = _corpus()
    assert write_jsonl(path, recs) == len(recs)
    assert list(read_jsonl(path)) == recs


def test_write_jsonl_creates_parent_directories(tmp_path) -> None:
    path = tmp_path / "a" / "b" / "c" / "test.jsonl.gz"
    write_jsonl(path, _corpus())
    assert path.exists()


def test_write_jsonl_output_is_gzip(tmp_path) -> None:
    path = tmp_path / "test.jsonl.gz"
    write_jsonl(path, _corpus())
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    with gzip.open(path, "rb") as fh:
        assert len(fh.read().splitlines()) == len(_corpus())


def test_write_jsonl_leaves_no_part_file(tmp_path) -> None:
    path = tmp_path / "test.jsonl.gz"
    write_jsonl(path, _corpus())
    assert list(tmp_path.iterdir()) == [path]
    assert not (tmp_path / "test.jsonl.gz.part").exists()


def test_write_jsonl_failure_leaves_the_target_untouched(tmp_path) -> None:
    """Atomicity: a half-written corpus must never replace a good one."""
    path = tmp_path / "test.jsonl.gz"
    good = _corpus()
    write_jsonl(path, good)

    def exploding():
        yield good[0]
        raise RuntimeError("loader blew up mid-stream")

    with pytest.raises(RuntimeError):
        write_jsonl(path, exploding())
    assert list(read_jsonl(path)) == good


def test_write_jsonl_of_empty_iterable_returns_zero(tmp_path) -> None:
    path = tmp_path / "test.jsonl.gz"
    assert write_jsonl(path, []) == 0
    assert list(read_jsonl(path)) == []


def test_read_jsonl_reads_plain_jsonl_too(tmp_path) -> None:
    path = tmp_path / "test.jsonl"
    recs = _corpus()
    path.write_bytes(b"\n".join(r.to_json() for r in recs) + b"\n")
    assert list(read_jsonl(path)) == recs


def test_read_jsonl_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "test.jsonl"
    recs = _corpus()
    path.write_bytes(b"\n\n".join(r.to_json() for r in recs) + b"\n\n")
    assert list(read_jsonl(path)) == recs


def test_read_jsonl_streams_lazily(tmp_path) -> None:
    path = tmp_path / "test.jsonl.gz"
    write_jsonl(path, _corpus())
    stream = read_jsonl(path)
    assert next(stream).id == "fake/test/000000"


# ==========================================================================
# 3. validate_record fails closed
# ==========================================================================


def test_validate_accepts_a_tiling_record() -> None:
    validate_record(_seg("我喜欢吃苹果"), " ")


def test_validate_accepts_spans_that_do_not_tile_the_text() -> None:
    """Spans need not cover ``text``; the gaps must be declared, that is all."""
    rec = _seg("我 喜欢 吃", spans=((0, 1), (2, 4), (5, 6)))
    validate_record(rec, " ")


def test_validate_accepts_thai_phrase_space_as_a_gap() -> None:
    text = "สวัสดี ครับ"
    rec = _seg(text, spans=((0, 6), (7, len(text))))
    validate_record(rec, " ")


def test_validate_accepts_khmer_zwsp_as_a_gap() -> None:
    # Offsets derived, never hand-written: Khmer COENG sequences make manual
    # codepoint arithmetic a reliable source of silent errors.
    text = KHMER_ZWSP
    spans, pos = [], 0
    for word in text.split("​"):
        spans.append((pos, pos + len(word)))
        pos += len(word) + 1
    validate_record(_seg(text, spans=tuple(spans)), "​")


def test_validate_without_gap_charset_skips_the_coverage_check() -> None:
    validate_record(_seg("我 喜欢", spans=((0, 1), (2, 4))), None)


def test_validate_accepts_a_record_with_no_spans_at_all() -> None:
    validate_record(_seg("  ", spans=()), " ")


def test_validate_rejects_empty_text() -> None:
    with pytest.raises(BuildValidationError, match="empty text"):
        validate_record(Segmented(id="fake/test/000000", text="", spans=()))


def test_validate_rejects_overlapping_spans() -> None:
    rec = _seg("我喜欢吃", spans=((0, 2), (1, 4)))
    with pytest.raises(BuildValidationError, match="overlaps"):
        validate_record(rec)


def test_validate_rejects_abutting_duplicate_spans() -> None:
    rec = _seg("我喜欢吃", spans=((0, 2), (0, 2)))
    with pytest.raises(BuildValidationError, match="overlaps"):
        validate_record(rec)


def test_validate_rejects_unsorted_spans() -> None:
    """Same span set, wrong order: still a contract violation."""
    rec = _seg("我喜欢吃", spans=((2, 4), (0, 2)))
    with pytest.raises(BuildValidationError, match="overlaps or precedes"):
        validate_record(rec)


@pytest.mark.parametrize(
    ("spans", "why"),
    [
        (((0, 5),), "end past n"),
        (((3, 4),), "start past n"),
        (((-1, 2),), "negative start"),
        (((1, 1),), "empty span"),
        (((2, 1),), "reversed span"),
    ],
)
def test_validate_rejects_out_of_range_spans(spans, why: str) -> None:
    rec = _seg("我喜欢", spans=spans)
    with pytest.raises(BuildValidationError, match="out of range"):
        validate_record(rec)


def test_validate_rejects_uncovered_codepoint_not_in_gap_charset() -> None:
    """The check that catches a loader silently dropping content."""
    rec = _seg("我喜欢吃", spans=((0, 1), (3, 4)))
    with pytest.raises(BuildValidationError, match=r"U\+559C"):
        validate_record(rec, " ")


def test_validate_rejects_uncovered_gap_when_gap_charset_is_empty() -> None:
    rec = _seg("我 喜欢", spans=((0, 1), (2, 4)))
    with pytest.raises(BuildValidationError, match="not in gap_charset"):
        validate_record(rec, "")


def test_validate_error_names_the_offending_record_and_codepoint() -> None:
    rec = Segmented(id="fake/test/000042", text="我喜欢", spans=((0, 1),))
    with pytest.raises(BuildValidationError) as excinfo:
        validate_record(rec, " ")
    message = str(excinfo.value)
    assert "fake/test/000042" in message
    assert "U+559C" in message


def test_validate_accepts_gap_charset_as_a_frozenset() -> None:
    validate_record(_seg("我 喜欢", spans=((0, 1), (2, 4))), frozenset(" "))


def test_validate_corpus_returns_totals() -> None:
    recs = [
        _seg("我喜欢", spans=((0, 1), (1, 3))),
        _seg("吃苹果", spans=((0, 1), (1, 3))),
    ]
    assert validate_corpus(recs, "") == (2, 4, 6)


def test_validate_corpus_rejects_an_empty_corpus() -> None:
    with pytest.raises(BuildValidationError, match="corpus is empty"):
        validate_corpus([], "")


def test_validate_corpus_propagates_the_first_record_error() -> None:
    recs = [_seg("我喜欢", spans=((0, 1), (1, 3))), _seg("吃", spans=((0, 3),))]
    with pytest.raises(BuildValidationError, match="out of range"):
        validate_corpus(recs, "")


# ==========================================================================
# 4. STATS_COLUMNS and RowStats must not drift apart
# ==========================================================================


def test_stats_columns_match_rowstats_fields_in_name_and_order() -> None:
    """Drift here writes the wrong integer into the wrong parquet column."""
    assert tuple(f.name for f in fields(RowStats)) == STATS_COLUMNS


def test_rowstats_as_tuple_is_positionally_aligned_with_stats_columns() -> None:
    values = {c: i for i, c in enumerate(STATS_COLUMNS)}
    values["sent_id"] = "fake/test/000000"
    row = RowStats(**values)
    assert row.as_tuple() == tuple(values[c] for c in STATS_COLUMNS)
    for i, col in enumerate(STATS_COLUMNS):
        assert row.as_tuple()[i] == getattr(row, col)


def test_stats_columns_are_unique_and_start_with_sent_id() -> None:
    assert len(set(STATS_COLUMNS)) == len(STATS_COLUMNS)
    assert STATS_COLUMNS[0] == "sent_id"


def test_rowstats_counters_default_to_zero() -> None:
    row = RowStats(
        sent_id="fake/test/000000",
        n_chars=3,
        n_mask=2,
        n_gold_words=2,
        n_tokens=2,
        n_tokens_accepted=2,
        b_tp=1,
        b_fp=0,
        b_fn=0,
        b_tn=1,
        w_tp=2,
        w_pred=2,
        w_gold=2,
        w_intact=2,
        crossing_tokens=0,
    )
    assert row.f_midcodepoint == row.f_cluster_split == 0
    assert row.f_overlap_rejected == row.f_dropped_chars == row.f_prefix_space_trim == 0


def test_every_persisted_counter_is_an_integer() -> None:
    """CONTRACTS.md sec.5: keep them integers. A float in this schema is a bug."""
    values = dict.fromkeys(STATS_COLUMNS, 1)
    values["sent_id"] = "fake/test/000000"
    row = RowStats(**values)
    for col in STATS_COLUMNS[1:]:
        assert isinstance(getattr(row, col), int)
        assert not isinstance(getattr(row, col), bool)


def test_masks_are_the_three_declared_universes() -> None:
    assert MASKS == ("raw", "legal", "core")


# ==========================================================================
# 5. EncodeResult.boundaries excludes sentence edges
# ==========================================================================


def test_boundaries_exclude_the_first_spans_start() -> None:
    enc = EncodeResult(spans=((0, 2), (2, 5), (5, 7)), n_tokens=3)
    assert enc.boundaries == frozenset({2, 5})
    assert 0 not in enc.boundaries


def test_boundaries_exclude_a_nonzero_first_start_too() -> None:
    """Sentence edges are never boundaries, wherever the first token begins."""
    enc = EncodeResult(spans=((1, 3), (3, 6)), n_tokens=2)
    assert enc.boundaries == frozenset({3})
    assert 1 not in enc.boundaries


def test_boundaries_exclude_the_last_spans_end() -> None:
    enc = EncodeResult(spans=((0, 2), (2, 5)), n_tokens=2)
    assert 5 not in enc.boundaries


def test_boundaries_of_a_single_span_are_empty() -> None:
    assert EncodeResult(spans=((0, 7),), n_tokens=1).boundaries == frozenset()


def test_boundaries_of_no_spans_are_empty() -> None:
    assert EncodeResult(spans=(), n_tokens=0).boundaries == frozenset()


def test_boundaries_of_non_contiguous_spans_use_starts_only() -> None:
    enc = EncodeResult(spans=((0, 2), (3, 5), (6, 8)), n_tokens=3)
    assert enc.boundaries == frozenset({3, 6})


def test_boundaries_count_is_one_less_than_the_span_count() -> None:
    spans = tuple((i, i + 2) for i in range(0, 20, 2))
    assert len(EncodeResult(spans=spans, n_tokens=len(spans)).boundaries) == len(spans) - 1


def test_encode_result_flags_default_to_an_empty_counter() -> None:
    enc = EncodeResult(spans=((0, 1),), n_tokens=1)
    assert enc.flags == Counter()
    assert enc.flags.get("midcodepoint_split", 0) == 0


def test_encode_result_keeps_raw_token_count_separate_from_accepted_spans() -> None:
    """n_tokens must not shrink just because a boundary was rejected."""
    enc = EncodeResult(spans=((0, 3),), n_tokens=4, flags=Counter({"midcodepoint_split": 3}))
    assert enc.n_tokens == 4
    assert len(enc.spans) == 1


# ==========================================================================
# 6. CorpusManifest round-trips through JSON
# ==========================================================================


def test_corpus_manifest_roundtrips() -> None:
    manifest = _manifest()
    assert CorpusManifest.from_json(manifest.to_json()) == manifest


def test_corpus_manifest_roundtrip_preserves_field_types() -> None:
    back = CorpusManifest.from_json(_manifest().to_json())
    assert back.redistributable is True
    assert back.zwsp_present is True
    assert isinstance(back.gold_illegal_rate, float)
    assert back.gold_illegal_rate == 0.0009765625
    assert back.splits == {"test": "a" * 64, "train": "b" * 64}


def test_corpus_manifest_roundtrip_preserves_non_ascii_fields() -> None:
    back = CorpusManifest.from_json(_manifest(gap_charset=" ​。！　").to_json())
    assert back.gap_charset == " ​。！　"
    assert back.notes == "ノート ខ្មែរ 𠮷"


def test_corpus_manifest_serialisation_is_deterministic() -> None:
    """The digest of these bytes is a cache key; it must not wobble."""
    assert _manifest().to_json() == _manifest().to_json()
    assert _manifest().to_json() != _manifest(version="v2").to_json()


def test_corpus_manifest_json_contains_every_field() -> None:
    payload = orjson.loads(_manifest().to_json())
    assert set(payload) == {f.name for f in fields(CorpusManifest)}


def test_corpus_manifest_notes_default_to_empty() -> None:
    manifest = CorpusManifest(
        corpus_id="fake_km",
        lang="km",
        script="Khmr",
        convention="fake",
        license="CC0-1.0",
        redistributable=True,
        source_url="https://example.invalid",
        version="v1",
        splits={},
        n_sents=1,
        n_words=1,
        n_chars=1,
        gap_charset=" ",
        gold_illegal_rate=0.0,
        zwsp_present=False,
        builder_version="1",
    )
    assert manifest.notes == ""
    assert CorpusManifest.from_json(manifest.to_json()) == manifest

# Updated

# Updated
