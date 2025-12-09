"""The runner/build layer: cache keys, parquet schema, resume, aggregation, sampling.

Everything here is **offline and data-free**. The corpora are hand-built
`Segmented` records and the tokenizers are deterministic fake adapters, so this
whole module runs with no network, no licence acceptance and no real corpus in
the cache -- which is the point: the expensive stage has to be testable without
the expensive inputs.

Every test that touches the cache takes the ``tmp_cache`` fixture, which points
``UNSEGBENCH_CACHE`` at a tmp dir. Nothing here may write to a real cache.

The properties pinned here are the ones whose violation is silent:

* the persisted schema is integers only (a float means something was averaged
  before pooling, which breaks the monoid law that makes sharding safe);
* identity lives in parquet metadata, not per row;
* pooling shards then combining == pooling everything at once, bit-identical;
* the cache key is exactly ``(code_version, fingerprint, corpus_sha, mask)``;
* resume recomputes nothing and does not even load the tokenizer;
* `build_corpus` fails closed and leaves nothing that later looks cached.
"""

from __future__ import annotations

import io
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rich.console import Console

from unsegbench import build, clusters, positions, runner
from unsegbench.corpora.base import CorpusSpec
from unsegbench.errors import BuildValidationError, TokenizerUnavailable
from unsegbench.fetch import cache
from unsegbench.metrics import core as metrics
from unsegbench.tok.base import TokenizerAdapter, TokenizerSpec
from unsegbench.types import MASKS, STATS_COLUMNS, EncodeResult, RowStats, Segmented

# ==========================================================================
# Fake corpora
# ==========================================================================

_ZH = (
    "我喜欢吃苹果。",
    "今天天气很好。",
    "他在北京大学学习中文。",
    "这本书非常有意思！",
    "中国是一个伟大的国家。",
    "请把门关上，谢谢。",
    "我们明天去上海。",
    "她买了三本书。",
)
_TH = (
    "ฉันทำงานที่บ้าน",
    "ไปโรงเรียนทุกวัน",
    "เขาเรียนภาษาไทย",
    "สวัสดีครับ ผมชื่อสมชาย",
)


def _tile(text: str, width: int = 2) -> tuple[tuple[int, int], ...]:
    return tuple((s, min(s + width, len(text))) for s in range(0, len(text), width))


def _records(corpus_id: str = "fake_zh", texts=_ZH, split: str = "test") -> list[Segmented]:
    return [
        Segmented(id=f"{corpus_id}/{split}/{i:06d}", text=t, spans=_tile(t))
        for i, t in enumerate(texts)
    ]


def _bundle(
    corpus_id: str = "fake_zh", lang: str = "zh", sha: str = "sha-zh-0000", texts=_ZH
) -> runner.CorpusBundle:
    return runner.CorpusBundle(
        corpus_id=corpus_id, lang=lang, corpus_sha=sha, records=tuple(_records(corpus_id, texts))
    )


# ==========================================================================
# Fake tokenizer adapters -- module level so they stay picklable
# ==========================================================================


class FixedWidthAdapter(TokenizerAdapter):
    """Deterministic offline tokenizer: fixed-width chunks over codepoints."""

    def __init__(
        self,
        tokenizer_id: str = "fake-w2",
        width: int = 2,
        fingerprint: str = "fp-w2",
        flags: dict[str, int] | None = None,
        extra_tokens: int = 0,
    ) -> None:
        self.spec = TokenizerSpec(tokenizer_id=tokenizer_id, source="builtin", ref=f"w{width}")
        self.width = width
        self._fingerprint = fingerprint
        self._flags = Counter(flags or {})
        self.extra_tokens = extra_tokens
        self.encode_calls = 0
        self.load_calls = 0

    def load(self) -> FixedWidthAdapter:
        self.load_calls += 1
        return self

    def encode(self, text: str) -> EncodeResult:
        self.encode_calls += 1
        spans = _tile(text, self.width)
        return EncodeResult(
            spans=spans, n_tokens=len(spans) + self.extra_tokens, flags=Counter(self._flags)
        )

    def fingerprint(self) -> str:
        return self._fingerprint


class GoldAdapter(TokenizerAdapter):
    """A tokenizer that reproduces the gold segmentation exactly."""

    def __init__(self, records, tokenizer_id: str = "fake-gold", fingerprint: str = "fp-gold"):
        self.spec = TokenizerSpec(tokenizer_id=tokenizer_id, source="builtin", ref="gold")
        self._table = {r.text: r.spans for r in records}
        self._fingerprint = fingerprint
        self.encode_calls = 0

    def load(self) -> GoldAdapter:
        return self

    def encode(self, text: str) -> EncodeResult:
        self.encode_calls += 1
        spans = self._table[text]
        return EncodeResult(spans=spans, n_tokens=len(spans))

    def fingerprint(self) -> str:
        return self._fingerprint


class Factory:
    """Adapter factory that records every ``(tokenizer_id, lang)`` it is asked for."""
# 
    def __init__(self, **adapters) -> None:
        self.adapters = adapters
        self.calls: list[tuple[str, str]] = []

    def __call__(self, tokenizer_id: str, lang: str):
        self.calls.append((tokenizer_id, lang))
        obj = self.adapters[tokenizer_id.replace("-", "_")]
        if isinstance(obj, Exception):
            raise obj
        return obj


class ExplodingFactory:
    """Fails loudly if anything tries to load a tokenizer. Used to prove resume is free."""

    def __call__(self, tokenizer_id: str, lang: str):
        raise AssertionError(f"tokenizer {tokenizer_id!r} was loaded, but resume should be free")


class Loader:
    """Corpus loader that records every call, so we can prove resume reads nothing."""

    def __init__(self, *bundles: runner.CorpusBundle) -> None:
        self.bundles = {b.corpus_id: b for b in bundles}
        self.calls: list[tuple] = []

    def __call__(self, corpus_id: str, split: str, sample: int | None, seed: int):
        self.calls.append((corpus_id, split, sample, seed))
        return self.bundles[corpus_id]


class Meta:
    """The cheap ``(lang, corpus_sha)`` lookup the resume pre-check uses."""

    def __init__(self, *bundles: runner.CorpusBundle) -> None:
        self.bundles = {b.corpus_id: b for b in bundles}
        self.calls: list[tuple] = []

    def __call__(self, corpus_id: str, split: str, sample: int | None, seed: int):
        self.calls.append((corpus_id, split, sample, seed))
        b = self.bundles[corpus_id]
        return b.lang, b.corpus_sha


def _run(tokenizers, corpora, masks=MASKS, **kw) -> runner.RunResult:
    kw.setdefault("jobs", 1)
    kw.setdefault("progress", False)
    kw.setdefault("console", Console(file=io.StringIO(), width=120))
    return runner.run(tokenizers, corpora, masks, **kw)


def _shards(root: Path) -> list[Path]:
    return sorted((root / "stats").rglob("*.parquet"))


@pytest.fixture
def zh_bundle() -> runner.CorpusBundle:
    return _bundle()


@pytest.fixture
def th_bundle() -> runner.CorpusBundle:
    return _bundle(corpus_id="fake_th", lang="th", sha="sha-th-0000", texts=_TH)


# ==========================================================================
# 7. The persisted parquet schema: integers only
# ==========================================================================


def test_stats_schema_names_match_stats_columns_in_order() -> None:
    assert [f.name for f in runner.STATS_SCHEMA] == list(STATS_COLUMNS)


def test_stats_schema_sent_id_is_a_string_and_everything_else_is_int64() -> None:
    assert runner.STATS_SCHEMA.field("sent_id").type == pa.string()
    for name in STATS_COLUMNS[1:]:
        assert runner.STATS_SCHEMA.field(name).type == pa.int64(), name


def test_stats_schema_declares_no_floating_point_field() -> None:
    """CONTRACTS.md sec.5: a float in this schema is a bug."""
    assert not any(pa.types.is_floating(f.type) for f in runner.STATS_SCHEMA)


def test_stats_schema_forbids_nulls() -> None:
    assert all(not f.nullable for f in runner.STATS_SCHEMA)


def test_a_fractional_counter_cannot_be_written_under_this_schema() -> None:
    """A float is not merely discouraged here -- arrow refuses to store it."""
    columns = [pa.array(["fake/test/000000"], type=pa.string())]
    for name in STATS_COLUMNS[1:]:
        value = 1.5 if name == "b_tp" else 1
        columns.append(pa.array([value], type=pa.float64() if name == "b_tp" else pa.int64()))
    with pytest.raises(pa.ArrowInvalid, match="truncated"):
        pa.Table.from_arrays(columns, schema=runner.STATS_SCHEMA)


def test_an_integral_float_is_coerced_back_to_int64_never_stored_as_float() -> None:
    columns = [pa.array(["fake/test/000000"], type=pa.string())]
    for name in STATS_COLUMNS[1:]:
        value = 3.0 if name == "b_tp" else 1
        columns.append(pa.array([value], type=pa.float64() if name == "b_tp" else pa.int64()))
    table = pa.Table.from_arrays(columns, schema=runner.STATS_SCHEMA)
    assert table.schema.field("b_tp").type == pa.int64()


def test_written_shard_has_exactly_stats_columns_in_order(tmp_cache, zh_bundle) -> None:
    adapter = FixedWidthAdapter()
    _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=Factory(fake_w2=adapter),
        corpus_loader=Loader(zh_bundle),
    )
    (path,) = [p for p in _shards(tmp_cache) if p.name == "core.parquet"]
    assert pq.read_schema(path).names == list(STATS_COLUMNS)


def test_written_shard_dtypes_are_string_plus_integers(tmp_cache, zh_bundle) -> None:
    _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=Factory(fake_w2=FixedWidthAdapter()),
        corpus_loader=Loader(zh_bundle),
    )
    for path in _shards(tmp_cache):
        schema = pq.read_schema(path)
        assert schema.field("sent_id").type == pa.string()
        for name in STATS_COLUMNS[1:]:
            assert schema.field(name).type == pa.int64(), (path.name, name)


def test_written_shard_contains_no_float_column(tmp_cache, zh_bundle) -> None:
    _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=Factory(fake_w2=FixedWidthAdapter()),
        corpus_loader=Loader(zh_bundle),
    )
    for path in _shards(tmp_cache):
        frame = pq.read_table(path).to_pandas()
        assert pd.api.types.is_string_dtype(frame["sent_id"])
        for name in STATS_COLUMNS[1:]:
            assert frame[name].dtype.kind == "i", (path.name, name, frame[name].dtype)


def test_written_shard_row_values_survive_the_roundtrip(tmp_cache, zh_bundle) -> None:
    adapter = FixedWidthAdapter()
    _run(
        ["fake-w2"],
        ["fake_zh"],
        ["core"],
        adapter_factory=Factory(fake_w2=adapter),
        corpus_loader=Loader(zh_bundle),
    )
    (path,) = _shards(tmp_cache)
    frame = pq.read_table(path).to_pandas()
    expected = [
        runner.score_sentence(rec, "zh", FixedWidthAdapter(), ("core",))["core"]
        for rec in zh_bundle.records
    ]
    assert list(frame["sent_id"]) == [r.sent_id for r in expected]
    assert [tuple(t) for t in frame[list(STATS_COLUMNS)].itertuples(index=False)] == [
        r.as_tuple() for r in expected
    ]


def test_shard_write_is_atomic_and_leaves_no_part_file(tmp_cache, zh_bundle) -> None:
    _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=Factory(fake_w2=FixedWidthAdapter()),
        corpus_loader=Loader(zh_bundle),
    )
    assert list((tmp_cache / "stats").rglob("*.part")) == []


# ==========================================================================
# 8. Identity lives in file metadata, never per row
# ==========================================================================


def _one_shard(tmp_cache, bundle, mask="core", **kw) -> Path:
    _run(
        ["fake-w2"],
        [bundle.corpus_id],
        [mask],
        adapter_factory=Factory(fake_w2=FixedWidthAdapter(**kw)),
        corpus_loader=Loader(bundle),
    )
    (path,) = [p for p in _shards(tmp_cache) if p.name == f"{mask}.parquet"]
    return path


def test_shard_columns_carry_no_identity(tmp_cache, zh_bundle) -> None:
    """Repeating three strings across millions of rows would dominate the file."""
    path = _one_shard(tmp_cache, zh_bundle)
    names = pq.read_schema(path).names
    for column in ("tokenizer_id", "corpus_id", "mask", "tokenizer_fingerprint", "corpus_sha"):
        assert column not in names


def test_shard_metadata_carries_the_full_identity(tmp_cache, zh_bundle) -> None:
    path = _one_shard(tmp_cache, zh_bundle)
    meta = pq.read_schema(path).metadata
    assert meta[b"unsegbench.tokenizer_id"] == b"fake-w2"
    assert meta[b"unsegbench.corpus_id"] == b"fake_zh"
    assert meta[b"unsegbench.mask"] == b"core"
    assert meta[b"unsegbench.tokenizer_fingerprint"] == b"fp-w2"
    assert meta[b"unsegbench.corpus_sha"] == zh_bundle.corpus_sha.encode()
    assert meta[b"unsegbench.code_version"] == runner.CODE_VERSION.encode()


def test_shard_metadata_records_split_sample_and_seed(tmp_cache, zh_bundle) -> None:
    _run(
        ["fake-w2"],
        ["fake_zh"],
        ["raw"],
        split="train",
        sample=4,
        seed=17,
        adapter_factory=Factory(fake_w2=FixedWidthAdapter()),
        corpus_loader=Loader(zh_bundle),
    )
    (path,) = _shards(tmp_cache)
# improved
    meta = pq.read_schema(path).metadata
    assert meta[b"unsegbench.split"] == b"train"
    assert meta[b"unsegbench.sample"] == b"4"
    assert meta[b"unsegbench.seed"] == b"17"


def test_read_stats_restores_identity_as_columns(tmp_cache, zh_bundle) -> None:
    path = _one_shard(tmp_cache, zh_bundle)
    frame = runner.read_stats(path)
    assert set(frame["tokenizer_id"]) == {"fake-w2"}
# improved
    assert set(frame["corpus_id"]) == {"fake_zh"}
    assert set(frame["mask"]) == {"core"}
    assert set(frame["tokenizer_fingerprint"]) == {"fp-w2"}
    assert set(frame["corpus_sha"]) == {zh_bundle.corpus_sha}
    assert set(frame["code_version"]) == {runner.CODE_VERSION}


def test_read_stats_keeps_every_counter_column(tmp_cache, zh_bundle) -> None:
    frame = runner.read_stats(_one_shard(tmp_cache, zh_bundle))
    assert set(STATS_COLUMNS) <= set(frame.columns)
    assert len(frame) == len(zh_bundle.records)


def test_read_stats_falls_back_to_the_path_when_metadata_is_absent(tmp_cache) -> None:
    """A shard written by an older revision still knows its mask and corpus."""
    directory = cache.stats_dir(runner.CODE_VERSION, "fp-bare", "sha-bare")
    directory.mkdir(parents=True, exist_ok=True)
    columns = [pa.array(["fake/test/000000"], type=pa.string())]
    columns += [pa.array([1], type=pa.int64()) for _ in STATS_COLUMNS[1:]]
    pq.write_table(
        pa.Table.from_arrays(columns, schema=runner.STATS_SCHEMA), directory / "legal.parquet"
    )
    frame = runner.read_stats(directory / "legal.parquet")
    assert frame["mask"].iloc[0] == "legal"
    assert frame["corpus_sha"].iloc[0] == "sha-bare"
    assert frame["code_version"].iloc[0] == runner.CODE_VERSION
    assert frame["tokenizer_id"].iloc[0] == ""


# ==========================================================================
# 9. Aggregation is a monoid: shard then combine == combine everything
# ==========================================================================


def _rows(records, lang="zh", adapter=None, mask="core") -> list[RowStats]:
    adapter = adapter or FixedWidthAdapter(width=3)
    return [runner.score_sentence(r, lang, adapter, (mask,))[mask] for r in records]


def _write(path: Path, rows: list[RowStats], **meta_kw) -> Path:
    meta = runner._metadata(
        code_version=runner.CODE_VERSION,
        tokenizer_id=meta_kw.get("tokenizer_id", "fake-w3"),
        tokenizer_fingerprint=meta_kw.get("fingerprint", "fp-w3"),
        corpus_id=meta_kw.get("corpus_id", "fake_zh"),
        corpus_sha=meta_kw.get("corpus_sha", "sha-zh-0000"),
        mask=meta_kw.get("mask", "core"),
        split="test",
        sample=None,
        seed=0,
    )
    runner._write_shard(path, rows, meta)
    return path


def test_aggregating_per_shard_then_pooling_equals_aggregating_at_once(tmp_path) -> None:
    """The monoid law that makes tokenizer-major sharding safe."""
    rows = _rows(_records())
    whole = _write(tmp_path / "whole" / "core.parquet", rows)
    left = _write(tmp_path / "a" / "core.parquet", rows[:3])
    right = _write(tmp_path / "b" / "core.parquet", rows[3:])

    combined = runner.aggregate([left, right])
    at_once = runner.aggregate([whole])
    pd.testing.assert_frame_equal(combined, at_once, check_exact=True)


def test_pooled_counters_are_the_exact_sum_of_the_shard_counters(tmp_path) -> None:
    rows = _rows(_records())
    left = _write(tmp_path / "a" / "core.parquet", rows[:5])
    right = _write(tmp_path / "b" / "core.parquet", rows[5:])
    counters = [c for c in STATS_COLUMNS if c != "sent_id"]

    pooled = runner.aggregate([left, right])
    per_shard = [runner.aggregate([left]), runner.aggregate([right])]
    for column in counters:
        assert int(pooled[column].iloc[0]) == sum(int(f[column].iloc[0]) for f in per_shard)
    assert int(pooled["n_sents"].iloc[0]) == len(rows)


def test_metrics_recomputed_from_summed_counters_are_bit_identical(tmp_path) -> None:
    """Metrics must be a pure function of the pooled integers, not an average of averages."""
    rows = _rows(_records())
    left = _write(tmp_path / "a" / "core.parquet", rows[:4])
    right = _write(tmp_path / "b" / "core.parquet", rows[4:])
    pooled = runner.aggregate([left, right]).iloc[0]

    total = {c: 0 for c in STATS_COLUMNS if c != "sent_id"}
    for frame in (runner.aggregate([left]), runner.aggregate([right])):
        for column in total:
            total[column] += int(frame[column].iloc[0])
    expected = metrics.compute_row(
        metrics.Counts(tp=total["b_tp"], fp=total["b_fp"], fn=total["b_fn"], tn=total["b_tn"]),
        w_tp=total["w_tp"],
        w_pred=total["w_pred"],
        w_gold=total["w_gold"],
        w_intact=total["w_intact"],
        n_tokens=total["n_tokens"],
        n_chars=total["n_chars"],
        n_gold_words=total["n_gold_words"],
        crossing=total["crossing_tokens"],
    )
    for field in metrics.MetricRow.__slots__:
        assert pooled[field] == getattr(expected, field), field


def test_aggregate_is_independent_of_shard_order(tmp_path) -> None:
    rows = _rows(_records())
    left = _write(tmp_path / "a" / "core.parquet", rows[:3])
    right = _write(tmp_path / "b" / "core.parquet", rows[3:])
    pd.testing.assert_frame_equal(
        runner.aggregate([left, right]), runner.aggregate([right, left]), check_exact=True
    )


def test_aggregate_gives_one_row_per_tokenizer_corpus_mask(tmp_path) -> None:
    rows = _rows(_records())
    paths = [
        _write(tmp_path / "s0" / "core.parquet", rows, mask="core"),
        _write(tmp_path / "s1" / "raw.parquet", rows, mask="raw"),
        _write(tmp_path / "s2" / "core.parquet", rows, corpus_id="fake_th"),
        _write(tmp_path / "s3" / "core.parquet", rows, tokenizer_id="fake-gold"),
    ]
    frame = runner.aggregate(paths)
    assert len(frame) == 4
    assert list(frame.columns[: len(runner.GROUP_COLUMNS)]) == list(runner.GROUP_COLUMNS)
    assert set(frame["mask"]) == {"core", "raw"}


def test_aggregate_accepts_a_directory(tmp_path) -> None:
    rows = _rows(_records())
    _write(tmp_path / "d" / "core.parquet", rows[:3])
    _write(tmp_path / "d" / "raw.parquet", rows[3:], mask="raw")
    assert len(runner.aggregate(tmp_path / "d")) == 2


def test_aggregate_accepts_a_single_path_and_a_dataframe(tmp_path) -> None:
    rows = _rows(_records())
    path = _write(tmp_path / "d" / "core.parquet", rows)
    from_path = runner.aggregate(path)
    from_frame = runner.aggregate(runner.read_stats(path))
    pd.testing.assert_frame_equal(from_path, from_frame, check_exact=True)


def test_aggregate_of_nothing_is_an_empty_frame_with_group_columns(tmp_path) -> None:
    empty = runner.aggregate([])
    assert empty.empty
    assert list(empty.columns) == list(runner.GROUP_COLUMNS)
    (tmp_path / "empty-dir").mkdir()
    assert runner.aggregate(tmp_path / "empty-dir").empty
    assert runner.aggregate(pd.DataFrame()).empty


def test_aggregate_touches_no_tokenizer(tmp_path, monkeypatch) -> None:
    """The report layer must be reproducible with no network and no licences."""
    monkeypatch.setattr(runner, "default_adapter_factory", ExplodingFactory())
    path = _write(tmp_path / "d" / "core.parquet", _rows(_records()))
    assert len(runner.aggregate(path)) == 1


# ==========================================================================
# 10. Resume recomputes nothing
# ==========================================================================


def test_second_run_resumes_every_shard(tmp_cache, zh_bundle, th_bundle) -> None:
    factory = Factory(fake_w2=FixedWidthAdapter())
    loader = Loader(zh_bundle, th_bundle)
    first = _run(["fake-w2"], ["fake_zh", "fake_th"], adapter_factory=factory, corpus_loader=loader)
    assert {o.status for o in first.outcomes} == {"computed"}

    second = _run(
        ["fake-w2"], ["fake_zh", "fake_th"], adapter_factory=factory, corpus_loader=loader
    )
    assert {o.status for o in second.outcomes} == {"resumed"}
    assert len(second.outcomes) == len(first.outcomes) == 2 * len(MASKS)


def test_resume_does_not_rewrite_a_single_shard(tmp_cache, zh_bundle) -> None:
    factory = Factory(fake_w2=FixedWidthAdapter())
    loader = Loader(zh_bundle)
    _run(["fake-w2"], ["fake_zh"], adapter_factory=factory, corpus_loader=loader)
    before = {p: p.stat().st_mtime_ns for p in _shards(tmp_cache)}
    assert len(before) == len(MASKS)

    time.sleep(0.01)
    _run(["fake-w2"], ["fake_zh"], adapter_factory=factory, corpus_loader=loader)
    after = {p: p.stat().st_mtime_ns for p in _shards(tmp_cache)}
    assert after == before


def test_resume_never_loads_the_tokenizer(tmp_cache, zh_bundle) -> None:
    """Memoised fingerprints turn "prove it is done" into a stat() per file."""
    meta = Meta(zh_bundle)
    _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=Factory(fake_w2=FixedWidthAdapter()),
        corpus_loader=Loader(zh_bundle),
        corpus_meta_fn=meta,
    )
    result = _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=ExplodingFactory(),
        corpus_loader=Loader(zh_bundle),
        corpus_meta_fn=meta,
    )
    assert {o.status for o in result.outcomes} == {"resumed"}


def test_resume_never_reads_the_corpus(tmp_cache, zh_bundle) -> None:
    factory = Factory(fake_w2=FixedWidthAdapter())
    meta = Meta(zh_bundle)
    first_loader = Loader(zh_bundle)
    _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=factory,
        corpus_loader=first_loader,
        corpus_meta_fn=meta,
    )
    assert first_loader.calls  # the first pass had to read it

    second_loader = Loader(zh_bundle)
    _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=factory,
        corpus_loader=second_loader,
        corpus_meta_fn=meta,
    )
    assert second_loader.calls == []


def test_resume_memoises_the_fingerprint_to_disk(tmp_cache, zh_bundle) -> None:
    _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=Factory(fake_w2=FixedWidthAdapter()),
        corpus_loader=Loader(zh_bundle),
    )
    store = tmp_cache / "fingerprints.json"
    assert store.exists()
    assert b"fake-w2|zh" in store.read_bytes()


def test_resume_false_recomputes_everything(tmp_cache, zh_bundle) -> None:
    factory = Factory(fake_w2=FixedWidthAdapter())
    loader = Loader(zh_bundle)
    _run(["fake-w2"], ["fake_zh"], adapter_factory=factory, corpus_loader=loader)
    before = {p: p.stat().st_mtime_ns for p in _shards(tmp_cache)}

    time.sleep(0.01)
    result = _run(
        ["fake-w2"], ["fake_zh"], adapter_factory=factory, corpus_loader=loader, resume=False
    )
    assert {o.status for o in result.outcomes} == {"computed"}
    after = {p: p.stat().st_mtime_ns for p in _shards(tmp_cache)}
    assert set(after) == set(before)
    assert after != before


def test_resume_recomputes_when_a_shard_is_deleted(tmp_cache, zh_bundle) -> None:
    factory = Factory(fake_w2=FixedWidthAdapter())
    loader = Loader(zh_bundle)
    _run(["fake-w2"], ["fake_zh"], adapter_factory=factory, corpus_loader=loader)
    _shards(tmp_cache)[0].unlink()

    result = _run(["fake-w2"], ["fake_zh"], adapter_factory=factory, corpus_loader=loader)
    assert {o.status for o in result.outcomes} == {"computed"}
    assert len(_shards(tmp_cache)) == len(MASKS)


def test_a_truncated_shard_is_not_treated_as_complete(tmp_cache, zh_bundle) -> None:
    factory = Factory(fake_w2=FixedWidthAdapter())
    loader = Loader(zh_bundle)
    _run(["fake-w2"], ["fake_zh"], adapter_factory=factory, corpus_loader=loader)
    victim = _shards(tmp_cache)[0]
    victim.write_bytes(b"PAR1 truncated")

    result = _run(["fake-w2"], ["fake_zh"], adapter_factory=factory, corpus_loader=loader)
    assert {o.status for o in result.outcomes} == {"computed"}
    assert runner._shard_is_valid(victim, {}) is True  # rewritten and parseable again


def test_shard_is_valid_rejects_foreign_metadata(tmp_cache, zh_bundle) -> None:
    path = _one_shard(tmp_cache, zh_bundle)
    assert runner._shard_is_valid(path, runner._metadata(corpus_id="fake_zh")) is True
    assert runner._shard_is_valid(path, runner._metadata(corpus_id="sighan_pku")) is False
    assert runner._shard_is_valid(path.with_name("nope.parquet"), {}) is False


# ==========================================================================
# 11. Cache key == (code_version, fingerprint, corpus_sha, mask)
# ==========================================================================


def test_cache_path_is_stable_when_nothing_changes(tmp_cache) -> None:
    assert runner._shard_path("fp-a", "sha-a", "core") == runner._shard_path(
        "fp-a", "sha-a", "core"
    )


def test_cache_path_changes_with_the_tokenizer_fingerprint(tmp_cache) -> None:
    assert runner._shard_path("fp-a", "sha-a", "core") != runner._shard_path(
        "fp-b", "sha-a", "core"
    )


def test_cache_path_changes_with_the_corpus_sha(tmp_cache) -> None:
    assert runner._shard_path("fp-a", "sha-a", "core") != runner._shard_path(
        "fp-a", "sha-b", "core"
    )


def test_cache_path_changes_with_the_mask(tmp_cache) -> None:
    paths = {runner._shard_path("fp-a", "sha-a", m) for m in MASKS}
    assert len(paths) == len(MASKS)


def test_cache_path_changes_with_the_code_version(tmp_cache, monkeypatch) -> None:
    before = runner._shard_path("fp-a", "sha-a", "core")
    monkeypatch.setattr(runner, "CODE_VERSION", "c99-b99-a99")
    assert runner._shard_path("fp-a", "sha-a", "core") != before


def test_code_version_folds_in_the_builder_and_adapter_versions() -> None:
    from unsegbench.corpora.base import BUILDER_VERSION
    from unsegbench.tok.base import ADAPTER_VERSION

    assert BUILDER_VERSION in runner.CODE_VERSION
    assert ADAPTER_VERSION in runner.CODE_VERSION


def test_a_new_fingerprint_forces_a_recompute(tmp_cache, zh_bundle) -> None:
    loader = Loader(zh_bundle)
    _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=Factory(fake_w2=FixedWidthAdapter(fingerprint="fp-old")),
        corpus_loader=loader,
    )
    # The fingerprint memo is keyed on (tokenizer_id, lang) and invalidated by
    # ADAPTER_VERSION, so a warm memo short-circuits before the adapter is ever
    # asked. Cold memo == what a fresh machine sees, and that is the path where
    # the fingerprint's role in the cache key has to hold.
    (tmp_cache / "fingerprints.json").unlink()
    result = _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=Factory(fake_w2=FixedWidthAdapter(fingerprint="fp-new")),
        corpus_loader=loader,
    )
    assert {o.status for o in result.outcomes} == {"computed"}
    assert len(_shards(tmp_cache)) == 2 * len(MASKS)


def test_a_warm_fingerprint_memo_short_circuits_before_the_adapter(tmp_cache, zh_bundle) -> None:
    """Documented consequence of memoising fingerprints, and why ADAPTER_VERSION exists.

    Resume proves a shard is done from ``(tokenizer_id, lang)`` alone, so a
    tokenizer whose *content* changed under an unchanged id is NOT noticed. The
    memo is invalidated by `ADAPTER_VERSION`; anything else that can change a
    fingerprint must bump it.
    """
    loader = Loader(zh_bundle)
    _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=Factory(fake_w2=FixedWidthAdapter(fingerprint="fp-old")),
        corpus_loader=loader,
    )
    result = _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=Factory(fake_w2=FixedWidthAdapter(fingerprint="fp-new")),
        corpus_loader=loader,
    )
    assert {o.status for o in result.outcomes} == {"resumed"}
    assert len(_shards(tmp_cache)) == len(MASKS)


def test_a_new_corpus_sha_forces_a_recompute(tmp_cache, zh_bundle) -> None:
    factory = Factory(fake_w2=FixedWidthAdapter())
    _run(["fake-w2"], ["fake_zh"], adapter_factory=factory, corpus_loader=Loader(zh_bundle))
    rebuilt = _bundle(sha="sha-zh-REBUILT")
    result = _run(["fake-w2"], ["fake_zh"], adapter_factory=factory, corpus_loader=Loader(rebuilt))
    assert {o.status for o in result.outcomes} == {"computed"}
    assert len(_shards(tmp_cache)) == 2 * len(MASKS)


def test_corpus_key_is_exactly_the_manifest_sha_for_the_full_test_split() -> None:
    """CONTRACTS.md sec.5: the canonical case is the manifest digest, unmodified."""
    sha = "e" * 64
    assert build.subset_key(sha, "test", None, 0) == sha


@pytest.mark.parametrize(
    ("split", "sample", "seed"),
    [("train", None, 0), ("test", 100, 0), ("test", 100, 1), ("test", 200, 0)],
)
def test_subset_key_binds_split_sample_and_seed(split: str, sample, seed: int) -> None:
    sha = "e" * 64
    assert build.subset_key(sha, split, sample, seed) != sha


def test_subset_keys_are_distinct_across_sampling_parameters() -> None:
    sha = "e" * 64
    keys = {
        build.subset_key(sha, sp, sm, sd)
        for sp in ("test", "train")
        for sm in (None, 100, 200)
        for sd in (0, 1)
    }
    assert len(keys) == 2 * 3 * 2 - 1  # (test, None, 0) and (test, None, 1) coincide by design


# ==========================================================================
# 12. All three masks come out of one tokenization
# ==========================================================================


def _clear_position_caches() -> None:
    for module in (positions, clusters):
        for name in dir(module):
            fn = getattr(module, name)
            if hasattr(fn, "cache_clear"):
                fn.cache_clear()


def test_run_tokenizes_each_sentence_exactly_once_for_three_masks(tmp_cache, zh_bundle) -> None:
    adapter = FixedWidthAdapter()
    result = _run(
        ["fake-w2"],
        ["fake_zh"],
        MASKS,
        adapter_factory=Factory(fake_w2=adapter),
        corpus_loader=Loader(zh_bundle),
    )
    assert adapter.encode_calls == len(zh_bundle.records)
    assert len({o.mask for o in result.outcomes}) == len(MASKS)
    assert len(_shards(tmp_cache)) == len(MASKS)


def test_three_masks_are_computed_in_one_pass(monkeypatch) -> None:
    calls = Counter()
    real_masks, real_mask = positions.compute_masks, positions.compute_mask

    def spy_masks(text, lang):
        calls["masks"] += 1
        return real_masks(text, lang)

    def spy_mask(text, lang, mask):
        calls["mask"] += 1
        return real_mask(text, lang, mask)

    monkeypatch.setattr(positions, "compute_masks", spy_masks)
    monkeypatch.setattr(positions, "compute_mask", spy_mask)

    records = _records()
    adapter = FixedWidthAdapter()
    for rec in records:
        runner.score_sentence(rec, "zh", adapter, MASKS)
    assert calls["masks"] == len(records)
    assert calls["mask"] == 0
    assert adapter.encode_calls == len(records)


def test_a_single_mask_skips_the_all_masks_pass(monkeypatch) -> None:
    calls = Counter()
    real_masks, real_mask = positions.compute_masks, positions.compute_mask
    monkeypatch.setattr(
        positions, "compute_masks", lambda t, lg: (calls.update(masks=1), real_masks(t, lg))[1]
    )
    monkeypatch.setattr(
        positions, "compute_mask", lambda t, lg, m: (calls.update(mask=1), real_mask(t, lg, m))[1]
    )
    records = _records()
    for rec in records:
        runner.score_sentence(rec, "zh", FixedWidthAdapter(), ("core",))
    assert calls["mask"] == len(records)
    assert calls["masks"] == 0

# improved

def test_score_sentence_returns_one_row_per_requested_mask() -> None:
    rec = _records()[2]
    out = runner.score_sentence(rec, "zh", FixedWidthAdapter(), MASKS)
    assert set(out) == set(MASKS)
    assert all(isinstance(v, RowStats) for v in out.values())
    assert {v.sent_id for v in out.values()} == {rec.id}
    assert {v.n_chars for v in out.values()} == {rec.n}


def test_mask_universes_are_nested_core_subset_legal_subset_raw() -> None:
    rec = _records("fake_th", _TH)[3]
    out = runner.score_sentence(rec, "th", FixedWidthAdapter(), MASKS)
    assert out["core"].n_mask <= out["legal"].n_mask <= out["raw"].n_mask


@pytest.mark.slow
def test_three_masks_cost_well_under_three_times_one_mask() -> None:
    """The whole performance story: masks are shared, only intersections repeat."""
    texts = ["".join(_ZH)[: 40 + (i % 37)] + f"{i:04d}" for i in range(120)]
    records = [
        Segmented(id=f"perf/test/{i:06d}", text=t, spans=_tile(t)) for i, t in enumerate(texts)
    ]
    adapter = FixedWidthAdapter(width=3)

    def measure(masks, reps=5):
        best = float("inf")
        for _ in range(reps):
            _clear_position_caches()
            start = time.perf_counter()
            for rec in records:
                runner.score_sentence(rec, "zh", adapter, masks)
            best = min(best, time.perf_counter() - start)
        return best

    three = measure(MASKS)
    singles = {m: measure((m,)) for m in MASKS}
    mean_single = sum(singles.values()) / len(singles)

    # Headline mask as the baseline: three universes for ~1.4x the cost of one.
    assert three < 2.5 * singles["core"], (three, singles)
    # And strictly sub-linear against the average single-mask pass.
    assert three < 3.0 * mean_single, (three, singles)


# ==========================================================================
# 13. Sampling is deterministic, order-stable and length-stratified
# ==========================================================================


def _population(n: int = 400) -> list[Segmented]:
    """Records whose lengths span 4..99 in a fixed, non-monotonic pattern."""
    out = []
    for i in range(n):
        length = 4 + (i * 37) % 96
        text = "字" * length
        out.append(Segmented(id=f"pop/test/{i:06d}", text=text, spans=_tile(text)))
    return out


def _ids(records) -> list[str]:
    return [r.id for r in records]


def test_sampling_is_deterministic_for_a_given_seed() -> None:
    population = _population()
    assert _ids(build.select_sample(population, 100, seed=7)) == _ids(
        build.select_sample(population, 100, seed=7)
    )


def test_sampling_changes_with_the_seed() -> None:
    population = _population()
    a = set(_ids(build.select_sample(population, 100, seed=0)))
    b = set(_ids(build.select_sample(population, 100, seed=1)))
    assert a != b
    assert len(a) == len(b) == 100


def test_sampling_is_independent_of_input_order() -> None:
    """Explicitly not reservoir sampling: a reordered loader must not move the sample."""
    population = _population()
    shuffled = population[137:] + population[:137]
    assert _ids(build.select_sample(population, 100, seed=3)) == _ids(
        build.select_sample(shuffled, 100, seed=3)
    )


def test_sample_is_returned_in_sent_id_order() -> None:
    sample = build.select_sample(_population(), 100, seed=3)
    assert _ids(sample) == sorted(_ids(sample))


def test_sample_preserves_the_length_quartiles() -> None:
    population = _population()
    n, k = len(population), 100
    ranked = sorted(population, key=lambda r: (r.n, r.id))
    quartile = {rec.id: min(3, rank * 4 // n) for rank, rec in enumerate(ranked)}

    counts = Counter(quartile[r.id] for r in build.select_sample(population, k, seed=11))
    assert set(counts) == {0, 1, 2, 3}
    for q in range(4):
        assert abs(counts[q] - k // 4) <= 6, counts


def test_sample_fills_every_length_decile(tmp_path) -> None:
    population = _population()
    n, k = len(population), 100
    ranked = sorted(population, key=lambda r: (r.n, r.id))
    decile = {rec.id: min(9, rank * 10 // n) for rank, rec in enumerate(ranked)}
# improved
    counts = Counter(decile[r.id] for r in build.select_sample(population, k, seed=5))
    assert sorted(counts) == list(range(10))
    assert set(counts.values()) == {k // 10}


def test_sample_median_length_tracks_the_population() -> None:
    population = _population()
    sample = build.select_sample(population, 100, seed=2)
    pop_median = sorted(r.n for r in population)[len(population) // 2]
    sample_median = sorted(r.n for r in sample)[len(sample) // 2]
    assert abs(sample_median - pop_median) <= 5


@pytest.mark.parametrize("size", [1, 7, 40, 100, 399])
def test_sample_returns_exactly_the_requested_size(size: int) -> None:
    assert len(build.select_sample(_population(), size, seed=0)) == size


def test_sample_none_returns_the_population_unchanged() -> None:
    population = _population()
    assert build.select_sample(population, None) is population


def test_sample_larger_than_the_population_returns_everything_sorted() -> None:
    population = _population(20)
    sample = build.select_sample(population, 500, seed=0)
    assert _ids(sample) == sorted(_ids(population))


def test_sample_of_zero_or_fewer_is_empty() -> None:
    assert build.select_sample(_population(20), 0) == []
    assert build.select_sample(_population(20), -3) == []


# ==========================================================================
# 14. build_corpus fails closed
# ==========================================================================


class _FakeSpec(CorpusSpec):
    """A corpus with no artifacts, so building it needs no network."""

    corpus_id = "fake_ok"
    lang = "zh"
    script = "Hans"
    convention = "fake"
    license = "CC0-1.0"
    redistributable = True
    source_url = "https://example.invalid/fake.zip"
    version = "v1"
    gap_charset = " ​。！，"
    notes = "synthetic"

    @property
    def artifacts(self):
        return ()

    def splits(self):
        return ("test",)

    def parse(self, raw_dir, split):
        yield from _records(self.corpus_id, _ZH, split)


class _BadSpec(_FakeSpec):
    """Its parse drops content: the second record leaves 在北京 uncovered."""

    corpus_id = "fake_bad"
    gap_charset = " "

    def parse(self, raw_dir, split):
        yield Segmented(
            id=f"{self.corpus_id}/{split}/000000", text="我喜欢", spans=((0, 1), (1, 3))
        )
        yield Segmented(id=f"{self.corpus_id}/{split}/000001", text="他在北京", spans=((0, 1),))


class _EmptySpec(_FakeSpec):
    corpus_id = "fake_empty"

    def parse(self, raw_dir, split):
        return iter(())


def test_build_corpus_raises_on_an_invalid_record(tmp_cache) -> None:
    with pytest.raises(BuildValidationError, match="not in gap_charset"):
        build.build_corpus(_BadSpec(), progress=False)


def test_a_failed_build_leaves_nothing_that_looks_cached(tmp_cache) -> None:
    with pytest.raises(BuildValidationError):
        build.build_corpus(_BadSpec(), progress=False)

    canon = cache.canonical_dir("fake_bad", "v1")
    assert not (canon / build.MANIFEST_NAME).exists()
    assert list(tmp_cache.rglob("*.jsonl.gz")) == []
    assert list(tmp_cache.rglob("manifest.json")) == []
    # Whatever survives is staging only -- never a file a later run would load.
    assert {p.name for p in canon.iterdir()} <= {"test.jsonl.gz.part"}
    assert build.is_built("fake_bad") is False


def test_a_failed_build_is_not_resumable_as_success(tmp_cache) -> None:
    with pytest.raises(BuildValidationError):
        build.build_corpus(_BadSpec(), progress=False)
    with pytest.raises(BuildValidationError):
        build.build_corpus(_BadSpec(), progress=False)
    with pytest.raises(BuildValidationError, match="not built"):
        build.load_manifest("fake_bad")


def test_build_corpus_raises_when_every_split_is_empty(tmp_cache) -> None:
    with pytest.raises(BuildValidationError, match="zero records"):
        build.build_corpus(_EmptySpec(), progress=False)
    assert build.is_built("fake_empty") is False


def test_build_corpus_happy_path_writes_a_manifest(tmp_cache) -> None:
    manifest = build.build_corpus(_FakeSpec(), progress=False)
    records = _records("fake_ok")
    assert manifest.corpus_id == "fake_ok"
    assert manifest.n_sents == len(records)
    assert manifest.n_words == sum(len(r.spans) for r in records)
#     assert manifest.n_chars == sum(r.n for r in records)
    assert manifest.gap_charset == _FakeSpec.gap_charset
    assert set(manifest.splits) == {"test"}
    assert build.is_built("fake_ok") is True


def test_build_corpus_output_round_trips_through_load_corpus(tmp_cache) -> None:
    build.build_corpus(_FakeSpec(), progress=False)
    assert build.load_corpus("fake_ok", "test") == _records("fake_ok")


def test_build_corpus_is_idempotent(tmp_cache) -> None:
    first = build.build_corpus(_FakeSpec(), progress=False)
    path = cache.canonical_dir("fake_ok", "v1") / build.MANIFEST_NAME
    mtime = path.stat().st_mtime_ns
    second = build.build_corpus(_FakeSpec(), progress=False)
    assert second == first
    assert path.stat().st_mtime_ns == mtime


def test_manifest_sha_is_the_corpus_side_of_the_cache_key(tmp_cache) -> None:
    manifest = build.build_corpus(_FakeSpec(), progress=False)
    assert build.corpus_key("fake_ok") == build.manifest_sha(manifest)
    assert build.manifest_sha(manifest) != build.manifest_sha(
        build.CorpusManifest.from_json(manifest.to_json().replace(b'"v1"', b'"v2"'))
    )


def test_load_corpus_rejects_a_split_that_was_never_built(tmp_cache) -> None:
    build.build_corpus(_FakeSpec(), progress=False)
    with pytest.raises(BuildValidationError, match="no split"):
        build.load_corpus("fake_ok", "train")


# ==========================================================================
# Scoring and sweep behaviour
# ==========================================================================


def test_score_sentence_of_a_perfect_tokenizer_has_no_errors() -> None:
    records = _records()
    adapter = GoldAdapter(records)
    for rec in records:
        row = runner.score_sentence(rec, "zh", adapter, ("raw",))["raw"]
        assert row.b_fn == 0
        assert row.b_fp == 0
        assert row.w_tp == row.w_gold == row.w_pred
        assert row.crossing_tokens == 0
        assert row.w_intact == row.n_gold_words


def test_score_sentence_keeps_raw_and_accepted_token_counts_apart() -> None:
    rec = _records()[0]
    adapter = FixedWidthAdapter(extra_tokens=3)
    row = runner.score_sentence(rec, "zh", adapter, ("raw",))["raw"]
    assert row.n_tokens == row.n_tokens_accepted + 3


def test_score_sentence_copies_integrity_flags_into_counter_columns() -> None:
    adapter = FixedWidthAdapter(
        flags={
            "midcodepoint_split": 2,
            "cluster_split": 3,
            "overlap_rejected": 4,
            "dropped_chars": 5,
            "prefix_space_trim": 6,
            "normaliser_mutated": 7,  # not persisted: no column for it
        }
    )
    row = runner.score_sentence(_records()[0], "zh", adapter, ("core",))["core"]
    assert (row.f_midcodepoint, row.f_cluster_split) == (2, 3)
    assert (row.f_overlap_rejected, row.f_dropped_chars, row.f_prefix_space_trim) == (4, 5, 6)


def test_run_rejects_an_unknown_mask(tmp_cache, zh_bundle) -> None:
    with pytest.raises(ValueError, match="unknown mask"):
        _run(
            ["fake-w2"],
            ["fake_zh"],
            ["core", "bogus"],
            adapter_factory=Factory(fake_w2=FixedWidthAdapter()),
            corpus_loader=Loader(zh_bundle),
        )


def test_run_canonicalises_the_mask_order(tmp_cache, zh_bundle) -> None:
    result = _run(
        ["fake-w2"],
        ["fake_zh"],
        ["core", "raw"],
        adapter_factory=Factory(fake_w2=FixedWidthAdapter()),
        corpus_loader=Loader(zh_bundle),
    )
    assert [o.mask for o in result.outcomes] == ["raw", "core"]


def test_run_skips_an_unavailable_tokenizer(tmp_cache, zh_bundle) -> None:
    """One gated repo must never take down a six-hour sweep."""
    factory = Factory(
        fake_w2=FixedWidthAdapter(), fake_gated=TokenizerUnavailable("gated repo, no token")
    )
    result = _run(
        ["fake-w2", "fake-gated"],
        ["fake_zh"],
        ["core"],
        adapter_factory=factory,
        corpus_loader=Loader(zh_bundle),
    )
    statuses = Counter(o.status for o in result.outcomes)
    assert statuses == {"computed": 1, "skipped": 1}
    assert any("gated repo" in o.reason for o in result.outcomes)
    assert set(result.frame["tokenizer_id"]) == {"fake-w2"}


def test_strict_makes_an_unavailable_tokenizer_fatal(tmp_cache, zh_bundle) -> None:
    factory = Factory(fake_gated=TokenizerUnavailable("gated repo"))
    with pytest.raises(TokenizerUnavailable):
        _run(
            ["fake-gated"],
            ["fake_zh"],
            ["core"],
            strict=True,
            adapter_factory=factory,
            corpus_loader=Loader(zh_bundle),
        )


def test_run_returns_one_aggregate_row_per_tokenizer_corpus_mask(
    tmp_cache, zh_bundle, th_bundle
) -> None:
    result = _run(
        ["fake-w2", "fake-gold"],
        ["fake_zh", "fake_th"],
        MASKS,
        adapter_factory=Factory(
            fake_w2=FixedWidthAdapter(),
# improved
            fake_gold=GoldAdapter(list(zh_bundle.records) + list(th_bundle.records)),
        ),
        corpus_loader=Loader(zh_bundle, th_bundle),
    )
    assert len(result.frame) == 2 * 2 * len(MASKS)
    assert result.fingerprints[("fake-w2", "zh")] == "fp-w2"
    assert result.fingerprints[("fake-gold", "th")] == "fp-gold"


def test_run_writes_only_aggregates_to_the_output_directory(tmp_cache, tmp_path, zh_bundle) -> None:
    out = tmp_path / "results"
    result = _run(
        ["fake-w2"],
        ["fake_zh"],
        ["core"],
        out=out,
        adapter_factory=Factory(fake_w2=FixedWidthAdapter()),
        corpus_loader=Loader(zh_bundle),
    )
    assert result.out_dir == out
    assert {p.name for p in out.iterdir()} == {"results.csv", "results.parquet"}
    written = pd.read_parquet(out / "results.parquet")
    assert "sent_id" not in written.columns  # per-sentence rows never leave the cache
    assert len(written) == 1


def test_run_with_no_work_returns_an_empty_frame(tmp_cache) -> None:
    result = _run([], [], MASKS, adapter_factory=ExplodingFactory())
    assert result.frame.empty
    assert result.outcomes == []
    assert result.paths == []


def test_run_result_paths_point_at_the_written_shards(tmp_cache, zh_bundle) -> None:
    result = _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=Factory(fake_w2=FixedWidthAdapter()),
        corpus_loader=Loader(zh_bundle),
    )
    assert sorted(result.paths) == _shards(tmp_cache)
    assert all(p.exists() for p in result.paths)


def test_run_never_writes_outside_the_configured_cache(tmp_cache, zh_bundle) -> None:
    _run(
        ["fake-w2"],
        ["fake_zh"],
        adapter_factory=Factory(fake_w2=FixedWidthAdapter()),
        corpus_loader=Loader(zh_bundle),
    )
    assert tmp_cache.exists()
    assert all(str(p).startswith(str(tmp_cache)) for p in _shards(tmp_cache))

# Updated

# Updated

# Enhanced

# Enhanced

# Refined
