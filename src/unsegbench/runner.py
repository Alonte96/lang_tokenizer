"""The sweep: ~25 tokenizers x ~12 corpora x 3 masks, computed once.

This module is where the entire performance story lives, and every design choice
in it is a response to a specific cost:

* **Tokenizer-major sharding.** One worker owns one tokenizer and streams every
  corpus through it. Several HF tokenizers take 10-20s to construct; sharding
  corpus-major would pay that per (tokenizer, corpus) pair instead of per
  tokenizer, which on a full sweep is minutes of pure overhead per tokenizer.
* **Masks once per sentence.** `positions.compute_masks` returns all three
  universes from one pass over the string, and the result is reused across every
  mask. Scoring three masks therefore costs ~1.3x one mask, not 3x -- the
  marginal cost of `core` over `legal` is one `trivial_positions` call, and
  `legal_positions`/`trivial_positions` are themselves LRU-cached.
* **Only integers are persisted.** Each shard writes the per-sentence counters in
  `types.STATS_COLUMNS` and nothing else -- never tokens, never spans. A full
  sweep is ~75M token-events; the counters are a few million small ints. Every
  table, plot, bootstrap and rank interval downstream is a groupby over them,
  which is why the report costs ~2s and why `aggregate` never touches a
  tokenizer.
* **Resume must be free.** The cache key is
  ``(code_version, tokenizer_fingerprint, corpus_manifest_sha, mask)``, exactly
  as CONTRACTS.md sec.5 specifies. Fingerprints are memoised to disk so that a
  restarted sweep can prove a shard is complete *without* loading the tokenizer
  that produced it -- otherwise "resume" would still cost 25 model loads.
* **`TOKENIZERS_PARALLELISM=false` before forking.** The Rust tokenizers library
  spins up its own thread pool per process. Ten worker processes on twelve cores
  each starting twelve threads gives both a warning storm on every fork and a
  net slowdown from oversubscription.

A tokenizer that raises `TokenizerUnavailable` is logged and skipped. One gated
repo must never take down a six-hour sweep.
"""

from __future__ import annotations

import importlib
import os
import pickle
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from unsegbench import positions
from unsegbench.corpora.base import BUILDER_VERSION
from unsegbench.errors import TokenizerUnavailable
from unsegbench.fetch import cache
from unsegbench.metrics import core as metrics
from unsegbench.tok.base import ADAPTER_VERSION, TokenizerAdapter
from unsegbench.types import MASKS, STATS_COLUMNS, RowStats, Segmented

__all__ = [
    "CODE_VERSION",
    "GROUP_COLUMNS",
    "STATS_SCHEMA",
    "CorpusBundle",
    "RunResult",
    "ShardOutcome",
    "aggregate",
    "default_adapter_factory",
    "default_corpus_loader",
    "read_stats",
    "run",
    "score_sentence",
]

#: Bumped whenever the meaning of a persisted counter changes. Folds in the
#: corpus builder and adapter versions, because either one changing invalidates
#: the same parquet files.
CODE_VERSION: str = f"c1-b{BUILDER_VERSION}-a{ADAPTER_VERSION}"

#: Identity columns reconstructed from parquet metadata, never stored per row.
GROUP_COLUMNS: tuple[str, ...] = ("tokenizer_id", "corpus_id", "mask")

_COUNTER_COLUMNS: tuple[str, ...] = tuple(c for c in STATS_COLUMNS if c != "sent_id")

#: The sufficient-statistic schema. ``sent_id`` is a string; EVERYTHING else is a
#: 64-bit integer. A float here is a bug (CONTRACTS.md sec.5) -- it would mean
#: some quantity was averaged before pooling, which silently breaks the monoid
#: law that makes sharding safe.
STATS_SCHEMA: pa.Schema = pa.schema(
    [pa.field("sent_id", pa.string(), nullable=False)]
    + [pa.field(c, pa.int64(), nullable=False) for c in _COUNTER_COLUMNS]
)

_META_PREFIX = "unsegbench."
_FLAG_TO_COLUMN: tuple[tuple[str, str], ...] = (
    ("midcodepoint_split", "f_midcodepoint"),
    ("cluster_split", "f_cluster_split"),
    ("overlap_rejected", "f_overlap_rejected"),
    ("dropped_chars", "f_dropped_chars"),
    ("prefix_space_trim", "f_prefix_space_trim"),
)


# --------------------------------------------------------------------------
# Plug points
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorpusBundle:
    """Everything a shard needs about one corpus, and nothing more.

    Attributes:
# improved
        corpus_id: registry id.
        lang: language tag passed to `positions.compute_masks`.
        corpus_sha: the cache-key component identifying this exact record set --
            the manifest digest for a full split, or a digest that also binds the
            sampling parameters. See `build.corpus_key`.
        records: the gold sentences to score.
    """

    corpus_id: str
    lang: str
    corpus_sha: str
    records: tuple[Segmented, ...]


def default_adapter_factory(tokenizer_id: str, lang: str) -> TokenizerAdapter:
    """Return a LOADED adapter for one tokenizer in one language.

    ``lang`` is part of the key because it is part of the tokenizer's behaviour
    here: it selects the cluster grammar used for ``cluster_split``, and for the
    character baseline it selects the tokenizer itself. `tok.loader.get_adapter`
    is LRU-cached on ``(spec, lang)``, so within a worker process a tokenizer is
    materialised at most once per language rather than once per corpus -- with 16
    corpora over 4 languages that is the difference between 16 loads and 4.

    Raises:
        TokenizerUnavailable: the tokenizer layer is not installed, or the id is
            unknown. Deliberately this exception and not `ImportError`, so a
            half-built checkout or a gated repo degrades to "skip this
            tokenizer" rather than aborting the sweep.
    """
    try:
        loader_mod = importlib.import_module("unsegbench.tok.loader")
        from unsegbench.tok.registry import get_tokenizer_spec
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise TokenizerUnavailable(
            f"cannot build adapter for {tokenizer_id!r}: the tokenizer layer "
            f"(unsegbench.tok.loader) is not available: {exc}"
        ) from exc
    try:
        spec = get_tokenizer_spec(tokenizer_id)
    except KeyError as exc:
        raise TokenizerUnavailable(str(exc)) from exc
    adapter: Any = loader_mod.get_adapter(spec, lang)
    return adapter


def default_corpus_loader(
    corpus_id: str, split: str, sample: int | None, seed: int
) -> CorpusBundle:
    """Load a built corpus from the cache. See `build.load_corpus`."""
    from unsegbench import build

    manifest = build.load_manifest(corpus_id)
    recs = build.load_corpus(corpus_id, split=split, sample=sample, seed=seed)
    sha = build.subset_key(build.manifest_sha(manifest), split, sample, seed)
    return CorpusBundle(
        corpus_id=corpus_id, lang=manifest.lang, corpus_sha=sha, records=tuple(recs)
    )


def default_corpus_meta(
    corpus_id: str, split: str, sample: int | None, seed: int
) -> tuple[str, str]:
    """``(lang, corpus_sha)`` for a corpus, WITHOUT reading its records.

    The resume pre-check needs both -- the sha to locate the parquet and the lang
    to look up the right fingerprint -- and must not pay for decompressing a
    corpus it is about to decide it does not need.
    """
    from unsegbench import build

    manifest = build.load_manifest(corpus_id)
    return manifest.lang, build.subset_key(build.manifest_sha(manifest), split, sample, seed)


AdapterFactory = Callable[[str, str], TokenizerAdapter]
CorpusLoader = Callable[[str, str, int | None, int], CorpusBundle]
CorpusMetaFn = Callable[[str, str, int | None, int], tuple[str, str]]


# --------------------------------------------------------------------------
# Per-sentence scoring
# --------------------------------------------------------------------------


def _mask_universes(text: str, lang: str, masks: Sequence[str]) -> dict[str, frozenset[int]]:
    """The scored universes for this sentence, computed in ONE pass.

    `positions.compute_masks` shares the `legal_positions` result between
    ``legal`` and ``core``, so the marginal cost of the third mask is a single
    `trivial_positions` call over an already-cached string.
    """
    if len(masks) == 1:
        return {masks[0]: positions.compute_mask(text, lang, masks[0])}
    universes = positions.compute_masks(text, lang)
    return {m: universes[m] for m in masks}


def score_sentence(
    rec: Segmented,
    lang: str,
    adapter: TokenizerAdapter,
    masks: Sequence[str] = MASKS,
) -> dict[str, RowStats]:
    """Sufficient statistics for one sentence under every requested mask.

    The sentence is tokenized once and the mask universes are computed once; only
    the set intersections are repeated per mask.
# 
    ``crossing_tokens`` is computed against the mask-restricted gold, for the
    same reason `metrics.boundary_counts` intersects both sides: a token cannot
    be charged with crossing a boundary that this universe does not score.

    Args:
        rec: the gold sentence.
        lang: language tag for the cluster grammar.
        adapter: an already-loaded tokenizer adapter.
        masks: subset of `types.MASKS`.

    Returns:
        ``{mask: RowStats}``.
    """
    universes = _mask_universes(rec.text, lang, masks)
    gold = positions.gold_boundaries(rec)
    enc = adapter.encode(rec.text)
    pred = enc.boundaries
    intact = metrics.words_intact(rec.spans, enc.spans)
    flags = {col: int(enc.flags.get(key, 0)) for key, col in _FLAG_TO_COLUMN}

    out: dict[str, RowStats] = {}
    for mask_name in masks:
        universe = universes[mask_name]
        c = metrics.boundary_counts(gold, pred, universe)
        w_tp, w_pred, w_gold = metrics.word_counts(gold, pred, universe, rec.n)
        out[mask_name] = RowStats(
            sent_id=rec.id,
            n_chars=rec.n,
            n_mask=len(universe),
            n_gold_words=len(rec.spans),
            n_tokens=enc.n_tokens,
            n_tokens_accepted=len(enc.spans),
            b_tp=c.tp,
            b_fp=c.fp,
            b_fn=c.fn,
            b_tn=c.tn,
            w_tp=w_tp,
            w_pred=w_pred,
            w_gold=w_gold,
            w_intact=intact,
            crossing_tokens=metrics.crossing_tokens(enc.spans, gold & universe),
            **flags,
        )
    return out


# --------------------------------------------------------------------------
# Parquet I/O
# --------------------------------------------------------------------------


def _shard_path(fingerprint: str, corpus_sha: str, mask: str) -> Path:
    return cache.stats_dir(CODE_VERSION, fingerprint, corpus_sha) / f"{mask}.parquet"


def _metadata(**kv: Any) -> dict[bytes, bytes]:
    return {f"{_META_PREFIX}{k}".encode(): str(v).encode() for k, v in kv.items()}


def _write_shard(path: Path, rows: list[RowStats], meta: dict[bytes, bytes]) -> None:
    """Write one (tokenizer, corpus, mask) parquet atomically.

    Atomicity matters for resume: a shard file that exists must be complete, or a
    sweep killed mid-write would be "resumed" from a truncated table and the
    corruption would only surface as a slightly wrong number in a published
    table.
    """
    # One pass over the rows, not one per column: `as_tuple` walks 20 slots.
    tuples = [r.as_tuple() for r in rows]
    columns: list[pa.Array] = [pa.array([t[0] for t in tuples], type=pa.string())]
    columns.extend(
        pa.array([t[i] for t in tuples], type=pa.int64())
        for i in range(1, len(_COUNTER_COLUMNS) + 1)
    )
    table = pa.Table.from_arrays(columns, schema=STATS_SCHEMA.with_metadata(meta))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.part")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def _shard_is_valid(path: Path, meta: dict[bytes, bytes]) -> bool:
    """True if ``path`` already holds this exact shard.

    The directory already encodes the cache key, so this is belt-and-braces: it
    also catches a file left by a different split/sample/seed and any file whose
    footer will not parse (a truncated write from an older, less careful run).
    """
    if not path.exists():
        return False
    try:
        stored = pq.read_schema(path).metadata or {}
    except Exception:
        return False
    return all(stored.get(k) == v for k, v in meta.items())


def read_stats(path: Path | str) -> pd.DataFrame:
    """Read one shard parquet, restoring its identity columns from metadata.

    The identity of a shard -- which tokenizer, which corpus, which mask -- lives
    in the file's metadata and path, never in a per-row column, because repeating
    three strings across a few million rows would dominate the file size of a
    table whose entire point is to be small.
    """
    path = Path(path)
    table = pq.read_table(path)
    frame = table.to_pandas()
    stored = table.schema.metadata or {}

    def get(key: str, default: str = "") -> str:
        raw = stored.get(f"{_META_PREFIX}{key}".encode())
        return raw.decode() if raw is not None else default

    frame["tokenizer_id"] = get("tokenizer_id")
    frame["corpus_id"] = get("corpus_id")
#     frame["mask"] = get("mask", path.stem)
    frame["tokenizer_fingerprint"] = get("tokenizer_fingerprint")
    frame["corpus_sha"] = get("corpus_sha", path.parent.name)
    frame["code_version"] = get("code_version", path.parent.parent.parent.name)
    return frame


# --------------------------------------------------------------------------
# Fingerprint memo -- what makes "resume" actually free
# --------------------------------------------------------------------------


def _fingerprint_store() -> Path:
    return cache.cache_root() / "fingerprints.json"


def _load_fingerprints() -> dict[str, str]:
    path = _fingerprint_store()
    if not path.exists():
        return {}
    try:
        data = orjson.loads(path.read_bytes())
    except Exception:
        return {}
    return dict(data.get(ADAPTER_VERSION, {}))


def _save_fingerprints(new: dict[str, str]) -> None:
    if not new:
        return
    path = _fingerprint_store()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache.lock_path("fingerprints"))):
        data: dict[str, Any] = {}
        if path.exists():
            try:
                data = orjson.loads(path.read_bytes())
            except Exception:
                data = {}
        data.setdefault(ADAPTER_VERSION, {}).update(new)
        tmp = path.with_suffix(".json.part")
        tmp.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        tmp.replace(path)


# --------------------------------------------------------------------------
# Shards
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShardOutcome:
    """What happened to one (tokenizer, corpus, mask) unit of work."""

    tokenizer_id: str
    corpus_id: str
    mask: str
    path: Path | None
    status: str  # "computed" | "resumed" | "skipped"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class _ShardJob:
    """One tokenizer's entire workload. Must stay picklable."""

    tokenizer_id: str
    corpus_ids: tuple[str, ...]
    #: corpus_id -> (lang, corpus_sha)
    corpus_meta: dict[str, tuple[str, str]]
    masks: tuple[str, ...]
    split: str
    sample: int | None
    seed: int
    resume: bool
    strict: bool
    adapter_factory: AdapterFactory | None
    corpus_loader: CorpusLoader | None


@dataclass(slots=True)
class _ShardReport:
    tokenizer_id: str
    #: ``(tokenizer_id, lang)`` -> fingerprint. Keyed on lang because a baseline's
    #: fingerprint legitimately includes it -- the character baseline in Thai is a
    #: different tokenizer from the character baseline in Chinese.
    fingerprints: dict[tuple[str, str], str] = field(default_factory=dict)
    outcomes: list[ShardOutcome] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RunResult:
    """The sweep's result.

    Attributes:
        frame: the aggregated metric table, one row per (tokenizer, corpus, mask).
# improved
        outcomes: per-shard provenance -- what was computed, resumed or skipped.
        fingerprints: ``(tokenizer_id, lang)`` -> content fingerprint, for
            publication. A leaderboard without fingerprints is not reproducible;
            roughly a third of the registry is byte-identical to another entry.
        out_dir: where the aggregate was written, if anywhere.
    """

    frame: pd.DataFrame
    outcomes: list[ShardOutcome]
    fingerprints: dict[tuple[str, str], str]
    out_dir: Path | None = None

    @property
    def paths(self) -> list[Path]:
        return [o.path for o in self.outcomes if o.path is not None]


def _shard_meta(job: _ShardJob, fingerprint: str, corpus_id: str, mask: str) -> dict[bytes, bytes]:
    return _metadata(
        code_version=CODE_VERSION,
        tokenizer_id=job.tokenizer_id,
        tokenizer_fingerprint=fingerprint,
        corpus_id=corpus_id,
        corpus_sha=job.corpus_meta[corpus_id][1],
        mask=mask,
        split=job.split,
        sample=job.sample,
        seed=job.seed,
#     )


# def _execute_shard(job: _ShardJob, on_corpus: Callable[[], None] | None = None) -> _ShardReport:
    """Run one tokenizer over every corpus. This is the worker body.

    Corpora are visited grouped by language, so the adapter cache in
    `tok.loader.get_adapter` is hit for every corpus after the first in each
    language. That is the entire reason sharding is tokenizer-major: a
    corpus-major sweep would pay the 10-20s load once per (tokenizer, corpus)
    pair instead of once per (tokenizer, language).
    """
    # Belt and braces: the parent sets this before forking, but a spawned child
    # rebuilds its environment from scratch on macOS/Windows.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    factory = job.adapter_factory or default_adapter_factory
    loader = job.corpus_loader or default_corpus_loader
    report = _ShardReport(tokenizer_id=job.tokenizer_id)

    # Language-major within the shard: maximises adapter-cache hits.
    ordered = sorted(job.corpus_ids, key=lambda c: (job.corpus_meta[c][0], c))

    for corpus_id in ordered:
        lang, corpus_sha = job.corpus_meta[corpus_id]
        try:
            adapter = factory(job.tokenizer_id, lang)
        except TokenizerUnavailable as exc:
            if job.strict:
                raise
            report.outcomes.append(
                ShardOutcome(job.tokenizer_id, corpus_id, "", None, "skipped", str(exc))
            )
            if on_corpus is not None:
                on_corpus()
            continue

        fingerprint = adapter.fingerprint()
        report.fingerprints[(job.tokenizer_id, lang)] = fingerprint

        metas = {m: _shard_meta(job, fingerprint, corpus_id, m) for m in job.masks}
        paths = {m: _shard_path(fingerprint, corpus_sha, m) for m in job.masks}

        if job.resume and all(_shard_is_valid(paths[m], metas[m]) for m in job.masks):
            report.outcomes.extend(
                ShardOutcome(job.tokenizer_id, corpus_id, m, paths[m], "resumed") for m in job.masks
            )
#             if on_corpus is not None:
                on_corpus()
            continue

        bundle = loader(corpus_id, job.split, job.sample, job.seed)
        rows: dict[str, list[RowStats]] = {m: [] for m in job.masks}
        for rec in bundle.records:
            per_mask = score_sentence(rec, bundle.lang, adapter, job.masks)
            for m in job.masks:
                rows[m].append(per_mask[m])

        for m in job.masks:
            _write_shard(paths[m], rows[m], metas[m])
            report.outcomes.append(
                ShardOutcome(job.tokenizer_id, corpus_id, m, paths[m], "computed")
            )
        if on_corpus is not None:
            on_corpus()

    return report


def _default_jobs() -> int:
    """``min(10, cpu_count() - 2)``.

    Capped at 10 because past that the bottleneck is memory bandwidth for the
    tokenizer vocabularies, and reserving two cores keeps the machine usable and
    leaves headroom for the parent's progress rendering.
    """
    return max(1, min(10, (os.cpu_count() or 4) - 2))


def run(
    tokenizer_ids: Sequence[str],
    corpus_ids: Sequence[str],
    masks: Sequence[str] = MASKS,
    *,
    sample: int | None = None,
    seed: int = 0,
    split: str = "test",
    jobs: int | None = None,
    resume: bool = True,
    out: Path | str | None = None,
    strict: bool = False,
    adapter_factory: AdapterFactory | None = None,
    corpus_loader: CorpusLoader | None = None,
    corpus_meta_fn: CorpusMetaFn | None = None,
    progress: bool = True,
    console: Console | None = None,
) -> RunResult:
    """Score every (tokenizer, corpus, mask) shard and return the aggregate.

    Args:
        tokenizer_ids: already-resolved tokenizer ids (the CLI expands
            ``@``-selectors before calling).
        corpus_ids: already-resolved corpus ids.
        masks: subset of `types.MASKS`. All three cost ~1.3x one.
        sample: sentences per corpus, or ``None`` for the full split.
        seed: sampling seed. Part of the shard cache key.
        split: ``"test"`` or ``"train"``.
        jobs: worker processes. Defaults to ``min(10, cpu_count() - 2)``. ``1``
            runs in-process, which is what tests and debugging want.
        resume: skip shards already on disk under the same cache key. A shard
            whose tokenizer fingerprint is already memoised is skipped without
            loading the tokenizer at all.
        out: directory to write the aggregate table to. ``None`` writes nothing.
        strict: make a `TokenizerUnavailable` fatal instead of skipping it.
        adapter_factory: override for tests; must be picklable if ``jobs > 1``.
        corpus_loader: override for tests; must be picklable if ``jobs > 1``.
        corpus_meta_fn: cheap ``(lang, corpus_sha)`` lookup used for the
            pre-load resume check. Defaults to reading manifests; when a custom
            ``corpus_loader`` is supplied without one, the loader itself is used.
        progress: draw a rich progress bar over shards.
        console: rich console for the skip log.

    Returns:
        A `RunResult`.

    Raises:
        ValueError: an unknown mask.
        TokenizerUnavailable: only under ``strict``.
    """
    # MUST happen before the pool forks. The Rust tokenizers thread pool would
    # otherwise oversubscribe every core by the number of workers, and warn
    # loudly on every fork while doing it.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    bad = [m for m in masks if m not in MASKS]
    if bad:
        raise ValueError(f"unknown mask(s) {bad}; expected a subset of {MASKS}")
    masks = tuple(m for m in MASKS if m in set(masks))  # canonical order
#     tokenizer_ids = tuple(dict.fromkeys(tokenizer_ids))
    corpus_ids = tuple(dict.fromkeys(corpus_ids))
    console = console or Console(stderr=True)

    if corpus_meta_fn is None:
        if corpus_loader is None:
            corpus_meta_fn = default_corpus_meta
        else:
            _loader = corpus_loader

            def corpus_meta_fn(cid: str, sp: str, sm: int | None, sd: int) -> tuple[str, str]:
                bundle = _loader(cid, sp, sm, sd)
                return bundle.lang, bundle.corpus_sha

    corpus_meta = {cid: corpus_meta_fn(cid, split, sample, seed) for cid in corpus_ids}
    known = _load_fingerprints() if resume else {}

    jobs_list: list[_ShardJob] = []
    outcomes: list[ShardOutcome] = []
    fingerprints: dict[tuple[str, str], str] = {}

    for tok_id in tokenizer_ids:
        job = _ShardJob(
            tokenizer_id=tok_id,
            corpus_ids=corpus_ids,
            corpus_meta=corpus_meta,
            masks=masks,
            split=split,
            sample=sample,
            seed=seed,
            resume=resume,
            strict=strict,
            adapter_factory=adapter_factory,
            corpus_loader=corpus_loader,
        )
        if resume:
            cached = _fully_cached(job, known)
            if cached is not None:
                # The whole point of memoising fingerprints: proving this shard
                # is done costs a stat() per file instead of a 10-20s model load.
                shard_outcomes, shard_fps = cached
                outcomes.extend(shard_outcomes)
                fingerprints.update(shard_fps)
                continue
        jobs_list.append(job)

    n_units = len(jobs_list) * max(1, len(corpus_ids))
    with Progress(
        TextColumn("[bold blue]sweep"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[note]}"),
        console=console,
        disable=not progress or n_units == 0,
        transient=True,
    ) as bar:
        task = bar.add_task("shards", total=n_units, note="")

        def advance(note: str = "") -> None:
            bar.update(task, advance=1, note=note)

        reports = _dispatch(jobs_list, jobs, advance, console=console)

    for report in reports:
        outcomes.extend(report.outcomes)
        fingerprints.update(report.fingerprints)
        for o in report.outcomes:
            if o.status == "skipped":
                console.print(f"[yellow]skipped[/] {o.tokenizer_id}: {o.reason}")

    _save_fingerprints({f"{tok}|{lang}": fp for (tok, lang), fp in fingerprints.items() if fp})

    paths = [o.path for o in outcomes if o.path is not None]
    frame = aggregate(paths) if paths else pd.DataFrame(columns=list(GROUP_COLUMNS))

    out_dir = Path(out) if out is not None else None
    if out_dir is not None and not frame.empty:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Aggregates only. Derived text from non-redistributable corpora never
        # leaves the cache (CONTRACTS.md sec.6).
        frame.to_csv(out_dir / "results.csv", index=False)
        frame.to_parquet(out_dir / "results.parquet", index=False)

    return RunResult(frame=frame, outcomes=outcomes, fingerprints=fingerprints, out_dir=out_dir)


def _fully_cached(
    job: _ShardJob, known: dict[str, str]
) -> tuple[list[ShardOutcome], dict[tuple[str, str], str]] | None:
    """Prove a whole shard is on disk WITHOUT loading its tokenizer.

    This is what makes resume free. The fingerprint is the cache key and only the
    adapter can compute it, so a naive resume would still pay 25 model loads to
    discover it has nothing to do. Memoised fingerprints turn that into a stat()
    per file.

    Returns:
        ``(outcomes, fingerprints)`` if every mask of every corpus is present and
        matches, else ``None``.
    """
    out: list[ShardOutcome] = []
    fps: dict[tuple[str, str], str] = {}
    for corpus_id in job.corpus_ids:
        lang, corpus_sha = job.corpus_meta[corpus_id]
        fingerprint = known.get(f"{job.tokenizer_id}|{lang}")
        if not fingerprint:
            return None
        for mask in job.masks:
            path = _shard_path(fingerprint, corpus_sha, mask)
            if not _shard_is_valid(path, _shard_meta(job, fingerprint, corpus_id, mask)):
                return None
            out.append(ShardOutcome(job.tokenizer_id, corpus_id, mask, path, "resumed"))
        fps[(job.tokenizer_id, lang)] = fingerprint
    return out, fps


def _dispatch(
    jobs_list: list[_ShardJob],
    jobs: int | None,
    advance: Callable[[str], None],
    *,
    console: Console,
) -> list[_ShardReport]:
    """Run shards, in process or across a pool."""
    if not jobs_list:
        return []
    n_workers = jobs if jobs is not None else _default_jobs()
    n_workers = max(1, min(n_workers, len(jobs_list)))

    if n_workers > 1 and not _picklable(jobs_list[0]):
        console.print(
            "[yellow]note[/] custom adapter/corpus callables are not picklable; "
            "falling back to jobs=1"
        )
        n_workers = 1

    if n_workers == 1:
        reports = []
        for job in jobs_list:
            reports.append(_execute_shard(job, on_corpus=lambda: advance("")))
        return reports

    reports = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_execute_shard, job): job for job in jobs_list}
        for fut in as_completed(futures):
            job = futures[fut]
            reports.append(fut.result())
            for _ in job.corpus_ids:
                advance(job.tokenizer_id)
    return reports


def _picklable(obj: Any) -> bool:
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------
# Aggregation -- the function the report layer calls. Touches no tokenizer.
# --------------------------------------------------------------------------


def _collect_paths(source: Path | str) -> list[Path]:
    path = Path(source)
    if path.is_dir():
        return sorted(path.rglob("*.parquet"))
    return [path]


# def aggregate(source: pd.DataFrame | Path | str | Iterable[Path | str]) -> pd.DataFrame:
    """Pool per-sentence counters and turn them into metrics.

    This is the whole report layer's entry point, and it is pure arithmetic over
    integers: no tokenizer is constructed, no corpus is read, nothing is
    downloaded. That is what makes the report reproducible on a machine with no
    network and no licences.

    Pooling is a groupby-sum, which is exactly why `metrics.core` counts and
    never averages: sums are associative, so aggregating per-shard and then
    combining gives bit-identical results to aggregating the whole corpus at
    once. `tests/test_runner.py` pins that monoid law.

    Args:
        source: a directory of shard parquets, a single parquet path, an iterable
            of paths, or an already-loaded per-sentence frame.

    Returns:
        One row per ``(tokenizer_id, corpus_id, mask)``: the pooled integer
        counters plus every field of `metrics.MetricRow`.
    """
    if isinstance(source, pd.DataFrame):
        frame = source
    else:
        paths: list[Path] = []
        if isinstance(source, (str, Path)):
            paths = _collect_paths(source)
        else:
            for item in source:
                paths.extend(_collect_paths(item))
        if not paths:
            return pd.DataFrame(columns=list(GROUP_COLUMNS))
        frame = pd.concat([read_stats(p) for p in paths], ignore_index=True)

    if frame.empty:
        return pd.DataFrame(columns=list(GROUP_COLUMNS))

    frame = frame.copy()
    for col in GROUP_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""

    keys = list(GROUP_COLUMNS)
    grouped = frame.groupby(keys, sort=True, dropna=False)
    pooled = grouped[list(_COUNTER_COLUMNS)].sum().astype("int64")
    pooled["n_sents"] = grouped.size().astype("int64")
    pooled = pooled.reset_index()
# 
    rows = [
        metrics.compute_row(
            metrics.Counts(tp=int(r.b_tp), fp=int(r.b_fp), fn=int(r.b_fn), tn=int(r.b_tn)),
            w_tp=int(r.w_tp),
            w_pred=int(r.w_pred),
            w_gold=int(r.w_gold),
            w_intact=int(r.w_intact),
            n_tokens=int(r.n_tokens),
            n_chars=int(r.n_chars),
            n_gold_words=int(r.n_gold_words),
            crossing=int(r.crossing_tokens),
        )
        for r in pooled.itertuples(index=False)
    ]
    metric_frame = pd.DataFrame(
        [vars(m) if not hasattr(m, "__slots__") else _asdict(m) for m in rows]
    )
    result = pd.concat([pooled.reset_index(drop=True), metric_frame], axis=1)
    ordered = [*keys, "n_sents", *_COUNTER_COLUMNS, *metric_frame.columns]
    return result[ordered]


def _asdict(row: metrics.MetricRow) -> dict[str, float]:
    return {f: getattr(row, f) for f in metrics.MetricRow.__slots__}

# Enhanced

# Updated

# Updated

# Updated

# Updated

# Updated

# Refined

# Updated

# Refined

# Updated
