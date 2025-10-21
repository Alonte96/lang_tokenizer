"""Canonical corpus builder: raw bytes in, validated `Segmented` JSONL out.

WHY this is a module and not a step inside each loader:

* **Validation must be uniform and fail-closed.** Every loader is written by a
  different person against a different upstream format. If each one validated
  its own output, "the loader silently dropped content" would be caught in eight
  places with eight severities. Here it is caught in exactly one, and a
  violation is fatal -- a corpus that half-parsed is worse than one that failed.
* **The canonical form is the cache key.** Everything downstream keys on the
  manifest digest, so the manifest has to be produced by one function that sees
  the whole corpus, not assembled incrementally by loaders.
* **Builds are concurrent.** A sweep may start eight workers that all want
  ``sighan_pku``. A per-corpus `filelock` plus double-checked existence makes
  the build idempotent and collision-free, and a second caller pays only the
  cost of reading a manifest.

Sampling lives here too, and is deliberately NOT reservoir sampling. A
reservoir's output depends on input order, so re-running after a loader change
that reorders records would silently change which sentences the leaderboard was
computed on. Instead each sentence gets a keyed BLAKE2b hash of its ``sent_id``
and we take the lowest hashes within each sentence-length decile. That is stable
under reordering, reproducible across machines, and keeps the length
distribution of the sample close to the corpus -- which matters because short
sentences have systematically fewer scorable positions.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock

from unsegbench import positions, types
from unsegbench.corpora.base import CorpusSpec
from unsegbench.errors import BuildValidationError
from unsegbench.fetch import cache
from unsegbench.types import CorpusManifest, Segmented

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = [
    "MANIFEST_NAME",
    "build_corpus",
    "corpus_key",
    "fetch_artifacts",
    "is_built",
    "load_corpus",
    "load_manifest",
    "manifest_sha",
    "select_sample",
]

#: Filename of the per-corpus manifest inside ``canonical/<corpus_id>/<version>/``.
MANIFEST_NAME = "manifest.json"

#: Number of strata used by `select_sample`. Deciles: fine enough that the
#: sample tracks the length distribution, coarse enough that even a 200-sentence
#: corpus has ~20 candidates per stratum.
_N_STRATA = 10

_ZWSP = "​"


# --------------------------------------------------------------------------
# Digests
# --------------------------------------------------------------------------


def manifest_sha(manifest: CorpusManifest) -> str:
    """Content digest of a manifest.

    Computed from the manifest's own canonical JSON rather than from the file on
    disk, so it does not change if the file is ever rewritten with different
    whitespace. This is the ``corpus_manifest_sha`` component of the
    sufficient-statistic cache key (CONTRACTS.md sec.5).

    Args:
        manifest: the corpus manifest.

    Returns:
        Lowercase hex sha256 of the manifest's canonical serialisation.
    """
    return hashlib.sha256(manifest.to_json()).hexdigest()


def corpus_key(
    corpus_id: str, split: str = "test", sample: int | None = None, seed: int = 0
) -> str:
    """Cache-key component identifying the exact record set a shard will score.

    For the canonical case -- the full ``test`` split -- this is exactly the
    manifest digest, as CONTRACTS.md sec.5 specifies. Any other case additionally
    binds the split and the sampling parameters, because a run over 500 sampled
    sentences and a run over the whole corpus are different data and must never
    share a parquet.

    Args:
        corpus_id: registry id of a built corpus.
        split: ``"test"`` or ``"train"``.
        sample: number of sentences to sample, or ``None`` for all.
        seed: sampling seed.

    Returns:
        A filesystem-safe key string.
    """
    sha = manifest_sha(load_manifest(corpus_id))
    return subset_key(sha, split, sample, seed)


def subset_key(sha: str, split: str, sample: int | None, seed: int) -> str:
    """`corpus_key` without the manifest lookup. See `corpus_key`."""
    if sample is None and split == "test":
        return sha
    suffix = hashlib.blake2b(f"{split}|{sample}|{seed}".encode(), digest_size=6).hexdigest()
    return f"{sha[:32]}-{suffix}"


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _Accumulator:
    """Streaming corpus statistics, so a build never holds a whole corpus."""

    n_sents: int = 0
    n_words: int = 0
    n_chars: int = 0
    gold_total: int = 0
    gold_illegal: int = 0
    zwsp_present: bool = False

    def update(self, rec: Segmented, lang: str) -> None:
        self.n_sents += 1
        self.n_words += len(rec.spans)
        self.n_chars += rec.n
        if not self.zwsp_present and _ZWSP in rec.text:
            self.zwsp_present = True
        gold = positions.gold_boundaries(rec)
        if gold:
            legal = positions.legal_positions(rec.text, lang)
            self.gold_total += len(gold)
            self.gold_illegal += len(gold - legal)

    @property
    def gold_illegal_rate(self) -> float:
        return (self.gold_illegal / self.gold_total) if self.gold_total else 0.0


def _validated(
    records: Iterable[Segmented], spec: CorpusSpec, acc: _Accumulator
) -> Iterator[Segmented]:
    """Validate and accumulate as records stream past on their way to disk.

    Fail-closed: `types.validate_record` raises `BuildValidationError` on the
    first bad record, before the partial output is moved into place (write_jsonl
    writes to ``.part`` and only renames on success), so a failed build leaves no
    corpus that later looks cached.
    """
    for rec in records:
# improved
        types.validate_record(rec, spec.gap_charset)
        acc.update(rec, spec.lang)
        yield rec


def _license_key(corpus_id: str) -> str | None:
    """Which `cache.LICENSE_GATED` group, if any, this corpus belongs to."""
    for key in cache.LICENSE_GATED:
        if corpus_id == key or corpus_id.startswith(f"{key}_"):
            return key
    return None


def fetch_artifacts(spec: CorpusSpec, raw: Path, *, force: bool, progress: bool) -> None:
    """Download (and optionally unpack) everything `spec.parse` will need."""
    from unsegbench.fetch.http import download  # local: keeps httpx off the import path

    raw.mkdir(parents=True, exist_ok=True)
    extracted = raw / "extracted"
    for art in spec.artifacts:
        dest = raw / art.name
        download(art.url, dest, sha256=art.sha256, force=force, progress=progress)
        if not art.extract:
            continue
        marker = extracted / f".{art.name}.unpacked"
        if marker.exists() and not force:
            continue
        extracted.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(dest), str(extracted))
        marker.write_text("ok\n", encoding="utf-8")


def build_corpus(spec: CorpusSpec, force: bool = False, *, progress: bool = True) -> CorpusManifest:
    """Fetch, parse, validate and freeze one corpus into the cache.

    Idempotent: an already-built corpus is returned from its manifest without
    re-parsing. Concurrency-safe: the whole body runs under a per-corpus
    `filelock`, with a second existence check inside the lock, so eight parallel
    workers that all want the same corpus do the work once.

    Fails closed. Every record goes through `types.validate_record` with the
    spec's ``gap_charset``; an uncovered codepoint outside that set means the
    loader lost content and the build raises rather than writing a corpus that
    would quietly under-report gold words forever after.

    Args:
        spec: the corpus loader.
        force: re-download and rebuild even if a manifest already exists.
        progress: draw download progress bars.

    Returns:
        The written `CorpusManifest`.

    Raises:
        BuildValidationError: a record violates the IR contract, or every split
            was empty.
        LicenseNotAccepted: a non-redistributable corpus whose licence has not
            been acknowledged.
    """
    canon = cache.canonical_dir(spec.corpus_id, spec.version)
    manifest_path = canon / MANIFEST_NAME
    if manifest_path.exists() and not force:
        return CorpusManifest.from_json(manifest_path.read_bytes())

    key = _license_key(spec.corpus_id)
    if key is not None:
        cache.require_license(key)

    lock = FileLock(str(cache.lock_path(f"build-{spec.corpus_id}")))
    with lock:
        # Double-checked: another worker may have built it while we waited.
        if manifest_path.exists() and not force:
            return CorpusManifest.from_json(manifest_path.read_bytes())

        raw = cache.raw_dir(spec.corpus_id, spec.version)
        fetch_artifacts(spec, raw, force=force, progress=progress)

        canon.mkdir(parents=True, exist_ok=True)
        acc = _Accumulator()
        split_shas: dict[str, str] = {}
        for split in spec.splits():
            out = canon / f"{split}.jsonl.gz"
            before = acc.n_sents
            written = types.write_jsonl(out, _validated(spec.parse(raw, split), spec, acc))
            if written == 0:
                # A corpus with no such split yields nothing rather than raising.
                out.unlink(missing_ok=True)
                acc.n_sents = before
                continue
            split_shas[split] = _sha256_file(out)

        if acc.n_sents == 0:
            raise BuildValidationError(
                f"{spec.corpus_id}: every split parsed to zero records -- "
                f"check the artifacts under {raw}"
            )

        manifest = spec.manifest(
            splits=split_shas,
            n_sents=acc.n_sents,
            n_words=acc.n_words,
            n_chars=acc.n_chars,
            gold_illegal_rate=acc.gold_illegal_rate,
            zwsp_present=acc.zwsp_present,
        )
        tmp = manifest_path.with_suffix(".json.part")
        tmp.write_bytes(manifest.to_json())
        tmp.replace(manifest_path)
        return manifest


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------


def _canonical_root(corpus_id: str) -> Path:
    return cache.cache_root() / "canonical" / corpus_id


def load_manifest(corpus_id: str, version: str | None = None) -> CorpusManifest:
    """Read a built corpus's manifest.

    The version is normally taken from the registry. When the loader module is
    not importable -- a partially built checkout, or a corpus built by an older
    revision -- we fall back to scanning the cache, so that reading results never
    depends on the loader that produced them still existing.

    Args:
        corpus_id: registry id.
        version: explicit corpus version; inferred if omitted.

    Raises:
        BuildValidationError: the corpus has not been built.
    """
    if version is None:
        try:
            from unsegbench.corpora.registry import get_corpus

            version = get_corpus(corpus_id).version
        except (KeyError, ModuleNotFoundError, ImportError):
            version = None
    candidates: list[Path] = []
    if version is not None:
        candidates.append(cache.canonical_dir(corpus_id, version) / MANIFEST_NAME)
    root = _canonical_root(corpus_id)
    if root.is_dir():
        candidates.extend(sorted(root.glob(f"*/{MANIFEST_NAME}")))
    for path in candidates:
        if path.exists():
            return CorpusManifest.from_json(path.read_bytes())
    raise BuildValidationError(
        f"corpus {corpus_id!r} is not built (no {MANIFEST_NAME} under {root}); "
        f"run `unsegbench build {corpus_id}` first"
    )


def is_built(corpus_id: str, version: str | None = None) -> bool:
    """True if a manifest for this corpus exists in the cache."""
    try:
        load_manifest(corpus_id, version)
    except BuildValidationError:
        return False
    return True


def load_corpus(
    corpus_id: str, split: str = "test", sample: int | None = None, seed: int = 0
) -> list[Segmented]:
    """Read a built corpus's split, optionally sampled.

    Args:
        corpus_id: registry id of a built corpus.
        split: ``"test"`` or ``"train"``.
        sample: number of sentences to keep, or ``None`` for all. Selection is
            deterministic and order-stable -- see `select_sample`.
        seed: sampling seed.

    Returns:
        The records, in corpus order when unsampled and in ``sent_id`` order
        when sampled.

    Raises:
        BuildValidationError: the corpus is not built, or has no such split.
    """
    manifest = load_manifest(corpus_id)
    path = cache.canonical_dir(corpus_id, manifest.version) / f"{split}.jsonl.gz"
    if not path.exists():
        have = ", ".join(sorted(manifest.splits)) or "(none)"
        raise BuildValidationError(f"{corpus_id}: no split {split!r}; built splits: {have}")
    return select_sample(list(types.read_jsonl(path)), sample, seed)


# --------------------------------------------------------------------------
# Deterministic, order-stable, length-stratified sampling
# --------------------------------------------------------------------------


def _hash_u(sent_id: str, seed: int) -> int:
    """Keyed 64-bit hash of a sentence id. The sample's only source of randomness."""
    digest = hashlib.blake2b(
        sent_id.encode("utf-8"), key=str(seed).encode("ascii"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _allocate(sizes: list[int], total: int) -> list[int]:
    """Largest-remainder apportionment of ``total`` across strata of ``sizes``.

    Deterministic: remainders are compared as exact integer pairs and ties break
    on stratum index, so the allocation never depends on float rounding or on
    dict iteration order.
    """
    n = sum(sizes)
    if n == 0:
        return [0] * len(sizes)
    base = [min(s, (total * s) // n) for s in sizes]
    remaining = total - sum(base)
    if remaining <= 0:
        return base
    # Remainder of (total * size / n), kept integral: total*size - base*n.
    order = sorted(
        range(len(sizes)),
        key=lambda i: (-(total * sizes[i] - base[i] * n), i),
    )
    for i in order:
        if remaining == 0:
            break
        if base[i] < sizes[i]:
            base[i] += 1
            remaining -= 1
    # A stratum may have been capped; spill the rest anywhere with room.
    while remaining > 0:
        progressed = False
        for i in range(len(sizes)):
            if remaining == 0:
                break
            if base[i] < sizes[i]:
                base[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return base


def select_sample(records: list[Segmented], sample: int | None, seed: int = 0) -> list[Segmented]:
    """Deterministic length-stratified subsample.

    Explicitly **not** reservoir sampling. A reservoir's output is a function of
    input order, so a loader change that merely reordered records would silently
    change which sentences the published numbers were computed on. Here:

    1. Records are ranked by ``(length, sent_id)`` -- a total order independent
       of input order -- and cut into `_N_STRATA` equal-count strata. Stratifying
       on length matters because short sentences have systematically fewer
       scorable positions, so an unstratified sample shifts every density-derived
       metric.
    2. Each stratum's quota is apportioned by largest remainder.
    3. Within a stratum, the records with the smallest ``blake2b(sent_id,
       key=seed)`` win.

    The result depends only on the *set* of records and on ``(sample, seed)``, so
    it is reproducible across runs, machines and corpus orderings.

    Args:
        records: the full split.
        sample: how many to keep; ``None`` returns ``records`` unchanged.
        seed: BLAKE2b key.

    Returns:
        The selected records in ``sent_id`` order.
    """
    if sample is None:
        return records
    if sample <= 0:
        return []
    n = len(records)
    if sample >= n:
        return sorted(records, key=lambda r: r.id)

    ranked = sorted(records, key=lambda r: (r.n, r.id))
    strata: list[list[Segmented]] = [[] for _ in range(_N_STRATA)]
    for rank, rec in enumerate(ranked):
        strata[min(_N_STRATA - 1, rank * _N_STRATA // n)].append(rec)

    quotas = _allocate([len(s) for s in strata], sample)
    chosen: list[Segmented] = []
    for stratum, quota in zip(strata, quotas, strict=True):
        if quota <= 0:
            continue
        stratum.sort(key=lambda r: (_hash_u(r.id, seed), r.id))
        chosen.extend(stratum[:quota])
    return sorted(chosen, key=lambda r: r.id)

# Enhanced

# Enhanced

# Enhanced

# Enhanced

# Updated

# Refined

# Refined

# Refined

# Updated
