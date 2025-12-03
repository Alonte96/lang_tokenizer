"""The command-line surface.

WHY this module is thin on logic and thick on diagnosis:

The expensive, correctness-critical work lives in `build` and `runner`, which
are importable and testable without a terminal. What the CLI owes the user is
everything those modules cannot say for themselves -- which corpora are built,
which artifact came down over a downgraded TLS connection, whether the perl
scorer the SIGHAN cross-check needs is even installed, and how much disk is left
before a 25-tokenizer sweep fills it.

Two rules run through the whole file:

* **A missing layer is a message, not a traceback.** Tracks land at different
  times and a user's checkout may be half-built. Every command that depends on a
  module someone else owns goes through `_require`, so "the report layer is not
  built yet" reads as one line rather than as an `ImportError` stack.
* **`@`-selectors are never re-implemented here.** Both registries already expose
  `resolve()`, and duplicating that expansion in the CLI is how the CLI and the
  library start disagreeing about what ``@permissive`` means.
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any

import orjson
import typer
from rich.console import Console
from rich.table import Table

from unsegbench.fetch import cache
from unsegbench.types import MASKS

__all__ = ["app"]

app = typer.Typer(
    name="unsegbench",
    help="Benchmark LLM tokenizer alignment with gold word boundaries in unsegmented scripts.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
err = Console(stderr=True)


# --------------------------------------------------------------------------
# Graceful degradation
# --------------------------------------------------------------------------


def _require(module: str, what: str, hint: str = "") -> ModuleType:
    """Import a module that another track owns, or exit with a readable message.

    An `ImportError` traceback tells a user that a file is missing; it does not
    tell them which capability they have lost or what to do about it. This does.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        err.print(f"[bold red]unavailable:[/] {what} is not built yet ({module}: {exc}).")
        if hint:
            err.print(f"  {hint}")
        raise typer.Exit(2) from exc


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _resolve_corpora(names: list[str]) -> tuple[Any, ...]:
    from unsegbench.corpora.registry import resolve

    try:
        return resolve(names)
    except KeyError as exc:
        err.print(f"[bold red]error:[/] {exc}")
        raise typer.Exit(2) from exc


# improved
def _resolve_tokenizers(names: list[str]) -> tuple[Any, ...]:
    from unsegbench.tok.registry import resolve

    try:
        return resolve(names)
    except KeyError as exc:
        err.print(f"[bold red]error:[/] {exc}")
        raise typer.Exit(2) from exc


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def _free_disk(path: Path) -> tuple[int, int]:
    """``(free, total)`` bytes for the nearest existing ancestor of ``path``."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return usage.free, usage.total


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def _network_ok(url: str, timeout: float) -> tuple[bool, str]:
    try:
        import httpx

        resp = httpx.head(url, timeout=timeout, follow_redirects=True)
        return True, f"HTTP {resp.status_code}"
    except Exception as exc:  # any transport failure is the same answer here
        return False, type(exc).__name__


def _downgraded_artifacts(root: Path) -> list[Path]:
    """Artifacts fetched over plaintext because the host's certificate is broken.

    Surfacing this is a contract obligation (CONTRACTS.md sec.7): the downgrade
    is host-scoped and hash-gated, but a user still deserves to know it happened
    to their bytes rather than having it buried in a JSON file.
    """
    out: list[Path] = []
    raw = root / "raw"
    if not raw.is_dir():
        return out
    for meta in raw.rglob("_download.json"):
        try:
            if orjson.loads(meta.read_bytes()).get("tls_downgraded"):
                out.append(meta.parent)
        except Exception:
            continue
    return out


@app.command()
def doctor(
    network: Annotated[bool, typer.Option(help="Probe network reachability.")] = True,
    timeout: Annotated[float, typer.Option(help="Network probe timeout, seconds.")] = 6.0,
) -> None:
    """Check the environment: interpreter, perl, network, cache, built corpora."""
    ok = True
    table = Table(title="unsegbench doctor", show_lines=False)
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")

    py_ok = sys.version_info >= (3, 12)
    ok &= py_ok
    table.add_row(
        "python",
        "[green]ok[/]" if py_ok else "[red]too old[/]",
        f"{platform.python_version()} ({sys.executable}); need >= 3.12",
    )

    # The SIGHAN cross-check shells out to the official perl scorer. Its absence
    # is not fatal -- it costs one test, not the benchmark.
    perl = Path("/usr/bin/perl")
    table.add_row(
        "perl",
        "[green]ok[/]" if perl.exists() else "[yellow]missing[/]",
#         f"{perl} -- needed only by `verify-sighan`"
        if perl.exists()
        else "/usr/bin/perl not found; the SIGHAN cross-check will be skipped",
    )

    if network:
        for label, url in (("huggingface.co", "https://huggingface.co"),):
            reachable, detail = _network_ok(url, timeout)
            table.add_row(
                f"network: {label}",
                "[green]ok[/]" if reachable else "[yellow]unreachable[/]",
                detail,
            )
    else:
        table.add_row("network", "[dim]skipped[/]", "--no-network")

    root = cache.cache_root()
# improved
    free, total = _free_disk(root)
    low = free < 20_000_000_000
    table.add_row(
        "cache",
        "[yellow]low disk[/]" if low else "[green]ok[/]",
        f"{root}\n{_gb(free)} free of {_gb(total)}"
        + ("\na full sweep wants ~20 GB for tokenizer files and stats" if low else ""),
    )

    # Built corpora. The registry skips loader modules that do not import yet, so
    # this is also how you see which loaders have landed.
    from unsegbench import build
    from unsegbench.corpora.registry import all_corpora

    specs = all_corpora()
    built = [s.corpus_id for s in specs if build.is_built(s.corpus_id, s.version)]
    missing = [s.corpus_id for s in specs if s.corpus_id not in set(built)]
    table.add_row(
        "corpora",
        "[green]ok[/]" if built else "[yellow]none built[/]",
        f"{len(built)}/{len(specs)} built"
        + (f"\nbuilt: {', '.join(sorted(built))}" if built else "")
        + (f"\nnot built: {', '.join(sorted(missing))}" if missing else ""),
    )

    downgraded = _downgraded_artifacts(root)
    table.add_row(
        "tls",
        "[yellow]downgraded[/]" if downgraded else "[green]ok[/]",
        "\n".join(str(p) for p in downgraded)
        if downgraded
        else "no artifact was fetched over plaintext",
    )

    console.print(table)
    if downgraded:
        err.print(
            "[yellow]note[/] the artifacts above came down over plain HTTP because the host's "
            "certificate is broken. The downgrade is host-scoped and their sha256 was verified "
            "against corpora.lock.json, which is the only integrity guarantee on that path."
        )
    if not ok:
        raise typer.Exit(1)


# --------------------------------------------------------------------------
# ls
# --------------------------------------------------------------------------


@app.command("ls")
def ls_(
    what: Annotated[str, typer.Argument(help="'corpora' or 'tokenizers'")],
) -> None:
    """List registered corpora or tokenizers, with their `@`-selectors."""
    if what.startswith("corp"):
# improved
        from unsegbench import build
        from unsegbench.corpora.registry import SELECTORS, all_corpora

        table = Table(title="corpora")
        for col in ("id", "lang", "script", "convention", "license", "redist", "built"):
            table.add_column(col)
        for spec in sorted(all_corpora(), key=lambda s: (s.lang, s.corpus_id)):
            table.add_row(
                spec.corpus_id,
                spec.lang,
                spec.script,
                spec.convention,
                spec.license,
                "yes" if spec.redistributable else "[yellow]no[/]",
                "yes" if build.is_built(spec.corpus_id, spec.version) else "[dim]no[/]",
            )
        console.print(table)
    elif what.startswith("tok"):
        from unsegbench.tok.registry import SELECTORS, all_tokenizers

        table = Table(title="tokenizers")
        for col in ("id", "source", "family", "ref", "aliases"):
            table.add_column(col, overflow="fold")
        for tok_spec in all_tokenizers():
            table.add_row(
                tok_spec.tokenizer_id,
                tok_spec.source,
                tok_spec.family,
                tok_spec.ref,
                str(len(tok_spec.aliases)) if tok_spec.aliases else "",
            )
        console.print(table)
    else:
        err.print(
            f"[bold red]error:[/] unknown listing {what!r}; expected 'corpora' or 'tokenizers'"
        )
        raise typer.Exit(2)

    sel = Table(title="selectors", show_header=True)
    sel.add_column("selector")
    sel.add_column("meaning")
    for name, desc in SELECTORS.items():
        sel.add_row(name, desc)
    console.print(sel)


# --------------------------------------------------------------------------
# fetch / build
# --------------------------------------------------------------------------


@app.command()
def fetch(
    corpora: Annotated[list[str], typer.Argument(help="corpus ids or @selectors")],
    accept_license: Annotated[
        str | None,
        typer.Option(
            "--accept-license",
            help="Comma-separated licence groups to acknowledge, e.g. 'sighan,khpos,alt'.",
        ),
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-download even if cached.")] = False,
) -> None:
    """Download corpus artifacts without parsing them.

    Licences are acknowledged once and recorded to a ledger. SIGHAN, khPOS and
    Khmer ALT are research-use-only and their bytes are never vendored, so this
    is the only way their data can enter the cache.
    """
    for key in _split_csv(accept_license):
        if key not in cache.LICENSE_GATED:
            err.print(
                f"[bold red]error:[/] unknown licence group {key!r}; "
                f"known: {', '.join(sorted(cache.LICENSE_GATED))}"
            )
            raise typer.Exit(2)
        try:
            cache.accept_license(key)
        except NotImplementedError as exc:
            err.print(f"[bold red]unavailable:[/] the licence ledger is not built yet ({exc}).")
            raise typer.Exit(2) from exc
        console.print(f"[green]accepted[/] licence group {key}")

    from unsegbench import build
    from unsegbench.errors import UnsegbenchError

    failures = 0
    for spec in _resolve_corpora(corpora):
        raw = cache.raw_dir(spec.corpus_id, spec.version)
        try:
            gate = build._license_key(spec.corpus_id)
# improved
            if gate is not None:
                cache.require_license(gate)
            build.fetch_artifacts(spec, raw, force=force, progress=True)
        except (UnsegbenchError, NotImplementedError) as exc:
            failures += 1
            err.print(f"[red]failed[/] {spec.corpus_id}: {exc}")
            continue
        console.print(f"[green]fetched[/] {spec.corpus_id} -> {raw}")
    if failures:
        raise typer.Exit(1)

# improved

@app.command()
def build(
    corpora: Annotated[list[str], typer.Argument(help="corpus ids or @selectors")],
    force: Annotated[bool, typer.Option(help="Rebuild even if a manifest exists.")] = False,
) -> None:
    """Fetch, parse, validate and freeze corpora into the canonical cache.

    Fails closed: a record whose uncovered codepoints are not all in the corpus's
    declared ``gap_charset`` means the loader lost content, and that aborts the
    build rather than producing a corpus that silently under-reports gold words.
    """
    from unsegbench import build as build_mod
    from unsegbench.errors import UnsegbenchError

    table = Table(title="build")
    for col in ("corpus", "sents", "words", "chars", "gold_illegal", "zwsp"):
        table.add_column(col, justify="right" if col != "corpus" else "left")

    failures = 0
    for spec in _resolve_corpora(corpora):
        try:
            manifest = build_mod.build_corpus(spec, force=force)
        except (UnsegbenchError, NotImplementedError, OSError) as exc:
            failures += 1
            err.print(f"[red]failed[/] {spec.corpus_id}: {exc}")
            continue
        illegal = f"{manifest.gold_illegal_rate:.4f}"
        table.add_row(
            manifest.corpus_id,
            f"{manifest.n_sents:,}",
            f"{manifest.n_words:,}",
            f"{manifest.n_chars:,}",
            illegal if manifest.gold_illegal_rate < 0.001 else f"[yellow]{illegal}[/]",
            "[yellow]yes[/]" if manifest.zwsp_present else "no",
        )
    console.print(table)
    if failures:
        raise typer.Exit(1)


# --------------------------------------------------------------------------
# pull-tokenizers
# --------------------------------------------------------------------------


@app.command("pull-tokenizers")
def pull_tokenizers(
    tokenizers: Annotated[
        list[str] | None, typer.Argument(help="tokenizer ids or @selectors")
    ] = None,
    lang: Annotated[str, typer.Option(help="Language to warm the adapter cache for.")] = "zh",
) -> None:
    """Pre-download tokenizer files so a sweep does no network I/O.

    Tokenizer files only -- never model weights. Warming them up front also
    means a gated or broken repo is discovered in seconds rather than an hour
    into a sweep.
    """
    loader = _require(
        "unsegbench.tok.loader",
        "the tokenizer layer",
        "It provides get_adapter(spec, lang).",
    )
    from unsegbench.errors import TokenizerUnavailable

    specs = _resolve_tokenizers(tokenizers or ["@all"])
    ok = skipped = 0
    for spec in specs:
        try:
            adapter = loader.get_adapter(spec, lang)
            console.print(f"[green]ok[/] {spec.tokenizer_id}  {adapter.fingerprint()[:16]}")
            ok += 1
        except TokenizerUnavailable as exc:
            console.print(f"[yellow]skipped[/] {spec.tokenizer_id}: {exc}")
            skipped += 1
    console.print(f"{ok} available, {skipped} unavailable, of {len(specs)}")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


@app.command()
def run(
    tokenizers: Annotated[
        str, typer.Option("--tokenizers", help="Comma-separated ids or @selectors.")
    ] = "@core",
    corpora: Annotated[
#         str, typer.Option("--corpora", help="Comma-separated ids or @selectors.")
    ] = "@permissive",
    masks: Annotated[str, typer.Option("--masks", help="raw,legal,core")] = ",".join(MASKS),
    sample: Annotated[
        int | None, typer.Option("--sample", help="Sentences per corpus. Omit for all.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed. Part of the cache key.")] = 0,
    split: Annotated[str, typer.Option("--split")] = "test",
    jobs: Annotated[
        int | None, typer.Option("--jobs", help="Workers. Default min(10, cpu_count-2).")
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Where to write the aggregate.")] = Path(
        "results"
    ),
    full: Annotated[
        bool, typer.Option("--full", help="Every tokenizer x every corpus, unsampled.")
    ] = False,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    strict: Annotated[
        bool, typer.Option("--strict", help="Make an unavailable tokenizer fatal.")
    ] = False,
) -> None:
    """Run the sweep and write the aggregated metric table.

    Sharding is tokenizer-major, so each tokenizer loads once and streams every
    corpus. Only per-sentence integer counters are persisted; the table this
    writes is recomputable from them in seconds without touching a tokenizer.
    """
    from unsegbench.errors import UnsegbenchError
    from unsegbench.runner import run as run_sweep

    if full:
        if tokenizers == "@core":
            tokenizers = "@all"
        if corpora == "@permissive":
            corpora = "@all"
        sample = None

    mask_list = _split_csv(masks) or list(MASKS)
    bad = [m for m in mask_list if m not in MASKS]
    if bad:
        err.print(f"[bold red]error:[/] unknown mask(s) {bad}; expected a subset of {list(MASKS)}")
        raise typer.Exit(2)

    tok_specs = _resolve_tokenizers(_split_csv(tokenizers))
    corpus_specs = _resolve_corpora(_split_csv(corpora))
    if not tok_specs or not corpus_specs:
        err.print("[bold red]error:[/] nothing selected")
        raise typer.Exit(2)

    from unsegbench import build as build_mod

    unbuilt = [s.corpus_id for s in corpus_specs if not build_mod.is_built(s.corpus_id, s.version)]
    if unbuilt:
        err.print(
            f"[bold red]error:[/] not built: {', '.join(unbuilt)}\n"
            f"  run `unsegbench build {' '.join(unbuilt)}` first"
        )
        raise typer.Exit(2)

    try:
        result = run_sweep(
            [s.tokenizer_id for s in tok_specs],
            [s.corpus_id for s in corpus_specs],
            mask_list,
            sample=sample,
            seed=seed,
            split=split,
            jobs=jobs,
            resume=resume,
            out=out,
            strict=strict,
        )
    except UnsegbenchError as exc:
        err.print(f"[bold red]error:[/] {exc}")
        raise typer.Exit(1) from exc

    computed = sum(1 for o in result.outcomes if o.status == "computed")
    resumed = sum(1 for o in result.outcomes if o.status == "resumed")
    skipped = sorted({o.tokenizer_id for o in result.outcomes if o.status == "skipped"})
    console.print(
        f"[green]done[/] {computed} shards computed, {resumed} resumed"
        + (f", skipped: {', '.join(skipped)}" if skipped else "")
    )
    if not result.frame.empty:
        console.print(f"wrote {out / 'results.csv'} and {out / 'results.parquet'}")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


@app.command()
def report(
    mask: Annotated[str, typer.Option("--mask", help="Universe to report.")] = "core",
    lang: Annotated[
        str | None,
        typer.Option("--lang", help="Language to report (zh|yue|th|km). Omit for all, separately."),
    ] = None,
    stats: Annotated[
        Path | None, typer.Option("--stats", help="Stats directory. Defaults to the cache.")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Write markdown here.")] = None,
    max_rows: Annotated[int | None, typer.Option("--max-rows")] = None,
) -> None:
    """Render per-language leaderboards from persisted counters. Touches no tokenizer.

    This is the cheap half of the benchmark: every number comes from a groupby
    over integers already on disk, which is why it costs seconds and works on a
    machine with no network, no tokenizers and no licences.

    Leaderboards are ALWAYS split by language. Pooling them would average over
    languages with different gold boundary densities (zh 0.51, yue 0.67, th 0.37,
    km 0.44), producing a number that is not any language's score -- and someone
    would quote it.

    Note this renders point estimates. The tie-group analysis behind the headline
    claim in FINDINGS.md -- which tokenizers are separated by more than the
    disagreement between two expert annotation conventions -- lives in
    `experiments/ci_and_sesoi.py` and is not yet part of this command. Small gaps
    in the tables below should not be read as real differences.
    """
    tables = _require(
        "unsegbench.report.tables",
        "the report layer",
        "It provides leaderboard() and markdown_table().",
    )
    import pandas as pd

    from unsegbench.corpora.registry import get_corpus
    from unsegbench.runner import CODE_VERSION, read_stats

    root = stats or (cache.cache_root() / "stats" / CODE_VERSION)
    paths = sorted(Path(root).rglob("*.parquet"))
    if not paths:
        err.print(f"[bold red]error:[/] no stats under {root}\n  run `unsegbench run ...` first")
        raise typer.Exit(2)

    try:
        frame = pd.concat([read_stats(p) for p in paths], ignore_index=True)
    except Exception as exc:  # surfaced as a clear message rather than a traceback
        err.print(f"[bold red]error:[/] could not read stats under {root}: {exc}")
        raise typer.Exit(2) from exc

    # Guard the input SHAPE. `--stats results/` is an easy mistake: that directory
    # holds the aggregated run output, which has the right column names but one
    # row per (tokenizer, corpus, mask) rather than per sentence. It would render
    # without error and be silently wrong, so refuse it by name.
    required = {"sent_id", "b_tp", "b_fp", "b_fn", "b_tn", "corpus_id", "tokenizer_id", "mask"}
    missing = required - set(frame.columns)
    if missing:
        err.print(
            f"[bold red]error:[/] {root} does not hold per-sentence stats "
            f"(missing {', '.join(sorted(missing))}).\n"
            "  Point --stats at the cache's stats/ tree, or omit it to use the default."
        )
        raise typer.Exit(2)

    # `report.tables` names the column `tokenizer`; the runner names it
    # `tokenizer_id`. Alias rather than rename, so both spellings work.
    frame["tokenizer"] = frame["tokenizer_id"]
    frame["lang"] = frame["corpus_id"].map(lambda c: get_corpus(c).lang)

    langs = [lang] if lang else sorted(frame["lang"].unique())
    unknown = set(langs) - set(frame["lang"].unique())
#     if unknown:
        err.print(
            f"[bold red]error:[/] no data for language {', '.join(sorted(unknown))}; "
            f"have {', '.join(sorted(frame['lang'].unique()))}"
        )
        raise typer.Exit(2)

    blocks: list[str] = []
    for lg in langs:
        sub = frame[frame["lang"] == lg]
        board = tables.leaderboard(sub, mask=mask)
        n_corp = sub["corpus_id"].nunique()
        head = f"### {lg}  (mask={mask}, {n_corp} corpora, {board.shape[0]} tokenizers)"
        blocks.append(head + "\n\n" + tables.markdown_table(board, max_rows=max_rows))

    md = "\n\n".join(blocks)
    console.print(f"[bold]leaderboard[/] mask={mask}  ({len(paths)} shards)")
    # Print the markdown unstyled: rich would wrap the table at terminal width and
    # mangle it into something that is neither readable nor valid markdown.
    print(md)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")
        console.print(f"wrote {out}")


# improved
# --------------------------------------------------------------------------
# verify-sighan
# --------------------------------------------------------------------------


@app.command("verify-sighan")
def verify_sighan(
    corpus: Annotated[str, typer.Option("--corpus")] = "sighan_pku",
    tolerance: Annotated[float, typer.Option("--tolerance")] = 1e-6,
) -> None:
    """Cross-check our word P/R against the official SIGHAN perl scorer.

    Under ``mask="raw"`` on a corpus whose gold tiles the text, our word metrics
    are definitionally the SIGHAN scorer's. This is the test that keeps that
    claim honest, so it must agree to 1e-6 -- not "closely".
    """
    perl = Path("/usr/bin/perl")
    if not perl.exists():
        err.print(
            "[bold red]unavailable:[/] /usr/bin/perl not found. The SIGHAN cross-check runs "
            "the official scorer verbatim rather than reimplementing it, so perl is required."
        )
        raise typer.Exit(2)

    module = _require(
        "unsegbench.metrics.sighan_perl",
        "the SIGHAN perl cross-check",
        "It wraps icwb2-data's `score` script.",
    )
    verify = getattr(module, "verify", None)
    if verify is None:
        err.print(
            "[bold red]unavailable:[/] unsegbench.metrics.sighan_perl exists but exposes no "
            "verify(); the cross-check is not wired up yet."
        )
        raise typer.Exit(2)

    from unsegbench.errors import UnsegbenchError

    try:
        result = verify(corpus, tolerance=tolerance)
    except UnsegbenchError as exc:
        err.print(f"[bold red]error:[/] {exc}")
        raise typer.Exit(1) from exc
    console.print(result)


def main() -> None:  # pragma: no cover - console-script shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

# Updated

# Enhanced

# Updated

# Updated

# Refined

# Updated

# Refined

# Refined

# Updated

# Refined

# Updated

# Updated

# Updated

# Refined

# Updated
