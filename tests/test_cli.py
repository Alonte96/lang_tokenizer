"""CLI surface tests.

The CLI's contract is not "it computes the right numbers" -- `runner` and
`report.tables` own that -- but "it refuses bad input with a sentence instead of
a traceback, and it never pools languages into one table". Both are properties of
the terminal output, so they are tested through `typer.testing.CliRunner` against
the real `app`.

Everything here is offline and touches no tokenizer. The handful of tests that
# read the user's real stats cache are marked ``slow`` and skip cleanly when it is
absent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from unsegbench.cli import app
from unsegbench.runner import CODE_VERSION
from unsegbench.types import STATS_COLUMNS

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

#: Every command registered on the app. `--help` for each must work even when
#: the layer behind it is unbuilt -- that is the whole point of `_require`.
SUBCOMMANDS: tuple[str, ...] = (
    "doctor",
    "ls",
    "fetch",
    "build",
    "pull-tokenizers",
    "run",
    "report",
    "verify-sighan",
)

#: The user's real sweep output, if they have one.
REAL_STATS = Path.home() / "Library" / "Caches" / "unsegbench" / "stats" / CODE_VERSION

runner = CliRunner()


def _flat(text: str) -> str:
    """Collapse whitespace.

    rich hard-wraps stderr at the terminal width, so an error message that reads
    as one sentence arrives with newlines in arbitrary places. Asserting on the
    wrapped form would make these tests depend on the width of the message.
    """
    return " ".join(text.split())


def _invoke(*args: str):
    return runner.invoke(app, list(args))


def write_shard(
    path: Path,
    *,
    tokenizer_id: str,
    corpus_id: str,
    mask: str = "core",
    tp: int = 1,
    fp: int = 1,
    fn: int = 1,
    tn: int = 1,
    n_sents: int = 4,
) -> Path:
    """Write a per-sentence stats shard the way the runner writes one.

    The identity columns live in parquet metadata rather than in the rows, so a
    shard written without that metadata reads back with ``corpus_id == ""`` and
    the language lookup blows up. Building these by hand rather than by running a
    sweep is what keeps these tests offline.
    """
    rows = []
    for i in range(n_sents):
        row = dict.fromkeys(STATS_COLUMNS, 0)
        row["sent_id"] = f"s{i}"
        row["b_tp"], row["b_fp"], row["b_fn"], row["b_tn"] = tp, fp, fn, tn
        row["n_mask"] = tp + fp + fn + tn
        row["n_chars"] = 10
        row["n_tokens"] = 5
        row["n_gold_words"] = 4
        row["w_tp"], row["w_pred"], row["w_gold"], row["w_intact"] = 2, 4, 4, 3
        rows.append(row)
    frame = pd.DataFrame(rows)[list(STATS_COLUMNS)]
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {
            b"unsegbench.tokenizer_id": tokenizer_id.encode(),
            b"unsegbench.corpus_id": corpus_id.encode(),
            b"unsegbench.mask": mask.encode(),
            b"unsegbench.code_version": CODE_VERSION.encode(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


@pytest.fixture
def synth_stats(tmp_path: Path) -> Path:
    """Two languages, one shared tokenizer id, deliberately different counts.

    `char` scores very differently in zh and th here. If the report ever pooled
    languages there would be ONE `char` row carrying neither number, which is
    exactly the regression `test_lang_blocks_are_not_pooled` pins down.
    """
    root = tmp_path / "stats"
    write_shard(
        root / "zh-char" / "core.parquet",
        tokenizer_id="char",
        corpus_id="sighan_pku",
        tp=9,
        fp=1,
        fn=1,
        tn=9,
    )
    write_shard(
        root / "th-char" / "core.parquet",
        tokenizer_id="char",
        corpus_id="ud_th_pud",
        tp=2,
        fp=8,
        fn=8,
        tn=2,
    )
    write_shard(
        root / "zh-whole" / "core.parquet",
        tokenizer_id="whole",
        corpus_id="ud_zh_gsd",
        tp=5,
        fp=5,
        fn=5,
        tn=5,
    )
    return root


def parse_tables(md: str) -> list[list[list[str]]]:
    """Split markdown into tables, each a list of rows, each a list of cells."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current.append([c.strip() for c in stripped.strip("|").split("|")])
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def lang_headings(md: str) -> list[str]:
    return re.findall(r"^### (\S+)", md, flags=re.MULTILINE)


requires_real_stats = pytest.mark.skipif(
    not REAL_STATS.is_dir() or not any(REAL_STATS.rglob("*.parquet")),
    reason=f"no cached stats under {REAL_STATS}; run `unsegbench run` first",
)


# --------------------------------------------------------------------------
# 1. --help everywhere
# --------------------------------------------------------------------------


def test_app_help_exits_zero():
    result = _invoke("--help")
    assert result.exit_code == 0
    assert "unsegbench" in result.output


def test_app_help_lists_every_subcommand():
    output = _invoke("--help").output
    missing = [c for c in SUBCOMMANDS if c not in _flat(output)]
    assert not missing, f"not advertised in --help: {missing}"

# improved

@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_subcommand_help_exits_zero(command: str):
    result = _invoke(command, "--help")
    assert result.exit_code == 0, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_subcommand_help_shows_usage(command: str):
    output = _flat(_invoke(command, "--help").output)
    assert f"Usage: unsegbench {command}" in output


def test_no_args_is_help_not_a_crash():
    result = _invoke()
    # `no_args_is_help=True` -- a bare invocation prints help rather than failing
    # with a stack trace.
    assert "Usage" in result.output
    assert not isinstance(result.exception, (KeyError, AttributeError, TypeError))


# --------------------------------------------------------------------------
# 2. ls
# --------------------------------------------------------------------------


def test_ls_corpora_renders(tmp_cache):
    result = _invoke("ls", "corpora")
    assert result.exit_code == 0, result.output
    assert "corpora" in result.output


def test_ls_corpora_lists_a_known_corpus(tmp_cache):
    assert "sighan_pku" in _flat(_invoke("ls", "corpora").output)


def test_ls_corpora_shows_selectors(tmp_cache):
    assert "@permissive" in _flat(_invoke("ls", "corpora").output)


def test_ls_tokenizers_renders():
    result = _invoke("ls", "tokenizers")
    assert result.exit_code == 0, result.output
    assert "tokenizers" in result.output


def test_ls_tokenizers_lists_the_baselines():
    output = _flat(_invoke("ls", "tokenizers").output)
    for tok in ("char", "whole", "whitespace"):
        assert tok in output


def test_ls_tokenizers_shows_selectors():
    assert "@baselines" in _flat(_invoke("ls", "tokenizers").output)


def test_ls_unknown_kind_exits_two():
    result = _invoke("ls", "kittens")
    assert result.exit_code == 2
    assert "corpora" in _flat(result.stderr) and "tokenizers" in _flat(result.stderr)


# --------------------------------------------------------------------------
# 3. report guards
# --------------------------------------------------------------------------


def test_report_on_aggregated_parquet_exits_two(tmp_path: Path):
    """`--stats results/` is the easy mistake: right columns, wrong granularity."""
    agg = pd.DataFrame(
        [
            {
                "tokenizer_id": "char",
                "corpus_id": "sighan_pku",
                "mask": "core",
                "b_tp": 100,
                "b_fp": 10,
                "b_fn": 10,
                "b_tn": 100,
                "n_tokens": 50,
                "n_sents": 5,
            }
        ]
    )
    stats = tmp_path / "results"
    stats.mkdir()
    agg.to_parquet(stats / "results.parquet")

    result = _invoke("report", "--stats", str(stats))
    assert result.exit_code == 2


def test_report_on_aggregated_parquet_names_the_missing_column(tmp_path: Path):
    agg = pd.DataFrame(
        [
            {
                "tokenizer_id": "char",
                "corpus_id": "sighan_pku",
                "mask": "core",
                "b_tp": 1,
                "b_fp": 1,
                "b_fn": 1,
                "b_tn": 1,
            }
        ]
    )
    stats = tmp_path / "results"
    stats.mkdir()
    agg.to_parquet(stats / "results.parquet")

    message = _flat(_invoke("report", "--stats", str(stats)).stderr)
    assert "sent_id" in message
    assert "per-sentence" in message


def test_report_on_aggregated_parquet_is_not_a_traceback(tmp_path: Path):
    agg = pd.DataFrame([{"tokenizer_id": "char", "corpus_id": "sighan_pku", "mask": "core"}])
    stats = tmp_path / "results"
    stats.mkdir()
    agg.to_parquet(stats / "results.parquet")

    result = _invoke("report", "--stats", str(stats))
    assert result.exit_code == 2
    # A SystemExit is the CLI exiting; anything else escaped as a stack trace.
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output


def test_report_unknown_lang_exits_two(synth_stats: Path):
    result = _invoke("report", "--stats", str(synth_stats), "--lang", "xx")
    assert result.exit_code == 2


def test_report_unknown_lang_lists_the_languages_that_exist(synth_stats: Path):
    message = _flat(_invoke("report", "--stats", str(synth_stats), "--lang", "xx").stderr)
    assert "xx" in message
    assert "zh" in message and "th" in message


def test_report_unknown_lang_is_not_a_traceback(synth_stats: Path):
    result = _invoke("report", "--stats", str(synth_stats), "--lang", "xx")
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output


def test_report_empty_stats_dir_exits_two(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _invoke("report", "--stats", str(empty))
    assert result.exit_code == 2


def test_report_empty_stats_dir_says_to_run_the_sweep(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    message = _flat(_invoke("report", "--stats", str(empty)).stderr)
    assert "no stats" in message
    assert "unsegbench run" in message


def test_report_empty_cache_exits_two(tmp_cache):
    """With no `--stats` the command falls back to the cache, which is empty here."""
    result = _invoke("report")
    assert result.exit_code == 2
    assert "unsegbench run" in _flat(result.stderr)


# --------------------------------------------------------------------------
# 4. --lang actually filters, and languages are never pooled
# --------------------------------------------------------------------------


def test_report_lang_emits_only_that_language(synth_stats: Path):
    result = _invoke("report", "--stats", str(synth_stats), "--lang", "zh")
    assert result.exit_code == 0, result.stderr
    assert lang_headings(result.stdout) == ["zh"]


def test_report_lang_th_emits_only_that_language(synth_stats: Path):
    result = _invoke("report", "--stats", str(synth_stats), "--lang", "th")
    assert result.exit_code == 0, result.stderr
    assert lang_headings(result.stdout) == ["th"]


def test_report_without_lang_emits_one_block_per_language(synth_stats: Path):
    result = _invoke("report", "--stats", str(synth_stats))
    assert result.exit_code == 0, result.stderr
    assert sorted(lang_headings(result.stdout)) == ["th", "zh"]


def test_report_emits_exactly_one_table_per_language_block(synth_stats: Path):
    """One heading, one table. A pooled table would show up as a count mismatch."""
    stdout = _invoke("report", "--stats", str(synth_stats)).stdout
    assert len(parse_tables(stdout)) == len(lang_headings(stdout))


def test_lang_blocks_are_not_pooled(synth_stats: Path):
    """The regression guard: no single table may mix languages.

    `char` is scored on zh counts (9/1/1/9) and th counts (2/8/8/2). Pooling
    would collapse those into one row whose phi is neither language's. Splitting
    gives two rows with opposite signs, which is what this asserts.
    """
    stdout = _invoke("report", "--stats", str(synth_stats)).stdout
    blocks = dict(zip(lang_headings(stdout), parse_tables(stdout), strict=True))

    def phi_of(block: list[list[str]], tokenizer: str) -> float:
        header = block[0]
        for row in block[2:]:
            cells = dict(zip(header, row, strict=True))
            if cells["tokenizer"] == tokenizer:
                return float(cells["phi"])
        raise AssertionError(f"{tokenizer!r} absent from block")

    assert phi_of(blocks["zh"], "char") == pytest.approx(0.8)
    assert phi_of(blocks["th"], "char") == pytest.approx(-0.6)
    # And a pooled table would have one `char` row, not two across two blocks.
    assert sum(len(t) - 2 for t in parse_tables(stdout)) == 3

# 
def test_each_table_belongs_to_exactly_one_language(synth_stats: Path):
    """No tokenizer id appears twice within a single table.

    Pooling zh and th shards for the same tokenizer without a language groupby is
    the other shape the bug could take: duplicated rows inside one table.
    """
    for table in parse_tables(_invoke("report", "--stats", str(synth_stats)).stdout):
        header = table[0]
        col = header.index("tokenizer")
        names = [row[col] for row in table[2:]]
        assert len(names) == len(set(names)), f"duplicate tokenizers in one table: {names}"


def test_report_block_reports_its_own_corpus_count(synth_stats: Path):
    """The `N corpora` in a heading counts that language's corpora only."""
    stdout = _invoke("report", "--stats", str(synth_stats)).stdout
    counts = dict(re.findall(r"### (\S+)\s+\(mask=\w+, (\d+) corpora", stdout))
    assert counts["zh"] == "2"  # sighan_pku + ud_zh_gsd
    assert counts["th"] == "1"  # ud_th_pud


def test_report_max_rows_truncates(synth_stats: Path):
    stdout = _invoke("report", "--stats", str(synth_stats), "--max-rows", "1").stdout
    for table in parse_tables(stdout):
        assert len(table) - 2 <= 1


# --------------------------------------------------------------------------
# 5. the emitted markdown is valid
# --------------------------------------------------------------------------


def test_markdown_has_a_separator_row(synth_stats: Path):
    for table in parse_tables(_invoke("report", "--stats", str(synth_stats)).stdout):
        assert all(set(c) <= {"-", ":"} and c for c in table[1]), table[1]


def test_markdown_separator_matches_header_width(synth_stats: Path):
    for table in parse_tables(_invoke("report", "--stats", str(synth_stats)).stdout):
        assert len(table[1]) == len(table[0])


def test_markdown_body_rows_match_header_width(synth_stats: Path):
    """rich wrapping the table at terminal width would show up right here."""
    for table in parse_tables(_invoke("report", "--stats", str(synth_stats)).stdout):
        width = len(table[0])
        for row in table[2:]:
            assert len(row) == width, row


def test_markdown_table_has_a_body(synth_stats: Path):
    tables = parse_tables(_invoke("report", "--stats", str(synth_stats)).stdout)
    assert tables
    for table in tables:
        assert len(table) >= 3


def test_markdown_header_starts_with_tokenizer(synth_stats: Path):
    for table in parse_tables(_invoke("report", "--stats", str(synth_stats)).stdout):
        assert table[0][0] == "tokenizer"


def test_markdown_lines_are_not_wrapped(synth_stats: Path):
    """No table line may be shorter than its header: that is what wrapping does."""
    stdout = _invoke("report", "--stats", str(synth_stats)).stdout
    lines = [ln for ln in stdout.splitlines() if ln.startswith("|")]
    assert lines
    # Every row of a given table has the same cell count, so a stray short line
    # is a wrapped fragment rather than a row.
    assert all(ln.rstrip().endswith("|") for ln in lines)


# --------------------------------------------------------------------------
# 6. --out
# --------------------------------------------------------------------------

# improved

def test_report_out_writes_the_file(synth_stats: Path, tmp_path: Path):
    out = tmp_path / "nested" / "report.md"
    result = _invoke("report", "--stats", str(synth_stats), "--out", str(out))
    assert result.exit_code == 0, result.stderr
    assert out.is_file()
    assert out.read_text(encoding="utf-8").strip()


def test_report_out_content_matches_stdout(synth_stats: Path, tmp_path: Path):
    out = tmp_path / "report.md"
    stdout = _invoke("report", "--stats", str(synth_stats), "--out", str(out)).stdout
    assert out.read_text(encoding="utf-8").strip() in stdout


def test_report_out_tables_match_stdout_tables(synth_stats: Path, tmp_path: Path):
    out = tmp_path / "report.md"
    stdout = _invoke("report", "--stats", str(synth_stats), "--out", str(out)).stdout
    assert parse_tables(out.read_text(encoding="utf-8")) == parse_tables(stdout)


def test_report_out_is_announced(synth_stats: Path, tmp_path: Path):
    out = tmp_path / "report.md"
    stdout = _invoke("report", "--stats", str(synth_stats), "--out", str(out)).stdout
    assert "wrote" in stdout


# --------------------------------------------------------------------------
# The same properties, against the user's real stats. Slow, and optional.
# --------------------------------------------------------------------------


@pytest.mark.slow
@requires_real_stats
def test_real_report_lang_zh_emits_only_zh():
    result = _invoke("report", "--stats", str(REAL_STATS), "--lang", "zh")
    assert result.exit_code == 0, result.stderr
    assert lang_headings(result.stdout) == ["zh"]


@pytest.mark.slow
@requires_real_stats
def test_real_report_without_lang_emits_every_language():
    result = _invoke("report", "--stats", str(REAL_STATS))
    assert result.exit_code == 0, result.stderr
    headings = lang_headings(result.stdout)
    assert len(headings) == len(set(headings)) >= 2


@pytest.mark.slow
@requires_real_stats
def test_real_report_never_pools_languages():
    stdout = _invoke("report", "--stats", str(REAL_STATS)).stdout
    tables = parse_tables(stdout)
    assert len(tables) == len(lang_headings(stdout))
    for table in tables:
        col = table[0].index("tokenizer")
        names = [row[col] for row in table[2:]]
        assert len(names) == len(set(names))


@pytest.mark.slow
@requires_real_stats
def test_real_report_markdown_is_well_formed():
    for table in parse_tables(_invoke("report", "--stats", str(REAL_STATS)).stdout):
        width = len(table[0])
        assert len(table[1]) == width
        assert all(len(row) == width for row in table[2:])


@pytest.mark.slow
@requires_real_stats
def test_real_report_per_language_scores_differ():
    """Sanity: the blocks are genuinely different tables, not one table repeated."""
    stdout = _invoke("report", "--stats", str(REAL_STATS)).stdout
    bodies = ["\n".join(str(r) for r in t[2:]) for t in parse_tables(stdout)]
    assert len(set(bodies)) == len(bodies)
# 
# Updated

# Updated

# Enhanced

# Refined

# Enhanced

# Updated

# Refined

# Updated

# Enhanced

# Refined

# Refined

# Enhanced

# Updated
