"""Report tables and the tokenizer registry.

`report.tables` is pure: frame in, frame or string out, no cache, no network, no
tokenizer. So everything here is a synthetic frame with counts chosen so the
expected number can be worked out by hand, which is the only way a metric test
is worth anything.

The registry tests guard the two invariants the rest of the benchmark assumes:
ids are unique (or a duplicate silently becomes two data points for one
tokenizer) and `@`-selectors mean what `SELECTORS` says they mean.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from unsegbench.report.tables import (
    REPORT_COLUMNS,
    leaderboard,
    markdown_table,
    tier0_table,
)
from unsegbench.tok.registry import (
    ENTRIES,
    SELECTORS,
    all_tokenizers,
    get_tokenizer_spec,
    resolve,
)

# --------------------------------------------------------------------------
# synthetic frames
# --------------------------------------------------------------------------


def stat_row(
    tokenizer: str,
    *,
    lang: str = "zh",
    mask: str = "core",
    tp: int = 1,
    fp: int = 1,
    fn: int = 1,
    tn: int = 1,
    n_tokens: int = 10,
    n_chars: int = 20,
    n_gold_words: int = 5,
    crossing: int = 0,
    midcodepoint: int = 0,
    cluster_split: int = 0,
    dropped: int = 0,
) -> dict[str, object]:
    """One per-sentence counter row, in the shape `leaderboard` consumes."""
    return {
        "tokenizer": tokenizer,
        "lang": lang,
        "mask": mask,
        "b_tp": tp,
        "b_fp": fp,
        "b_fn": fn,
        "b_tn": tn,
        "w_tp": 3,
        "w_pred": 5,
        "w_gold": 5,
        "w_intact": 4,
        "n_tokens": n_tokens,
        "n_chars": n_chars,
        "n_gold_words": n_gold_words,
        "crossing_tokens": crossing,
        "f_midcodepoint": midcodepoint,
        "f_cluster_split": cluster_split,
        "f_dropped_chars": dropped,
    }


@pytest.fixture
def three_tokenizers() -> pd.DataFrame:
    """Counts chosen so phi is exactly 0.8 / 0.0 / -0.6, in that order.

    Deliberately inserted worst-first so a leaderboard that forgot to sort would
    fail rather than pass by luck.
    """
    return pd.DataFrame(
        [
            stat_row("bad", tp=2, fp=8, fn=8, tn=2),  # phi = -0.6
            stat_row("mid", tp=5, fp=5, fn=5, tn=5),  # phi =  0.0
            stat_row("good", tp=9, fp=1, fn=1, tn=9),  # phi =  0.8
        ]
    )


# --------------------------------------------------------------------------
# 7. leaderboard()
# --------------------------------------------------------------------------


def test_leaderboard_is_ordered_by_descending_phi(three_tokenizers: pd.DataFrame):
    board = leaderboard(three_tokenizers)
    assert list(board["tokenizer"]) == ["good", "mid", "bad"]


def test_leaderboard_phi_values_are_the_pooled_ones(three_tokenizers: pd.DataFrame):
    board = leaderboard(three_tokenizers).set_index("tokenizer")
    assert board.loc["good", "phi"] == pytest.approx(0.8)
    assert board.loc["mid", "phi"] == pytest.approx(0.0)
    assert board.loc["bad", "phi"] == pytest.approx(-0.6)


def test_leaderboard_phi_is_monotone_non_increasing(three_tokenizers: pd.DataFrame):
    phis = list(leaderboard(three_tokenizers)["phi"])
    assert phis == sorted(phis, reverse=True)


def test_leaderboard_columns_are_a_subset_of_report_columns(three_tokenizers: pd.DataFrame):
    assert set(leaderboard(three_tokenizers).columns) <= set(REPORT_COLUMNS)


def test_leaderboard_columns_keep_report_column_order(three_tokenizers: pd.DataFrame):
    cols = list(leaderboard(three_tokenizers).columns)
    assert cols == [c for c in REPORT_COLUMNS if c in cols]


def test_leaderboard_carries_the_density_context(three_tokenizers: pd.DataFrame):
    """`phi` without `rho`/`delta_s`/`fertility` invites a bogus comparison."""
    cols = set(leaderboard(three_tokenizers).columns)
    assert {"rho", "delta_s", "fertility"} <= cols


def test_leaderboard_without_rank_intervals_has_no_rank_columns(three_tokenizers: pd.DataFrame):
    """Omitting the bootstrap must drop the columns, not invent integer ranks."""
    cols = set(leaderboard(three_tokenizers).columns)
    assert "rank_lo" not in cols
    assert "rank_hi" not in cols
    assert not any(c.startswith("rank") for c in cols)


def test_leaderboard_with_rank_intervals_has_rank_columns(three_tokenizers: pd.DataFrame):
    board = leaderboard(
        three_tokenizers,
        rank_intervals={"good": (1, 2), "mid": (1, 3), "bad": (2, 3)},
    ).set_index("tokenizer")
    assert board.loc["good", "rank_lo"] == 1
    assert board.loc["bad", "rank_hi"] == 3


def test_leaderboard_rank_columns_sit_right_after_phi(three_tokenizers: pd.DataFrame):
    board = leaderboard(three_tokenizers, rank_intervals={"good": (1, 1)})
    cols = list(board.columns)
    assert cols[:4] == ["tokenizer", "phi", "rank_lo", "rank_hi"]


def test_leaderboard_filters_by_mask(three_tokenizers: pd.DataFrame):
    other = three_tokenizers.assign(mask="raw", tokenizer="raw_only")
    both = pd.concat([three_tokenizers, other], ignore_index=True)
    assert "raw_only" not in set(leaderboard(both, mask="core")["tokenizer"])
    assert set(leaderboard(both, mask="raw")["tokenizer"]) == {"raw_only"}


def test_leaderboard_pools_rows_within_a_tokenizer(three_tokenizers: pd.DataFrame):
    """Two identical sentences must give the same phi as one -- phi is a ratio."""
    doubled = pd.concat([three_tokenizers, three_tokenizers], ignore_index=True)
    one = leaderboard(three_tokenizers).set_index("tokenizer")["phi"]
    two = leaderboard(doubled).set_index("tokenizer")["phi"]
    assert two.to_dict() == pytest.approx(one.to_dict())


def test_leaderboard_row_count_is_the_tokenizer_count(three_tokenizers: pd.DataFrame):
    assert leaderboard(three_tokenizers).shape[0] == 3


# --------------------------------------------------------------------------
# 8. markdown_table()
# --------------------------------------------------------------------------


def test_markdown_table_merges_ranks_into_one_column():
    df = pd.DataFrame([{"tokenizer": "a", "rank_lo": 1, "rank_hi": 4, "phi": 0.5}])
    md = markdown_table(df)
    header = [c.strip() for c in md.splitlines()[0].strip("|").split("|")]
    assert "rank" in header
    assert "rank_lo" not in header and "rank_hi" not in header


def test_markdown_table_renders_a_point_rank_as_one_integer():
    df = pd.DataFrame([{"tokenizer": "a", "rank_lo": 1, "rank_hi": 1, "phi": 0.5}])
    assert "| 1 |" in markdown_table(df)
    assert "1-1" not in markdown_table(df)


def test_markdown_table_renders_an_interval_rank_as_a_range():
    df = pd.DataFrame([{"tokenizer": "a", "rank_lo": 1, "rank_hi": 4, "phi": 0.5}])
    assert "1-4" in markdown_table(df)


def test_markdown_table_puts_rank_second():
    df = pd.DataFrame([{"tokenizer": "a", "phi": 0.5, "rank_lo": 2, "rank_hi": 3}])
    header = [c.strip() for c in markdown_table(df).splitlines()[0].strip("|").split("|")]
    assert header[:2] == ["tokenizer", "rank"]


def test_markdown_table_formats_floats_to_the_given_precision():
    df = pd.DataFrame([{"tokenizer": "a", "phi": 0.123456}])
    assert "0.123" in markdown_table(df, floats=3)
    assert "0.1235" in markdown_table(df, floats=4)
    assert "0.1" in markdown_table(df, floats=1)


def test_markdown_table_float_precision_is_exact():
    df = pd.DataFrame([{"tokenizer": "a", "phi": 0.5}])
    body = markdown_table(df, floats=3).splitlines()[2]
    assert body == "| a | 0.500 |"


def test_markdown_table_renders_nan_as_a_dash():
    df = pd.DataFrame([{"tokenizer": "a", "phi": float("nan")}])
    assert "--" in markdown_table(df)
    assert "nan" not in markdown_table(df).lower()


def test_markdown_table_renders_nan_beside_real_values():
    df = pd.DataFrame([{"tokenizer": "a", "phi": 0.25}, {"tokenizer": "b", "phi": float("nan")}])
    lines = markdown_table(df).splitlines()
    assert lines[2] == "| a | 0.250 |"
    assert lines[3] == "| b | -- |"


def test_markdown_table_leaves_integers_alone():
    df = pd.DataFrame([{"tokenizer": "a", "n_tokens": 12345}])
    assert "12345" in markdown_table(df)


def test_markdown_table_structure_is_valid_markdown():
    df = pd.DataFrame(
        [{"tokenizer": "a", "phi": 0.5, "rho": 1.0}, {"tokenizer": "b", "phi": 0.1, "rho": 0.9}]
    )
    lines = markdown_table(df).splitlines()
    widths = {len(ln.strip("|").split("|")) for ln in lines}
    assert len(widths) == 1
    assert set(lines[1].strip("|")) == {"-", "|"}


def test_markdown_table_max_rows_truncates():
    df = pd.DataFrame([{"tokenizer": t, "phi": 0.1} for t in "abcde"])
    assert len(markdown_table(df, max_rows=2).splitlines()) == 2 + 2


def test_markdown_table_max_rows_none_keeps_everything():
    df = pd.DataFrame([{"tokenizer": t, "phi": 0.1} for t in "abcde"])
    assert len(markdown_table(df, max_rows=None).splitlines()) == 5 + 2


def test_markdown_table_of_a_leaderboard_round_trips(three_tokenizers: pd.DataFrame):
    board = leaderboard(three_tokenizers)
    lines = markdown_table(board).splitlines()
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    assert header == list(board.columns)
    assert len(lines) == board.shape[0] + 2

# 
# --------------------------------------------------------------------------
# 9. tier0_table()
# --------------------------------------------------------------------------


@pytest.fixture
def tier0_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            stat_row("a", lang="zh", n_tokens=100, n_chars=200, midcodepoint=5, cluster_split=10),
            stat_row("a", lang="zh", n_tokens=100, n_chars=200, midcodepoint=5, cluster_split=10),
            stat_row("a", lang="th", n_tokens=50, n_chars=150, midcodepoint=25, cluster_split=0),
            stat_row("b", lang="zh", n_tokens=200, n_chars=400, midcodepoint=0, dropped=7),
        ]
    )


def test_tier0_groups_by_tokenizer_and_lang(tier0_frame: pd.DataFrame):
    out = tier0_table(tier0_frame)
    assert set(zip(out["tokenizer"], out["lang"], strict=True)) == {
        ("a", "zh"),
        ("a", "th"),
        ("b", "zh"),
    }


def test_tier0_subchar_rate_is_flags_over_tokens(tier0_frame: pd.DataFrame):
    out = tier0_table(tier0_frame).set_index(["tokenizer", "lang"])
    assert out.loc[("a", "zh"), "subchar_rate"] == pytest.approx(10 / 200)
    assert out.loc[("a", "th"), "subchar_rate"] == pytest.approx(25 / 50)
    assert out.loc[("b", "zh"), "subchar_rate"] == pytest.approx(0.0)


def test_tier0_cluster_violation_rate_is_flags_over_tokens(tier0_frame: pd.DataFrame):
    out = tier0_table(tier0_frame).set_index(["tokenizer", "lang"])
    assert out.loc[("a", "zh"), "cluster_violation_rate"] == pytest.approx(20 / 200)
    assert out.loc[("a", "th"), "cluster_violation_rate"] == pytest.approx(0.0)


def test_tier0_sums_tokens_within_a_group(tier0_frame: pd.DataFrame):
    out = tier0_table(tier0_frame).set_index(["tokenizer", "lang"])
    assert out.loc[("a", "zh"), "n_tokens"] == 200


def test_tier0_reports_dropped_chars(tier0_frame: pd.DataFrame):
    out = tier0_table(tier0_frame).set_index(["tokenizer", "lang"])
    assert out.loc[("b", "zh"), "dropped_chars"] == 7


def test_tier0_zero_tokens_yields_zero_not_a_zero_division():
    frame = pd.DataFrame(
        [stat_row("empty", lang="km", n_tokens=0, n_chars=0, midcodepoint=0, cluster_split=0)]
    )
    out = tier0_table(frame).set_index(["tokenizer", "lang"])
    assert out.loc[("empty", "km"), "subchar_rate"] == 0.0
    assert out.loc[("empty", "km"), "cluster_violation_rate"] == 0.0


def test_tier0_zero_tokens_cpt_is_nan_not_zero():
    """Zero tokens means cpt is undefined; 0.0 would be a lie about granularity."""
    frame = pd.DataFrame([stat_row("empty", lang="km", n_tokens=0, n_chars=0)])
    out = tier0_table(frame).set_index(["tokenizer", "lang"])
    assert math.isnan(out.loc[("empty", "km"), "cpt"])


def test_tier0_zero_tokens_alongside_real_rows_is_safe(tier0_frame: pd.DataFrame):
    frame = pd.concat(
        [tier0_frame, pd.DataFrame([stat_row("z", lang="km", n_tokens=0, n_chars=0)])],
        ignore_index=True,
    )
    out = tier0_table(frame).set_index(["tokenizer", "lang"])
    assert out.loc[("z", "km"), "subchar_rate"] == 0.0
    assert out.loc[("a", "th"), "subchar_rate"] == pytest.approx(0.5)


def test_tier0_cpt_is_chars_over_tokens(tier0_frame: pd.DataFrame):
    out = tier0_table(tier0_frame).set_index(["tokenizer", "lang"])
    assert out.loc[("a", "th"), "cpt"] == pytest.approx(150 / 50)


def test_tier0_is_sorted_by_lang_then_worst_first(tier0_frame: pd.DataFrame):
    out = tier0_table(tier0_frame)
    assert list(out["lang"]) == sorted(out["lang"])
    for _, grp in out.groupby("lang", sort=False):
        rates = list(grp["subchar_rate"])
        assert rates == sorted(rates, reverse=True)


# --------------------------------------------------------------------------
# 10. the tokenizer registry
# --------------------------------------------------------------------------


def test_all_tokenizers_is_non_empty():
    assert len(all_tokenizers()) > 0


def test_all_tokenizer_ids_are_unique():
    """A duplicate id would silently become two data points for one tokenizer."""
    ids = [s.tokenizer_id for s in all_tokenizers()]
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]


def test_all_tokenizers_matches_entries():
    assert all_tokenizers() == ENTRIES


def test_baselines_selector_is_exactly_the_three_baselines():
# improved
    assert [s.tokenizer_id for s in resolve(["@baselines"])] == ["char", "whole", "whitespace"]


def test_baselines_all_declare_the_baseline_family():
    assert {s.family for s in resolve(["@baselines"])} == {"baseline"}


def test_core_selector_is_non_empty():
    assert len(resolve(["@core"])) > 0


def test_core_selector_is_a_subset_of_all():
    core = {s.tokenizer_id for s in resolve(["@core"])}
    assert core <= {s.tokenizer_id for s in all_tokenizers()}


def test_all_selector_returns_every_entry():
    assert len(resolve(["@all"])) == len(ENTRIES)


def test_resolve_deduplicates_across_selectors():
    both = resolve(["@baselines", "char", "@all"])
    ids = [s.tokenizer_id for s in both]
    assert len(ids) == len(set(ids)) == len(ENTRIES)


def test_resolve_preserves_request_order():
    assert [s.tokenizer_id for s in resolve(["whole", "char"])] == ["whole", "char"]


def test_resolve_unknown_id_raises_key_error():
    with pytest.raises(KeyError):
        resolve(["not-a-tokenizer"])


def test_resolve_unknown_id_names_the_known_ids():
    with pytest.raises(KeyError) as excinfo:
        resolve(["not-a-tokenizer"])
    message = str(excinfo.value)
    assert "not-a-tokenizer" in message
    for known in ("char", "whole", "cl100k_base"):
        assert known in message
# improved


def test_get_tokenizer_spec_unknown_id_names_the_known_ids():
    with pytest.raises(KeyError) as excinfo:
        get_tokenizer_spec("nope")
    assert "cl100k_base" in str(excinfo.value)


def test_resolve_unknown_selector_raises_key_error():
    with pytest.raises(KeyError) as excinfo:
        resolve(["@nonesuch"])
    assert "@nonesuch" in str(excinfo.value)


# def test_every_documented_selector_resolves():
    for name in SELECTORS:
        assert len(resolve([name])) > 0, name


def test_specs_with_aliases_have_a_non_empty_family():
    """An alias collapses several repos into one entry; the family is how the
    collapsed entry is still findable by `@family` selector."""
    for spec in all_tokenizers():
        if spec.aliases:
            assert spec.family, spec.tokenizer_id


def test_every_spec_has_a_non_empty_family():
    for spec in all_tokenizers():
        assert spec.family, spec.tokenizer_id


def test_aliases_never_collide_with_a_registered_id():
    ids = {s.tokenizer_id for s in all_tokenizers()}
    for spec in all_tokenizers():
        assert not (set(spec.aliases) & ids), spec.tokenizer_id


def test_aliases_are_unique_across_the_registry():
    seen: list[str] = []
    for spec in all_tokenizers():
        seen.extend(spec.aliases)
    assert len(seen) == len(set(seen))


def test_family_selectors_resolve_to_that_family():
    families = {s.family for s in all_tokenizers()} - {"baseline"}
    for family in sorted(families):
        assert {s.family for s in resolve([f"@{family}"])} == {family}

# Enhanced

# Enhanced
