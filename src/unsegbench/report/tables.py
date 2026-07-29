"""Leaderboard and analysis tables.

Reads only the per-sentence integer counters the runner persisted, so nothing
here ever loads a tokenizer or a corpus. That is why the whole report regenerates
in seconds and why changing a table costs nothing.

Two presentation rules are enforced here rather than left to the caller, because
both are easy to get wrong in a way that overstates the result:

1. **Every row carries its density context.** A phi without its boundary density
   ratio invites the reader to compare tokenizers that chop at completely
   different granularities as though the comparison were like-for-like.
2. **Ranks are intervals, not integers.** The leaderboard reports "Rank 1-4",
   because that is what the data supports. A table of unique integer ranks
   implies a precision the bootstrap does not license.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from unsegbench.metrics.core import Counts, compute_row

__all__ = [
    "REPORT_COLUMNS",
    "convention_table",
    "leaderboard",
    "markdown_table",
    "tier0_table",
]

#: Column order for the headline leaderboard. `rho`, `delta_s` and `fertility`
#: are NOT optional -- they are the context that makes `phi` interpretable.
REPORT_COLUMNS: tuple[str, ...] = (
    "tokenizer",
    "phi",
    "rank_lo",
    "rank_hi",
    "informedness",
    "markedness",
    "b_f1",
    "w_f1",
    "boundary_miss_rate",
    "purity",
    "delta_s",
    "delta_g",
    "rho",
    "fertility",
    "cpt",
)


def _pool(df: pd.DataFrame) -> Counts:
    return Counts(
        tp=int(df["b_tp"].sum()),
        fp=int(df["b_fp"].sum()),
        fn=int(df["b_fn"].sum()),
        tn=int(df["b_tn"].sum()),
    )


def _row_for(df: pd.DataFrame) -> dict[str, float]:
    c = _pool(df)
    row = compute_row(
        c,
        w_tp=int(df["w_tp"].sum()),
        w_pred=int(df["w_pred"].sum()),
        w_gold=int(df["w_gold"].sum()),
        w_intact=int(df["w_intact"].sum()),
        n_tokens=int(df["n_tokens"].sum()),
        n_chars=int(df["n_chars"].sum()),
        n_gold_words=int(df["n_gold_words"].sum()),
        crossing=int(df["crossing_tokens"].sum()),
    )
    return {f: getattr(row, f) for f in row.__slots__}


def leaderboard(
    stats: pd.DataFrame,
    *,
    mask: str = "core",
    rank_intervals: dict[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Per-tokenizer leaderboard for one (language, corpus, mask).

    Args:
        stats: long frame of per-sentence counters with a ``tokenizer`` column.
        mask: which universe to report. Defaults to ``core``, the headline.
        rank_intervals: ``{tokenizer: (lo, hi)}`` from
            `unsegbench.metrics.stats.rank_intervals`. Omitting these produces a
            table without rank columns rather than one with fake precision.

    Returns:
        A frame ordered by descending phi.
    """
    df = stats[stats["mask"] == mask] if "mask" in stats.columns else stats
    rows = []
    for tok, grp in df.groupby("tokenizer", sort=False):
        r = _row_for(grp)
        r["tokenizer"] = tok
        if rank_intervals and tok in rank_intervals:
            r["rank_lo"], r["rank_hi"] = rank_intervals[tok]
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("phi", ascending=False).reset_index(drop=True)
    cols = [c for c in REPORT_COLUMNS if c in out.columns]
    return out[cols]


def tier0_table(stats: pd.DataFrame) -> pd.DataFrame:
    """Gold-free defect rates. Needs no annotation convention at all.

    ``subchar_rate`` is the fraction of token boundaries that fall strictly
    inside a Unicode codepoint; ``cluster_violation_rate`` the fraction that fall
    inside an orthographic cluster. Thai and Khmer characters are three bytes in
    UTF-8, so byte-level BPE splits them routinely.

    These numbers depend on no gold data, no convention and no metric choice, so
    they survive every objection that can be raised against the rest of the
    benchmark. Report them first.
    """
    rows = []
    for (tok, lang), grp in stats.groupby(["tokenizer", "lang"], sort=False):
        n_tok = int(grp["n_tokens"].sum())
        rows.append(
            {
                "tokenizer": tok,
                "lang": lang,
                "n_tokens": n_tok,
                "subchar_rate": int(grp["f_midcodepoint"].sum()) / n_tok if n_tok else 0.0,
                "cluster_violation_rate": (
                    int(grp["f_cluster_split"].sum()) / n_tok if n_tok else 0.0
                ),
                "dropped_chars": int(grp["f_dropped_chars"].sum()),
                "cpt": int(grp["n_chars"].sum()) / n_tok if n_tok else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["lang", "subchar_rate"], ascending=[True, False])


def convention_table(
    stats: pd.DataFrame, *, mask: str = "core", floor: float | None = None
) -> pd.DataFrame:
    """Tokenizer x convention phi matrix for one language.

    Args:
        stats: long frame including ``tokenizer`` and ``convention`` columns.
        mask: universe to score on.
        floor: the inter-convention noise floor from
            `metrics.agreement.ConventionMatrix.floor`. When supplied, it is
            attached as frame metadata so downstream renderers can mark
            differences that fall below it -- a gap smaller than the disagreement
            between two expert conventions is not a real gap.
    """
    df = stats[stats["mask"] == mask] if "mask" in stats.columns else stats
    cells: dict[tuple[str, str], float] = {}
    for (tok, conv), grp in df.groupby(["tokenizer", "convention"], sort=False):
        cells[(tok, conv)] = _row_for(grp)["phi"]
    toks = sorted({t for t, _ in cells})
    convs = sorted({c for _, c in cells})
    mat = pd.DataFrame(
        [[cells.get((t, c), np.nan) for c in convs] for t in toks], index=toks, columns=convs
    )
    mat.attrs["noise_floor"] = floor
    mat.attrs["mask"] = mask
    return mat


def markdown_table(df: pd.DataFrame, *, floats: int = 3, max_rows: int | None = None) -> str:
    """Render a frame as a GitHub markdown table.

    Rank columns are merged into a single ``Rank`` cell (``1-4``) so the output
    cannot be misread as a strict ordering.
    """
    d = df.copy()
    if "rank_lo" in d.columns and "rank_hi" in d.columns:
        d["rank"] = [
            f"{int(lo)}" if lo == hi else f"{int(lo)}-{int(hi)}"
            for lo, hi in zip(d["rank_lo"], d["rank_hi"], strict=True)
        ]
        d = d.drop(columns=["rank_lo", "rank_hi"])
        d = d[["tokenizer", "rank", *[c for c in d.columns if c not in ("tokenizer", "rank")]]]
    if max_rows is not None:
        d = d.head(max_rows)
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "--" if pd.isna(v) else f"{v:.{floats}f}")
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)
    )
    return f"{header}\n{sep}\n{body}"


def sanity_checks(stats: pd.DataFrame, *, mask: str = "core") -> dict[str, object]:
    """The metric-validity checks that PREREGISTRATION.md declares as blockers.

    Returns a dict including ``phi_density_spearman``: the rank correlation
    between phi and predicted boundary density across tokenizers. If its
    magnitude reaches 0.7 the metric is fertility in disguise and the leaderboard
    is trivial -- in which case the pre-registered response is to report that
    finding instead of the leaderboard.
    """
    from scipy.stats import spearmanr

    out: dict[str, object] = {}
    df = stats[stats["mask"] == mask] if "mask" in stats.columns else stats
    per_lang: dict[str, float] = {}
    for lang, grp in df.groupby("lang", sort=False):
        rows = [_row_for(g) for _, g in grp.groupby("tokenizer", sort=False)]
        if len(rows) < 3:
            continue
        rho = spearmanr([r["phi"] for r in rows], [r["delta_s"] for r in rows]).statistic
        per_lang[str(lang)] = float(rho)
    out["phi_density_spearman"] = per_lang
    out["blocker_triggered"] = any(abs(v) >= 0.7 for v in per_lang.values())
    return out
