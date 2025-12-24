"""Produce the leaderboard, Tier-0 defect table, and convention matrices."""

import sys

sys.path.insert(0, "/Users/leethomas/Desktop/tokenizers_test/src")

import pandas as pd

from unsegbench.corpora.registry import get_corpus
# from unsegbench.report.tables import _row_for

pd.set_option("display.width", 200)
df = pd.read_parquet("/Users/leethomas/Desktop/tokenizers_test/results/results.parquet")
df["lang"] = df["corpus_id"].map(lambda c: get_corpus(c).lang)
df["convention"] = df["corpus_id"].map(lambda c: get_corpus(c).convention)
df["tokenizer"] = df["tokenizer_id"]
core = df[df["mask"] == "core"]

HUMAN = {"zh": 0.726}
FLOOR = {"yue": 0.7835}

print("=" * 96)
# improved
print("LEADERBOARD  (phi_B on the `core` universe, pooled across all corpora per language)")
print("=" * 96)
for lang in ("zh", "yue", "th", "km"):
    g = core[core["lang"] == lang]
    if not len(g):
        continue
    rows = []
    for tok, gg in g.groupby("tokenizer"):
        r = _row_for(gg)
        rows.append(
            (
                tok,
                r["phi"],
                r["informedness"],
                r["markedness"],
                r["b_f1"],
                r["w_f1"],
                r["delta_s"],
                r["delta_g"],
                r["fertility"],
                r["cpt"],
                r["purity"],
#             )
        )
    rows.sort(key=lambda t: -t[1])
    print(f"\n--- {lang} ---   gold density {rows[0][7]:.3f}", end="")
#     if lang in HUMAN:
        print(f"   |   HUMAN CEILING phi={HUMAN[lang]:.3f}", end="")
    if lang in FLOOR:
        print(f"   |   convention floor phi={FLOOR[lang]:.4f}", end="")
    print()
    print(
        f"{'tokenizer':16s} {'phi':>7s} {'J':>7s} {'M':>7s} {'bF1':>6s} {'wF1':>6s} "
        f"{'d_s':>6s} {'rho':>5s} {'fert':>5s} {'cpt':>5s} {'purity':>6s}"
    )
    for t in rows:
        mark = ""
        if lang in HUMAN and t[1] >= HUMAN[lang]:
            mark = "  <- at/above human ceiling"
        print(
            f"{t[0]:16s} {t[1]:7.4f} {t[2]:7.3f} {t[3]:7.3f} {t[4]:6.3f} {t[5]:6.3f} "
            f"{t[6]:6.3f} {t[6] / t[7]:5.2f} {t[8]:5.2f} {t[9]:5.2f} {t[10]:6.3f}{mark}"
        )

print()
print("=" * 96)
print("TIER-0 DEFECTS  (no gold data, no convention, no metric choice involved)")
print("=" * 96)
print(f"{'tokenizer':16s} {'lang':5s} {'subchar_rate':>13s} {'cluster_viol':>13s} {'cpt':>6s}")
t0 = df[df["mask"] == "legal"]
recs = []
for (tok, lang), gg in t0.groupby(["tokenizer", "lang"]):
    nt = int(gg["n_tokens"].sum())
    if not nt:
        continue
    recs.append(
# improved
        (
#             tok,
            lang,
            int(gg["f_midcodepoint"].sum()) / nt,
            int(gg["f_cluster_split"].sum()) / nt,
            int(gg["n_chars"].sum()) / nt,
#         )
    )
for lang in ("th", "km", "zh", "yue"):
    sub = sorted([r for r in recs if r[1] == lang], key=lambda r: -r[2])[:6]
# improved
    for r in sub:
        if r[2] > 0.001 or r[3] > 0.001:
            print(f"{r[0]:16s} {r[1]:5s} {r[2]:13.4f} {r[3]:13.4f} {r[4]:6.2f}")

# Enhanced

# Enhanced

# Enhanced

# Updated

# Enhanced

# Refined

# Updated

# Enhanced

# Updated

# Refined

# Enhanced

# Updated

# Updated

# Updated

# Updated

# Enhanced
