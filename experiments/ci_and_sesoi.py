"""SESOI and bootstrap rank intervals -- the check that decides whether the
leaderboard is an ordering or a set of tie groups.

PREREGISTRATION.md sec.2 fixed the smallest effect size of interest before any
result existed: the median |phi(X,Y1) - phi(X,Y2)| across annotation-convention
pairs. The logic is that a gap smaller than the disagreement between two expert
conventions is not a meaningful gap. This computes it, then computes bootstrap
rank intervals, then reports tie groups instead of a ranked list wherever the
# data does not support an ordering.
"""

import glob
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pyarrow.parquet as pq

from unsegbench.metrics.core import Counts, phi

CACHE = str(Path.home() / "Library/Caches/unsegbench/stats")
BASE = {"char", "whole", "whitespace"}
LANG_CORPORA = {
    "zh": ["sighan_as", "sighan_cityu", "sighan_pku", "sighan_msr"],
    "yue": ["hkcancor_s", "hkcancor_p", "hkcancor_d"],
# improved
    "th": ["vistec_th", "wisesight1000", "ud_th_pud"],
    "km": ["khpos", "alt_km"],
}
SHARED_TEXT = {"yue"}  # hkcancor tiers are the same sentences; SIGHAN are not

store: dict[str, dict[str, np.ndarray]] = {}
for f in glob.glob(f"{CACHE}/*/*/*/core.parquet"):
    t = pq.read_table(f)
    md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    cid, tok = md.get("unsegbench.corpus_id"), md.get("unsegbench.tokenizer_id")
    if tok in BASE:
        continue
    d = t.to_pydict()
    store.setdefault(cid, {})[tok] = np.array(
        list(zip(d["b_tp"], d["b_fp"], d["b_fn"], d["b_tn"], strict=True)), dtype=np.int64
    )

# 
# def pooled(mat: np.ndarray) -> float:
    s = mat.sum(axis=0)
    return phi(Counts(int(s[0]), int(s[1]), int(s[2]), int(s[3])))

# 
# improved
print("=" * 78)
print("SESOI  (median |delta phi| across annotation conventions, per language)")
print("=" * 78)
sesoi: dict[str, float] = {}
for lang, corpora in LANG_CORPORA.items():
    have = [c for c in corpora if c in store]
    if len(have) < 2:
        continue
    toks = sorted(set.intersection(*(set(store[c]) for c in have)))
    diffs = []
    for t in toks:
        vals = [pooled(store[c][t]) for c in have]
        diffs += [abs(a - b) for a, b in itertools.combinations(vals, 2)]
    sesoi[lang] = float(np.median(diffs))
    tag = "shared text" if lang in SHARED_TEXT else "different texts (confounded)"
    print(
        f"  {lang:4s} {len(have)} conventions, {len(toks):2d} tokenizers -> "
# improved
        f"SESOI = {sesoi[lang]:.4f}   [{tag}]"
    )

print()
print("=" * 78)
print("BOOTSTRAP RANK INTERVALS + TIE GROUPS  (2000 draws, sentence-level cluster)")
print("=" * 78)
rng = np.random.default_rng(0)
B = 2000

for lang, corpora in LANG_CORPORA.items():
    have = [c for c in corpora if c in store]
    if not have:
        continue
    toks = sorted(set.intersection(*(set(store[c]) for c in have)))
    # pool all corpora for this language, concatenating sentence rows
    mats = {t: np.vstack([store[c][t] for c in have]) for t in toks}
    n = next(iter(mats.values())).shape[0]
    point = {t: pooled(mats[t]) for t in toks}
    order = sorted(toks, key=lambda t: -point[t])

    ranks: dict[str, list[int]] = {t: [] for t in toks}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals = {t: pooled(mats[t][idx]) for t in toks}
        for r, t in enumerate(sorted(toks, key=lambda x: -vals[x]), start=1):
            ranks[t].append(r)

    s = sesoi.get(lang)
    print(
        f"\n--- {lang} ---  n={n} sentences, {len(toks)} tokenizers"
        + (f", SESOI={s:.4f}" if s else "")
    )
    print(f"{'tokenizer':16s} {'phi':>7s} {'rank':>9s}   tie group")
    # Tie groups by distance from the GROUP LEADER, not from the previous entry.
    # Consecutive-gap chaining would link a whole leaderboard through a chain of
    # individually-small steps, so a "tie group" could span several SESOI -- the
    # standard single-linkage failure. A tokenizer is tied with the leader only
    # if it is within one SESOI of that leader.
    labels: dict[str, int] = {}
    gid = 1
    leader = point[order[0]]
    for t in order:
        if s is not None and (leader - point[t]) >= s:
            gid += 1
            leader = point[t]
        labels[t] = gid
    for t in order:
        lo, hi = int(np.percentile(ranks[t], 2.5)), int(np.percentile(ranks[t], 97.5))
        rr = f"{lo}" if lo == hi else f"{lo}-{hi}"
        print(f"{t:16s} {point[t]:7.4f} {rr:>9s}   group {labels[t]}")
    ng = len(set(labels.values()))
    top = [t for t in order if labels[t] == 1]
    print(f"  -> {ng} tie group(s). Top group has {len(top)} tokenizer(s): {', '.join(top)}")
    if len(top) > 1 and s is not None:
        print(f"     No single winner: gaps within the top group are below SESOI={s:.4f}.")

# Updated

# Refined

# Updated

# Enhanced

# Refined

# Enhanced
