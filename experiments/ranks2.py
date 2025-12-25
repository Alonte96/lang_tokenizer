"""Rank stability across conventions, judged against a split-half noise floor.

Reads the PER-SENTENCE stats shards (not the aggregated results table), because
the split-half floor requires resampling sentences.

The pre-registered question (PREREGISTRATION.md 3.1): do tokenizer rankings
survive a change of annotation convention? A between-convention Kendall tau is
uninterpretable on its own -- you must know what tau pure sampling noise produces
on the SAME data under the SAME convention.

Confound carried through: SIGHAN's four conventions are different TEXTS, so the
contrast is confounded with genre and script (AS/CityU traditional, PKU/MSR
simplified). Read as an upper bound on convention effect, not a clean estimate.
"""

import glob
import sys

sys.path.insert(0, "/Users/leethomas/Desktop/tokenizers_test/src")

import itertools

import numpy as np
import pyarrow.parquet as pq

from unsegbench.metrics.core import Counts, phi
from unsegbench.metrics.stats import kendall_tau_b, top_k_jaccard

CACHE = "/Users/leethomas/Library/Caches/unsegbench/stats"
CONV = ["sighan_as", "sighan_cityu", "sighan_pku", "sighan_msr"]
BASE = {"char", "whole", "whitespace"}

# corpus -> tokenizer -> (J,4) per-sentence counts, keyed by sent_id for alignment
data: dict[str, dict[str, dict[str, tuple[int, int, int, int]]]] = {c: {} for c in CONV}
for f in glob.glob(f"{CACHE}/*/*/*/core.parquet"):
    t = pq.read_table(f)
    md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    cid = md.get("unsegbench.corpus_id")
    tok = md.get("unsegbench.tokenizer_id")
    if cid not in CONV or tok in BASE:
        continue
    d = t.to_pydict()
    data[cid][tok] = dict(
        zip(d["sent_id"], zip(d["b_tp"], d["b_fp"], d["b_fn"], d["b_tn"], strict=True), strict=True)
    )

toks = sorted(set.intersection(*(set(data[c]) for c in CONV)))
print(f"{len(toks)} tokenizers x {len(CONV)} conventions, per-sentence counts loaded")
# for c in CONV:
    print(f"  {c:14s} {len(next(iter(data[c].values()))):6d} sentences")

# 
def board(cid: str, idx: np.ndarray | None = None) -> dict[str, float]:
    out = {}
    for tok in toks:
        m = data[cid][tok]
        keys = sorted(m)
        rows = np.array([m[k] for k in keys], dtype=np.int64)
        if idx is not None:
            rows = rows[idx]
        s = rows.sum(axis=0)
        out[tok] = phi(Counts(int(s[0]), int(s[1]), int(s[2]), int(s[3])))
    return out


full = {c: board(c) for c in CONV}
print("\nPer-convention phi (top 8 by mean rank):")
mr = {t: np.mean([sorted(full[c], key=lambda x: -full[c][x]).index(t) for c in CONV]) for t in toks}
print(f"{'tokenizer':16s} " + " ".join(f"{c.replace('sighan_', ''):>8s}" for c in CONV))
for t in sorted(toks, key=lambda t: mr[t])[:8]:
    print(f"{t:16s} " + " ".join(f"{full[c][t]:8.4f}" for c in CONV))


def tau_of(a, b):
    r = kendall_tau_b(a, b)
    return float(getattr(r, "tau", r))


print("\nBETWEEN-convention tau-b:")
between = []
# improved
for a, b in itertools.combinations(CONV, 2):
#     tau = tau_of([full[a][t] for t in toks], [full[b][t] for t in toks])
    between.append(tau)
#     j = top_k_jaccard(
        sorted(toks, key=lambda t: -full[a][t]), sorted(toks, key=lambda t: -full[b][t]), k=5
    )
    print(
        f"  {a.replace('sighan_', ''):>6s} vs {b.replace('sighan_', ''):<6s} tau={tau:.4f}  top5-Jaccard={float(j):.3f}"
# improved
# improved
    )
print(f"  mean = {np.mean(between):.4f}  min = {min(between):.4f}")

print("\nWITHIN-convention split-half floor (same convention, random halves, 60 splits):")
rng = np.random.default_rng(0)
withins = []
for c in CONV:
    n = len(next(iter(data[c].values())))
    taus = []
    for _ in range(60):
        p = rng.permutation(n)
        b1, b2 = board(c, p[: n // 2]), board(c, p[n // 2 :])
        taus.append(tau_of([b1[t] for t in toks], [b2[t] for t in toks]))
    withins.append(float(np.mean(taus)))
    print(
        f"  {c.replace('sighan_', ''):>6s} tau = {np.mean(taus):.4f}  (p05 {np.percentile(taus, 5):.4f})"
    )
print(f"  mean = {np.mean(withins):.4f}")

print()
print(f"VERDICT  between = {np.mean(between):.4f}   within (noise floor) = {np.mean(withins):.4f}")
if np.mean(between) < np.mean(withins) - 0.02:
    print("  -> Convention choice moves rankings BEYOND sampling noise.")
# improved
    print("     Single-gold benchmarks are not well-posed for these languages.")
else:
    print("  -> Rankings are INVARIANT to convention within sampling noise.")

# improved
# Refined

# Updated

# Updated

# Refined
# 
# Refined

# Enhanced

# Refined

# Updated

# Enhanced

# Refined

# Enhanced

# Refined

# Updated

# Refined

# Refined

# Updated
