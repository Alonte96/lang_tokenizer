"""E3: isolate SCRIPT from CONVENTION.

The SIGHAN rank-stability result is triple-confounded: the four corpora differ in
annotation convention, genre AND script (AS/CityU traditional, PKU/MSR
simplified). The observed pattern -- high tau within a script, low across it --
is exactly what a pure script effect would produce, so the convention claim
cannot stand without this control.

UD_Chinese-GSD and UD_Chinese-GSDSimp are the SAME 4,997 sentences under the SAME
annotation convention, differing ONLY in traditional vs simplified script. Any
rank instability here is script alone, with convention held constant.
"""

import glob
import sys

sys.path.insert(0, "/Users/leethomas/Desktop/tokenizers_test/src")

import numpy as np
import pyarrow.parquet as pq

from unsegbench.metrics.core import Counts, phi
from unsegbench.metrics.stats import kendall_tau_b, top_k_jaccard

CACHE = "/Users/leethomas/Library/Caches/unsegbench/stats"
PAIRS = [
    ("ud_zh_gsd", "ud_zh_gsdsimp", "SCRIPT  (same sentences, same convention, trad vs simp)"),
    ("ud_zh_hk", "ud_yue_hk", "LANGUAGE (parallel sentences, zh vs yue rendering)"),
]
BASE = {"char", "whole", "whitespace"}

store: dict[str, dict[str, np.ndarray]] = {}
for f in glob.glob(f"{CACHE}/*/*/*/core.parquet"):
    t = pq.read_table(f)
    md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    cid, tok = md.get("unsegbench.corpus_id"), md.get("unsegbench.tokenizer_id")
    if tok in BASE:
        continue
    if cid not in {c for p in PAIRS for c in p[:2]}:
        continue
    d = t.to_pydict()
    store.setdefault(cid, {})[tok] = np.array(
        list(zip(d["b_tp"], d["b_fp"], d["b_fn"], d["b_tn"], strict=True)), dtype=np.int64
    )


def board(cid):
    return {t: phi(Counts(*map(int, m.sum(axis=0)))) for t, m in store[cid].items()}


def tau_of(a, b):
    r = kendall_tau_b(a, b)
    return float(getattr(r, "tau", r))


for a, b, label in PAIRS:
    if a not in store or b not in store:
        print(f"skip {a}/{b}: not scored")
        continue
    ba, bb = board(a), board(b)
    toks = sorted(set(ba) & set(bb))
    tau = tau_of([ba[t] for t in toks], [bb[t] for t in toks])
    j = top_k_jaccard(sorted(toks, key=lambda t: -ba[t]), sorted(toks, key=lambda t: -bb[t]), k=5)
    dphi = np.mean([abs(ba[t] - bb[t]) for t in toks])
    print(f"\n=== {label} ===")
    print(f"  {a} vs {b}:  n_tok={len(toks)}")
    print(f"  Kendall tau-b       = {tau:.4f}")
    print(f"  top5-Jaccard        = {float(j):.3f}")
    print(f"  mean |delta phi|    = {dphi:.4f}")
    print(f"  {'tokenizer':16s} {'A':>8s} {'B':>8s} {'diff':>8s}")
    for t in sorted(toks, key=lambda t: -abs(ba[t] - bb[t]))[:6]:
        print(f"  {t:16s} {ba[t]:8.4f} {bb[t]:8.4f} {ba[t] - bb[t]:+8.4f}")

print()
print("Compare against the SIGHAN numbers:")
print("  between-convention tau (SIGHAN, confounded)      = 0.6973")
print("  within-convention split-half floor               = 0.9769")
print("  If the SCRIPT-only tau above is also ~0.70, the SIGHAN result is")
print("  substantially a script effect, not a convention effect.")

# Enhanced

# Enhanced
