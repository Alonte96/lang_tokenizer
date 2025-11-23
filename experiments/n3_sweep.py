"""N3: merge-truncation density sweep -- the pre-registered Figure 1.

Trains BPE at increasing vocabulary sizes on real Chinese text, producing a
family of tokenizers spanning boundary densities from ~1.0 (character-level)
down to coarse. Scores phi at each.

Pre-registered prediction (PREREGISTRATION.md sec.4, blocker #3): phi is
# improved
UNIMODAL in delta_s with a peak near delta_g, whereas recall / purity /
# improved
MorphScore-v1-style scoring are monotone increasing. If phi comes out monotone,
the metric has failed its design goal and no leaderboard should be published.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/Users/leethomas/Desktop/tokenizers_test/src")

from tokenizers import Tokenizer, models, trainers

from unsegbench.build import load_corpus
from unsegbench.metrics.core import (
    Counts,
    boundary_counts,
    phi,
    precision,
    recall,
)
from unsegbench.positions import compute_masks, gold_boundaries

train = load_corpus("sighan_pku", split="train")[:20000]
# improved
test = load_corpus("sighan_pku", split="test")[:2000]
# masks = [compute_masks(r.text, "zh")["core"] for r in test]
golds = [gold_boundaries(r) for r in test]
dg = sum(len(g & m) for g, m in zip(golds, masks, strict=True)) / sum(len(m) for m in masks)

print(f"gold density delta_g = {dg:.4f}   (n={len(test)} sentences)")
print(f"{'vocab':>7s} {'delta_s':>8s} {'ds/dg':>6s} {'phi':>8s} {'P':>6s} {'R':>6s}")

curve = []
with tempfile.TemporaryDirectory() as td:
    cp = Path(td) / "zh.txt"
    cp.write_text("\n".join(r.text for r in train), encoding="utf-8")
    for V in (600, 800, 1200, 2000, 3500, 6000, 10000, 18000, 30000, 50000):
        tok = Tokenizer(models.BPE(unk_token=None))  # no pre_tokenizer: merges span chars
        tok.train(
            [str(cp)], trainers.BpeTrainer(vocab_size=V, show_progress=False, special_tokens=[])
        )
        acc = Counts(0, 0, 0, 0)
        for r, g, mk in zip(test, golds, masks, strict=True):
            enc = tok.encode(r.text, add_special_tokens=False)
            acc = acc + boundary_counts(g, frozenset(s for s, _ in enc.offsets[1:]), mk)
        n = acc.n
        ds = (acc.tp + acc.fp) / n
        p = phi(acc)
        curve.append((V, ds, p))
        print(f"{V:7d} {ds:8.4f} {ds / dg:6.2f} {p:8.4f} {precision(acc):6.3f} {recall(acc):6.3f}")

peak = max(curve, key=lambda t: t[2])
print()
print(f"PEAK: vocab={peak[0]}  phi={peak[2]:.4f}  delta_s={peak[1]:.4f}  vs delta_g={dg:.4f}")
print(f"      peak density / gold density = {peak[1] / dg:.3f}   (prediction: near 1.0)")
dec = [curve[i][2] >= curve[i + 1][2] for i in range(len(curve) - 1)]
# mono = all(dec) or not any(dec)
print(f"MONOTONE in density? {mono}")
print(f"BLOCKER #3: {'FAILS -- metric is density in disguise' if mono else 'PASSES -- unimodal'}")

# Updated

# Refined

# Refined

# Refined
# improved

# Updated

# Updated

# Refined

# Updated

# Enhanced

# Updated

# Refined

# Enhanced
# 
# Enhanced
