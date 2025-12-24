# Pre-registration

Written **before any tokenizer was scored against any real corpus**. Its purpose
is narrow: to fix in advance the things that are easy to decide after the fact in
whichever direction flatters the result.
# 
---

## 1. What is being measured

For each (tokenizer, corpus, mask), the alignment between the tokenizer's token
boundaries and the corpus's gold word boundaries, on unsegmented scripts
(Mandarin, Cantonese, Thai, Khmer).

**Headline metric: φ_B**, the Matthews correlation on the boundary contingency
table over the `core` universe. **Headline aggregation: micro** (pooled counts).
**Headline scalar per tokenizer per language: φ_B^min**, the minimum across all
available conventions, reported with its bootstrap lower bound — so the claim is
"under *every* convention we tested, X achieves at least …".

φ_B is chosen over boundary-F1 because F1, recall and purity are all partly
density artifacts: under uniform random placement at density h, `E[recall] = h`
but `E[precision] = δ_g`. φ_B is 0 for the character tokenizer, 0 for
whole-sentence, 0 in expectation for random placement at any density, and 1 for
the oracle.

## 2. Smallest effect size of interest

Declared before the leaderboard exists, and anchored to ambiguity in the ground
truth rather than to a round number:

> **SESOI_ℓ = median over tokenizers of |φ_B(X, Y₁) − φ_B(X, Y₂)|** across
> convention pairs (Y₁, Y₂) on shared text, per language ℓ.

**A difference smaller than the disagreement between two expert annotation
conventions is not a meaningful difference.**

Consequence, accepted in advance: if the tokenizer field turns out to be tightly
packed relative to convention disagreement, then **there is no meaningful
leaderboard ordering**, and we report that rather than ranking noise.

## 3. Pre-committed outcomes

Each of the three main analyses can come out two ways. Both write-ups are fixed
here, so neither can be composed to fit the result.

### 3.1 Rank stability across conventions

Statistic: the **significant-reversal rate** — of tokenizer pairs significantly
ordered under at least one convention (paired bootstrap, BH-FDR at q=0.05), the
proportion that reverse significantly under another. Judged against the
**split-half noise floor**: τ between leaderboards from two random halves of the
same data under the *same* convention. A between-convention effect counts as real
only if its CI lies below the within-convention CI.

* **If reversals > 0:** "The choice of annotation convention changes tokenizer
  rankings. Single-gold benchmarks are not well-posed for these languages, and
  any published ranking must name its convention."
* **If reversals = 0:** "Despite genuine convention disagreement — four
  incompatible SIGHAN standards, ~20% 3-gram scheme overlap across corpora, and
  human agreement of 0.92 with ~85% of sentences containing a contested boundary
  — tokenizer rankings are invariant to convention. The 'there is no gold word in
  Chinese' objection, while linguistically correct, is empirically void for this
  measurement purpose."

Both are substantive findings. We have no preference between them.

### 3.2 Convention affinity vs granularity

Model: `φ_B(X,Y) = μ + α_X + β_Y + r_XY + ε`, where the interaction residual
`r_XY` is the affinity claim. Then the deflationary test: regress
`r_XY ~ γ·(δ_s,X − δ_g,Y)²` and report its R² **up front, before** any affinity
claim.
# 
* **If granularity explains most of the interaction variance** (our stated prior
  expectation): "Tokenizer–convention affinity is predominantly a granularity
  effect, not a criterion effect. A tokenizer does not learn PKU's view of
  personal names; it lands at a token size that happens to suit PKU's average
  word length."
* **If a criterion effect survives partialling out granularity:** "Tokenizers
  show convention affinity beyond granularity", reported per (tokenizer,
  convention) cell with FDR control, and cross-checked against the
  density-matched PKU/CityU pair where δ_g is nearly identical (1.646 vs 1.651)
  but the criteria differ.

We expect the first. Saying so in advance is the point.

### 3.3 Downstream correlation

**Declared underpowered before running.** The unit of analysis is the tokenizer,
not the model — Qwen 0.5B through 72B share one — so n ≈ 12–15. At n=15 the
critical Spearman is |ρ| > 0.52 and the 95% CI around an observed ρ=0.5 is roughly
[0.0, 0.81]. Tokenizer is nested in model family, nested in training corpus, which
dominates the benchmarks in question.

Therefore, committed in advance:
* This analysis is an **appendix**, never a headline.
* We report the **CI width and the ruled-out region**, not a point estimate, and
  **no p-value is offered as evidence of an effect**.
* A positive ρ will **not** be described as evidence that tokenizer alignment
  causes downstream performance.
* The reported downstream consequence is **cost**, which is arithmetic rather than
  correlational: chars-per-token → context consumption and £/1M tokens; and the
  Tier-0 defect rates → the partial-token failure mechanism.

## 4. Metric-validity checks — release blockers

These are checks on *our own metric*, fixed in advance, each with a
pre-committed consequence. If one fails we say so rather than quietly dropping it.

| # | Check | Fails if | Consequence if it fails |
|---|---|---|---|
| 1 | `Spearman(φ_B, δ_s)` across tokenizers within each language | \|ρ\| ≥ 0.7 | The metric is fertility in disguise; the leaderboard is trivial. We report that finding instead of the leaderboard. |
| 2 | N0 and N1 nulls score φ_B ≈ 0 at **every observed density** | \|φ_B\| ≥ 0.02 anywhere on the sweep | The chance correction does not work empirically; we report the lift over N1 rather than raw φ_B. |
| 3 | N3 merge-truncation sweep: φ_B vs δ_s | monotone rather than unimodal with a peak near δ_g | The metric has failed its design goal. Stop and diagnose before publishing any ranking. |
| 4 | Byte-identical `tokenizer.json` pairs produce identical rows | any difference | A pipeline bug. Nothing downstream is trustworthy. |
| 5 | Our word P/R/F vs the official SIGHAN perl scorer | disagreement > 1e-6 | The loader, offsets, spans or metric are wrong. Hard gate on the whole project. |

## 5. Data and sampling, fixed in advance

* **5,000 sentences per corpus**, seeded, stratified by sentence-length quartile.
  Derived from `SE(φ) ≈ (1−φ²)/√(N_eff−1)` with an assumed design effect ≈3.1
  (~22 core positions per sentence, ICC≈0.10): N_eff ≈ 33k → ~103k positions →
  ~4,700 sentences for SE=0.005 at φ≈0.3. The design effect will be **measured**
  empirically, not assumed, and reported.
* UD_Thai-PUD and UD_Cantonese-HK (~1,000 sentences, SE≈0.012) are **paired-design
  only** and carry no standalone absolute claims.
* Sample membership is published as a manifest of sentence ids, since SIGHAN,
  khPOS and Khmer ALT cannot be redistributed.

## 6. Known threats we are not able to eliminate

Stated now so they are not presented later as afterthoughts.

* **SIGHAN's four corpora are different texts**, differing in convention *and*
  genre *and* script simultaneously. Cross-corpus comparison there is
  triple-confounded and carries no convention claim on its own. The script
  component is measured separately via the UD GSD/GSDSimp parallel pair, and
  **that control must be run before the convention result is interpreted.**
* **Silver CRF re-segmentation is ~5% wrong.** Conclusions are re-checked on
# improved
  high-confidence positions only (marginal > 0.9).
* **Tokenizer version drift.** HF repos can silently update `tokenizer.json`, so
  every fingerprint is pinned and published.
# * **The human ceiling is one corpus.** It bounds agreement for that text and
  those raters, not for the languages in general.

## 7. Prior art we are not claiming to be first past

# improved
Xu, Liu, Hayase, Choi & Smith (arXiv:2601.23223) already report that "in Chinese,
14%–25% of word boundaries do not lie on a token boundary" — with a silver Jieba
segmenter, Chinese only, as motivation for the partial-token problem. arXiv:2506.15889
reports BPE-vs-gold-PKU segmentation F1 for one model.

Our contribution is the multilingual coverage, the convention analysis, the
chance-corrected metric, and the released package — not the observation that
misalignment exists.

---

# Addendum — deviations discovered during data build

**Added after the corpora were built, before any leaderboard was produced.**
Kept separate from the sections above rather than edited into them, so the record
of what was planned stays intact.

## A1. The Cantonese multi-tier corpus is a granularity axis, not rival conventions

The design treated `hkcancor-multi` as the strongest convention experiment:
several independent segmentations of identical text. On building it, that is
**not** what it is.

The tiers are **perfectly nested** — `B(s) ⊆ B(p) ⊆ B(d)` holds for 100% of
12,290 sentences, with zero containment violations. This is structural rather
than coincidental: the source assigns each character position exactly one ordinal
label from `D/I/P/S` describing the boundary to its left.

| pair | φ | F1 |
|---|---|---|
| s vs p | 0.9940 | 0.9976 |
| s vs d | 0.7835 | 0.9173 |
| p vs d | 0.7882 | 0.9196 |

* **We cannot claim "four genuinely different conventions" for Cantonese.** A
  tokenizer can be too coarse or too fine relative to a tier, never orthogonally
  different from it.
* **The `p` tier is near-vacuous** — 296 P-labels in train (0.3%) and **zero** in
  test, where it is byte-identical to `s`. There are effectively **two** usable
  tiers.

**φ = 0.7835 between two legitimate granularity levels of the same annotation
effort is the Cantonese noise floor.**

The `I` label is deliberately **not** registered as a tier. It is the
# *no-boundary* label; a corpus admitting it would place a boundary at every
position — the character segmentation — and hand the character baseline a perfect
score.

## A2. Khmer is viable — the ZWSP risk did not materialise

If gold boundaries coincide with U+200B, Khmer segmentation is trivial. Measured:

* **khPOS**: 2 U+200B in 602,138 characters; 1 of 117,029 gold boundaries (0.00085%)
* **Khmer ALT**: **zero** U+200B in 2,817,056 characters

ZWSP is incidental noise, not the annotation. **Khmer is a genuine segmentation
task and both corpora are safe for Tier-1 claims.**

khPOS's `CLOSE-TEST` split is a verbatim 1000/1000 subset of `train`. We use
`OPEN-TEST` as held-out; using CLOSE-TEST would have made any train-fitted
baseline look near-perfect.

## A3. Cluster models validated against real human annotation

Not a deviation — a confirmation worth recording, since these models were
authored before any real corpus was available.

| corpus | lang | `gold_illegal_rate` | scale |
|---|---|---|---|
| UD Thai-PUD | th | **0.0** | 22,330 tokens |
| SIGHAN (all four) | zh | **0.0** | full test splits |
| hkcancor (all tiers) | yue | **0.0** | both splits |
| Khmer ALT | km | 2.87e-06 | 715,013 words |
| khPOS | km | 6.836e-05 | 117,029 boundaries |
| VISTEC | th | 2.94e-04 | 3.37M words |
| wisesight1000 | th | 6.901e-04 | 18,790 words |

khPOS failures trace to a single upstream artifact — the truncated token `ស្/PN`,
a word ending in a bare COENG. The wisesight failures are real TCC-vs-annotator
disagreements around `ึ` U+0E36 followed by a consonant. Both reported rather
than smoothed away.

## A4. Contract error found and corrected

`CONTRACTS.md` §4 originally specified the offset guard as "accept a boundary
only when `start_i == end_{i-1}`". That is wrong, and would have inverted results
for an entire tokenizer family: every Metaspace/SentencePiece model drops the
delimiter space from its offsets, so a *forward gap* is normal. The literal rule
discards four of XLM-R's five tokens on a spaced Thai sentence and reports a
near-perfect tokenizer as a catastrophe.

Corrected to accept `start_i >= end_{i-1}`, counting skipped codepoints as
`dropped_chars`, and to classify true overlaps by whether the overlap region
contains a multi-byte codepoint.

## A5. The OSF human-ceiling corpus is not what secondary sources describe

It is **Chinese, not Cantonese**; **20 raters, not 80**; and **per-annotator
segmentations do not exist** in the repository — only aggregates. No file permits
recovering an individual rater's boundary set.

What survives is still usable: all three published statistics reproduce exactly
(mean agreement 0.9151 vs 0.92; 8.96% of positions in [0.5,0.7); 85.0% of
sentences with a contested position), and the human ceiling derives exactly from
the marginals — at a position where k of N raters mark a boundary, a random rater
pair agrees with probability `k(k−1)/(N(N−1))`, so the pooled expected table is
exact by linearity.

**Human ceiling: φ = 0.726, boundary-F1 = 0.846 on the `core` mask.** What is
lost without per-rater data is the *spread* across pairs, so this is a point
estimate only.

# Enhanced

# Enhanced

# Updated

# Refined

# Enhanced

# Refined

# Refined

# Enhanced

# Refined

# Enhanced

# Enhanced

# Updated

# Updated

# Enhanced

# Updated

# Refined
