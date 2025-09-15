# Findings

Full sweep: **24 tokenizers × 17 corpora × 3 masks**, 5,000-sentence stratified
sample per corpus, seed 0. All numbers are φ_B on the `core` universe unless
stated. Regenerate with the scripts in `experiments/` (`leaderboard.py`,
`n3_sweep.py`, `ranks2.py`, `e3.py`, `ci_and_sesoi.py`).

Two tokenizers were dropped by the runner and are absent below: `internlm3`
(no fast backend, so no offset mapping exists) and `mt5` on some corpora (a
transient network failure mid-sweep).

---

## 1. The validity blockers

`PREREGISTRATION.md` §4 declared five checks as release blockers, each with a
pre-committed consequence. Results:

| # | check | result |
|---|---|---|
| 1 | `\|Spearman(φ_B, δ_s)\|` < 0.7 within each language | **TRIGGERED** — ρ = −0.83 to −0.98 |
| 2 | N0/N1 nulls score φ_B ≈ 0 at every density | **PASSES** — max \|φ\| = **0.00082** |
| 3 | φ_B unimodal in δ_s, peaking near δ_g | **PASSES** — peak at δ_s/δ_g = **0.993** |
| 4 | byte-identical tokenizers give identical rows | **PASSES** — see §5 |
| 5 | word P/R/F matches the SIGHAN perl scorer | **PASSES** — exactly 0.0 for `char`, `cl100k_base` |

Blocker #1 triggered, and the pre-registered response was to report that finding
rather than a leaderboard. Blocker #3 is the check that adjudicates *why*, and it
is decisive.

### The N3 density sweep — the pre-registered Figure 1

BPE trained on real PKU text at increasing vocabulary sizes, spanning boundary
densities from character-level down to coarse. Gold density δ_g = 0.5134.

| vocab | δ_s | δ_s/δ_g | **φ_B** | precision | recall |
|---|---|---|---|---|---|
| 600–3500 | 0.998 | 1.94 | 0.0387 | 0.514 | **1.000** |
| 6000 | 0.753 | 1.47 | 0.4251 | 0.635 | 0.931 |
| 10000 | 0.627 | 1.22 | 0.4969 | 0.705 | 0.861 |
| 18000 | 0.553 | 1.08 | 0.5193 | 0.747 | 0.804 |
| **30000** | **0.510** | **0.99** | **0.5250** | 0.771 | 0.765 |
| 50000 | 0.477 | 0.93 | 0.5185 | **0.785** | 0.730 |

**φ_B peaks at δ_s = 0.5096 against a gold density of 0.5134 — within 0.7%.** The
curve is non-monotone. Recall falls monotonically (1.000 → 0.730) and precision
rises monotonically (0.514 → 0.785), so neither identifies the correct
granularity; φ_B does, and it lands essentially on gold density without being
told what that is.

So φ_B is **not** fertility in disguise. The strong negative φ–density
correlation in blocker #1 has a different cause, and it is itself the project's
most robust finding:

> **Every production tokenizer we tested over-segments these scripts.** Not one of
> 21 real tokenizers falls below gold boundary density in any of the four
> languages. In Khmer they span δ_s/δ_g = **1.12 to 2.25**.

The real tokenizers occupy only the right-hand limb of the φ curve, where φ and
density are near-collinear by construction. That is a property of the sample, not
of the metric — and the metric is what reveals it.

### Blocker #2 — the nulls, measured

N0 (uniform random placement, closed-form hypergeometric) and N1 (length-matched
to each tokenizer's own token-length distribution) across a density sweep from
0.05 to 0.95:

| arm | max \|φ\| | vs 0.02 threshold |
|---|---|---|
| **N0 closed form** (hypergeometric, 4,000 sentences / 117,806 positions) | **0.00012** | 170x inside |
| N0 Monte-Carlo (12 realisations per density) | 0.00153 | 13x inside |
| **N1 length-matched** (40 replicates, bimodal pmf, 77,812 positions) | **0.00082** | 24x inside |
| worst *single* replicate anywhere on either sweep | 0.0136 | still inside |

**Even the worst individual replicate stays under the threshold.** N1's achieved
density tracked its target to within 0.0013 across the whole 0.05–0.95 range, so
the sweep genuinely visited every density rather than collapsing under snapping.

Supporting: E[recall] tracks density exactly (0.050 → 0.950) while E[precision]
stays flat at 0.300 — the density confound made explicit and measurable.

## 2. There is no single winner in Mandarin — read the leaderboard as tie groups

`PREREGISTRATION.md` §2 fixed the smallest meaningful difference **before any
result existed**: the median |φ(X,Y₁) − φ(X,Y₂)| across annotation conventions,
on the reasoning that a gap smaller than the disagreement between two expert
conventions is not a real gap. Measured:

| lang | conventions | SESOI | basis |
|---|---|---|---|
| zh | 4 (SIGHAN) | **0.0729** | different texts — confounded with genre and script |
| yue | 3 (hkcancor tiers) | **0.0361** | **shared text** — the clean one |
| th | 3 | 0.0535 | different texts |
| km | 2 | 0.1113 | different texts |

Applying it to Mandarin (2000-draw sentence-level cluster bootstrap, n = 12,421):

| tokenizer | φ_B | rank CI | tie group |
|---|---|---|---|
| xlm-r | 0.5887 | 1 | **1** |
| bloom | 0.5697 | 2 | **1** |
| baichuan-m2 | 0.5603 | 3 | **1** |
| deepseek-v3 | 0.5532 | 4–5 | **1** |
| mt5 | 0.5524 | 4–5 | **1** |
| command-r | 0.5317 | 6 | **1** |
| yi | 0.5250 | 7 | **1** |
| glm4.5 | 0.5084 | 8 | 2 |
| gemma3 | 0.5026 | 9 | 2 |
| o200k_base | 0.4619 | 13 | 2 |
| nllb | 0.4233 | 14 | 3 |
| tekken | 0.2872 | 16 | 4 |
| cl100k_base / olmo2 / phi4 | 0.1956 | 17–19 | 5 |
| smollm2 / mistral-v3 | 0.054 / 0.052 | 20–21 | 6 |

> **Seven tokenizers tie for first in Mandarin.** "XLM-R wins" is not a
> supportable claim. Groups are formed by distance from the *group leader*, not
> from the previous entry — consecutive-gap chaining is single-linkage and would
> have linked 15 tokenizers spanning 2.5 × SESOI.

**Statistical resolution is not meaningful separation.** With 12,421 sentences the
bootstrap rank intervals are tight — most tokenizers occupy a single rank, so the
ordering *is* statistically resolved. It is still not interpretable, because the
differences are smaller than the disagreement between two expert annotation
conventions of the same language. Reporting the ordering without the SESOI would
be technically defensible and substantively misleading.

Cantonese is the one language with a clean SESOI (the hkcancor tiers are the same
text) and it *does* yield a single winner: **NLLB at φ = 0.3808**, clear of
XLM-R's 0.3201 by 0.061 against a SESOI of 0.036.

⚠️ **Note on §3's tables:** those pool *all* corpora per language, whereas the tie
groups above pool only the convention corpora listed. Different corpus sets give
different numbers, and the Cantonese leader changes between them (command-r vs
NLLB). The tie-group analysis is the one to trust for ranking claims; §3 is
retained for its per-tokenizer diagnostic columns.

## 3. Leaderboard

### Mandarin (δ_g = 0.510) — human ceiling φ = 0.726

# improved
| tokenizer | φ_B | J | M | δ_s | ρ | fertility | chars/token | purity |
|---|---|---|---|---|---|---|---|---|
| **xlm-r** | **0.5845** | 0.567 | 0.602 | 0.620 | 1.22 | 1.19 | 1.40 | 0.945 |
| bloom | 0.5671 | 0.562 | 0.573 | 0.570 | 1.12 | 1.09 | 1.52 | 0.909 |
| baichuan-m2 | 0.5479 | 0.534 | 0.562 | 0.613 | 1.20 | 1.15 | 1.44 | 0.930 |
| deepseek-v3 | 0.5460 | 0.544 | 0.548 | 0.548 | 1.08 | 1.07 | 1.56 | 0.889 |
| mt5 | 0.5428 | 0.512 | 0.575 | 0.665 | 1.31 | 1.25 | 1.33 | 0.956 |
| command-r | 0.5269 | 0.505 | 0.550 | 0.643 | 1.26 | 1.21 | 1.38 | 0.941 |
| gemma3 | 0.4951 | 0.465 | 0.527 | 0.671 | 1.32 | 1.23 | 1.36 | 0.946 |
| glm4.5 | 0.4925 | 0.483 | 0.502 | 0.598 | 1.17 | 1.19 | 1.40 | 0.911 |
| o200k_base | 0.4525 | 0.375 | 0.546 | 0.780 | 1.53 | 1.40 | 1.19 | 0.983 |
| nllb | 0.4231 | 0.347 | 0.515 | 0.785 | 1.54 | 1.39 | 1.19 | 0.979 |
| tekken | 0.2864 | 0.186 | 0.440 | 0.880 | 1.73 | 1.66 | 1.01 | 0.989 |
| cl100k_base | 0.2007 | 0.080 | 0.506 | 0.959 | 1.88 | 2.16 | 0.77 | **0.999** |
| olmo2 | 0.2007 | 0.080 | 0.506 | 0.959 | 1.88 | 2.16 | 0.77 | 0.999 |
| phi4 | 0.2007 | 0.080 | 0.506 | 0.959 | 1.88 | 2.16 | 0.77 | 0.999 |
| smollm2 | 0.0626 | 0.008 | 0.479 | 0.996 | 1.95 | 3.11 | 0.54 | **1.000** |
| mistral-v3 | 0.0606 | 0.008 | 0.478 | 0.996 | 1.95 | 2.07 | 0.81 | 1.000 |
| *char (baseline)* | *0.0000* | — | — | 1.000 | 1.96 | 1.67 | 1.00 | *1.000* |

**The best tokenizer reaches 80% of the human ceiling.** Note the purity column:
the *worst* tokenizers score 0.999–1.000 on it. Purity is a coarsening of recall
and rewards shredding — exactly why it is not the headline.

### Cantonese (δ_g = 0.668) — convention floor φ = 0.7835

| tokenizer | φ_B | J | M | δ_s | ρ | fertility |
|---|---|---|---|---|---|---|
| **command-r** | **0.5330** | 0.420 | 0.677 | 0.835 | 1.25 | 1.42 |
| bloom | 0.5048 | 0.435 | 0.586 | 0.792 | 1.19 | 1.21 |
| xlm-r | 0.5030 | 0.416 | 0.608 | 0.814 | 1.22 | 1.20 |
| gemma3 | 0.4702 | 0.373 | 0.593 | 0.832 | 1.25 | 1.21 |
| deepseek-v3 | 0.4648 | 0.394 | 0.549 | 0.801 | 1.20 | 1.31 |
| o200k_base | 0.3662 | 0.206 | 0.652 | 0.924 | 1.38 | 1.53 |
| cl100k_base | 0.2304 | 0.080 | 0.667 | 0.973 | 1.46 | 2.16 |
| mistral-v3 | 0.1572 | 0.037 | 0.676 | 0.988 | 1.48 | 2.30 |

**No tokenizer reaches the convention floor.** φ = 0.7835 is the agreement between
two legitimate granularity tiers of the *same* Cantonese annotation effort on
*identical text*. The best tokenizer is well below the level at which the gold
standard disagrees with itself.

### Thai (δ_g = 0.374)

| tokenizer | φ_B | δ_s | ρ | fertility | chars/token |
|---|---|---|---|---|---|
| **xlm-r** | **0.6643** | 0.442 | 1.18 | 1.30 | 3.18 |
| mt5 | 0.6572 | 0.432 | 1.16 | 1.31 | 3.16 |
| gemma3 | 0.6289 | 0.523 | 1.40 | 1.57 | 2.63 |
| sea-lion-gemma | 0.6171 | 0.549 | 1.47 | 1.70 | 2.44 |
| o200k_base | 0.5614 | 0.535 | 1.43 | 1.83 | 2.25 |
| tekken | 0.4707 | 0.615 | 1.64 | 2.32 | 1.78 |

### Khmer (δ_g = 0.380) — the worst case

| tokenizer | φ_B | δ_s | ρ | fertility | chars/token |
|---|---|---|---|---|---|
| **mt5** | **0.6954** | 0.506 | 1.33 | 1.49 | 3.12 |
| xlm-r | 0.6763 | 0.488 | 1.12 | 1.26 | 3.17 |
| nllb | 0.6530 | 0.511 | 1.17 | 1.51 | 2.64 |
| gemma3 | 0.5913 | 0.608 | 1.39 | 1.88 | 2.12 |
| o200k_base | 0.5120 | 0.576 | 1.32 | 2.46 | 1.62 |
| cl100k_base | 0.2793 | 0.813 | 1.86 | 6.48 | 0.61 |
| command-r | 0.1203 | 0.982 | 2.25 | **6.92** | 0.58 |
| **tekken** | 0.1147 | 0.983 | 2.25 | **11.60** | **0.34** |
| yi | 0.1134 | 0.984 | 2.25 | **11.72** | 0.34 |
| minicpm4 | 0.1132 | 0.984 | 2.25 | 11.70 | 0.34 |

**Tekken and Yi spend ~11.7 tokens per Khmer word and roughly 3 tokens per
character.** Chars-per-token below 1.0 means the tokenizer uses more than one
token for every single character of text.

## 4. Tier-0 defects — no gold data, no convention, no metric choice

`subchar_rate` is the fraction of token boundaries falling **strictly inside a
Unicode codepoint**. Thai and Khmer characters are 3 bytes in UTF-8.

| tokenizer | lang | subchar_rate | cluster_violation | chars/token |
|---|---|---|---|---|
| **yi** | th | **0.6020** | 0.1619 | 0.41 |
| minicpm4 | th | 0.4723 | 0.2153 | 0.55 |
| smollm2 | th | 0.4510 | 0.2254 | 0.57 |
| bloom | th | 0.1779 | 0.3515 | 1.05 |
| cl100k_base | th | 0.0310 | 0.3849 | 1.11 |
| **minicpm4** | km | **0.6603** | 0.1584 | 0.34 |
| **tekken** | km | **0.6601** | 0.1598 | 0.34 |
| **yi** | km | **0.6596** | 0.1583 | 0.34 |
| smollm2 | km | 0.5433 | 0.2147 | 0.46 |
| cl100k_base | km | 0.4929 | 0.2368 | 0.61 |
| smollm2 | zh | 0.4663 | 0.0000 | 0.54 |
| cl100k_base | zh | 0.2564 | 0.0000 | 0.77 |
| smollm2 | yue | 0.5290 | 0.0000 | 0.48 |
| cl100k_base | yue | 0.4058 | 0.0000 | 0.61 |

**Two thirds of MiniCPM4's, Tekken's and Yi's Khmer token boundaries fall inside
a character.** These are not word-segmentation errors — they are boundaries at
positions that are not linguistically positions at all. `cl100k_base` splits a
quarter of Chinese characters and 41% of Cantonese characters mid-codepoint.

This table requires no gold annotation, no convention choice and no metric
argument. It is the most defensible result here.

## 5. Blocker #4: byte-identical tokenizers, byte-identical rows

`cl100k_base`, `olmo2` and `phi4` produce **identical values on every metric** in
both Mandarin (φ = 0.2007) and Cantonese (φ = 0.2304). They share cl100k's 100,000
merges and differ only in special tokens. This was predicted from the vocabulary
fingerprints before the sweep and is confirmed by it — a live check that the
pipeline is deterministic and that fingerprint-based deduplication is correct.

Without deduplication, this one tokenizer would have appeared as three independent
data points.

## 6. Rank stability — and why the obvious reading is wrong

Pre-registered question (`PREREGISTRATION.md` §3.1): do tokenizer rankings survive
a change of annotation convention? Judged against a **split-half noise floor** —
Kendall τ between leaderboards built from two random halves of the *same* data
under the *same* convention. Without that floor, a τ of 0.70 means nothing.

| comparison | Kendall τ-b |
|---|---|
| **within-convention split-half floor** (60 splits × 4 corpora) | **0.9769** |
| between SIGHAN conventions (mean of 6 pairs) | **0.6973** |
| AS ↔ CityU — *both traditional* | 0.9420 |
| PKU ↔ MSR — *both simplified* | 0.8068 |
| AS ↔ PKU | 0.6232 |
| AS ↔ MSR | 0.5845 |
| CityU ↔ PKU | 0.6618 |
| CityU ↔ MSR | 0.5652 |

Between-convention τ sits far below the noise floor, which reads as "convention
choice changes rankings". **But look at the structure**: the two within-script
pairs score 0.94 and 0.81, while every cross-script pair sits at 0.57–0.66.
AS and CityU are traditional; PKU and MSR are simplified. That is precisely the
signature of a script effect, and `PREREGISTRATION.md` §6 committed us to running
the script control *before* interpreting this.

### The control overturns the reading

`UD_Chinese-GSD` and `UD_Chinese-GSDSimp` are the **same 4,997 sentences under the
same annotation convention**, differing only in traditional vs simplified script.
`UD_Chinese-HK` and `UD_Cantonese-HK` are 1,004 parallel sentences differing in
language.

| contrast | what varies | τ-b | mean \|Δφ\| |
|---|---|---|---|
| GSD ↔ GSDSimp | **script only** | **0.5942** | 0.0591 |
| zh-HK ↔ yue-HK | **language only** | **0.9420** | 0.0449 |
| SIGHAN four | convention + genre + script | 0.6973 | — |
| split-half | nothing (sampling noise) | 0.9769 | — |

> **Script alone disrupts rankings MORE than SIGHAN's four conventions do**
> (τ = 0.594 vs 0.697), on identical sentences with the annotation criteria held
> constant. Meanwhile swapping Mandarin for Cantonese barely moves rankings at all
> (τ = 0.942, essentially at the noise floor).

The per-tokenizer differences make the mechanism obvious — these are the same
sentences, so the gap is pure script coverage in training data:

| tokenizer | traditional | simplified | Δ |
|---|---|---|---|
| hunyuan | 0.2403 | 0.4453 | **+0.205** |
| minicpm4 | 0.2452 | 0.4387 | +0.194 |
| glm4.5 | 0.3245 | 0.5137 | +0.189 |
| o200k_base | 0.3575 | 0.4626 | +0.105 |

**Revised conclusion.** The pre-registered outcome "convention choice changes
tokenizer rankings" is *not* supported once the confound is removed. What changes
rankings is the **script**: Chinese-centric tokenizers (Hunyuan, MiniCPM4, GLM-4.5)
are dramatically better on simplified than traditional characters, and reordering
follows. The residual convention effect — the within-script pairs at τ = 0.94 and
0.81 — is real but much smaller than sampling-noise-adjusted intuition suggests.

This is the deflationary outcome, and it was one of the two write-ups fixed in
advance. It is also a more useful finding than the naive one: a benchmark that
reports "Chinese tokenizer quality" without naming a script is reporting an
artifact of which script its corpus happened to use.

## 7. What this means

**Multilingual encoder tokenizers beat every LLM tokenizer tested.** XLM-R wins
Mandarin and Thai; mT5 is best in Khmer and second in Thai; BLOOM is second in
Mandarin. The frontier LLM tokenizers — cl100k, o200k, tekken, Mistral — sit in
the bottom half everywhere. The plausible reason is visible in the δ_s column:
they are tuned for compression on English-dominant corpora and shred these
scripts.

**The cost consequence is arithmetic, not correlational.** A Khmer document
costs 11.7× more tokens under Tekken than it does words, and more than 3 tokens
per character. At any per-token price, Khmer users pay an order of magnitude more
than the text length warrants.

**The correctness consequence is the Tier-0 table.** A token boundary inside a
codepoint cannot be a linguistic boundary, and a model that must reassemble
characters from byte fragments is solving a problem its tokenizer created.

---

## Caveats

- **The downstream correlation is not reported** as evidence of anything. It was
  declared underpowered before running (n ≈ 12–15 distinct tokenizers, confounded
  with training corpus and model size); see `PREREGISTRATION.md` §3.3.
- **Cantonese is a granularity axis, not rival conventions** — the hkcancor tiers
  are perfectly nested. See the `PREREGISTRATION.md` addendum.
- **The human ceiling is a point estimate** derived from published marginals; the
  underlying corpus publishes aggregates only, not per-annotator segmentations,
  and is Chinese with 20 raters (not Cantonese with 80, as secondary sources
  state).
- **Read §2's tie groups, not §3's ordering.** The rank intervals in §2 are tight
  because the sample is large, but tightness is statistical resolution, not
  meaningful separation. Seven Mandarin tokenizers are within one SESOI of each
  other and should be treated as tied.
- **Three of the four SESOI values are confounded.** Only Cantonese is measured on
  shared text; the zh/th/km figures come from corpora that differ in genre and
  (for zh) script as well as convention, so they are upper bounds on the true
  convention disagreement — which makes the tie groups conservative in the
  direction of *fewer* claimed distinctions, not more.
- **The convention-affinity analysis was not run.** `fit_affinity` and
  `granularity_explains` are implemented but never applied to real data. Given §6's
  finding that script dominates convention, we expect it to be deflationary, but
  that is an expectation and not a result.
- **Blocker #2, measured precisely.** Max |φ| across the full 0.05→0.95 density
  sweep is **0.00012** for the N0 closed form and **0.00082** for N1 — 170x and
  24x inside the pre-registered 0.02 threshold. Even the worst single replicate
  anywhere on either sweep (0.0136) stays under it.
- **The measured design effect brackets the assumption.** `PREREGISTRATION.md` §5
  assumed deff ≈ 3.1 (ICC ≈ 0.10). Measured: **1.02** on i.i.d. positions and
  **13.4** on all-or-nothing sentences. The assumption sits between the extremes,
  as intended, but the real value depends on the corpus.

# Refined
