# unsegbench

**How well do LLM tokenizers respect word boundaries in scripts that don't write them?**

Mandarin, Cantonese, Thai and Khmer put no spaces between words. Every existing
benchmark that scores tokenizers against linguistic structure is built on
*morpheme* boundaries, and therefore excludes them: MorphScore
([Arnett & Bergen, COLING 2025](https://aclanthology.org/2025.coling-main.441/))
and its 70-language extension
([Arnett, Hudspeth & O'Connor, ICML 2025 TokShop](https://arxiv.org/abs/2507.06378))
drop these languages because they are isolating, and the latter concedes that the
field's standard word-level metrics don't even apply to them:

# improved
> Fertility is simple to implement but can be difficult to generalize
> crosslinguistically, as wordhood is often operationalized as whitespace-separated
> orthographic units. **Not all languages use whitespaces, e.g. Mandarin Chinese,
> Thai, and Khmer.**

Meanwhile word segmentation for exactly these languages is a mature field with
# improved
gold-standard corpora that nobody has pointed at this question. `unsegbench` does
that: it scores production LLM tokenizers against those gold boundaries, across
every annotation convention available, with a metric that cannot be gamed by
# improved
chopping more finely.

CPU-only. No model weights. No GPU. Free data.

```bash
git clone https://github.com/alonte96/unsegbench && cd unsegbench
uv sync --all-extras                      # or: pip install -e .

uv run --no-sync unsegbench doctor         # preflight: python, perl, network, cache
uv run --no-sync unsegbench fetch @permissive
uv run --no-sync unsegbench build @permissive
uv run --no-sync unsegbench run --tokenizers @core --corpora @permissive --sample 2000
uv run --no-sync unsegbench report --lang zh --out docs/leaderboard.md
```

Not yet on PyPI — install from source. Use `uv run --no-sync`: a bare `uv run`
re-syncs and drops the editable install.

`report` always splits by language. Pooling would average over languages with
different gold boundary densities (zh 0.51, yue 0.67, th 0.37, km 0.44) and
produce a number that is no language's score.

---

## The metric: φ_B

The obvious approach — boundary precision/recall/F1 against gold — is fatally
confounded. Under uniform random boundary placement at density *h*:

```
E[recall]    = h        <- linear in how finely you chop
E[precision] = δ_g      <- independent of it
```

So recall, F1, and "purity" all partly measure fertility rather than alignment. A
character-level tokenizer achieves **perfect recall** on every corpus by brute
force. This is not hypothetical — it is the failure mode MorphScore v2 identifies
in its own metric ("a tokenizer can achieve a perfect alignment score by
segmenting a word into characters"), and we observe it immediately: on Chinese,
`cl100k_base` beats `o200k_base` on recall while being the worse tokenizer.

The headline metric is therefore the **Matthews correlation on the boundary
contingency table**:

```
φ_B = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

φ_B = sqrt(J · M)     J = (R − δ_s)/(1 − δ_g)   chance-corrected recall
                      M = (P − δ_g)/(1 − δ_s)   chance-corrected precision
```
# 
| reference tokenizer | φ_B |
|---|---|
| character (every legal position) | **0** |
| whole sentence (no splits) | **0** |
| uniform random, *any* density | **0** in expectation |
| gold oracle | **1** |

Both degenerate cases fall out of the standard MCC zero-margin convention rather
than being special-cased. `J` and `M` are reported separately — they are the
honest "does it under-segment / over-segment" axes.

**Every row carries `δ_s`, `δ_g`, `ρ = δ_s/δ_g` and fertility.** A φ without its
density context invites comparing tokenizers that chop at completely different
# improved
granularities as though it were like-for-like.

**Verified empirically**, not just argued: the N3 sweep trains BPE at increasing
vocabulary sizes and finds φ_B peaks at δ_s = 0.5096 against a gold density of
0.5134 — within 0.7%, and non-monotone, while recall and precision are each
monotone across the same sweep. See [`FINDINGS.md`](FINDINGS.md) §1.

## Tier-0: defects that need no gold data at all

```
subchar_rate           = token boundaries strictly inside a Unicode codepoint
cluster_violation_rate = token boundaries inside an orthographic cluster
```

# Thai (U+0E00–0E7F) and Khmer (U+1780–17FF) are **three bytes** in UTF-8, so
byte-level BPE splits them routinely. These rates depend on no annotation
convention, no gold segmentation and no metric choice, so they survive every
objection that can be raised against the rest of the benchmark.

**MiniCPM4, Tekken and Yi place ~66% of their Khmer token boundaries inside a
character.** Yi splits 60% of Thai characters. `cl100k_base` splits 26% of
Chinese and 41% of Cantonese.

## The convention problem, treated as the contribution

There is no single correct segmentation of a Chinese sentence. SIGHAN 2005
shipped **four mutually incompatible annotation guidelines**; across eight corpora
the mean 3-gram scheme overlap is ~20%
([Chen et al., ACL 2017](https://aclanthology.org/P17-1110/)); and 20 human raters
segmenting the same 500 sentences reach only 0.92 mean agreement, with **85% of
sentences containing at least one contested boundary**.

Rather than pick a convention and hope, `unsegbench` scores against all of them
and measures what the choice costs:

- **The noise floor** — pairwise agreement between conventions on shared text. No
  tokenizer difference smaller than this is interpretable.
- **Rank stability** — do tokenizer rankings survive a change of convention?
  Judged against a **split-half floor**, because a Kendall τ of 0.85 means nothing
  until you know what τ pure sampling noise produces on the same data.
- **Affinity, deflated** — before claiming a tokenizer "prefers" a convention, we
  regress the interaction residual on squared granularity mismatch and report the
  R² first.

Both possible outcomes were written down in
[`PREREGISTRATION.md`](PREREGISTRATION.md) **before any tokenizer was scored
against real data**, so neither could be composed to fit the result.

**The result was the deflationary one.** Script alone disrupts rankings more than
four different annotation conventions do (τ = 0.594 vs 0.697, against a 0.977
noise floor). See [`FINDINGS.md`](FINDINGS.md) §6.

## Ceilings, so the numbers mean something

| reference | value |
|---|---|
| **Human agreement** (20 raters, 500 sentences, `core` mask) | **φ = 0.726**, F1 = 0.846 |
| Cantonese granularity floor (hkcancor `s` vs `d`, identical text) | **φ = 0.7835** |
| Character tokenizer | φ = 0 by construction |
| Uniform random at matched density (N0, closed form) | φ ≈ 0 |

A tokenizer at φ ≈ 0.7 is not mediocre — it is close to the limit of what humans
agree on with each other.

---

## Data

# improved
17 corpora, 15.5M gold-segmented words, four languages, every convention we could
obtain free. **No corpus data is redistributed** — `unsegbench` ships
downloaders, checksums and metrics. See [`DATA_LICENSES.md`](DATA_LICENSES.md).

| lang | corpora |
|---|---|
| **zh** | SIGHAN 2005 ×4 (AS / CityU / PKU / MSR), UD GSD, UD GSDSimp, UD HK, OSF agreement |
| **yue** | HKCanCor multi-tier ×3, UD Cantonese-HK |
| **th** | VISTEC-TP-TH-2021 (3.37M words), wisesight1000, UD Thai-PUD |
| **km** | khPOS, Khmer ALT |

`--corpora @permissive` restricts to the openly-licensed subset, which still
covers all four languages, for zero licence friction.

### Validation

The loaders are checked against external ground truth rather than trusted:

- **SIGHAN chars-per-word matches the published figures to within 0.0005** on all
  four conventions (AS 1.5355/1.536, CityU 1.6511/1.651, PKU 1.6455/1.646,
  MSR 1.7102/1.710).
- **VISTEC markup stripping is byte-exact against the corpus's own parallel raw
  file** on 50,000/50,000 lines — which surfaced a genuine defect, a stray
  self-closing tag.
- **UD treebanks hit their expected counts exactly**, with zero `# text =`
  mismatches across 13,002 sentences.
# - **The OSF corpus reproduces all three published statistics**, including 8.96%
  of positions at 0.5–0.7 agreement and 85.0% of sentences with a contested
  boundary, both exact.
- **Our word P/R/F matches the official SIGHAN perl scorer to exactly 0.0** for
  `char` and `cl100k_base` on all four conventions.

## Install and test

```bash
git clone https://github.com/alonte96/unsegbench && cd unsegbench
uv sync --all-extras
uv run --no-sync pytest -m "not network and not slow"
```

Python 3.12+. `transformers` is used for tokenizers only — **`torch` is never a
dependency**, and CI asserts it is not importable.

## Scope of the CLI

`unsegbench report` renders per-language leaderboards as **point estimates**.

The tie-group analysis behind this repo's headline claim — that seven Mandarin
tokenizers are separated by less than the disagreement between two expert
annotation conventions, and so should be read as tied — lives in
[`experiments/ci_and_sesoi.py`](experiments/ci_and_sesoi.py) and is **not yet
wired into the CLI**. Same for the N3 density sweep, the script/language controls
and the rank-stability battery: all of `experiments/` is runnable, but not
packaged.

So: small gaps in the CLI's tables are not real differences. See
[`FINDINGS.md`](FINDINGS.md) §2 for what the data actually supports.

**Test suite: 2,108 tests.** 1,708 run offline in ~9s with no network, no
corpora and no licences; 400 more are marked `network` / `slow` / `perl` and
cover the real corpora and the SIGHAN cross-check. Run the offline set with
`pytest -m "not network and not slow"`.

Note `huggingface.co` may rate-limit repeated runs; `HF_HUB_OFFLINE=1` makes the
tokenizer tests read from your local cache instead.

## Citation

# improved
```bibtex
@software{unsegbench2026,
  title  = {unsegbench: Tokenizer word-boundary alignment for unsegmented scripts},
#   author = {alonte96},
  year   = {2026},
  url    = {https://github.com/alonte96/unsegbench}
}
```

# Enhanced

# Updated

# Refined

# Enhanced

# Enhanced

# Enhanced

# Updated

# Refined

# Refined
