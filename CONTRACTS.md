# CONTRACTS

The frozen interfaces. Read this before writing any module; do not infer these
rules from other code. If something here is wrong, change it here first and say
so — do not work around it locally.

---

## 1. Canonical IR

One gold sentence is a `Segmented` (`unsegbench/types.py`):

```jsonc
{"id": "sighan_pku/test/000123",
 "text": "我喜欢吃苹果。",
 "spans": [[0,1],[1,3],[3,4],[4,6]],
 "meta": {}}
```

* `text` is the source string **verbatim**. Loaders **never** call
  `unicodedata.normalize`. Thai U+0E33 changes codepoint count under NFKD/NFKC;
  NFKC reorders Khmer marks. Normalisation is a tokenizer-side concern.
* `spans` are `(start, end)` **codepoint** offsets, sorted, non-overlapping,
  `0 <= start < end <= len(text)`.
* Spans need **not** tile `text`. Uncovered codepoints are inter-word gaps and
  must all be in the corpus's declared `gap_charset` — anything else uncovered
  is a build error meaning the loader dropped content.
* Store nothing derivable. `words`, boundary sets and masks are pure functions
  in `positions.py`, never serialised.
* `id` is `f"{corpus_id}/{split}/{i:06d}"`.

Corpus-level constants live once in `CorpusManifest`, never per record.

**Delimiter whitespace introduced by the annotation format is removed.** `text`
is what a tokenizer would realistically see: SIGHAN's inter-word spaces are
delimiters, not content, so they do not appear in `text` and the spans tile it.
Whitespace that is genuinely in the source — Thai phrase spaces, spaces around
embedded Latin — stays, and is declared in `gap_charset`.

## 2. Position universes

**Indexing (frozen):** position `i` is the gap between `text[i-1]` and `text[i]`.
Valid positions are `1 .. n-1`. Positions `0` and `n` are **excluded** from every
universe — every tokenizer gets sentence edges for free and counting them
inflates recall badly on short sentences.

A gold span `(s, e)` contributes boundaries at `s` and `e`, minus any that are
`0` or `n`. Both ends matter because spans need not tile.

| mask | universe | use |
|---|---|---|
| `raw` | all of `1..n-1` | appendix; comparability with Xu et al. (2026) |
| `legal` | on a legal orthographic cluster edge (𝓛) | denominator for Tier-0 defect rates |
| **`core`** | **𝓛 minus trivial positions (𝓣)** | **every headline claim** |

* **𝓛** = UAX#29 grapheme-cluster starts ∩ script grammar ∩ non-starter/non-final
  rules. Thai uses the full TCC grammar (`clusters.thai_cluster_starts`,
  verified against PyThaiNLP); Khmer uses the orthographic-syllable rules.
* **𝓣** = either side whitespace (**including U+200B ZWSP**, which `str.isspace()`
  misses), or either side Unicode category `P*`/`S*`, or a script/class transition.
* Trivial **positions** are removed from the scored universe. The corresponding
  gold **boundaries** are never removed from the gold — that would change the
  gold density and the word spans.

`compute_masks(text, lang)` returns all three in one pass; the runner calls it
once per sentence and reuses it across all ~25 tokenizers.

## 3. Metrics

`boundary_counts(gold, pred, mask)` intersects **both** gold and pred with the
mask before counting. This is the only definition under which precision and
recall stay commensurable.

**Headline metric is `phi` (Matthews correlation on the boundary table).** It is
0 for the character tokenizer, 0 for whole-sentence, 0 in expectation for random
placement at any density, and 1 for the oracle. Boundary-F1, recall and purity
are all partly density artifacts — under random placement `E[recall] = h` (linear
in how finely you chop) while `E[precision] = δ_g` (density-invariant).

Word metrics score the partition **induced** by each boundary set, not raw token
spans. Under `mask="raw"` on a corpus whose gold tiles the text this is exactly
the SIGHAN scorer's word P/R, which the perl cross-check verifies to 1e-6.

**0/0 conventions (frozen — tested explicitly):**

| case | value |
|---|---|
| precision with no predictions | `1.0` |
| recall with no gold boundaries | `1.0` |
| F1 when `P + R == 0` | `0.0` |
| `phi` with a zero margin | `0.0` (standard MCC convention) |

These conventions interact in two corners. Both are consequences of the table
above rather than separate rules, and both are pinned by name in the test suite:

1. **`phi² != J·M` on an empty universe.** With no gold *and* no predictions in
   the mask, `precision = recall = 1.0` so `informedness = markedness = 1.0`,
   while `phi = 0.0` by the zero-margin rule.
2. **`f1(Counts(0,0,0,0)) == 1.0`, not `0.0`.** The `P + R == 0` guard never
   fires, because both are pushed to `1.0` first.

Both are individually correct. Only sentences with no gold boundary in the mask
are affected, and they contribute nothing to any pooled table — but do not
"fix" either one without reading this note, because each looks like a bug in
isolation.

`informedness` and `markedness` are clamped to `[-1, 1]`. They are exactly
bounded there mathematically, but dividing by `1 - delta` loses precision when
delta approaches 1 (at `delta_g = 36/37` the raw value overshoots -1.0 by ~2e-15).
`phi` needs no clamp — verified across every `Counts` with all cells in 0..39
plus 200k random draws.

## 4. Tokenizer adapters

`encode(text) -> EncodeResult(spans, n_tokens, flags)`.

* `spans` are **accepted** token spans: strictly ascending, non-overlapping,
  each `text[s:e]` non-empty. They need not cover `text`.
* `n_tokens` is the **raw** count including tokens whose boundary was rejected.
  Fertility and chars-per-token derive from it, and they must not improve just
  because we failed to place a boundary.
* **The mid-codepoint guard.** Thai (U+0E00–0E7F) and Khmer (U+1780–17FF) are 3
  bytes in UTF-8, so byte-level BPE splits them constantly. HF's
  `return_offsets_mapping` **collapses** all bytes of a codepoint onto that
  codepoint, so consecutive tokens come back with identical or overlapping
  spans. The guard is:

  | relation | meaning | action |
  |---|---|---|
  | `start_i == end_{i-1}` | contiguous | accept |
  | `start_i > end_{i-1}` | **forward gap** | **accept**, and add the skipped codepoints to `flags["dropped_chars"]` |
  | `start_i < end_{i-1}` | overlap | reject the boundary |

  A **forward gap is not a defect.** Every Metaspace/SentencePiece tokenizer
  drops the delimiter space from its offsets: XLM-R on `ฉัน ทำ งาน ที่ บ้าน`
  returns `[(0,3),(4,6),(7,10),(11,14),(15,19)]` — a perfect segmentation with
  four one-character gaps. An earlier version of this contract said "accept only
  when `start_i == end_{i-1}`", which would have discarded four of those five
  tokens and reported a near-perfect Thai tokenizer as a catastrophe. A gap is a
  verified boundary plus unaccounted-for characters, and `dropped_chars` already
  exists to record exactly that.

  On overlap, classify by whether the overlap region `text[s:min(e, prev_end)]`
  contains a multi-byte codepoint: if so it is a UTF-8 artefact →
  `flags["midcodepoint_split"]`; if the region is pure ASCII, no UTF-8 artefact
  can explain it, so the offset list is genuinely disordered →
  `flags["overlap_rejected"]`. These two must not leak into each other —
  `midcodepoint_split` is a headline Tier-0 number.

  Note a token can straddle *two* codepoints and so end **beyond** the previous
  token's end (observed: BLOOM on Khmer). Classify on the overlap region, not on
  the end offset, or such cases are misfiled as disorder and the headline
  mid-codepoint rate is silently deflated.
* `flags["prefix_space_trim"]`: a token composed only of space markers (`▁`, `Ġ`)
  whose span **overlaps a neighbour** is zero-width in the source and is zeroed.
  HF assigns XLM-R's bare `▁` the first character of the *following* word — for
  `สวัสดี…` it returns `("▁", (0,1))` overlapping `("สว", (0,2))`. Untreated, the
  guard accepts the fabricated `(0,1)` and rejects the real `(0,2)`, planting a
  boundary one character inside the first word. Only overlapping markers are
  zeroed; a `▁` sitting on a genuine space is left alone.
* If the normaliser mutates the string, validate offsets against the **original**
  text and set `flags["normaliser_mutated"]`. Never score against the normalised
  string.

Four points of precision the table above leaves implicit, all pinned by tests:

* **`dropped_chars` is wider than the forward-gap row suggests.** It is computed
  as `len(text) - covered`, so it also absorbs uncovered leading/trailing text and
  the codepoints of *rejected* overlapping tokens. That matches `FLAG_KEYS`
  ("codepoints not covered by any accepted token span"); the gap row is just the
  commonest cause, not the only one.
* **Malformed spans** (`s < 0`, `e > n`, `s > e`) have no row of their own and are
  counted as `overlap_rejected`. The flag vocabulary is closed, so they share the
  nearest bucket.
* **`prefix_space_trim` covers more markers than `▁` and `Ġ`** — the
  implementation also admits `Ċ`, `ĉ` and literal whitespace.
* **Predicted boundaries come from span *starts* only** (`EncodeResult.boundaries`),
  whereas gold spans contribute at both ends (§2). So a token *end* followed by a
  forward gap is never a predicted boundary even though the same position in gold
  would be one. This is deliberate — a gap means we do not know where the token
  truly ended — but it is an asymmetry, and it slightly depresses recall for
  tokenizers that drop delimiters.

**No silent repair.** Anything the layer cannot do honestly goes in `flags` and
becomes a results column.

### `flags` vocabulary (closed set — see `types.FLAG_KEYS`)

`midcodepoint_split` · `cluster_split` · `overlap_rejected` · `dropped_chars` ·
`prefix_space_trim` · `normaliser_mutated`

## 5. Sufficient statistics

The expensive stage persists **only** the per-sentence integer counters in
`types.STATS_COLUMNS` — never tokens or spans. Every table, plot, bootstrap,
Kendall τ and rank interval is a groupby over those integers, which is why the
report costs ~2s. Cache key is
`(code_version, tokenizer_fingerprint, corpus_manifest_sha, mask)`.

Keep them integers. A float in this schema is a bug.

## 6. Licensing

* **SIGHAN 2005**, **khPOS**, **Khmer ALT** are `redistributable=False`. Never
  vendor their bytes — not in the repo, not in fixtures, not in test data.
  Fetching requires `--accept-license`, recorded once to a ledger.
* Their canonical build output stays in the cache and never lands in `results/`.
  Only aggregate metrics are committed.
* The benchmark must **degrade gracefully to the permissive subset**
  (`@permissive`: HKCanCor, hkcancor-multi, wisesight1000, UD, OSF) so a fresh
  user gets a leaderboard with zero licence friction.
* `zwsp_present` in the manifest is load-bearing for Khmer: if gold boundaries
  coincide with U+200B the task is trivial there and the corpus is **not** usable
  for Tier-1 claims. (Measured: khPOS has 2 ZWSP in 602,138 chars, Khmer ALT has
  zero. Both are safe.)

## 7. Network

* Integrity by hash. Expected sha256 lives in `corpora.lock.json`; a mismatch is
  fatal, never a warning.
* SIGHAN's HTTPS certificate is broken. The downgrade is **host-scoped and
  hash-gated** (`TLS_DOWNGRADE_ALLOWLIST`), recorded in `_download.json`, and
  surfaced by `unsegbench doctor`. Never a global `verify=False`, never an
  import-time warning suppression.
* Tokenizer downloads use `allow_patterns` for tokenizer files only. **Never**
  model weights. CI asserts `torch` is not importable.

## 8. Merge discipline

One agent, one directory. The only shared files are the two registries, and each
loader module exposes its own `ENTRIES` tuple that the registry imports — so
nobody edits anyone else's lines.

Every test track runs against `tests/fixtures/mini.py` and **only** that. It
is committed, network-free, licence-free, and covers every Unicode hazard:
Thai SARA AM, Thai leading vowels, a Thai multi-character vowel form, Khmer
COENG, Khmer spacing dependent vowels, ZWSP, Thai phrase spaces, full-width CJK
punctuation, Latin/digit script transitions, and single-word sentences with no
interior gold boundary.

# Updated

# Refined

# Updated

# Updated

# Refined

# Enhanced
