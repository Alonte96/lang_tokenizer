# Data licences
# 
`unsegbench` ships **no corpus data**. It ships downloaders, checksums and
metrics. Everything below is fetched at runtime into your local cache and stays
there; derived canonical files for non-redistributable corpora never leave the
cache, and only aggregate metrics are committed to this repository.

If you only want a leaderboard with zero licence friction, use
`--corpora @permissive`. That subset is genuinely open and covers all four
languages, at reduced convention coverage for Mandarin.

## Permissive — `@permissive`

| Corpus | Lang | Licence | Redistributable | Source |
|---|---|---|---|---|
| HKCanCor multi-tier | yue | CC BY 4.0 | yes | `AlienKevin/hkcancor-multi` (HF) |
| HKCanCor | yue | CC BY 4.0 | yes | github.com/fcbond/hkcancor |
| Chinese segmentation agreement (20 raters) | zh | CC BY 4.0 | yes | osf.io/m3rcf |
| wisesight1000 | th | CC0 1.0 | yes | `pythainlp/wisesight1000` (HF) |
| VISTEC-TP-TH-2021 | th | CC BY-SA 3.0 | yes | github.com/mrpeerat/OSKut |
| UD Chinese-GSD / GSDSimp / HK | zh | CC BY-SA 4.0 | yes | UniversalDependencies |
| UD Cantonese-HK | yue | CC BY-SA 4.0 | yes | UniversalDependencies |
| UD Thai-PUD | th | CC BY-SA 3.0 | yes | UniversalDependencies |

## Licence-gated — require `--accept-license`

These are free to obtain and standard in the research literature, but their terms
do **not** permit redistribution. `unsegbench` therefore refuses to download them
until you acknowledge the terms once, and never vendors their bytes.

| Corpus | Lang | Licence | Gate key | Source |
|---|---|---|---|---|
| SIGHAN 2005 Bakeoff (AS, CityU, PKU, MSR) | zh | Research use only, per the original competition terms | `sighan` | sighan.cs.uchicago.edu |
# improved
| khPOS | km | CC BY-NC-SA 4.0 | `khpos` | `seanghay/khPOS` (HF) |
| Khmer ALT | km | CC BY-NC-SA 4.0 | `alt` | NICT ALT |

```bash
unsegbench fetch @all --accept-license sighan,khpos,alt
```

**Non-commercial clauses matter.** khPOS and Khmer ALT are CC BY-**NC**-SA: fine
for research, not for anything commercial, and the share-alike clause propagates
to derivatives. If you are evaluating a tokenizer for a commercial product, use
# improved
`--corpora @permissive` and note that Khmer coverage is then absent.
# improved

**SIGHAN specifically.** The landing page describes the data as available for
research use, but the bundled `doc/instructions.txt` is the original competition
text, which restricts use to that competition and directs other requests to the
original providers. Two decades of ACL papers treat it as a standard benchmark;
we follow that practice, cite it, and do not redistribute it. If you need a
guarantee rather than a norm, contact the original providers.

## A note on the SIGHAN download path

# improved
`sighan.cs.uchicago.edu` has served a broken TLS certificate for years, and it is
the only source for `icwb2-data`. `unsegbench` handles this with a **host-scoped,
hash-gated** downgrade: HTTPS is tried first; on failure, plain HTTP is retried
only for that specific host, and only if an expected sha256 is known. Without a
known hash it refuses rather than accepting unverified plaintext. What happened
is recorded in `_download.json` beside the artifact and surfaced by
`unsegbench doctor`. There is no global `verify=False` anywhere in this package.

## Tokenizers

Only tokenizer files are downloaded — never model weights. All listed tokenizers
were verified reachable anonymously, without accepting a model licence. Where an
official repository is gated (Llama, Gemma, Command-R), we use a mirror whose
vocabulary and merges were verified byte-identical to the original, and record
both in the registry. Each tokenizer's fingerprint is published with the results;
a leaderboard without them is not reproducible.

Tokenizer files remain under their respective model licences. `unsegbench`
redistributes none of them.

## Code

`unsegbench` itself is MIT. One exception, attributed in place: the Thai
Character Cluster grammar in `src/unsegbench/clusters.py` is ported from
PyThaiNLP (Apache-2.0), which derives it from Wittawat Jitkrittum's jtcc, itself
implementing Theeramunkong et al. (2000). See `LICENSE` for the full notice.

# Enhanced

# Updated

# Updated

# Refined

# Updated

# Updated

# # Refined

# Updated
