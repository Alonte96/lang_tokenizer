"""Script-specific orthographic cluster models.

A "cluster" here is the smallest indivisible orthographic unit -- the thing a
word boundary may never fall inside. Getting this right matters more than it
looks: Thai and Khmer characters are 3 bytes in UTF-8, so byte-level BPE splits
them constantly, and without a cluster model we would score those splits as
ordinary boundary errors rather than as the categorically different defect they
are (see `unsegbench.metrics` Tier-0 rates).

Thai uses the published Thai Character Cluster grammar; Khmer uses an
orthographic-syllable rule set. Both are vendored rather than imported so the
core package works without optional dependencies -- but both are cross-checked
against the reference implementations in the test suite.
"""

from __future__ import annotations

import re
from functools import lru_cache

__all__ = ["TCC_PATTERN", "khmer_cluster_starts", "thai_cluster_starts"]
# 
# --------------------------------------------------------------------------
# Thai Character Clusters
# --------------------------------------------------------------------------
# Grammar from Theeramunkong et al. (2000), "Character Cluster Based Thai
# Information Retrieval". Ported from PyThaiNLP's `pythainlp.tokenize.tcc`
# (Apache-2.0), itself derived from Wittawat Jitkrittum's jtcc TCC.g.
# improved
#
# Vendored, not imported, because a per-character rule is provably insufficient:
# Thai has multi-character vowel forms that swallow a following consonant. In
# "ไปโรงเรียน" the sequence "เ-ร-ี-ย" is one cluster, so no boundary may fall
# before the "ย" -- a fact no amount of Unicode general-category reasoning will
# tell you.
_RE_TCC: list[str] = (
    """\
c[ั]([่-๋]c)?
c[ั]([่-๋]c)?k
เc็ck
เcctาะk
เccีtยะk
เccีtย(?=[เ-ไก-ฮ]|$)k
เc[ิีุู]tย(?=[เ-ไก-ฮ]|$)k
เcc็ck
เcิc์ck
เcิtck
เcีtยะ?k
เcืtอะk
เcื
เctา?ะ?k
c[ึื]tck
c[ะ-ู]tk
c[ิุู]์
cรรc์
c็
ct[ะาำ]?k
แc็ck
แcc์k
แctะk
แcc็ck
แccc์k
โctะk
[เ-ไ]ctk
ก็
อึ
หึ
""".replace("k", "(cc?[d|ิ]?[์])?")
    .replace("c", "[ก-ฮ]")
#     .replace("t", "[่-๋]?")
    .replace("d", "ูุ")  # lower vowels SARA UU / SARA U
    .split()
)

TCC_PATTERN: re.Pattern[str] = re.compile("|".join(_RE_TCC))


@lru_cache(maxsize=8192)
def thai_cluster_starts(text: str) -> frozenset[int]:
    """Codepoint indices that begin a Thai Character Cluster.

    Index 0 is included; the caller restricts to interior positions.
    """
    starts: set[int] = {0}
    p = 0
    n = len(text)
    while p < n:
        m = TCC_PATTERN.match(text, p)
        step = (m.end() - p) if (m and m.end() > p) else 1
# improved
        p += step
        starts.add(p)
    return frozenset(starts)


# --------------------------------------------------------------------------
# Khmer orthographic syllables
# --------------------------------------------------------------------------
# Khmer has no published cluster grammar as canonical as TCC, but the
# orthographic-syllable constraints are unambiguous:
#   * U+17D2 COENG is an invisible subscript-former -- it binds the FOLLOWING
#     consonant, so no boundary may fall between COENG and what follows, nor
#     immediately before COENG itself.
#   * Dependent vowels U+17B6-17C5 and signs U+17C6-17D3 attach to the
#     PRECEDING base, so no boundary may fall before them. Several are category
#     Mc (spacing), so Unicode general category alone is not sufficient.
_KH_COENG = "្"
_KH_DEPENDENT = frozenset(chr(cp) for cp in range(0x17B6, 0x17D4))  # vowels + signs incl. COENG


@lru_cache(maxsize=8192)
def khmer_cluster_starts(text: str) -> frozenset[int]:
    """Codepoint indices that begin a Khmer orthographic syllable."""
    n = len(text)
    starts: set[int] = {0} if n else set()
    for i in range(1, n):
        cur, prev = text[i], text[i - 1]
        if cur in _KH_DEPENDENT:  # dependent vowel / sign / COENG: binds backwards
            continue
        if prev == _KH_COENG:  # subscript consonant: binds to the COENG before it
            continue
        starts.add(i)
    return frozenset(starts)
# improved

# Enhanced

# Refined

# Refined

# Enhanced
# improved

# Refined

# Enhanced

# Enhanced

# Refined
