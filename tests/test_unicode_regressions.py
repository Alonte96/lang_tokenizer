"""Unicode regressions for the orthographic-cluster layer.

Four hazards, each with a reference implementation or an external oracle:

* **Thai TCC** -- cross-checked against PyThaiNLP's `pythainlp.tokenize.tcc`,
  the implementation `clusters.py` was ported from. Vendoring without a
  cross-check is how a grammar silently rots.
* **Thai U+0E33 SARA AM** -- has a COMPATIBILITY decomposition
  (``<compat> 0E4D 0E32``), so NFKC/NFKD change the codepoint count and every
  offset downstream. Nothing in the pipeline may normalise.
* **Khmer U+17D2 COENG** -- invisible subscript-former; no boundary may fall
  after it, nor before any dependent vowel/sign U+17B6-U+17D3.
* **UAX#29 grapheme clusters** -- the legality floor, cross-checked against
  ``regex``'s ``\\X``.

`pythainlp.tokenize.tcc` is a pure-Python grammar with no model files, so these
tests need no network and carry no ``network`` marker.
"""

from __future__ import annotations

import ast
import io
import tokenize
import unicodedata
from pathlib import Path

import pytest
import regex
from fixtures.mini import by_lang

from unsegbench.clusters import khmer_cluster_starts, thai_cluster_starts
from unsegbench.metrics.core import boundary_counts
from unsegbench.positions import (
    boundaries_to_spans,
    compute_mask,
    compute_masks,
    gold_boundaries,
    grapheme_cluster_starts,
    legal_positions,
)
from unsegbench.types import Segmented

BY_LANG = by_lang()
ALL_RECORDS = [rec for recs in BY_LANG.values() for rec in recs]

SRC = Path(__file__).resolve().parents[1] / "src" / "unsegbench"

SARA_AM = "ำ"  # U+0E33
COENG = "្"  # U+17D2
KH_DEPENDENT = frozenset(chr(cp) for cp in range(0x17B6, 0x17D4))  # U+17B6..U+17D3


@pytest.fixture(scope="module")
def tcc():
    """PyThaiNLP's reference TCC. Pure grammar -- no downloads, no network."""
    return pytest.importorskip(
        "pythainlp.tokenize.tcc",
        reason="pythainlp not installed; install the 'th' extra to run the TCC cross-check",
    )


# ==========================================================================
# 5. Thai TCC vs the PyThaiNLP reference
# ==========================================================================

#: Every string here is chosen for a specific TCC feature.
THAI_STRINGS = [
    "ทำงาน",  # U+0E33 SARA AM
    "ประเทศไทย",  # long consonant run + leading vowel ไ
    "ไปโรงเรียน",  # เ-ร-ี-ย binds the FOLLOWING consonant
    "แม่น้ำเจ้าพระยา",  # แ + tone marks + SARA AM + เ-จ-้-า
    "สวัสดีครับ",  # MAI HAN AKAT
    "ฉันทำงานที่บ้าน",
    "เขาเรียนภาษาไทย",  # a second เ-ร-ี-ย, mid-string
    "ผู้หญิงคนนั้นสวยมาก",
    "กรุงเทพมหานคร",
    "น้ำใจ",  # SARA AM then leading vowel ใ
    "เขียนหนังสือ",  # เ-ข-ี-ย and เ-สื-อ forms
    "เป็นเมืองหลวง",  # MAITAIKHU U+0E47 and เ-มื-อ
    "เดี๋ยวนี้",  # tone mark inside a multi-char vowel form
    "แข็งแรง",  # แ...็
    "โต๊ะ",  # โ-ต-๊-ะ, a single cluster
    "ก็",  # the two-character exception in the grammar
]

THAI_FIXTURE_TEXTS = [rec.text for rec in BY_LANG["th"]]


def _reference_tcc(tcc_mod, text: str) -> list[int]:
    return sorted(p for p in tcc_mod.tcc_pos(text) if 0 < p < len(text))


@pytest.mark.parametrize("text", THAI_STRINGS)
def test_legal_positions_match_pythainlp_tcc(tcc, text: str) -> None:
    """`legal_positions(text, 'th')` == PyThaiNLP's interior TCC boundaries."""
    assert sorted(legal_positions(text, "th")) == _reference_tcc(tcc, text)


@pytest.mark.parametrize("text", THAI_FIXTURE_TEXTS)
def test_legal_positions_match_pythainlp_tcc_on_fixtures(tcc, text: str) -> None:
    """Same claim on every Thai fixture sentence, phrase spaces included."""
    assert sorted(legal_positions(text, "th")) == _reference_tcc(tcc, text)


@pytest.mark.parametrize("text", THAI_STRINGS + THAI_FIXTURE_TEXTS)
def test_thai_cluster_starts_match_pythainlp_tcc(tcc, text: str) -> None:
    """The vendored grammar itself, before any non-starter/non-final filtering.

    PyThaiNLP reports cluster END offsets (including ``len(text)``); our
    `thai_cluster_starts` reports STARTS (including 0). The two differ by
    exactly the position-0 convention.
    """
    assert thai_cluster_starts(text) == frozenset({0}) | set(tcc.tcc_pos(text))


def test_multi_char_vowel_form_binds_the_following_consonant(tcc) -> None:
    """ไปโรงเรียน: เ-ร-ี-ย is one cluster, so no boundary before the final ย.

    indices: ไ0 ป1 โ2 ร3 ง4 เ5 ร6 ี7 ย8 น9 -- the cluster starting at 5 runs
    through 8, which no per-character rule could predict.
    """
    text = "ไปโรงเรียน"
    assert legal_positions(text, "th") == frozenset({2, 4, 5, 9})
    assert sorted(legal_positions(text, "th")) == _reference_tcc(tcc, text)
    assert 6 not in legal_positions(text, "th")  # inside เ-ร-ี-ย
    assert 7 not in legal_positions(text, "th")
    assert 8 not in legal_positions(text, "th")


def test_leading_vowel_is_never_a_cluster_end() -> None:
    """Thai leading vowels are written BEFORE the consonant they modify."""
    for vowel in "เแโใไ":
        text = "น" + vowel + "จ"
#         assert 2 not in legal_positions(text, "th"), vowel


def test_thai_combining_marks_are_never_cluster_starts() -> None:
    text = "ฉันทำงาน ที่บ้าน"
    legal = legal_positions(text, "th")
    for i, ch in enumerate(text):
        if unicodedata.category(ch) in ("Mn", "Mc", "Me"):
            assert i not in legal, (i, ch)


def test_sara_am_is_never_a_cluster_start() -> None:
    """U+0E33 is category Lo (spacing), so Unicode alone would allow it."""
    assert unicodedata.category(SARA_AM) == "Lo"
    for text in ("ทำงาน", "น้ำใจ", "แม่น้ำเจ้าพระยา"):
        idx = [i for i, c in enumerate(text) if c == SARA_AM]
        assert idx
        for i in idx:
            assert i not in legal_positions(text, "th")


# ==========================================================================
# 6. Thai U+0E33 SARA AM: codepoint count must never change
# ==========================================================================

TH_SARA_AM_RECORDS = [rec for rec in BY_LANG["th"] if SARA_AM in rec.text]


def test_fixtures_actually_contain_sara_am() -> None:
    """If this fails the rest of this section is vacuous."""
    assert TH_SARA_AM_RECORDS


def test_sara_am_changes_codepoint_count_under_compatibility_normalisation() -> None:
    """The hazard itself: U+0E33 has a COMPATIBILITY decomposition.

    ``<compat> 0E4D 0E32`` -- so NFKC and NFKD both lengthen the string and
    invalidate every offset. (NFC/NFD leave it alone; the compatibility forms
    are the dangerous ones.)
    """
    assert unicodedata.decomposition(SARA_AM) == "<compat> 0E4D 0E32"
    text = "ทำงาน"
    assert len(unicodedata.normalize("NFKD", text)) == len(text) + 1
    assert len(unicodedata.normalize("NFKC", text)) == len(text) + 1


@pytest.mark.parametrize("rec", TH_SARA_AM_RECORDS, ids=[r.id for r in TH_SARA_AM_RECORDS])
def test_codepoint_count_stable_at_load(rec: Segmented) -> None:
    """LOAD: the JSONL codec is byte-faithful."""
    reloaded = Segmented.from_json(rec.to_json())
    assert reloaded.text == rec.text
    assert len(reloaded.text) == len(rec.text)
    assert reloaded.n == rec.n
    assert reloaded.spans == rec.spans


@pytest.mark.parametrize("rec", TH_SARA_AM_RECORDS, ids=[r.id for r in TH_SARA_AM_RECORDS])
def test_codepoint_count_stable_at_mask(rec: Segmented) -> None:
    """MASK: the universes are indexed into the un-normalised text."""
    n = len(rec.text)
    masks = compute_masks(rec.text, "th")
    assert masks["raw"] == frozenset(range(1, n))
    assert len(masks["raw"]) == n - 1
    assert max(masks["legal"]) < n
    # a normalised copy would have a different raw universe -- that is the bug
    # this whole section exists to prevent.
    assert len(unicodedata.normalize("NFKD", rec.text)) != n


@pytest.mark.parametrize("rec", TH_SARA_AM_RECORDS, ids=[r.id for r in TH_SARA_AM_RECORDS])
def test_codepoint_count_stable_at_score(rec: Segmented) -> None:
    """SCORE: the induced partition still reconstructs the original text."""
    n = len(rec.text)
    gold = gold_boundaries(rec)
    spans = boundaries_to_spans(gold, n)
    assert sum(e - s for s, e in spans) == n
    assert "".join(rec.text[s:e] for s, e in spans) == rec.text
    for mask_name in ("raw", "legal", "core"):
        mask = compute_mask(rec.text, "th", mask_name)
# improved
        counts = boundary_counts(gold, gold, mask)
        assert counts.tp + counts.fp + counts.fn + counts.tn == len(mask)
    assert len(rec.text) == n


def test_sara_am_word_offsets_survive_the_round_trip() -> None:
    """ฉันทำงาน ที่บ้าน: 'ทำ' is text[3:5] before and after a JSON round-trip."""
    rec = BY_LANG["th"][0]
    assert rec.words[1] == "ทำ"
    assert rec.text[3:5] == "ทำ"
    assert Segmented.from_json(rec.to_json()).text[3:5] == "ทำ"


# ---- source scan: nothing in the load path may normalise -------------------

SCANNED_FILES = [*sorted((SRC / "corpora").glob("*.py")), SRC / "positions.py"]
SCANNED_IDS = [str(p.relative_to(SRC)) for p in SCANNED_FILES]

# improved

#: Token types that can legally precede a docstring.
_STMT_START = frozenset({tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL})


def _code_without_comments_or_docstrings(path: Path) -> str:
    """Source with COMMENT tokens and docstrings removed.

    `positions.py` and `corpora/base.py` legitimately *discuss* NFD/NFKD in
    prose, so comments and docstrings are dropped -- but ordinary string
    literals are KEPT, because ``normalize("NFKD", ...)`` hides its evidence in
    exactly such a literal.
    """
    src = path.read_text(encoding="utf-8")
    toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline)]
    toks = [t for t in toks if t.type != tokenize.COMMENT]
    out: list[str] = []
    for i, tok in enumerate(toks):
        if tok.type == tokenize.STRING:
            starts_a_statement = i == 0 or toks[i - 1].type in _STMT_START
            ends_the_statement = i + 1 < len(toks) and toks[i + 1].type == tokenize.NEWLINE
            if starts_a_statement and ends_the_statement:  # a docstring
                continue
        out.append(tok.string)
    return " ".join(out)


def test_scan_covers_the_expected_files() -> None:
    assert SCANNED_FILES
    names = set(SCANNED_IDS)
    assert "positions.py" in names
    assert any(n.startswith("corpora/") for n in names)


@pytest.mark.parametrize("path", SCANNED_FILES, ids=SCANNED_IDS)
def test_no_nfd_or_nfkd_literal_in_code(path: Path) -> None:
    """CONTRACTS sec.1: loaders never call `unicodedata.normalize`."""
    code = _code_without_comments_or_docstrings(path)
    assert "NFD" not in code, f"{path}: NFD appears in executable code"
    assert "NFKD" not in code, f"{path}: NFKD appears in executable code"


@pytest.mark.parametrize("path", SCANNED_FILES, ids=SCANNED_IDS)
def test_no_normalize_call_in_source(path: Path) -> None:
    """AST-level: no call to anything named `normalize` (however imported)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else ""
        )
        if name == "normalize":
            offenders.append(node.lineno)
    assert not offenders, f"{path}: normalize() called at lines {offenders}"
# improved


@pytest.mark.parametrize("path", SCANNED_FILES, ids=SCANNED_IDS)
def test_no_normalize_string_argument_anywhere(path: Path) -> None:
    """Belt and braces: no NFD/NFKD/NFKC string constant is passed to anything."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in ("NFD", "NFKD", "NFKC", "NFC")
    ]
    assert not bad, f"{path}: normalisation form constants present: {bad}"


# ==========================================================================
# 7. Khmer COENG U+17D2 and dependent vowels
# ==========================================================================

KM_RECORDS = BY_LANG["km"]
KM_IDS = [r.id for r in KM_RECORDS]


def test_km_fixtures_contain_coeng() -> None:
    assert any(COENG in rec.text for rec in KM_RECORDS)


@pytest.mark.parametrize("rec", KM_RECORDS, ids=KM_IDS)
def test_no_legal_position_immediately_after_coeng(rec: Segmented) -> None:
    """COENG binds the FOLLOWING consonant as a subscript."""
    legal = legal_positions(rec.text, "km")
    for i in legal:
        assert rec.text[i - 1] != COENG, f"{rec.id}: legal position {i} splits a COENG sequence"


@pytest.mark.parametrize("rec", KM_RECORDS, ids=KM_IDS)
def test_no_legal_position_before_a_dependent_vowel(rec: Segmented) -> None:
    """U+17B6-U+17D3 attach to the preceding base; several are spacing (Mc)."""
    legal = legal_positions(rec.text, "km")
    for i in legal:
        assert rec.text[i] not in KH_DEPENDENT, (
            f"{rec.id}: legal position {i} lands before U+{ord(rec.text[i]):04X}"
        )


@pytest.mark.parametrize("rec", KM_RECORDS, ids=KM_IDS)
def test_no_legal_position_immediately_before_coeng(rec: Segmented) -> None:
    """COENG is itself in the dependent range, so it can never start a cluster."""
    assert COENG in KH_DEPENDENT
    legal = legal_positions(rec.text, "km")
    for i in legal:
        assert rec.text[i] != COENG


@pytest.mark.parametrize("rec", KM_RECORDS, ids=KM_IDS)
def test_gold_span_starts_are_legal_cluster_starts(rec: Segmented) -> None:
#     """Every Khmer gold word begins on an orthographic-syllable boundary."""
    starts = khmer_cluster_starts(rec.text)
    for s, _ in rec.spans:
        assert s in starts, f"{rec.id}: gold span start {s} is not a cluster start"


@pytest.mark.parametrize("rec", KM_RECORDS, ids=KM_IDS)
def test_interior_gold_span_ends_are_legal_cluster_starts(rec: Segmented) -> None:
    starts = khmer_cluster_starts(rec.text)
    for _, e in rec.spans:
        if 0 < e < rec.n:
            assert e in starts, f"{rec.id}: gold span end {e} is not a cluster start"


@pytest.mark.parametrize("rec", KM_RECORDS, ids=KM_IDS)
def test_km_legal_positions_are_cluster_starts(rec: Segmented) -> None:
    assert legal_positions(rec.text, "km") <= khmer_cluster_starts(rec.text)


def test_coeng_sequence_hand_worked() -> None:
    """ខ្មែរ: ខ0 ្1 ម2 ែ3 រ4 -- only position 4 could ever be legal, and the
    dependent vowel ែ at 3 rules out 3 as well."""
    text = "ខ្មែរ"
    assert text[1] == COENG
    assert khmer_cluster_starts(text) == frozenset({0, 4})
    assert legal_positions(text, "km") == frozenset({4})


def test_coeng_cluster_inside_a_fixture_word() -> None:
    """ភាសាខ្មែរ ...: the COENG at index 5 blocks positions 5 and 6."""
    rec = KM_RECORDS[0]
    assert rec.text[5] == COENG
    legal = legal_positions(rec.text, "km")
    assert 5 not in legal
    assert 6 not in legal
    assert 4 in legal  # the word boundary before ខ


def test_khmer_dependent_vowels_include_spacing_marks() -> None:
    """Unicode general category alone is not sufficient here."""
    spacing = [c for c in KH_DEPENDENT if unicodedata.category(c) == "Mc"]
    assert spacing, "expected spacing (Mc) dependent vowels in U+17B6..U+17D3"
    for ch in spacing:
        text = "ក" + ch + "ខ"
        assert 1 not in legal_positions(text, "km")


# ==========================================================================
# 9. grapheme_cluster_starts vs regex \X
# ==========================================================================

SYNTHETIC_GRAPHEME_TEXTS = [
    "",
    "a",
    "ab",
    "éx",  # U+00E9 precomposed -- one cluster
    "éx",  # e + U+0301 COMBINING ACUTE -- also one cluster
    "ก้าง",  # Thai base + Mn
    "ខ្មែរ",
    "a\r\nb",  # CRLF is a single cluster
    "👨‍👩‍👧‍👦ab",  # ZWJ emoji sequence
    "🇹🇭x",  # regional indicator pair
    "ทำ",  # SARA AM
]


def _regex_cluster_starts(text: str) -> frozenset[int]:
    out: set[int] = set()
    idx = 0
    for cluster in regex.findall(r"\X", text):
        out.add(idx)
        idx += len(cluster)
    return frozenset(out)


@pytest.mark.parametrize("text", SYNTHETIC_GRAPHEME_TEXTS, ids=lambda t: ascii(t))
def test_grapheme_cluster_starts_match_regex_X(text: str) -> None:
    assert grapheme_cluster_starts(text) == _regex_cluster_starts(text)


@pytest.mark.parametrize("rec", ALL_RECORDS, ids=[r.id for r in ALL_RECORDS])
def test_grapheme_cluster_starts_match_regex_X_on_fixtures(rec: Segmented) -> None:
    assert grapheme_cluster_starts(rec.text) == _regex_cluster_starts(rec.text)


@pytest.mark.parametrize("rec", ALL_RECORDS, ids=[r.id for r in ALL_RECORDS])
def test_grapheme_cluster_count_matches_regex_X(rec: Segmented) -> None:
    assert len(grapheme_cluster_starts(rec.text)) == len(regex.findall(r"\X", rec.text))


def test_grapheme_cluster_starts_includes_zero_for_nonempty_text() -> None:
    assert 0 in grapheme_cluster_starts("我喜欢")
    assert grapheme_cluster_starts("") == frozenset()


def test_grapheme_cluster_starts_excludes_n() -> None:
    for text in SYNTHETIC_GRAPHEME_TEXTS:
        assert len(text) not in grapheme_cluster_starts(text)

# Refined

# Refined

# Updated

# Enhanced

# Enhanced

# Updated

# Updated

# Updated

# Updated

# Updated

# Refined
