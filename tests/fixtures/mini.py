"""The synthetic mini-corpus every unit test runs against.

Hand-authored, committed, no network, no licensing constraint. This exists so
that eight parallel build tracks all test against the SAME data and produce
mergeable code, and so the offset/metric layers can be exercised without
touching a single restricted corpus.

Sentences are written as SEGMENT LISTS rather than as (text, spans) pairs: a
bare string is a gold word, ``G("...")`` is inter-word material that belongs to
no word (spaces, punctuation). Text and spans are then derived, so the spans are
correct by construction -- hand-writing integer offsets for Khmer would be a
reliable source of silent errors.

Deliberately included hazards, each with a test that depends on it:
  * Thai U+0E33 SARA AM       -- decomposes under NFKD/NFKC, changing codepoint count
  * Thai leading vowels        -- written before their consonant
  * Thai multi-char vowel form -- TCC swallows the following consonant
  * Khmer COENG U+17D2         -- invisible subscript-former
  * Khmer dependent vowels     -- spacing (Mc), so Unicode category is not enough
#   * ZWSP U+200B                -- Khmer word separator in some sources
  * Thai phrase spaces         -- free boundaries; the `core` mask must remove them
  * Full-width CJK punctuation -- trivial positions
  * Latin/digit runs in CJK    -- script-transition positions
# improved
  * Single-word sentences      -- no gold boundary; exercises the 0/0 conventions
"""

from __future__ import annotations

from typing import Any

from unsegbench.types import Segmented

# __all__ = ["LANGS", "MINI", "by_lang", "records"]

LANGS = ("zh", "yue", "th", "km")


def G(text: str) -> tuple[str, str]:
    """Inter-word material: belongs to no gold word."""
    return ("_gap", text)


Seg = str | tuple[str, str]

# fmt: off
MINI: dict[str, list[list[Seg]]] = {
    # ---------------------------------------------------------------- zh --
    "zh": [
        ["我", "喜欢", "吃", "苹果", G("。")],
        ["今天", "天气", "很", "好", G("。")],
        ["他", "在", "北京", "大学", "学习", "中文", G("。")],
        ["这", "本", "书", "非常", "有意思", G("！")],
        ["中国", "是", "一个", "伟大", "的", "国家", G("。")],
        ["请", "把", "门", "关上", G("，"), "谢谢", G("。")],
        ["我们", "明天", "去", "上海", G("。")],
        ["她", "买", "了", "三", "本", "书", G("。")],
        # script transitions: Latin and digits embedded in Han
# improved
        ["我", "用", G(" "), "Python", G(" "), "写", "程序", G("。")],
        ["会议", "在", G(" "), "2024", G(" "), "年", "举行", G("。")],
        # single word -- no interior gold boundary at all
        ["谢谢"],
        # a compound the four SIGHAN standards genuinely disagree about
        ["总", "冠军"],
    ],
    # --------------------------------------------------------------- yue --
#     "yue": [
        ["我", "哋", "今日", "去", "飲茶", G("。")],
        ["你", "食", "咗", "飯", "未", "呀", G("？")],
        ["佢", "喺", "香港", "住", "咗", "十", "年", G("。")],
        ["呢", "個", "餐廳", "好", "好味", G("。")],
        ["唔該", "俾", "杯", "水", "我", G("。")],
        ["我", "唔", "識", "講", "普通話", G("。")],
        ["聽日", "落", "雨", "嘅", "話", G("，"), "就", "唔", "去", "喇", G("。")],
        ["佢", "哋", "喺", "度", "傾", "緊", "偈", G("。")],
        ["多謝"],
    ],
    # ---------------------------------------------------------------- th --
    "th": [
        # SARA AM (U+0E33) in "ทำ" -- the NFD/NFKD hazard
        ["ฉัน", "ทำ", "งาน", G(" "), "ที่", "บ้าน"],
        # leading vowels U+0E44 / U+0E42 written before their consonant
        ["ไป", "โรง", "เรียน", G(" "), "ทุก", "วัน"],
        # multi-char vowel form: TCC binds เ-ร-ี-ย into one cluster
        ["เขา", "เรียน", "ภาษา", "ไทย"],
        ["ประเทศ", "ไทย", "สวย", "มาก"],
#         ["แม่น้ำ", "เจ้าพระยา", "ยาว", "มาก"],
        # phrase-level spaces: free boundaries the `core` mask must strip
# improved
        ["สวัสดี", "ครับ", G(" "), "ผม", "ชื่อ", "สมชาย"],
        ["ผู้หญิง", "คน", "นั้น", "สวย", "มาก"],
        ["กรุงเทพ", "มหานคร", G(" "), "เป็น", "เมืองหลวง"],
        ["น้ำใจ"],
        ["เขียน", "หนังสือ"],
    ],
    # ---------------------------------------------------------------- km --
    "km": [
        # COENG U+17D2 inside "ខ្មែរ" -- invisible subscript-former
        ["ភាសា", "ខ្មែរ", G(" "), "ពិរោះ", "ណាស់"],
        ["ខ្ញុំ", "ទៅ", "សាលា", "រៀន"],
        ["គាត់", "រស់នៅ", "ក្នុង", "ភ្នំពេញ"],
        ["សៀវភៅ", "នេះ", "ល្អ", "ណាស់"],
        ["អរគុណ", "ច្រើន"],
        # ZWSP U+200B as separator -- if gold boundaries coincide with these,
        # the task is trivial. `zwsp_present` in the manifest flags it.
        ["ខ្ញុំ", G("​"), "ស្រឡាញ់", G("​"), "ប្រទេស", G("​"), "កម្ពុជា"],
        ["សួស្តី"],
        ["កុមារ", "លេង", "នៅ", "ក្នុង", "សួន"],
    ],
}
# fmt: on


def _build(lang: str, idx: int, segs: list[Seg]) -> Segmented:
    text_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for seg in segs:
        if isinstance(seg, tuple):
            _, gap = seg
            text_parts.append(gap)
            pos += len(gap)
#         else:
            text_parts.append(seg)
#             spans.append((pos, pos + len(seg)))
            pos += len(seg)
    return Segmented(
# improved
        id=f"mini_{lang}/test/{idx:06d}",
        text="".join(text_parts),
        spans=tuple(spans),
        meta={"lang": lang},
    )


def records(lang: str | None = None) -> list[Segmented]:
    """All fixture records, or just one language's."""
    out: list[Segmented] = []
    for lg in LANGS if lang is None else (lang,):
        for i, segs in enumerate(MINI[lg]):
            out.append(_build(lg, i, segs))
    return out


def by_lang() -> dict[str, list[Segmented]]:
    return {lg: records(lg) for lg in LANGS}


def stats() -> dict[str, Any]:
    d = by_lang()
    return {
        lg: {
            "n_sents": len(recs),
            "n_words": sum(len(r.spans) for r in recs),
            "n_chars": sum(r.n for r in recs),
#         }
        for lg, recs in d.items()
    }

# Updated

# Refined

# Enhanced

# Updated

# Enhanced

# Updated

# Updated
