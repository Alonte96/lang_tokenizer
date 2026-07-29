"""hkcancor-multi: one Cantonese text, several defensible segmentations of it.

WHY THIS CORPUS IS THE PROJECT'S DIFFERENTIATOR. Every other corpus here hands
us one gold segmentation and invites the reader to treat it as ground truth.
Lam, Lau and Lee (LREC-COLING 2024) re-segmented HKCanCor on the explicit
premise that there is no consensus about what a Chinese word is, and encoded
that by labelling each position with one of four tiers rather than a binary
boundary. Registering one `CorpusSpec` per tier gives us several genuinely
different segmentations of the SAME string, which is the only clean way to
separate "this tokenizer is bad" from "this tokenizer answers to a different
convention".

WHAT THE LABELS ACTUALLY ARE (verified against the shipped parquet, not the
paper). Each row is ``chars: list[str]`` and ``labels: list[int]`` of equal
length, indexing ``["D", "I", "P", "S"]``. The label on character ``j`` describes
the boundary to its LEFT, i.e. exactly our position ``j``; character 0 is
labelled ``S`` by convention and is discarded, since position 0 is outside every
universe (CONTRACTS.md sec.2). ``I`` (Intermediate) means "no boundary". The
other three are the original notation's three separator strengths, weakest
first: ``D`` dash, ``P`` pipe, ``S`` space.

    original:  即係 噉樣 嗰-啲 呀 ?
    chars:     即 係 噉 樣 嗰 啲 呀 ?
    labels:    S  I  S  I  S  D  S  S

THE TIERS ARE A CHAIN, NOT FOUR CONVENTIONS. Because a position carries exactly
one ordinal label, ``B(s) subset B(p) subset B(d)`` holds by construction, and
`tier_boundaries` is checked against that in the test suite on real data. So
this is a pure GRANULARITY axis: a tokenizer cannot be "right for tier s and
wrong for tier d" in any way other than cutting too coarsely or too finely.
That is a weaker -- but much cleaner -- claim than "four rival conventions", and
downstream analysis must not overstate it.

TWO CONSEQUENCES WORTH KNOWING BEFORE USING THESE:

* ``P`` is nearly vacuous. It occurs 296 times in train against 108,105 ``S``,
  and ZERO times in the test split, so ``hkcancor_p`` is byte-identical to
  ``hkcancor_s`` on test. Treat ``s`` and ``d`` as the two real tiers.
* There is no ``hkcancor_i``. ``I`` is the "no boundary" label, so a tier that
  admitted it would put a boundary at every position -- the character
  segmentation, derivable from ``text`` alone with zero information from the
  annotation. Registering it would hand the character-tokenizer baseline a
  perfect score on a corpus in the ``@permissive`` leaderboard. If it is ever
  wanted as an explicit floor, add it as a BASELINE, not as gold.

The four-way split file also carries a ``validation`` split (221 rows). The IR
recognises only ``test`` and ``train`` (`CorpusSpec.splits`), so it is not
exposed; nothing downstream would consume it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from unsegbench.corpora.base import Artifact, CorpusSpec
from unsegbench.fetch.hfsrc import fetch_dataset_file
from unsegbench.types import Segmented, Span

__all__ = ["ENTRIES", "REPO_ID", "TIERS", "HkCanCorMulti", "tier_boundaries"]

REPO_ID = "AlienKevin/hkcancor-multi"
_SOURCE_URL = f"https://huggingface.co/datasets/{REPO_ID}"
_VERSION = "lrec-coling-2024"

#: ClassLabel index -> name, from the parquet's ``huggingface`` schema metadata.
_LABEL_NAMES: tuple[str, ...] = ("D", "I", "P", "S")

#: Parquet shard names, PINNED. Resolved once with
#: `unsegbench.fetch.hfsrc.list_dataset_files`; an upstream re-shard must fail
#: loudly rather than silently give us different data.
_FILES: dict[str, str] = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

#: Tier -> the label set that counts as a boundary. Strictly nested by
#: construction, coarsest first; see the module docstring.
TIERS: dict[str, frozenset[str]] = {
    "s": frozenset({"S"}),
    "p": frozenset({"S", "P"}),
    "d": frozenset({"S", "P", "D"}),
}


def tier_boundaries(labels: list[int] | tuple[int, ...], tier: str) -> frozenset[int]:
    """Interior boundary positions for one tier of one row.

    Position 0 is dropped: the leftmost character is labelled ``S`` by the
    dataset's own convention, and sentence edges are excluded from every
    universe anyway.

    Args:
        labels: the row's raw ClassLabel indices into `_LABEL_NAMES`.
        tier: a key of `TIERS`.

    Returns:
        Positions ``1 <= i < len(labels)`` whose label is a boundary at ``tier``.
    """
    admitted = TIERS[tier]
    return frozenset(i for i in range(1, len(labels)) if _LABEL_NAMES[labels[i]] in admitted)


class HkCanCorMulti(CorpusSpec):
    """One tier of the multi-tiered HKCanCor re-segmentation.

    All tiers read the same parquet and emit the same ``text`` in the same order,
    so records pair across tiers by index. That index alignment is what makes
    the convention experiment expressible, and it is asserted in the tests.
    """

    lang = "yue"
    script = "Hant"
    license = "CC BY 4.0"
    redistributable = True
    source_url = _SOURCE_URL
    version = _VERSION
    #: Empty on purpose. Every character carries a label, so the spans TILE the
    #: text and nothing may be left uncovered. If upstream ever introduces a
    #: space, `types.validate_record` fails closed rather than quietly dropping it.
    gap_charset = ""

    def __init__(self, tier: str) -> None:
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIERS)}")
        self.tier = tier
        self.corpus_id = f"hkcancor_{tier}"
        self.convention = f"hkcancor-multi-{tier}"
        self.notes = (
            f"Multi-tiered HKCanCor (Lam, Lau & Lee, LREC-COLING 2024), tier {tier.upper()}. "
            "Tiers are a nested chain S subset P subset D by construction -- a granularity "
            "axis, not rival conventions. P is near-vacuous (296 train labels, 0 in test), "
            "so it coincides with S on the test split. Upstream 'validation' split (221 rows) "
            "is not exposed. Tier I is deliberately unregistered: it is the no-boundary label, "
            "so it would yield the character segmentation."
        )

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        """Empty: the files come from the HF hub, not a plain URL.

        `unsegbench.fetch.hfsrc.fetch_dataset_file` resolves and caches them into
        ``raw_dir`` on first `parse`, because hub URLs are revision-resolved
        redirects that the generic hash-gated downloader has no way to pin.
        """
        return ()

    def parse(self, raw_dir: Path, split: str) -> Iterator[Segmented]:
        """Yield one record per row, spans tiling ``text``."""
        filename = _FILES.get(split)
        if filename is None:
            return
        import pyarrow.parquet as pq

        path = fetch_dataset_file(REPO_ID, filename, raw_dir)
        table = pq.read_table(path, columns=["chars", "labels"])
        chars_col = table.column("chars").to_pylist()
        labels_col = table.column("labels").to_pylist()

        for i, (chars, labels) in enumerate(zip(chars_col, labels_col, strict=True)):
            if len(chars) != len(labels):
                raise ValueError(
                    f"{self.corpus_id}/{split}/{i:06d}: {len(chars)} chars vs {len(labels)} labels"
                )
            text = "".join(chars)
            if not text:
                continue
            cuts = sorted(tier_boundaries(labels, self.tier))
            spans: list[Span] = []
            prev = 0
            for cut in (*cuts, len(text)):
                spans.append((prev, cut))
                prev = cut
            yield Segmented(
                id=f"{self.corpus_id}/{split}/{i:06d}",
                text=text,
                spans=tuple(spans),
                meta={"tier": self.tier},
            )


#: Coarsest first. Registered by `unsegbench.corpora.registry`.
ENTRIES: tuple[CorpusSpec, ...] = (
    HkCanCorMulti("s"),
    HkCanCorMulti("p"),
    HkCanCorMulti("d"),
)
