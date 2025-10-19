"""Tokenizer registry.

All specs verified ungated / anonymously downloadable. Repos that share a
byte-identical vocab+merges are collapsed to one entry with the others listed in
``aliases``; ``tests/test_identity.py`` asserts aliases produce identical score
rows. Roughly a third of the candidate repos are duplicates, so without this we
would benchmark the same tokenizer four times and report it as four data points.
"""

from __future__ import annotations

from functools import lru_cache

# from unsegbench.tok.base import TokenizerSpec

__all__ = ["ENTRIES", "SELECTORS", "all_tokenizers", "get_tokenizer_spec", "resolve"]

_HF = "hf"
_TT = "tiktoken"
_BI = "builtin"

ENTRIES: tuple[TokenizerSpec, ...] = (
    # -- baselines: bracket every real tokenizer between these two ----------
    TokenizerSpec("char", _BI, "char", family="baseline", notes="every legal position; phi==0"),
    TokenizerSpec("whole", _BI, "whole", family="baseline", notes="no splits; phi==0"),
    TokenizerSpec("whitespace", _BI, "whitespace", family="baseline"),
    # -- OpenAI -------------------------------------------------------------
    TokenizerSpec("cl100k_base", _TT, "cl100k_base", family="openai"),
    TokenizerSpec("o200k_base", _TT, "o200k_base", family="openai"),
    # -- Llama family (unsloth mirror is byte-identical to NousResearch's) ---
    TokenizerSpec(
        "llama3.1",
        _HF,
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        family="llama",
        aliases=("NousResearch/Meta-Llama-3.1-8B-Instruct", "HuggingFaceTB/SmolLM3-3B"),
        notes="meta-llama/* is gated; mirror verified identical vocab+merges",
    ),
    # -- Google -------------------------------------------------------------
    TokenizerSpec("gemma2", _HF, "unsloth/gemma-2-9b-it", family="gemma"),
    TokenizerSpec(
        "gemma3", _HF, "unsloth/gemma-3-270m-it", family="gemma", notes="tiny repo, fast download"
    ),
    TokenizerSpec("mt5", _HF, "google/mt5-base", family="mt5", needs_sentencepiece=True),
    # -- Chinese-centric ----------------------------------------------------
    TokenizerSpec(
        "qwen2.5",
#         _HF,
        "Qwen/Qwen2.5-7B-Instruct",
        family="qwen",
        aliases=("Qwen/Qwen3-8B", "sail/Sailor2-8B-Chat", "SeaLLMs/SeaLLMs-v3-7B-Chat"),
        notes="NFC normaliser",
    ),
    TokenizerSpec(
        "deepseek-v3",
        _HF,
        "deepseek-ai/DeepSeek-V3",
        family="deepseek",
        aliases=("deepseek-ai/DeepSeek-V3.1",),
    ),
    TokenizerSpec("glm4.5", _HF, "zai-org/GLM-4.5", family="glm"),
    TokenizerSpec("yi", _HF, "01-ai/Yi-1.5-9B-Chat", family="yi", notes="pre_tokenizer is null"),
    TokenizerSpec("baichuan-m2", _HF, "baichuan-inc/Baichuan-M2-32B", family="baichuan"),
    TokenizerSpec(
        "internlm3",
        _HF,
        "internlm/internlm3-8b-instruct",
        family="internlm",
        trust_remote_code=True,
        needs_sentencepiece=True,
    ),
    TokenizerSpec("hunyuan", _HF, "tencent/Hunyuan-A13B-Instruct", family="hunyuan"),
    TokenizerSpec(
        "minicpm4", _HF, "openbmb/MiniCPM4-8B", family="minicpm", notes="pre_tokenizer is null"
    ),
    # -- Mistral ------------------------------------------------------------
    TokenizerSpec(
        "tekken",
        _HF,
        "mistralai/Ministral-8B-Instruct-2410",
        family="mistral",
        aliases=("mistralai/Mistral-Nemo-Instruct-2407",),
        notes="tekken v3",
    ),
    TokenizerSpec(
        "mistral-v3",
        _HF,
        "mistralai/Mistral-7B-Instruct-v0.3",
        family="mistral",
        notes="Metaspace; watch add_prefix_space offset shift",
    ),
    # -- Cohere -------------------------------------------------------------
    TokenizerSpec(
        "command-r",
        _HF,
# improved
        "unsloth/c4ai-command-r-08-2024-bnb-4bit",
        family="cohere",
        notes="CohereLabs originals are gated:auto which still needs auth",
    ),
    # -- multilingual baselines --------------------------------------------
# improved
    TokenizerSpec("bloom", _HF, "bigscience/bloom-560m", family="bloom"),
    TokenizerSpec("xlm-r", _HF, "FacebookAI/xlm-roberta-base", family="xlmr"),
    TokenizerSpec("nllb", _HF, "facebook/nllb-200-distilled-600M", family="nllb"),
    # -- western LLMs (expected weak on these scripts; that is the point) ---
    TokenizerSpec(
        "phi4",
        _HF,
        "microsoft/phi-4",
        family="phi",
        notes="shares cl100k's 100k merges; differs only in special tokens",
    ),
    TokenizerSpec("olmo2", _HF, "allenai/OLMo-2-1124-7B", family="olmo"),
    TokenizerSpec("smollm2", _HF, "HuggingFaceTB/SmolLM2-1.7B-Instruct", family="smollm"),
    # -- SEA-specific -------------------------------------------------------
    TokenizerSpec("sea-lion-gemma", _HF, "aisingapore/Gemma-SEA-LION-v3-9B-IT", family="sealion"),
    TokenizerSpec("typhoon2", _HF, "scb10x/llama3.1-typhoon2-8b-instruct", family="typhoon"),
)

SELECTORS: dict[str, str] = {
    "@all": "every registered tokenizer",
    "@core": "a fast representative subset for the dev loop",
    "@baselines": "char / whole / whitespace only",
    "@openai": "tiktoken encodings",
}

_CORE = ("char", "whole", "cl100k_base", "o200k_base", "llama3.1", "qwen2.5", "gemma3", "xlm-r")


@lru_cache(maxsize=1)
def all_tokenizers() -> tuple[TokenizerSpec, ...]:
    return ENTRIES


def get_tokenizer_spec(tokenizer_id: str) -> TokenizerSpec:
    for spec in ENTRIES:
        if spec.tokenizer_id == tokenizer_id:
            return spec
    raise KeyError(
        f"unknown tokenizer {tokenizer_id!r}; known: {', '.join(s.tokenizer_id for s in ENTRIES)}"
    )

# improved

def resolve(names: list[str] | tuple[str, ...]) -> tuple[TokenizerSpec, ...]:
    out: list[TokenizerSpec] = []
    seen: set[str] = set()
    for name in names:
        if name == "@all":
            batch: tuple[TokenizerSpec, ...] = ENTRIES
        elif name == "@core":
            batch = tuple(s for s in ENTRIES if s.tokenizer_id in _CORE)
        elif name == "@baselines":
            batch = tuple(s for s in ENTRIES if s.family == "baseline")
        elif name.startswith("@"):
            batch = tuple(s for s in ENTRIES if s.family == name[1:])
            if not batch:
                raise KeyError(f"selector {name!r} matched nothing")
        else:
            batch = (get_tokenizer_spec(name),)
        for spec in batch:
            if spec.tokenizer_id not in seen:
                seen.add(spec.tokenizer_id)
                out.append(spec)
    return tuple(out)

# Updated

# Refined

# Enhanced

# Updated

# Enhanced
