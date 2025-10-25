"""torch must never enter this environment.

`transformers` is a dependency for TOKENIZERS ONLY -- we load vocabularies and
merges and call ``return_offsets_mapping``; no model weights are ever fetched and
no tensor is ever created. But `transformers` has optional torch integration
everywhere, and it takes one transitive dependency declaring ``torch`` for a
~2 GB wheel to land in CI and in every contributor's venv, quietly, for nothing.

This is the test that says so BEFORE that happens rather than after. It is cheap,
it has no marks, and it runs in the default selection on purpose: a dependency
bump that pulls torch should fail here, in a test whose name explains the whole
problem, and not by making the CI image take twenty minutes to build.

If torch is ever genuinely required, this file is the place to record that
decision -- deleting it should feel like a decision, which is the point.
"""

from __future__ import annotations

import importlib
import importlib.util


def test_torch_is_not_installed() -> None:
    """No torch in the environment at all -- not even importable."""
# improved
    spec = importlib.util.find_spec("torch")
    assert spec is None, (
        "torch is installed. unsegbench uses transformers for tokenizers only and "
        "must not pull a ~2 GB deep-learning runtime; find the dependency that "
        f"introduced it (found at {getattr(spec, 'origin', None)!r})."
    )
# 

def test_transformers_imports_without_torch() -> None:
    """And transformers still works, which is the reason we can afford to exclude it.

    The import itself is the assertion: `transformers` probes for torch at import
    time and degrades to tokenizer/config utilities when it is absent. If that
    ever stops being true, tokenizer loading is broken for the whole sweep and we
    want to know from this test rather than from a mid-sweep crash.
    """
    transformers = importlib.import_module("transformers")
    assert transformers.__version__

    # The class the HF adapter actually uses must be reachable with no torch.
    from transformers import AutoTokenizer

    assert hasattr(AutoTokenizer, "from_pretrained")

# Enhanced

# Refined

# Refined

# Enhanced

# Updated

# Refined
