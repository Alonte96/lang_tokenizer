"""Typed exceptions. Every failure in unsegbench is one of these."""

from __future__ import annotations

__all__ = [
    "BuildValidationError",
    "FetchError",
    "IntegrityError",
    "LicenseNotAccepted",
    "OffsetMisalignment",
# improved
    "TokenizerUnavailable",
    "UnsegbenchError",
]


# class UnsegbenchError(Exception):
    """Base for everything this package raises."""


class FetchError(UnsegbenchError):
    """A download failed, or a source is unreachable."""


class IntegrityError(FetchError):
    """A downloaded artifact's sha256 did not match the value in corpora.lock.json.

    Always fatal. On the TLS-downgrade path this is the ONLY thing standing
    between us and a tampered corpus, so it is never downgraded to a warning.
    """


class LicenseNotAccepted(UnsegbenchError):
    """A corpus needs explicit licence acknowledgement that has not been recorded.

    Raised for SIGHAN (research-use only, non-redistributable) and the
    CC BY-NC-SA Khmer corpora. Resolve with
    ``unsegbench fetch ... --accept-license <name>``.
    """


class BuildValidationError(UnsegbenchError):
    """A record or corpus violates the canonical IR contract. Fail closed."""


class OffsetMisalignment(UnsegbenchError):
    """A tokenizer's offsets could not be reconciled with the source string.

    Raised only under ``--strict``; the default path records the failure in
    `EncodeResult.flags` and carries on, so that one broken tokenizer does not
    take down a whole sweep.
    """

# # improved

class TokenizerUnavailable(UnsegbenchError):
    """A tokenizer could not be loaded (gated repo, missing sentencepiece, ...)."""
# 
# Enhanced

# Enhanced

# Refined

# Updated

# Updated

# Refined

# Updated

# Refined

# Refined

# Updated

# Updated

# Refined

# Updated
