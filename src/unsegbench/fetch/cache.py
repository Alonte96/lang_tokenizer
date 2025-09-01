"""Cache layout, licence ledger and locking.

STUB -- Phase 0 contract. Track B implements; every other track codes against
these signatures.

Layout::

    <cache>/
      licenses_accepted.json
      locks/<corpus_id>.lock
      raw/<corpus_id>/<version>/{artifact files}
                               /_download.json     provenance incl. tls_downgraded
                               /extracted/
      canonical/<corpus_id>/<version>/manifest.json
                                     /{test,train}.jsonl.gz
      tokenizers/                                   HF_HOME override, tokenizer files only
      stats/<code_ver>/<tok_fp>/<corpus_sha>/<mask>.parquet
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import orjson
import platformdirs
from filelock import FileLock

from unsegbench.errors import LicenseNotAccepted

__all__ = [
    "LICENSE_GATED",
    "accept_license",
    "cache_root",
    "canonical_dir",
    "is_license_accepted",
    "ledger_path",
    "lock_path",
    "raw_dir",
    "require_license",
    "stats_dir",
    "tokenizer_dir",
]

#: Corpora whose licence forbids redistribution and so needs explicit
#: acknowledgement before we will download. Value is the notice file under
#: ``licenses/`` that we print (a notice -- never the data itself).
LICENSE_GATED: dict[str, str] = {
    "sighan": "SIGHAN2005.txt",  # research use only; competition terms
    "khpos": "CC-BY-NC-SA-4.0.txt",  # khPOS
    "alt": "CC-BY-NC-SA-4.0.txt",  # Khmer ALT
}


def cache_root() -> Path:
    """``$UNSEGBENCH_CACHE`` or the platform cache dir."""
    env = os.environ.get("UNSEGBENCH_CACHE")
    return Path(env).expanduser() if env else Path(platformdirs.user_cache_dir("unsegbench"))


def raw_dir(corpus_id: str, version: str) -> Path:
    return cache_root() / "raw" / corpus_id / version


def canonical_dir(corpus_id: str, version: str) -> Path:
    return cache_root() / "canonical" / corpus_id / version


def tokenizer_dir() -> Path:
    return cache_root() / "tokenizers"


def stats_dir(code_version: str, tok_fingerprint: str, corpus_sha: str) -> Path:
    return cache_root() / "stats" / code_version / tok_fingerprint / corpus_sha


def lock_path(name: str) -> Path:
    p = cache_root() / "locks"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.lock"


def ledger_path() -> Path:
    """The licence-acceptance ledger. One file for the whole cache."""
    return cache_root() / "licenses_accepted.json"


def _read_ledger() -> dict[str, dict[str, object]]:
    """Load the ledger, treating anything unreadable as empty.

    Fails CLOSED: a corrupt or truncated ledger means "nothing was accepted",
    which costs the user one re-acknowledgement. The opposite default would let
    a damaged file silently authorise a restricted download.
    """
    path = ledger_path()
    try:
        data = orjson.loads(path.read_bytes())
    except (FileNotFoundError, orjson.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def is_license_accepted(key: str) -> bool:
    """True if ``key`` has been recorded as accepted in the ledger."""
    return key in _read_ledger()


def accept_license(key: str) -> None:
    """Record acceptance to the ledger. Idempotent.

    Locked with `filelock` and written via a temp file plus ``os.replace``, so
    concurrent ``unsegbench fetch`` processes cannot interleave a read-modify-
    write and lose an acceptance. Re-accepting keeps the ORIGINAL timestamp, so
    a repeat call is a genuine no-op rather than a rewrite.

    Args:
        key: a group name from `LICENSE_GATED`, e.g. ``"sighan"``.
    """
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path("licenses"))):
        ledger = _read_ledger()
        if key in ledger:
            return
        ledger[key] = {"accepted_at": time.time(), "notice": LICENSE_GATED.get(key, "")}
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(orjson.dumps(ledger, option=orjson.OPT_INDENT_2))
        os.replace(tmp, path)


def require_license(key: str) -> None:
    """Raise `LicenseNotAccepted` unless ``key`` has been accepted."""
    if key in LICENSE_GATED and not is_license_accepted(key):
        raise LicenseNotAccepted(
            f"corpus group {key!r} requires explicit licence acknowledgement; "
            f"re-run with --accept-license {key}"
        )
