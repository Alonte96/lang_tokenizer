"""Anonymous file-level fetch from a Hugging Face dataset repo.

WHY THIS EXISTS INSTEAD OF ``datasets``. The ``datasets`` library is
deliberately NOT a dependency. HF removed dataset-script support in
``datasets`` 3.0, so every script-based loader -- which is what
``pythainlp/wisesight1000`` and ``seanghay/khPOS`` still are -- hard-fails on
any modern version. Pinning ``datasets<3`` to work around that would drag a
large, fast-moving dependency into a package whose entire job is to be a stable
measuring instrument. Instead we treat a dataset repo as what it physically is:
a directory of files. `huggingface_hub` resolves and caches them; `pyarrow` or
the stdlib parses them. Nothing in this package ever executes a remote script.

TWO OTHER DELIBERATE CHOICES:

* **The HF cache is redirected**, to ``<unsegbench cache>/hf``. Clobbering the
  user's global ``~/.cache/huggingface`` with benchmark corpora -- or, worse,
  inheriting a half-populated one -- makes builds irreproducible and is rude.
  It sits beside the tokenizer cache (`cache.tokenizer_dir`) so that
  ``unsegbench doctor`` and a plain ``rm -rf`` both see one cache root.
* **Anonymous access only.** ``token=False`` is passed explicitly rather than
  left to default, because the default consults ``HF_TOKEN`` and the local login
  store. Every corpus we read here is public; if a build only succeeds because
  the operator happens to be logged in, that is a reproducibility bug we would
  rather fail on than inherit.

Filenames are PINNED by the callers. Resolve them once with
`list_dataset_files` during development, then hard-code them, so a repo
re-sharding upstream is a loud failure rather than a silently different corpus.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from unsegbench.errors import FetchError
from unsegbench.fetch import cache

__all__ = ["fetch_dataset_file", "hf_cache_dir", "list_dataset_files"]


def hf_cache_dir() -> Path:
    """Where `huggingface_hub` may write, inside our own cache root.

    Sibling of `cache.tokenizer_dir` so the whole download footprint lives under
    one directory the user can inspect or delete.
    """
    return cache.tokenizer_dir().parent / "hf"


def list_dataset_files(repo_id: str) -> list[str]:
    """Every file path in a dataset repo, sorted.

    Development aid: run this once to discover the real parquet shard names,
    then PIN them in the loader. Callers should not resolve filenames at build
    time -- an upstream re-shard must fail loudly, not silently change the data.

    Args:
        repo_id: e.g. ``"AlienKevin/hkcancor-multi"``.

    Returns:
        Sorted repo-relative paths.

    Raises:
        FetchError: the repo is unreachable or does not exist.
    """
    from huggingface_hub import HfApi

    try:
        files = HfApi().list_repo_files(repo_id, repo_type="dataset", token=False)
    except Exception as exc:
        raise FetchError(f"could not list files of HF dataset {repo_id!r}: {exc}") from exc
    return sorted(files)


def fetch_dataset_file(repo_id: str, filename: str, dest_dir: Path) -> Path:
    """Download one file from a dataset repo into ``dest_dir``.

    The file is materialised as a real copy under ``dest_dir`` rather than
    handed back as a path into the HF blob store. Loaders receive
    ``raw_dir = cache/raw/<corpus_id>/<version>/`` and every other corpus in the
    benchmark puts its bytes there, so provenance, ``_download.json`` and manual
    inspection all work the same way regardless of where a corpus came from.

    Args:
        repo_id: e.g. ``"pythainlp/wisesight1000"``.
        filename: repo-relative path, PINNED by the caller.
        dest_dir: destination directory; created if absent. The basename of
            ``filename`` is used, so pinned paths in subdirectories flatten.

    Returns:
        Path to the local copy.

    Raises:
        FetchError: download failed, or the file is not in the repo.
    """
    from huggingface_hub import hf_hub_download

    dest = dest_dir / Path(filename).name
    if dest.exists():
        return dest

    hf_cache = hf_cache_dir()
    hf_cache.mkdir(parents=True, exist_ok=True)
    try:
        src = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            cache_dir=str(hf_cache),
            token=False,
        )
    except Exception as exc:
        raise FetchError(
            f"could not fetch {filename!r} from HF dataset {repo_id!r}: {exc}"
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    shutil.copyfile(src, tmp)
    tmp.replace(dest)
    return dest
