"""Fetch a whole GitHub repository snapshot as a tarball.

WHY this exists instead of a per-file `Artifact`: Universal Dependencies ships
one GitHub repo per treebank, and the files inside are named after a release
(``zh_gsd-ud-train.conllu``) but live at a branch head that is re-tagged every
six months. Enumerating individual raw.githubusercontent URLs would hardcode a
file list that goes stale at the next UD release, and would cost one request per
split. A treebank is a few megabytes of plain text, so one codeload tarball is
both cheaper and more honest about what we actually pinned.

The other reason is branch drift: UD repos are historically on ``master``, but
newer treebanks are created on ``main``. We try the requested branch first and
fall back, rather than making every caller know which era its treebank is from.

Integrity: codeload tarballs are NOT byte-stable (git re-compresses), so a
recorded sha256 is only meaningful when the caller pins a tag rather than a
branch. ``sha256`` is therefore optional here and is passed straight through to
`unsegbench.fetch.http.download`, which treats a mismatch as fatal.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from unsegbench.errors import FetchError
from unsegbench.fetch.http import download

__all__ = ["CODELOAD", "fetch_repo", "tarball_url"]

#: GitHub's tarball endpoint. Not ``api.github.com``: codeload is unauthenticated
#: and not subject to the API rate limit, which matters when a fresh user builds
#: five treebanks in a row.
CODELOAD = "https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/{branch}"

#: Branches to try, in order, after the caller's own choice. UD repos are mostly
#: on ``master``; treebanks added after GitHub's default-branch change are on
#: ``main``. A 404 on one is not an error, it is the wrong guess.
_FALLBACK_BRANCHES: tuple[str, ...] = ("master", "main")


def tarball_url(owner: str, repo: str, branch: str) -> str:
    """The codeload URL for one branch snapshot.

    Args:
        owner: GitHub owner or organisation, e.g. ``"UniversalDependencies"``.
        repo: repository name, e.g. ``"UD_Chinese-GSD"``.
        branch: branch name, e.g. ``"master"``.

    Returns:
        The ``https://codeload.github.com/...`` tarball URL.
    """
    return CODELOAD.format(owner=owner, repo=repo, branch=branch)


def _extracted_root(dest_dir: Path, repo: str) -> Path | None:
    """An already-extracted snapshot of ``repo`` under ``dest_dir``, if any.

    GitHub names the tarball's single top-level directory ``<repo>-<branch>``, so
    any directory with that prefix is a previous extraction of this repo and the
    fetch can be skipped entirely.
    """
    if not dest_dir.is_dir():
        return None
    for child in sorted(dest_dir.iterdir()):
        if child.is_dir() and (child.name == repo or child.name.startswith(f"{repo}-")):
            return child
    return None


def _extract(tarball: Path, dest_dir: Path) -> Path:
    """Unpack ``tarball`` into ``dest_dir`` and return its top-level directory.

    Args:
        tarball: a ``.tar.gz`` with a single top-level directory (codeload's
            layout).
        dest_dir: where to unpack. Created if missing.

    Returns:
        The top-level directory, read from the archive rather than assumed, so a
        change in GitHub's naming shows up as a wrong path instead of silently
        extracting somewhere else.

    Raises:
        FetchError: the archive is empty or has no usable top-level directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        roots = {Path(m.name).parts[0] for m in tf.getmembers() if m.name.strip("./")}
        if len(roots) != 1:
            raise FetchError(f"{tarball}: expected one top-level directory, got {sorted(roots)}")
        root = roots.pop()
        # filter="data" rejects absolute paths, "..", devices and symlink escapes.
        tf.extractall(dest_dir, filter="data")
    return dest_dir / root


def fetch_repo(
    owner: str,
    repo: str,
    branch: str,
    dest_dir: Path,
    sha256: str | None = None,
) -> Path:
    """Download and extract a GitHub repo snapshot. Idempotent.

    Args:
        owner: GitHub owner, e.g. ``"UniversalDependencies"``.
        repo: repository name, e.g. ``"UD_Chinese-GSD"``.
        branch: preferred branch. ``"master"`` and ``"main"`` are tried after it
            if it 404s, because UD treebanks are split across both conventions.
        dest_dir: directory to extract into. Created if missing.
        sha256: expected digest of the tarball. Only meaningful for a pinned tag
            -- branch snapshots are not byte-stable. A mismatch is fatal.

    Returns:
        The extracted repository root, i.e. ``dest_dir/<repo>-<branch>``.

    Raises:
        FetchError: every candidate branch failed. The message carries the last
            failure so a genuinely missing repo is distinguishable from a wrong
            branch guess.
    """
    dest_dir = Path(dest_dir)
    existing = _extracted_root(dest_dir, repo)
    if existing is not None:
        return existing

    candidates: list[str] = [branch]
    candidates += [b for b in _FALLBACK_BRANCHES if b != branch]

    last: Exception | None = None
    for cand in candidates:
        tarball = dest_dir / f"{repo}-{cand}.tar.gz"
        try:
            download(tarball_url(owner, repo, cand), tarball, sha256=sha256, progress=False)
        except FetchError as exc:  # wrong branch, or the repo is really gone
            last = exc
            continue
        return _extract(tarball, dest_dir)

    raise FetchError(f"{owner}/{repo}: no snapshot for branches {candidates}; last error: {last}")
