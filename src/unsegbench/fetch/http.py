"""Streaming download with resume, hash verification and a scoped TLS downgrade.

Every other track calls `download` and nothing else.

WHY this is not three lines of ``httpx.get``:

* The corpora are tens of megabytes over academic hosting that stalls. A partial
  file must be resumable (``Range``) and must never be mistaken for a complete
  one, so bytes land in ``<dest>.part`` and only an already-verified file is
  ``os.replace``d into place. A crash therefore leaves a resumable ``.part``,
  never a truncated artifact that later looks cached.
* SIGHAN's HTTPS certificate has been broken for years, and it is the only way
  to obtain icwb2-data. The downgrade to plaintext is host-scoped
  (`TLS_DOWNGRADE_ALLOWLIST`) and HASH-GATED: without an expected sha256 there
  is no integrity guarantee left at all, so we refuse rather than accept
  unverified plaintext. There is no global ``verify=False`` and no import-time
  warning suppression anywhere in this package -- both would silently weaken
  every other download too.
* What actually happened is recorded next to the bytes in ``_download.json``
  (including ``tls_downgraded``), so ``unsegbench doctor`` can surface it and a
  reader of the cache can tell how each artifact was obtained.
"""

from __future__ import annotations

import hashlib
import os
import ssl
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import orjson
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from unsegbench.errors import FetchError, IntegrityError

__all__ = ["TLS_DOWNGRADE_ALLOWLIST", "download", "sha256_file"]

#: Hosts where an HTTPS failure may be retried over plain HTTP.
#:
#: SIGHAN's certificate has been broken for years. The downgrade is host-scoped
#: and HASH-GATED: on the plaintext path a sha256 match is mandatory, because it
#: is the only integrity guarantee left. Never a global ``verify=False``, never
#: an import-time warning suppression.
TLS_DOWNGRADE_ALLOWLIST: frozenset[str] = frozenset({"sighan.cs.uchicago.edu"})

#: Read in 1 MiB blocks: big enough that hashing 50 MB is not syscall-bound,
#: small enough that the progress bar still moves.
_CHUNK = 1 << 20

_TIMEOUT = httpx.Timeout(connect=20.0, read=60.0, write=60.0, pool=20.0)


def sha256_file(path: Path) -> str:
    """Hex digest of a file, streamed.

    Args:
        path: file to digest. Read in fixed-size blocks so a 50 MB corpus never
            has to be resident in memory.

    Returns:
        The lowercase hex sha256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _host(url: str) -> str:
    return urlsplit(url).hostname or ""


def _to_http(url: str) -> str:
    """Same URL over plain HTTP. Used only for allowlisted, hash-gated hosts."""
    parts = urlsplit(url)
    return urlunsplit(("http", parts.netloc, parts.path, parts.query, parts.fragment))


def _write_provenance(
    dest: Path,
    *,
    url_requested: str,
    url_used: str,
    digest: str,
    n_bytes: int,
    tls_downgraded: bool,
) -> None:
    """Record how this artifact was obtained, beside the artifact itself."""
    payload: dict[str, Any] = {
        "url_requested": url_requested,
        "url_used": url_used,
        "sha256": digest,
        "bytes": n_bytes,
        "fetched_at": time.time(),
        "tls_downgraded": tls_downgraded,
    }
    (dest.parent / "_download.json").write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def _stream_to_part(url: str, part: Path, *, progress: bool) -> None:
    """Fetch ``url`` into ``part``, resuming if it already holds bytes.

    Args:
        url: the exact URL to request (post-downgrade, if any).
        part: the ``.part`` scratch file. Its current size is the resume offset.
        progress: draw a rich progress bar. ``False`` disables it entirely
            rather than rendering to a null console, so nothing touches stdout.

    Raises:
        FetchError: any non-resumable HTTP status.
        httpx.ConnectError | ssl.SSLError: propagated, so the caller can decide
            whether a TLS downgrade is permitted for this host.
    """
    resume_from = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with (
        httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client,
        client.stream("GET", url, headers=headers) as resp,
    ):
        if resp.status_code == 416:
            # Range beyond EOF: the server considers what we already have complete.
            return
        if resp.status_code == 206:
            mode = "ab"
        elif resp.status_code == 200:
            # Server ignored the Range header (or we had nothing): start over.
            resume_from = 0
            mode = "wb"
        else:
            raise FetchError(f"{url}: HTTP {resp.status_code}")

        declared = resp.headers.get("Content-Length")
        total = (resume_from + int(declared)) if declared is not None else None

        with (
            Progress(
                TextColumn("[bold blue]{task.fields[name]}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                disable=not progress,
            ) as bar,
            open(part, mode) as fh,
        ):
            task = bar.add_task("download", total=total, name=part.name)
            bar.update(task, completed=resume_from)
            for chunk in resp.iter_bytes(_CHUNK):
                fh.write(chunk)
                bar.update(task, advance=len(chunk))


def download(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Fetch ``url`` to ``dest``, atomically.

    Behaviour:
        * no-op if ``dest`` exists and its digest matches ``sha256``
        * writes to ``dest.with_suffix(dest.suffix + ".part")`` then ``os.replace``
        * resumes a partial download with an HTTP ``Range`` request
        * on ``ssl.SSLError``/``ConnectError``, retries over http **only** if the
          host is in `TLS_DOWNGRADE_ALLOWLIST`, and then **requires** ``sha256``
        * writes ``_download.json`` beside ``dest`` recording url_used, digest,
          bytes, and ``tls_downgraded``

    Raises:
        IntegrityError: digest mismatch. Always fatal, never downgraded.
        FetchError: unreachable, or a downgrade was needed without a known hash.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and not force:
        if sha256 is None:
            return dest
        if sha256_file(dest) == sha256:
            return dest
        # A cached file that no longer matches the lock is stale, not fatal:
        # re-fetch and let the post-download check decide.
        dest.unlink()
    if force:
        part.unlink(missing_ok=True)

    url_used = url
    tls_downgraded = False
    try:
        _stream_to_part(url, part, progress=progress)
    except (ssl.SSLError, httpx.ConnectError, httpx.ConnectTimeout) as exc:
        host = _host(url)
        if urlsplit(url).scheme != "https" or host not in TLS_DOWNGRADE_ALLOWLIST:
            raise FetchError(f"{url}: {exc}") from exc
        if sha256 is None:
            raise FetchError(
                f"{url}: TLS failed and {host} is allowlisted for a plaintext retry, but no "
                f"expected sha256 was supplied. The hash is the only integrity guarantee on "
                f"the plaintext path, so the download is refused. Record the digest in "
                f"corpora.lock.json first."
            ) from exc
        url_used = _to_http(url)
        tls_downgraded = True
        try:
            _stream_to_part(url_used, part, progress=progress)
        except (ssl.SSLError, httpx.HTTPError) as exc2:
            raise FetchError(f"{url_used}: {exc2}") from exc2
    except httpx.HTTPError as exc:
        raise FetchError(f"{url}: {exc}") from exc

    if not part.exists():
        raise FetchError(f"{url}: no bytes were written")

    digest = sha256_file(part)
    if sha256 is not None and digest != sha256:
        # Do NOT keep the bytes around: a later resume would append to a body we
        # already know is wrong, and every subsequent digest would be garbage.
        part.unlink(missing_ok=True)
        raise IntegrityError(
            f"{url_used}: sha256 mismatch\n  expected {sha256}\n  got      {digest}"
        )

    n_bytes = part.stat().st_size
    os.replace(part, dest)
    _write_provenance(
        dest,
        url_requested=url,
        url_used=url_used,
        digest=digest,
        n_bytes=n_bytes,
        tls_downgraded=tls_downgraded,
    )
    return dest
