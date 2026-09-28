from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import contextlib
import fcntl
import os
import re
import time
from http.client import IncompleteRead, RemoteDisconnected
from typing import Callable

from orbit.native_llama.gguf_split import parse_split_name, shard_header_problems, validate_split_set
from orbit.native_llama.model_registry import (
    ModelManifest,
    ModelFileSpec,
    effective_models_dir,
    load_registry,
    local_model_path,
)


HF_RESOLVE_BASE = "https://huggingface.co"
DownloadProgress = Callable[[int, int], None]


@dataclass(frozen=True)
class DownloadRequest:
    repo: str
    file: str


@dataclass(frozen=True)
class DownloadResult:
    path: Path          # the model file the backend opens (the first shard of a split set)
    downloaded: bool    # True when at least one file was fetched in this call
    url: str
    shards: tuple[Path, ...] = ()  # every file of a split set, first shard first


# (shard index 1-based, shard count, filename, action) where action is one of
# "present" (reused as-is), "download" (fetched from scratch), "resume"
# (continued from a partial file).
ShardCallback = Callable[[int, int, str, str], None]


@dataclass(frozen=True)
class DownloadBatchResult:
    results: tuple[DownloadResult, ...]


def parse_huggingface_spec(spec: str, *, prefer: str = "target") -> DownloadRequest:
    cleaned = spec.strip().strip("/")
    if not cleaned:
        raise ValueError("empty Hugging Face model spec")
    parts = cleaned.split("/")
    if len(parts) < 2:
        raise ValueError("expected Hugging Face repo or repo/file")

    repo = "/".join(parts[:2])
    file_part = "/".join(parts[2:])
    if file_part:
        if not file_part.endswith(".gguf"):
            raise ValueError("only explicit .gguf files are supported")
        return DownloadRequest(repo=repo, file=file_part)

    manifest_match = _find_manifest_file_for_repo(repo, prefer=prefer)
    if manifest_match is None:
        raise ValueError(f"repo requires an explicit .gguf file: {repo}")
    return DownloadRequest(repo=manifest_match.repo, file=manifest_match.file)


def download_model(
    spec: str,
    *,
    models_dir: Path | None = None,
    prefer: str = "target",
    progress: DownloadProgress | None = None,
    opener=None,
    on_shard: ShardCallback | None = None,
) -> DownloadResult:
    """Fetch one model. A split GGUF (`<base>-00001-of-00003.gguf`) is fetched
    as its whole shard set: the siblings are derived from the requested name
    (zero padding preserved), complete shards are reused, partial ones resumed,
    and the set is validated before it is reported. Single files use the same
    persistent, size-verified HTTP transfer and atomic publication.
    """
    request = parse_huggingface_spec(spec, prefer=prefer)
    destination = local_model_path(
        ModelFileSpec(repo=request.repo, file=request.file, cache_glob=""),
        models_dir=models_dir or effective_models_dir(),
    )
    split = parse_split_name(request.file)  # raises on a malformed split name
    if split is not None:
        return _download_split_set(request, split, destination, progress=progress, opener=opener, on_shard=on_shard)
    url = huggingface_resolve_url(request)
    if destination.exists():
        return DownloadResult(path=destination, downloaded=False, url=url, shards=(destination,))

    fetch_resumable(url, destination, opener=opener, progress=progress)
    return DownloadResult(path=destination, downloaded=True, url=url, shards=(destination,))


# --- split GGUF sets ---------------------------------------------------------

_CHUNK = 1 << 20
_PART_SUFFIX = ".part"
_LOCK_SUFFIX = ".lock"
_READ_TIMEOUT = 30.0
_MAX_ATTEMPTS = 4
_RETRY_BACKOFF = 1.0
_TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}
_CONTENT_RANGE = re.compile(r"^bytes (?:(?P<start>\d+)-(?P<end>\d+)|\*)/(?:(?P<total>\d+)|\*)$")


class DownloadIncomplete(RuntimeError):
    """The transfer ended before the declared size; the partial file is kept
    so a re-run resumes it."""


def _download_split_set(request, split, requested: Path, *, progress, opener, on_shard) -> DownloadResult:
    directory = requested.parent
    prefix = request.file.rsplit("/", 1)[0] + "/" if "/" in request.file else ""
    shards = tuple(directory / name for name in split.siblings())
    first = shards[0]
    fetched: list[tuple[int, Path]] = []
    for position, shard in enumerate(shards, start=1):
        sibling_request = DownloadRequest(repo=request.repo, file=f"{prefix}{shard.name}")
        url = huggingface_resolve_url(sibling_request)
        if shard.exists() and _shard_is_sound(shard, position - 1, split.count):
            _notify(on_shard, position, split.count, shard.name, "present")
            continue
        if shard.exists():
            raise ValueError(
                f"{shard.name} is present but is not a valid shard {position} of {split.count}; "
                f"remove it and run the download again"
            )
        action = "resume" if _part_path(shard).exists() and _part_path(shard).stat().st_size > 0 else "download"
        _notify(on_shard, position, split.count, shard.name, action)
        fetch_resumable(url, shard, opener=opener, progress=progress)
        fetched.append((position, shard))
    validation = validate_split_set(first)
    if not validation.complete:
        # A shard WE just fetched whose own header is wrong is discarded so it
        # cannot pass for a sound shard later. A cross-shard disagreement
        # (tensor counts) does not say which file is wrong, so nothing is
        # deleted for it; pre-existing files are the user's and only reported.
        for position, shard in fetched:
            if _own_header_problems(shard, position - 1, split.count):
                shard.unlink(missing_ok=True)
        raise ValueError("split GGUF set is not consistent: " + "; ".join(validation.problems))
    return DownloadResult(path=first, downloaded=bool(fetched), url=huggingface_resolve_url(
        DownloadRequest(repo=request.repo, file=f"{prefix}{first.name}")), shards=shards)


def _notify(on_shard, index: int, count: int, name: str, action: str) -> None:
    if on_shard is not None:
        on_shard(index, count, name, action)


def _shard_is_sound(path: Path, number: int, count: int) -> bool:
    """A present shard is reused only when its header says what its name says."""
    return not _own_header_problems(path, number, count)


def _own_header_problems(path: Path, number: int, count: int) -> tuple[str, ...]:
    """What is wrong with one shard on its own (existence, header, split keys
    and their types); the same rules `validate_split_set` applies per shard."""
    return shard_header_problems(path, number, count)


def _part_path(destination: Path) -> Path:
    return destination.with_name(destination.name + _PART_SUFFIX)


class _TransferInterrupted(DownloadIncomplete):
    """A bounded retry may continue the bytes already committed to .part."""


def fetch_resumable(url: str, destination: Path, *, opener=None, progress: DownloadProgress | None = None,
                    chunk_size: int = _CHUNK) -> Path:
    """One locked, persistent .part for single files and shards.

    Network inactivity has a socket timeout. Reads return available bytes
    rather than waiting to fill a large block, so received bytes are persisted
    before another potentially stalled read. Only transient transport failures
    and short bodies retry, with bounded exponential backoff. Protocol errors,
    disk errors and cancellation preserve the partial and propagate immediately.
    A 200 restart uses .part.restart, preserving the original partial until
    publication. If that restart is interrupted, it is resumed first. Another
    200 must match its existing prefix; incompatible bytes fail closed rather
    than losing either partial or accumulating unbounded backup files.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(destination)
    restart = part.with_name(part.name + ".restart")
    lock_path = destination.with_name(destination.name + _LOCK_SUFFIX)
    with _destination_lock(lock_path):
        if destination.exists():
            return destination
        for attempt in range(_MAX_ATTEMPTS):
            try:
                active = restart if restart.exists() else part
                _fetch_attempt(url, destination, active, opener or urlopen, progress, chunk_size)
                # Only a verified final permits removal of obsolete partials.
                for obsolete in (part, restart):
                    obsolete.unlink(missing_ok=True)
                    _etag_path(obsolete).unlink(missing_ok=True)
                return destination
            except _TransferInterrupted as exc:
                if attempt + 1 == _MAX_ATTEMPTS:
                    raise DownloadIncomplete(
                        f"{destination.name}: download interrupted after {_MAX_ATTEMPTS} attempts "
                        f"({exc}); partial files kept beside {destination}; rerun to resume"
                    ) from exc
                time.sleep(_RETRY_BACKOFF * (2 ** attempt))
    raise AssertionError("download retry budget must be positive")


def _network_call(operation):
    """Scope retryability to network operations, never file writes/callbacks."""
    try:
        return operation()
    except HTTPError as exc:
        if exc.code == 416:
            return exc  # process and close it like an ordinary response
        if exc.code not in _TRANSIENT_HTTP:
            exc.close()
            raise
        exc.close()
        raise _TransferInterrupted(f"HTTP {exc.code}") from exc
    except (TimeoutError, ConnectionError, IncompleteRead, RemoteDisconnected) as exc:
        raise _TransferInterrupted(str(exc) or type(exc).__name__) from exc
    except URLError as exc:
        if not isinstance(exc.reason, (TimeoutError, ConnectionError)):
            raise
        raise _TransferInterrupted(str(exc.reason)) from exc


def _etag_path(part: Path) -> Path:
    return part.with_name(part.name + ".etag")


def _fetch_attempt(url, destination, part, opener, progress, chunk_size):
    etag_path = _etag_path(part)
    existing = part.stat().st_size if part.exists() else 0
    saved_etag = _strong_etag(_read_small(etag_path)) if existing else None
    headers = {"Accept-Encoding": "identity"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        if saved_etag:
            headers["If-Range"] = saved_etag
    response = _network_call(lambda: opener(Request(url, headers=headers), timeout=_READ_TIMEOUT))
    with response:
        status = int(getattr(response, "status", 200) or 200)
        if status in _TRANSIENT_HTTP:
            raise _TransferInterrupted(f"HTTP {status}")
        if status not in {200, 206, 416}:
            raise DownloadIncomplete(f"{destination.name}: unexpected HTTP {status}")
        if (response.headers.get("Content-Encoding") or "identity").lower() != "identity":
            raise DownloadIncomplete(f"{destination.name}: encoded representation cannot be resumed safely")
        etag_header = response.headers.get("ETag")
        current_etag = _strong_etag(etag_header)
        if status in {206, 416} and saved_etag and etag_header is not None and current_etag != saved_etag:
            raise DownloadIncomplete(f"{destination.name}: ETag changed in a ranged response")
        if status == 416:
            total = _content_range_total(response.headers)
            if not existing or total != existing:
                raise DownloadIncomplete(f"{destination.name}: 416 does not verify the partial size; partial kept")
            with part.open("rb") as handle:
                os.fsync(handle.fileno())
            if progress:
                progress(existing, total)
            _finalize(part, destination, etag_path)
            return
        length = _content_length(response.headers)
        if status == 206:
            start, end, total = _content_range(response.headers)
            if (not existing or start != existing or end is None or total is None
                    or not (start <= end < total)
                    or length is not None and length != end - start + 1):
                raise DownloadIncomplete(f"{destination.name}: inconsistent Content-Range; partial kept")
            mode = "ab"
            expected = end + 1
        else:
            total = length
            expected = total
            if existing and part == _part_path(destination):
                part = part.with_name(part.name + ".restart")
                etag_path = _etag_path(part)
                existing = 0
            mode = "r+b" if existing else "wb"
        if total is None or total <= 0:
            raise DownloadIncomplete(f"{destination.name}: the server did not declare the file size; transfer not started")
        prefix = existing if status == 200 else 0
        if prefix > total:
            raise DownloadIncomplete(f"{destination.name}: restarted representation is smaller than its partial; partials kept")
        written = 0 if status == 200 else existing
        # Unbuffered writes survive SIGTERM without relying on Python cleanup.
        with part.open(mode, buffering=0) as handle:
            if status == 200 and not prefix:
                _remember_etag(etag_path, current_etag)
            if progress:
                progress(written, total)
            read = getattr(response, "read1", response.read)
            while True:
                data = _network_call(lambda: read(min(chunk_size, expected - written + 1)))
                if not data:
                    break
                if written + len(data) > expected:
                    raise DownloadIncomplete(f"{destination.name}: response exceeds its declared range/size")
                # A server may repeatedly ignore Range. Re-read from byte zero,
                # but retain the existing prefix until it matches completely.
                compare = min(len(data), max(0, prefix - written))
                if compare and handle.read(compare) != data[:compare]:
                    raise DownloadIncomplete(f"{destination.name}: restarted representation changed; partials kept")
                if compare and written + compare == prefix:
                    _remember_etag(etag_path, current_etag)
                remaining = data[compare:]
                if remaining and handle.write(remaining) != len(remaining):
                    raise OSError(f"{destination.name}: short file write; partial kept")
                written += len(data)
                if progress:
                    progress(written, total)
            os.fsync(handle.fileno())
            if os.fstat(handle.fileno()).st_size != max(prefix, written):
                raise DownloadIncomplete(f"{destination.name}: partial size changed during transfer")
        if written != expected or written != total:
            raise _TransferInterrupted(f"received {written} of {total} bytes")
        _finalize(part, destination, etag_path)


@contextlib.contextmanager
def _destination_lock(lock_path: Path):
    """Exclusive, non-blocking ownership of a destination. The lock file is
    removed on release; because a waiter may have opened the old inode, the
    inode is re-checked after locking so two writers can never both own the
    same path."""
    for _attempt in range(16):
        lock = lock_path.open("a+b")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            raise RuntimeError(f"another download is already writing {lock_path.name[: -len(_LOCK_SUFFIX)]}; "
                               f"wait for it to finish") from None
        try:
            current = lock_path.stat().st_ino
        except FileNotFoundError:
            current = None
        if current != os.fstat(lock.fileno()).st_ino:
            lock.close()  # we locked an inode the previous owner already removed
            continue
        try:
            yield lock
        finally:
            # All writes are done by now (success or failure), so the lock
            # file can go while still held: a later writer creates a fresh one.
            lock_path.unlink(missing_ok=True)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        return
    raise RuntimeError(f"could not take the download lock for {lock_path.name[: -len(_LOCK_SUFFIX)]}")


def _finalize(part: Path, destination: Path, etag_path: Path) -> None:
    os.replace(part, destination)
    etag_path.unlink(missing_ok=True)
    _fsync_directory(destination.parent)


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _read_small(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _strong_etag(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if len(value) < 2 or not value.startswith('"') or not value.endswith('"'):
        return None
    return value if all(c == "!" or 35 <= ord(c) <= 126 or ord(c) >= 128 for c in value[1:-1]) else None


def _remember_etag(etag_path: Path, etag: str | None) -> None:
    if etag:
        etag_path.write_text(etag, encoding="utf-8")
    else:
        etag_path.unlink(missing_ok=True)


def _content_length(headers) -> int | None:
    value = (headers.get("Content-Length") or "").strip() if headers is not None else ""
    return int(value) if value.isdigit() else None


def _content_range(headers) -> tuple[int | None, int | None, int | None]:
    """Exact start/end/total; absent or malformed values are not authority."""
    value = (headers.get("Content-Range") or "").strip() if headers is not None else ""
    match = _CONTENT_RANGE.match(value)
    if match is None:
        return None, None, None
    start = int(match.group("start")) if match.group("start") else None
    total = int(match.group("total")) if match.group("total") else None
    end = int(match.group("end")) if match.group("end") else None
    return start, end, total


def _content_range_total(headers) -> int | None:
    start, end, total = _content_range(headers)
    return total if start is None and end is None else None


def download_all_for_repo(
    repo: str,
    *,
    models_dir: Path | None = None,
    progress: DownloadProgress | None = None,
    opener=None,
    on_shard: ShardCallback | None = None,
) -> DownloadBatchResult:
    manifest = _find_manifest_by_target_repo(repo)
    if manifest is None:
        raise ValueError(f"repo is not registered for --all: {repo}")

    requests = [DownloadRequest(repo=manifest.target.repo, file=manifest.target.file)]
    if manifest.mmproj is not None:
        requests.append(DownloadRequest(repo=manifest.mmproj.repo, file=manifest.mmproj.file))
    if manifest.mtp is not None:
        requests.append(DownloadRequest(repo=manifest.mtp.repo, file=manifest.mtp.file))

    results = tuple(
        download_model(
            f"{request.repo}/{request.file}",
            models_dir=models_dir,
            progress=progress,
            opener=opener,
            on_shard=on_shard,
        )
        for request in requests
    )
    return DownloadBatchResult(results=results)


def huggingface_resolve_url(request: DownloadRequest) -> str:
    return f"{HF_RESOLVE_BASE}/{request.repo}/resolve/main/{request.file}"


def _find_manifest_file_for_repo(repo: str, *, prefer: str = "target") -> ModelFileSpec | None:
    for manifest in load_registry():
        if prefer == "mmproj" and manifest.mmproj is not None and manifest.mmproj.repo == repo:
            return manifest.mmproj
        if manifest.target.repo == repo:
            return manifest.target
        if manifest.mtp is not None and manifest.mtp.repo == repo:
            return manifest.mtp
        if prefer != "mmproj" and manifest.mmproj is not None and manifest.mmproj.repo == repo:
            return manifest.mmproj
    return None


def _find_manifest_by_target_repo(repo: str) -> ModelManifest | None:
    for manifest in load_registry():
        if manifest.target.repo == repo:
            return manifest
    return None
