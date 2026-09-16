from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve
import fcntl
import os
import tempfile
from typing import Callable

from orbit.native_llama.gguf_split import parse_split_name, read_gguf_header, validate_split_set
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
    retrieve=urlretrieve,
    progress: DownloadProgress | None = None,
    opener=None,
    on_shard: ShardCallback | None = None,
) -> DownloadResult:
    """Fetch one model. A split GGUF (`<base>-00001-of-00003.gguf`) is fetched
    as its whole shard set: the siblings are derived from the requested name
    (zero padding preserved), complete shards are reused, partial ones resumed,
    and the set is validated before it is reported. Ordinary single files keep
    their original semantics (`retrieve` + temporary file + atomic rename).
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

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        if progress is None:
            retrieve(url, str(tmp_path))
        else:
            retrieve(url, str(tmp_path), _progress_hook(progress))
        tmp_path.replace(destination)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return DownloadResult(path=destination, downloaded=True, url=url, shards=(destination,))


# --- split GGUF sets ---------------------------------------------------------

_CHUNK = 1 << 20
_PART_SUFFIX = ".part"
_LOCK_SUFFIX = ".lock"


class DownloadIncomplete(RuntimeError):
    """The transfer ended before the declared size; the partial file is kept
    so a re-run resumes it."""


def _download_split_set(request, split, requested: Path, *, progress, opener, on_shard) -> DownloadResult:
    directory = requested.parent
    prefix = request.file.rsplit("/", 1)[0] + "/" if "/" in request.file else ""
    shards = tuple(directory / name for name in split.siblings())
    first = shards[0]
    fetched: list[Path] = []
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
        fetched.append(shard)
    validation = validate_split_set(first)
    if not validation.complete:
        # Anything WE just fetched that does not validate is discarded so it
        # cannot pass for a complete shard later; pre-existing files are the
        # user's and are only reported.
        for shard in fetched:
            if any(problem.startswith(f"{shard.name}:") for problem in validation.problems):
                shard.unlink(missing_ok=True)
        raise ValueError("split GGUF set is not consistent: " + "; ".join(validation.problems))
    return DownloadResult(path=first, downloaded=bool(fetched), url=huggingface_resolve_url(
        DownloadRequest(repo=request.repo, file=f"{prefix}{first.name}")), shards=shards)


def _notify(on_shard, index: int, count: int, name: str, action: str) -> None:
    if on_shard is not None:
        on_shard(index, count, name, action)


def _shard_is_sound(path: Path, number: int, count: int) -> bool:
    """A present shard is reused only when its header says what its name says."""
    try:
        if path.stat().st_size == 0:
            return False
        header = read_gguf_header(path, keys=frozenset({"split.no", "split.count"}))
    except (OSError, ValueError):
        return False
    return header.kv.get("split.no") == number and header.kv.get("split.count") == count


def _part_path(destination: Path) -> Path:
    return destination.with_name(destination.name + _PART_SUFFIX)


def fetch_resumable(url: str, destination: Path, *, opener=None, progress: DownloadProgress | None = None,
                    chunk_size: int = _CHUNK) -> Path:
    """Fetch `url` into `destination` through a persistent `<name>.part` file.

    A partial file is continued with an HTTP Range request (a server that
    ignores the range restarts it). The final name appears only after the
    declared size has been written, by an atomic rename, so a partial transfer
    never looks complete. One writer per destination is enforced with a lock
    file; a second concurrent caller fails instead of corrupting the file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(destination)
    lock_path = destination.with_name(destination.name + _LOCK_SUFFIX)
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f"another download is already writing {destination.name}; wait for it to finish") from None
        try:
            existing = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
            with (opener or urlopen)(Request(url, headers=headers), timeout=60) as response:
                status = int(getattr(response, "status", 200) or 200)
                total: int | None = None
                if status == 206 and existing > 0:
                    mode = "ab"
                    content_range = response.headers.get("Content-Range") or ""
                    if "/" in content_range and content_range.rsplit("/", 1)[1].isdigit():
                        total = int(content_range.rsplit("/", 1)[1])
                    elif (response.headers.get("Content-Length") or "").isdigit():
                        total = existing + int(response.headers.get("Content-Length"))
                elif status == 416 and existing > 0:
                    # The server says our offset is at or past the end: the
                    # partial file already holds everything it will ever send.
                    mode = None
                    total = existing
                else:
                    mode = "wb"
                    existing = 0
                    if (response.headers.get("Content-Length") or "").isdigit():
                        total = int(response.headers.get("Content-Length"))
                written = existing
                if mode is not None:
                    with part.open(mode) as handle:
                        if progress is not None:
                            progress(written, total or 0)
                        while True:
                            data = response.read(chunk_size)
                            if not data:
                                break
                            handle.write(data)
                            written += len(data)
                            if progress is not None:
                                progress(written, total or 0)
                        handle.flush()
                        os.fsync(handle.fileno())
            if total is not None and written != total:
                raise DownloadIncomplete(
                    f"{destination.name}: received {written} of {total} bytes; run the download again to resume"
                )
            if written == 0:
                raise DownloadIncomplete(f"{destination.name}: the server sent no data")
            os.replace(part, destination)
        finally:
            # All writes are done by now (success or failure), so the lock file
            # can go while still held: a later writer creates a fresh one.
            lock_path.unlink(missing_ok=True)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return destination


def download_all_for_repo(
    repo: str,
    *,
    models_dir: Path | None = None,
    retrieve=urlretrieve,
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
            retrieve=retrieve,
            progress=progress,
            opener=opener,
            on_shard=on_shard,
        )
        for request in requests
    )
    return DownloadBatchResult(results=results)


def huggingface_resolve_url(request: DownloadRequest) -> str:
    return f"{HF_RESOLVE_BASE}/{request.repo}/resolve/main/{request.file}"


def _progress_hook(progress: DownloadProgress):
    def report(block_count: int, block_size: int, total_size: int) -> None:
        downloaded = max(0, block_count * block_size)
        if total_size > 0:
            downloaded = min(downloaded, total_size)
        progress(downloaded, total_size)

    return report


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
