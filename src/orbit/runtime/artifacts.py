from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Callable

from orbit.backend.base import ChatResult, Message, StreamProgress


ARTIFACT_TOOL_NAME = "write_artifact"
ARTIFACT_VERIFY_TOOL_NAME = "verify_artifact"
ARTIFACT_CONTENT_MAX_TOKENS = 4_096
MAX_ARTIFACT_BYTES = 64 * 1024
MAX_ARTIFACT_PATH_CHARS = 512
MAX_ARTIFACT_VERIFICATION_CHARS = 12 * 1024
_TEMP_PREFIX = ".orbit-artifact-"
_UNSUPPORTED_TMPFILE_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EINVAL", None),
        getattr(errno, "ENOENT", None),
    )
    if value is not None
)
_UNSUPPORTED_RENAME_NOREPLACE_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EINVAL", None),
        getattr(errno, "ENOSYS", None),
    )
    if value is not None
)

ARTIFACT_VERIFICATION_PROMPT = (
    "Artifact content generation completed, but nothing has been published yet. "
    "Select exactly one verify_artifact call for the exact pending path. "
    "Choose content when bounded body evidence is needed, otherwise choose text_integrity. "
    "A passing check will publish the artifact atomically. Return only one tool call."
)


def write_artifact_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": ARTIFACT_TOOL_NAME,
            "description": (
                "Create one non-trivial UTF-8 text file through a dedicated content-only generation phase. "
                "Use this instead of embedding file content in shell, JSON, XML, or a heredoc. "
                "Missing parents are created only when create_parents is true. "
                "The file and any new parents are published atomically only after complete generation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative destination path inside the active workdir.",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Set true only when the user authorized replacing an existing regular file.",
                    },
                    "create_parents": {
                        "type": "boolean",
                        "description": "Set true only when missing parent directories should be created.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


def verify_artifact_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": ARTIFACT_VERIFY_TOOL_NAME,
            "description": (
                "Select a bounded verification for the generated content of the pending write_artifact request. "
                "A passing check atomically completes that already-authorized publication; the verifier never edits or executes the content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Exact relative path returned by write_artifact.",
                    },
                    "check": {
                        "type": "string",
                        "description": (
                            "Use content for bounded body evidence or text_integrity for UTF-8, byte-count, and hash verification."
                        ),
                        "enum": [
                            "content",
                            "text_integrity",
                        ],
                    },
                },
                "required": ["path", "check"],
                "additionalProperties": False,
            },
        },
    }


def validate_artifact_path(path: object, *, workdir: Path) -> str | None:
    if not isinstance(path, str) or not path.strip():
        return "error: artifact path must be a non-empty string"
    if len(path) > MAX_ARTIFACT_PATH_CHARS:
        return f"error: artifact path exceeds {MAX_ARTIFACT_PATH_CHARS} characters"
    if "\x00" in path or any(ord(char) < 32 for char in path):
        return "error: artifact path contains an unsafe control character"
    supplied = Path(path)
    if supplied.is_absolute():
        return "error: artifact path must be relative to the workdir"
    lexical_parts = path.split("/")
    if lexical_parts and lexical_parts[0] == ".":
        lexical_parts = lexical_parts[1:]
    if any(part in {"", ".", ".."} for part in lexical_parts):
        return "error: artifact path must identify one file inside the workdir"
    root = workdir.expanduser().resolve()
    lexical = Path(os.path.abspath(root / supplied))
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        return "error: artifact path escapes workdir"
    if not relative.parts or any(part in {"", ".", ".."} for part in supplied.parts):
        return "error: artifact path must identify one file inside the workdir"
    return None


def artifact_content_messages(
    *,
    user_request: str,
    path: str,
    overwrite: bool,
) -> list[Message]:
    return [
        {
            "role": "system",
            "content": (
                "Produce only the complete UTF-8 body to publish at the validated destination. "
                "The first output character must belong to the file body, and the last output character must complete that body. "
                "Do not output the path or discuss saving. Do not add a wrapper, fence, tool envelope, or transfer framing around the body. "
                "Preserve every piece of text required inside the artifact, even when it resembles control syntax. "
                "Fully implement every requirement in the original request without placeholders, stubs, abbreviations, or omissions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original request:\n{user_request}\n\n"
                f"Validated destination: {path}\n"
                f"Overwrite authorized: {'yes' if overwrite else 'no'}\n"
                f"Maximum UTF-8 size: {MAX_ARTIFACT_BYTES} bytes."
            ),
        },
    ]


@dataclass(frozen=True)
class ArtifactPublication:
    path: str
    bytes_written: int
    sha256: str
    overwrite: bool
    created_parent_directories: tuple[str, ...] = ()
    directory_sync_complete: bool = True

    def evidence(self) -> str:
        return "\n".join(
            [
                "artifact_publication: complete",
                f"path: {self.path}",
                f"bytes: {self.bytes_written}",
                f"sha256: {self.sha256}",
                f"overwrite: {'true' if self.overwrite else 'false'}",
                "created_parent_directories: "
                + (", ".join(self.created_parent_directories) if self.created_parent_directories else "none"),
                "directory_sync: " + ("complete" if self.directory_sync_complete else "unconfirmed"),
                "verification_completed: true",
            ]
        )


class ArtifactGenerationError(RuntimeError):
    def __init__(self, message: str, *, result: ChatResult) -> None:
        super().__init__(message)
        self.result = result


def _open_private_artifact_temp(parent_fd: int) -> tuple[int, str | None, bool]:
    tmpfile = getattr(os, "O_TMPFILE", 0)
    if tmpfile:
        try:
            fd = os.open(
                ".",
                os.O_RDWR | tmpfile | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            if exc.errno not in _UNSUPPORTED_TMPFILE_ERRNOS:
                raise
        else:
            return fd, None, True

    fd, name = _open_named_artifact_temp(parent_fd)
    return fd, name, False


def _open_named_artifact_temp(parent_fd: int) -> tuple[int, str]:
    name = f"{_TEMP_PREFIX}{secrets.token_hex(16)}.tmp"
    fd = os.open(
        name,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=parent_fd,
    )
    return fd, name


@dataclass
class PendingArtifact:
    target: PreparedArtifactTarget
    generation: ChatResult
    content: str
    raw: bytes
    _closed: bool = False

    @property
    def path(self) -> str:
        return self.target.relative_path.as_posix()

    def evidence(self) -> str:
        return "\n".join(
            [
                "artifact_generation: complete",
                f"path: {self.path}",
                f"bytes: {len(self.raw)}",
                f"sha256: {hashlib.sha256(self.raw).hexdigest()}",
                "publication_status: pending",
                "verification_required: true",
            ]
        )

    def verify_and_publish(
        self,
        arguments: dict[str, object],
        *,
        workdir: Path,
    ) -> tuple[ArtifactPublication, str]:
        if self._closed:
            raise ValueError("artifact request is no longer pending")
        path = arguments.get("path")
        check = arguments.get("check")
        if path != self.path:
            raise ValueError("artifact verification path does not match the pending request")
        detail, content_lines = _verify_artifact_bytes(
            self.raw,
            check=check,
        )
        try:
            publication = self.target.publish(self.content)
        finally:
            self.abort()
        lines = [
            publication.evidence(),
            "artifact_verification: complete",
            f"path: {publication.path}",
            f"check: {check}",
            f"bytes: {len(self.raw)}",
            f"sha256: {publication.sha256}",
            "status: pass",
            f"detail: {detail}",
            *content_lines,
        ]
        return publication, "\n".join(lines)

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.target.close()

    def __del__(self) -> None:
        self.abort()


@dataclass
class PreparedArtifactTarget:
    workdir: Path
    relative_path: Path
    parent_fd: int
    parent_device: int
    parent_inode: int
    existing_parent_relative: Path
    missing_parent_parts: tuple[str, ...]
    destination_version: tuple[int, int, int, int, int] | None
    destination_fd: int
    destination_sha256: str | None
    destination_mode: int | None
    overwrite: bool

    def close(self) -> None:
        if self.destination_fd >= 0:
            os.close(self.destination_fd)
            self.destination_fd = -1
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1

    def __enter__(self) -> PreparedArtifactTarget:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def publish(self, content: str) -> ArtifactPublication:
        raw = content.encode("utf-8", errors="strict")
        if not raw:
            raise ValueError("artifact generation returned empty content")
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ValueError(
                f"artifact content exceeds {MAX_ARTIFACT_BYTES} bytes: {len(raw)}"
            )
        generated_sha256 = hashlib.sha256(raw).hexdigest()
        if self.missing_parent_parts:
            return self._publish_with_new_parents(raw)
        self._attest_parent_and_destination()
        basename = self.relative_path.name
        replacing_existing = self.destination_version is not None
        temp_name: str | None = None
        temp_fd = -1
        unnamed_temp = False
        temp_cleanup_safe = False
        published_new_file = False
        directory_sync_complete = True
        try:
            temp_fd, temp_name, unnamed_temp = _open_private_artifact_temp(self.parent_fd)
            temp_cleanup_safe = temp_name is not None
            _write_all(temp_fd, raw)
            if replacing_existing:
                assert self.destination_mode is not None
                os.fchmod(temp_fd, self.destination_mode)
            os.fsync(temp_fd)
            generated_identity = _file_identity(os.fstat(temp_fd))
            self._attest_parent_and_destination()
            if unnamed_temp:
                link_name = (
                    f"{_TEMP_PREFIX}{secrets.token_hex(16)}.tmp"
                    if replacing_existing
                    else basename
                )
                try:
                    os.link(
                        f"/proc/self/fd/{temp_fd}",
                        link_name,
                        dst_dir_fd=self.parent_fd,
                        follow_symlinks=True,
                    )
                except OSError as exc:
                    if exc.errno not in _UNSUPPORTED_TMPFILE_ERRNOS:
                        raise
                    os.close(temp_fd)
                    temp_fd = -1
                    temp_fd, temp_name = _open_named_artifact_temp(self.parent_fd)
                    unnamed_temp = False
                    temp_cleanup_safe = True
                    _write_all(temp_fd, raw)
                    if replacing_existing:
                        assert self.destination_mode is not None
                        os.fchmod(temp_fd, self.destination_mode)
                    os.fsync(temp_fd)
                    generated_identity = _file_identity(os.fstat(temp_fd))
                    self._attest_parent_and_destination()
                else:
                    temp_name = link_name if replacing_existing else None
                    published_new_file = not replacing_existing
            if replacing_existing:
                assert temp_name is not None and self.destination_version is not None
                _rename_exchange(
                    old_dir_fd=self.parent_fd,
                    old_name=temp_name,
                    new_dir_fd=self.parent_fd,
                    new_name=basename,
                )
                temp_cleanup_safe = False
                try:
                    replaced_version = _file_version(
                        os.stat(temp_name, dir_fd=self.parent_fd, follow_symlinks=False)
                    )
                    published_identity = _file_identity(
                        os.stat(basename, dir_fd=self.parent_fd, follow_symlinks=False)
                    )
                    held_destination = os.fstat(self.destination_fd)
                    held_identity = _file_identity(held_destination)
                    held_sha256 = _sha256_fd(self.destination_fd)
                    self._attest_existing_parent()
                    if (
                        _file_identity_from_version(self.destination_version) != held_identity
                        or _file_identity_from_version(replaced_version) != held_identity
                        or held_sha256 != self.destination_sha256
                        or published_identity != generated_identity
                    ):
                        raise ValueError("artifact destination changed during atomic overwrite")
                except BaseException:
                    if _rollback_artifact_overwrite(
                        parent_fd=self.parent_fd,
                        temp_name=temp_name,
                        basename=basename,
                        generated_identity=generated_identity,
                        destination_fd=self.destination_fd,
                    ):
                        temp_cleanup_safe = True
                    else:
                        # The private name may hold displaced user data. Preserve
                        # it rather than guessing which concurrent entry is safe.
                        temp_name = None
                    raise
                directory_sync_complete = _try_fsync(self.parent_fd)
                os.unlink(temp_name, dir_fd=self.parent_fd)
                temp_name = None
                temp_cleanup_safe = False
                directory_sync_complete = _try_fsync(self.parent_fd) and directory_sync_complete
            elif not unnamed_temp:
                assert temp_name is not None
                _commit_named_file_noreplace(
                    old_dir_fd=self.parent_fd,
                    old_name=temp_name,
                    new_dir_fd=self.parent_fd,
                    new_name=basename,
                )
                temp_name = None
                temp_cleanup_safe = False
                published_new_file = True
                directory_sync_complete = _try_fsync(self.parent_fd)
            elif not replacing_existing:
                directory_sync_complete = _try_fsync(self.parent_fd)
            try:
                _attest_published_artifact(
                    self.parent_fd,
                    basename,
                    expected_identity=generated_identity,
                    expected_sha256=generated_sha256,
                )
                self._attest_existing_parent()
            except BaseException:
                if published_new_file:
                    _retract_published_file(
                        parent_fd=self.parent_fd,
                        basename=basename,
                        expected_identity=generated_identity,
                        expected_sha256=generated_sha256,
                    )
                    published_new_file = False
                raise
        except FileExistsError as exc:
            raise ValueError(f"artifact destination already exists: {self.relative_path}") from exc
        finally:
            if temp_fd >= 0:
                os.close(temp_fd)
            if temp_name is not None and temp_cleanup_safe:
                try:
                    os.unlink(temp_name, dir_fd=self.parent_fd)
                except FileNotFoundError:
                    pass
        return ArtifactPublication(
            path=self.relative_path.as_posix(),
            bytes_written=len(raw),
            sha256=generated_sha256,
            overwrite=replacing_existing,
            directory_sync_complete=directory_sync_complete,
        )

    def _publish_with_new_parents(self, raw: bytes) -> ArtifactPublication:
        self._attest_existing_parent()
        top_name = self.missing_parent_parts[0]
        try:
            os.stat(top_name, dir_fd=self.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("artifact parent path changed before atomic publication")

        stage_name = f"{_TEMP_PREFIX}{secrets.token_hex(16)}.tmp"
        os.mkdir(stage_name, 0o700, dir_fd=self.parent_fd)
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        dir_fds: list[int] = []
        file_fd = -1
        published = False
        directory_sync_complete = True
        try:
            stage_fd = os.open(stage_name, directory_flags, dir_fd=self.parent_fd)
            dir_fds.append(stage_fd)
            current_fd = stage_fd
            for part in self.missing_parent_parts[1:]:
                os.mkdir(part, 0o755, dir_fd=current_fd)
                try:
                    next_fd = os.open(part, directory_flags, dir_fd=current_fd)
                except BaseException:
                    try:
                        os.rmdir(part, dir_fd=current_fd)
                    except OSError:
                        pass
                    raise
                dir_fds.append(next_fd)
                os.fchmod(next_fd, 0o755)
                os.fsync(current_fd)
                current_fd = next_fd
            file_fd = os.open(
                self.relative_path.name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=current_fd,
            )
            _write_all(file_fd, raw)
            os.fsync(file_fd)
            generated_sha256 = _sha256_fd(file_fd)
            if generated_sha256 != hashlib.sha256(raw).hexdigest():
                raise ValueError("artifact changed before atomic parent publication")
            os.fsync(current_fd)
            self._attest_existing_parent()
            try:
                os.stat(top_name, dir_fd=self.parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError("artifact parent path changed before atomic publication")
            os.fchmod(stage_fd, 0o755)
            os.fsync(stage_fd)
            _rename_noreplace(
                old_dir_fd=self.parent_fd,
                old_name=stage_name,
                new_dir_fd=self.parent_fd,
                new_name=top_name,
            )
            published = True
            try:
                directory_sync_complete = _try_fsync(self.parent_fd)
                _attest_published_artifact_path(
                    self.parent_fd,
                    self.missing_parent_parts,
                    self.relative_path.name,
                    expected_identity=_file_identity(os.fstat(file_fd)),
                    expected_sha256=generated_sha256,
                )
                self._attest_existing_parent()
            except BaseException:
                try:
                    _attest_published_artifact_path(
                        self.parent_fd,
                        self.missing_parent_parts,
                        self.relative_path.name,
                        expected_identity=_file_identity(os.fstat(file_fd)),
                        expected_sha256=generated_sha256,
                    )
                except ValueError:
                    # Preserve a tree whose content or identity was changed by
                    # another actor; it is no longer safe for Orbit to remove.
                    pass
                else:
                    _retract_published_directory(
                        parent_fd=self.parent_fd,
                        published_name=top_name,
                        private_name=stage_name,
                        expected_directory_fd=stage_fd,
                    )
                    published = False
                raise
        finally:
            if not published:
                if file_fd >= 0 and dir_fds:
                    try:
                        os.unlink(self.relative_path.name, dir_fd=dir_fds[-1])
                    except FileNotFoundError:
                        pass
                for index in range(len(dir_fds) - 1, 0, -1):
                    os.close(dir_fds[index])
                    try:
                        os.rmdir(self.missing_parent_parts[index], dir_fd=dir_fds[index - 1])
                    except OSError:
                        pass
                if dir_fds:
                    os.close(dir_fds[0])
                try:
                    os.rmdir(stage_name, dir_fd=self.parent_fd)
                except OSError:
                    pass
            else:
                for directory_fd in reversed(dir_fds):
                    os.close(directory_fd)
            if file_fd >= 0:
                os.close(file_fd)

        created: list[str] = []
        current = self.existing_parent_relative
        for part in self.missing_parent_parts:
            current = current / part
            created.append(current.as_posix())
        return ArtifactPublication(
            path=self.relative_path.as_posix(),
            bytes_written=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            overwrite=False,
            created_parent_directories=tuple(created),
            directory_sync_complete=directory_sync_complete,
        )

    def _attest_existing_parent(self) -> None:
        opened_parent = os.fstat(self.parent_fd)
        if (
            opened_parent.st_dev,
            opened_parent.st_ino,
        ) != (self.parent_device, self.parent_inode) or not _directory_path_matches(
            self.workdir,
            self.existing_parent_relative,
            expected_device=self.parent_device,
            expected_inode=self.parent_inode,
        ):
            raise ValueError("artifact parent directory changed during generation")

    def _attest_parent_and_destination(self) -> None:
        self._attest_existing_parent()
        try:
            current = os.stat(
                self.relative_path.name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            current_version = None
        else:
            if not stat.S_ISREG(current.st_mode):
                raise ValueError("artifact destination is not a regular file")
            current_version = _file_version(current)
        if current_version != self.destination_version:
            raise ValueError("artifact destination changed during generation")


def prepare_artifact_target(
    path: object,
    *,
    overwrite: object,
    create_parents: object = False,
    workdir: Path,
) -> PreparedArtifactTarget:
    error = validate_artifact_path(path, workdir=workdir)
    if error:
        raise ValueError(error.removeprefix("error: "))
    if not isinstance(overwrite, bool):
        raise ValueError("artifact overwrite must be a boolean")
    if not isinstance(create_parents, bool):
        raise ValueError("artifact create_parents must be a boolean")
    assert isinstance(path, str)
    root = workdir.expanduser().resolve()
    relative = Path(os.path.abspath(root / path)).relative_to(root)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd = os.open(root, directory_flags)
    destination_fd = -1
    existing_parent_parts: list[str] = []
    missing_parent_parts: tuple[str, ...] = ()
    try:
        for index, part in enumerate(relative.parts[:-1]):
            try:
                next_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            except FileNotFoundError:
                if not create_parents:
                    raise ValueError(f"artifact parent directory does not exist: {relative.parent.as_posix()}")
                missing_parent_parts = tuple(relative.parts[index:-1])
                break
            os.close(parent_fd)
            parent_fd = next_fd
            existing_parent_parts.append(part)
        parent_stat = os.fstat(parent_fd)
        if missing_parent_parts:
            destination_version = None
            destination_sha256 = None
            destination_mode = None
        else:
            try:
                destination = os.stat(relative.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                destination_version = None
                destination_mode = None
            else:
                if not stat.S_ISREG(destination.st_mode):
                    raise ValueError("artifact destination is not a regular file")
                if not overwrite:
                    raise ValueError(f"artifact destination already exists: {relative.as_posix()}")
                destination_version = _file_version(destination)
                if destination.st_size > MAX_ARTIFACT_BYTES:
                    raise ValueError(
                        f"existing artifact exceeds {MAX_ARTIFACT_BYTES} bytes"
                    )
                destination_fd = os.open(
                    relative.name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=parent_fd,
                )
                opened_destination = os.fstat(destination_fd)
                if not stat.S_ISREG(opened_destination.st_mode):
                    raise ValueError("artifact destination is not a regular file")
                if _file_version(opened_destination) != destination_version:
                    raise ValueError("artifact destination changed during preparation")
                destination_mode = stat.S_IMODE(opened_destination.st_mode) & 0o777
                existing_raw = _read_fd(destination_fd, MAX_ARTIFACT_BYTES)
                existing_raw.decode("utf-8", errors="strict")
                destination_sha256 = hashlib.sha256(existing_raw).hexdigest()
            if destination_version is None:
                destination_sha256 = None
                destination_mode = None
        return PreparedArtifactTarget(
            workdir=root,
            relative_path=relative,
            parent_fd=parent_fd,
            parent_device=parent_stat.st_dev,
            parent_inode=parent_stat.st_ino,
            existing_parent_relative=Path(*existing_parent_parts),
            missing_parent_parts=missing_parent_parts,
            destination_version=destination_version,
            destination_fd=destination_fd,
            destination_sha256=destination_sha256,
            destination_mode=destination_mode,
            overwrite=overwrite,
        )
    except BaseException:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(parent_fd)
        raise


def begin_artifact_generation(
    *,
    backend: Any,
    user_request: str,
    path: str,
    overwrite: bool,
    create_parents: bool,
    workdir: Path,
    temperature: float,
    on_progress: Callable[[StreamProgress], None] | None,
) -> PendingArtifact:
    generate = getattr(backend, "artifact_content_stream", None)
    if not callable(generate):
        raise RuntimeError("artifact content generation requires the native Orbit backend")
    target = prepare_artifact_target(
        path,
        overwrite=overwrite,
        create_parents=create_parents,
        workdir=workdir,
    )
    try:
        messages = artifact_content_messages(
            user_request=user_request,
            path=target.relative_path.as_posix(),
            overwrite=overwrite,
        )
        result = generate(
            messages,
            temperature=temperature,
            max_tokens=ARTIFACT_CONTENT_MAX_TOKENS,
            on_delta=lambda _text: None,
            on_progress=on_progress,
        )
        if result.finish_reason != "stop":
            raise ArtifactGenerationError(
                f"artifact content was not published: finish_reason={result.finish_reason or 'unknown'}",
                result=result,
            )
        if result.tool_calls or result.reasoning_content:
            raise ArtifactGenerationError(
                "artifact content was not published: unexpected structured model output",
                result=result,
            )
        if (
            isinstance(result.completion_tokens, int)
            and result.completion_tokens > ARTIFACT_CONTENT_MAX_TOKENS
        ):
            raise ArtifactGenerationError(
                "artifact content was not published: output token limit exceeded",
                result=result,
            )
        try:
            raw = result.content.encode("utf-8", errors="strict")
            if not raw:
                raise ValueError("artifact generation returned empty content")
            if len(raw) > MAX_ARTIFACT_BYTES:
                raise ValueError(
                    f"artifact content exceeds {MAX_ARTIFACT_BYTES} bytes: {len(raw)}"
                )
        except (UnicodeError, ValueError) as exc:
            raise ArtifactGenerationError(
                f"artifact content failed validation: {exc}",
                result=result,
            ) from exc
        return PendingArtifact(
            target=target,
            generation=result,
            content=result.content,
            raw=raw,
        )
    except BaseException:
        target.close()
        raise


def _verify_artifact_bytes(
    raw: bytes,
    *,
    check: object,
) -> tuple[str, list[str]]:
    if check not in {
        "content",
        "text_integrity",
    }:
        raise ValueError("artifact verification check is invalid")
    text = raw.decode("utf-8", errors="strict")
    detail = "UTF-8 content, byte count, and SHA-256 are valid."
    content_lines: list[str] = []
    if check == "content":
        content_lines = [
            "content_coverage: "
            + ("complete" if len(raw) <= MAX_ARTIFACT_VERIFICATION_CHARS else "partial"),
            "content:",
            _bounded_verification_text(text, MAX_ARTIFACT_VERIFICATION_CHARS),
        ]
    return detail, content_lines


def _write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short artifact write")
        view = view[written:]


def _try_fsync(fd: int) -> bool:
    try:
        os.fsync(fd)
    except OSError:
        return False
    return True


def _bounded_verification_text(text: str, maximum: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= maximum:
        return text
    head_limit = max(0, (maximum * 2) // 3)
    tail_limit = max(0, maximum - head_limit)
    head = raw[:head_limit].decode("utf-8", errors="ignore")
    tail = raw[-tail_limit:].decode("utf-8", errors="ignore") if tail_limit else ""
    return f"{head}\n[verification content truncated]\n{tail}"


def _file_version(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _file_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size)


def _file_identity_from_version(value: tuple[int, int, int, int, int]) -> tuple[int, int, int]:
    return value[:3]


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _rollback_artifact_overwrite(
    *,
    parent_fd: int,
    temp_name: str,
    basename: str,
    generated_identity: tuple[int, int, int],
    destination_fd: int,
) -> bool:
    """Restore the entry displaced by the generated inode without deleting either."""

    del destination_fd
    try:
        published = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return False
    if (published.st_dev, published.st_ino, published.st_size) != generated_identity:
        return False
    try:
        _rename_exchange(
            old_dir_fd=parent_fd,
            old_name=temp_name,
            new_dir_fd=parent_fd,
            new_name=basename,
        )
        private = os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return False
    return (
        private.st_dev,
        private.st_ino,
        private.st_size,
    ) == generated_identity


def _retract_published_file(
    *,
    parent_fd: int,
    basename: str,
    expected_identity: tuple[int, int, int],
    expected_sha256: str,
) -> None:
    """Remove only exact, unmodified generated content after privatizing it."""

    private_name = f"{_TEMP_PREFIX}{secrets.token_hex(16)}.rollback"
    try:
        _rename_noreplace(
            old_dir_fd=parent_fd,
            old_name=basename,
            new_dir_fd=parent_fd,
            new_name=private_name,
        )
    except OSError as exc:
        raise ValueError("artifact publication rollback failed safely") from exc
    try:
        _attest_published_artifact(
            parent_fd,
            private_name,
            expected_identity=expected_identity,
            expected_sha256=expected_sha256,
        )
    except (OSError, ValueError):
        try:
            _rename_noreplace(
                old_dir_fd=parent_fd,
                old_name=private_name,
                new_dir_fd=parent_fd,
                new_name=basename,
            )
        except OSError as exc:
            raise ValueError(
                "artifact path changed and unrelated entry was preserved under a private name"
            ) from exc
        raise ValueError("artifact destination changed after atomic publication")
    os.unlink(private_name, dir_fd=parent_fd)
    _try_fsync(parent_fd)


def _retract_published_directory(
    *,
    parent_fd: int,
    published_name: str,
    private_name: str,
    expected_directory_fd: int,
) -> None:
    """Move an exact Orbit-created tree back to its private staging name."""

    try:
        _rename_noreplace(
            old_dir_fd=parent_fd,
            old_name=published_name,
            new_dir_fd=parent_fd,
            new_name=private_name,
        )
    except OSError as exc:
        raise ValueError("artifact parent publication rollback failed safely") from exc
    moved = os.stat(private_name, dir_fd=parent_fd, follow_symlinks=False)
    expected = os.fstat(expected_directory_fd)
    if not stat.S_ISDIR(moved.st_mode) or not _same_inode(moved, expected):
        try:
            _rename_noreplace(
                old_dir_fd=parent_fd,
                old_name=private_name,
                new_dir_fd=parent_fd,
                new_name=published_name,
            )
        except OSError as exc:
            raise ValueError(
                "artifact parent path changed and unrelated tree was preserved under a private name"
            ) from exc
        raise ValueError("artifact parent path changed after atomic publication")


def _read_fd(fd: int, maximum: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(64 * 1024, maximum + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            raise ValueError(f"artifact content exceeds {maximum} bytes")
    os.lseek(fd, 0, os.SEEK_SET)
    return b"".join(chunks)


def _sha256_fd(fd: int) -> str:
    return hashlib.sha256(_read_fd(fd, MAX_ARTIFACT_BYTES)).hexdigest()


def _attest_published_artifact(
    parent_fd: int,
    basename: str,
    *,
    expected_identity: tuple[int, int, int],
    expected_sha256: str,
) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(basename, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError("artifact destination changed after atomic publication") from exc
    try:
        current = os.fstat(fd)
        if not stat.S_ISREG(current.st_mode):
            raise ValueError("artifact destination changed after atomic publication")
        version_before_read = _file_version(current)
        digest = _sha256_fd(fd)
        version_after_read = _file_version(os.fstat(fd))
        path_version = _file_version(
            os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
        )
        if (
            _file_identity(current) != expected_identity
            or digest != expected_sha256
            or version_before_read != version_after_read
            or path_version != version_after_read
        ):
            raise ValueError("artifact destination changed after atomic publication")
    except OSError as exc:
        raise ValueError("artifact destination changed after atomic publication") from exc
    finally:
        os.close(fd)


def _attest_published_artifact_path(
    parent_fd: int,
    directory_parts: tuple[str, ...],
    basename: str,
    *,
    expected_identity: tuple[int, int, int],
    expected_sha256: str,
) -> None:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    opened_fds = [os.dup(parent_fd)]
    edges: list[tuple[int, str, tuple[int, int]]] = []
    try:
        for part in directory_parts:
            next_fd = os.open(part, directory_flags, dir_fd=opened_fds[-1])
            current = os.fstat(next_fd)
            identity = (current.st_dev, current.st_ino)
            edges.append((opened_fds[-1], part, identity))
            opened_fds.append(next_fd)
        _attest_published_artifact(
            opened_fds[-1],
            basename,
            expected_identity=expected_identity,
            expected_sha256=expected_sha256,
        )
        for ancestor_fd, name, expected in reversed(edges):
            current = os.stat(name, dir_fd=ancestor_fd, follow_symlinks=False)
            if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != expected:
                raise ValueError("artifact parent path changed after atomic publication")
    except OSError as exc:
        raise ValueError("artifact parent path changed after atomic publication") from exc
    finally:
        for fd in reversed(opened_fds):
            os.close(fd)


def _directory_path_matches(
    root: Path,
    relative: Path,
    *,
    expected_device: int,
    expected_inode: int,
) -> bool:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current_fd = -1
    try:
        current_fd = os.open(root, directory_flags)
        for part in relative.parts:
            next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        current = os.fstat(current_fd)
        return (current.st_dev, current.st_ino) == (expected_device, expected_inode)
    except OSError:
        return False
    finally:
        if current_fd >= 0:
            os.close(current_fd)


def _rename_noreplace(
    *,
    old_dir_fd: int,
    old_name: str,
    new_dir_fd: int,
    new_name: str,
) -> None:
    _renameat2(
        old_dir_fd=old_dir_fd,
        old_name=old_name,
        new_dir_fd=new_dir_fd,
        new_name=new_name,
        flags=1,  # RENAME_NOREPLACE
    )


def _commit_named_file_noreplace(
    *,
    old_dir_fd: int,
    old_name: str,
    new_dir_fd: int,
    new_name: str,
) -> None:
    try:
        _rename_noreplace(
            old_dir_fd=old_dir_fd,
            old_name=old_name,
            new_dir_fd=new_dir_fd,
            new_name=new_name,
        )
        return
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_RENAME_NOREPLACE_ERRNOS:
            raise
    source_fd = os.open(
        old_name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=old_dir_fd,
    )
    try:
        source = os.fstat(source_fd)
        if not stat.S_ISREG(source.st_mode):
            raise ValueError("artifact temporary file is not regular")
        expected_identity = _file_identity(source)
        expected_sha256 = _sha256_fd(source_fd)
    finally:
        os.close(source_fd)
    os.link(
        old_name,
        new_name,
        src_dir_fd=old_dir_fd,
        dst_dir_fd=new_dir_fd,
        follow_symlinks=False,
    )
    try:
        os.unlink(old_name, dir_fd=old_dir_fd)
    except BaseException:
        _retract_published_file(
            parent_fd=new_dir_fd,
            basename=new_name,
            expected_identity=expected_identity,
            expected_sha256=expected_sha256,
        )
        raise


def _rename_exchange(
    *,
    old_dir_fd: int,
    old_name: str,
    new_dir_fd: int,
    new_name: str,
) -> None:
    _renameat2(
        old_dir_fd=old_dir_fd,
        old_name=old_name,
        new_dir_fd=new_dir_fd,
        new_name=new_name,
        flags=2,  # RENAME_EXCHANGE
    )


def _renameat2(
    *,
    old_dir_fd: int,
    old_name: str,
    new_dir_fd: int,
    new_name: str,
    flags: int,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "atomic artifact publication requires renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        old_dir_fd,
        os.fsencode(old_name),
        new_dir_fd,
        os.fsencode(new_name),
        flags,
    )
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), new_name)
