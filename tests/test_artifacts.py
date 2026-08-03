from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import socket
import tempfile
import unittest
from unittest import mock

from orbit.backend.base import ChatResult
from orbit.runtime import artifacts as artifact_module
from orbit.runtime.artifacts import (
    ARTIFACT_CONTENT_MAX_TOKENS,
    MAX_ARTIFACT_BYTES,
    artifact_content_messages,
    begin_artifact_generation,
    cleanup_stale_artifact_entries,
    prepare_artifact_target,
    validate_artifact_path,
    verify_artifact_definition,
    write_artifact_definition,
)


def _result(content: str, *, finish_reason: str = "stop") -> ChatResult:
    return ChatResult(
        content=content,
        model="fake",
        finish_reason=finish_reason,
        tool_calls=[],
        prompt_tokens=10,
        completion_tokens=3,
        cached_tokens=0,
        prompt_tokens_per_second=None,
        generation_tokens_per_second=None,
    )


class _Backend:
    def __init__(self, result: ChatResult, *, during_call=None) -> None:
        self.result = result
        self.during_call = during_call
        self.calls = []

    def artifact_content_stream(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.during_call is not None:
            self.during_call()
        return self.result


class ArtifactPublicationTests(unittest.TestCase):
    def _generate(
        self,
        root: Path,
        backend: _Backend,
        *,
        path: str = "samples/game.js",
        overwrite: bool = False,
        create_parents: bool = True,
        check: str = "text_integrity",
    ):
        pending = begin_artifact_generation(
            backend=backend,
            user_request="create a browser game",
            path=path,
            overwrite=overwrite,
            create_parents=create_parents,
            workdir=root,
            temperature=0,
            on_progress=None,
        )
        try:
            pending.verify({"check": check})
            return pending.generation, pending.publication
        finally:
            pending.abort()

    def test_schema_is_small_and_content_is_not_an_argument(self) -> None:
        function = write_artifact_definition()["function"]
        properties = function["parameters"]["properties"]

        self.assertEqual(function["name"], "write_artifact")
        self.assertEqual(set(properties), {"path", "overwrite", "create_parents"})
        self.assertNotIn("content", properties)

        verify_function = verify_artifact_definition()["function"]
        self.assertEqual(verify_function["name"], "verify_artifact")
        self.assertEqual(
            set(verify_function["parameters"]["properties"]),
            {"check"},
        )
        self.assertEqual(
            verify_function["parameters"]["properties"]["check"]["enum"],
            ["content", "text_integrity"],
        )
        self.assertEqual(verify_function["parameters"]["required"], ["check"])

    def test_directory_form_destinations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in ("samples/", "samples/.", "samples//game.js"):
                with self.subTest(path=path):
                    self.assertIn("must identify one file", validate_artifact_path(path, workdir=root))
            self.assertIsNone(validate_artifact_path("./samples/game.js", workdir=root))

    def test_content_phase_contract_requires_the_file_body_without_format_routing(self) -> None:
        messages = artifact_content_messages(
            user_request="create one bounded text artifact",
            path="chosen/name.ext",
            overwrite=False,
        )

        system = messages[0]["content"]
        self.assertIn("one complete bounded UTF-8 text artifact", system)
        self.assertIn("Do not use tools, shell, JSON/XML envelopes", system)
        self.assertIn("Output the complete file body now", messages[1]["content"])
        self.assertIn("do not output or discuss the destination path", messages[1]["content"])
        self.assertNotIn("JavaScript", system)
        self.assertNotIn("Python", system)
        self.assertNotIn("HTML", system)
        self.assertEqual(messages[1]["content"].count("chosen/name.ext"), 1)

    def test_generation_publishes_before_model_selected_read_only_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pending = begin_artifact_generation(
                backend=_Backend(_result("console.log('ready');\n")),
                user_request="create a browser game",
                path="samples/games/snake.js",
                overwrite=False,
                create_parents=True,
                workdir=root,
                temperature=0,
                on_progress=None,
            )

            self.assertTrue((root / "samples/games/snake.js").is_file())
            self.assertIn("artifact_publication: complete", pending.evidence())
            before = (root / "samples/games/snake.js").stat()
            mutation_paths = (
                "orbit.runtime.artifacts._renameat2",
                "orbit.runtime.artifacts.os.unlink",
                "orbit.runtime.artifacts.os.mkdir",
                "orbit.runtime.artifacts.os.rmdir",
                "orbit.runtime.artifacts.os.rename",
                "orbit.runtime.artifacts.os.replace",
                "orbit.runtime.artifacts.os.link",
                "orbit.runtime.artifacts.os.write",
                "orbit.runtime.artifacts.os.fchmod",
                "orbit.runtime.artifacts.os.chmod",
            )
            patchers = [
                mock.patch(path, side_effect=AssertionError(f"verifier mutated through {path}"))
                for path in mutation_paths
            ]
            for patcher in patchers:
                patcher.start()
            try:
                evidence = pending.verify({"check": "text_integrity"})
            finally:
                for patcher in reversed(patchers):
                    patcher.stop()
            after = (root / "samples/games/snake.js").stat()

            self.assertEqual(pending.publication.created_parent_directories, ("samples", "samples/games"))
            self.assertEqual((root / "samples/games/snake.js").read_text(), "console.log('ready');\n")
            self.assertIn("artifact_verification: complete", evidence)
            self.assertEqual(
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns),
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            )

    def test_invalid_model_selected_verification_preserves_published_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pending = begin_artifact_generation(
                backend=_Backend(_result("literal content\n")),
                user_request="create text",
                path="new/parents/note.txt",
                overwrite=False,
                create_parents=True,
                workdir=root,
                temperature=0,
                on_progress=None,
            )
            try:
                with self.assertRaisesRegex(ValueError, "verification check is invalid"):
                    pending.verify({"check": "syntax_by_extension"})
            finally:
                pending.abort()

            self.assertEqual((root / "new/parents/note.txt").read_text(), "literal content\n")
            self.assertEqual(list(root.glob(".orbit-artifact-*")), [])

    def test_verification_never_executes_tool_like_generated_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = "require('fs').writeFileSync('executed.txt', 'bad');\n"
            pending = begin_artifact_generation(
                backend=_Backend(_result(content)),
                user_request="create JavaScript",
                path="samples/check-only.js",
                overwrite=False,
                create_parents=True,
                workdir=root,
                temperature=0,
                on_progress=None,
            )
            pending.verify({"check": "text_integrity"})

            self.assertFalse((root / "executed.txt").exists())
            self.assertEqual((root / "samples/check-only.js").read_text(), content)

    def test_invalid_verification_preserves_completed_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "samples").mkdir()
            target = root / "samples/game.js"
            target.write_text("original", encoding="utf-8")
            pending = begin_artifact_generation(
                backend=_Backend(_result("const broken = ;\n")),
                user_request="replace JavaScript",
                path="samples/game.js",
                overwrite=True,
                create_parents=False,
                workdir=root,
                temperature=0,
                on_progress=None,
            )
            try:
                with self.assertRaisesRegex(ValueError, "verification check is invalid"):
                    pending.verify({"check": "syntax_by_extension"})
            finally:
                pending.abort()

            self.assertEqual(target.read_text(encoding="utf-8"), "const broken = ;\n")

    def test_model_selected_generic_checks_gate_publication(self) -> None:
        cases = (
            ("notes/readme.md", "# Notes\n\nComplete.\n", "content", "content_coverage: complete"),
            ("data/config.json", '{"enabled": true, "count": 3}\n', "text_integrity", "UTF-8 content"),
            ("src/example.py", "def answer():\n    return 42\n", "text_integrity", "UTF-8 content"),
        )
        for path, content, check, marker in cases:
            with self.subTest(check=check), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pending = begin_artifact_generation(
                    backend=_Backend(_result(content)),
                    user_request=f"create {path}",
                    path=path,
                    overwrite=False,
                    create_parents=True,
                    workdir=root,
                    temperature=0,
                    on_progress=None,
                )
                evidence = pending.verify({"check": check})
                publication = pending.publication

                self.assertEqual((root / path).read_text(encoding="utf-8"), content)
                self.assertEqual(publication.sha256, hashlib.sha256(content.encode()).hexdigest())
                self.assertIn(marker, evidence)
                self.assertEqual(list(root.rglob(".orbit-artifact-*")), [])

    def test_missing_parent_requires_explicit_authorization_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = _Backend(_result("content"))
            with self.assertRaisesRegex(ValueError, "parent directory does not exist"):
                begin_artifact_generation(
                    backend=backend,
                    user_request="create a file",
                    path="missing/file.txt",
                    overwrite=False,
                    create_parents=False,
                    workdir=root,
                    temperature=0,
                    on_progress=None,
                )

            self.assertEqual(backend.calls, [])
            self.assertFalse((root / "missing").exists())

    def test_concurrent_content_in_created_parent_is_not_displaced_on_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_attest = artifact_module._attest_created_directories
            calls = 0

            def fail_after_file_publication(created) -> None:
                nonlocal calls
                calls += 1
                real_attest(created)
                if calls == 3:
                    (root / "samples/concurrent.txt").write_text(
                        "concurrent", encoding="utf-8"
                    )
                    (root / "samples/concurrent-dir").mkdir()
                    raise ValueError("injected post-publication parent failure")

            with mock.patch(
                "orbit.runtime.artifacts._attest_created_directories",
                side_effect=fail_after_file_publication,
            ):
                with self.assertRaisesRegex(ValueError, "injected post-publication"):
                    begin_artifact_generation(
                        backend=_Backend(_result("text\n")),
                        user_request="create text",
                        path="samples/note.txt",
                        overwrite=False,
                        create_parents=True,
                        workdir=root,
                        temperature=0,
                        on_progress=None,
                    )

            self.assertEqual((root / "samples/concurrent.txt").read_text(), "concurrent")
            self.assertTrue((root / "samples/concurrent-dir").is_dir())
            self.assertFalse((root / "samples/note.txt").exists())
            self.assertEqual(list(root.rglob(".orbit-artifact-*")), [])

    def test_path_validation_rejects_escape_absolute_and_control_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(validate_artifact_path("samples/game.js", workdir=root))
            for path in ("../game.js", "/tmp/game.js", "samples/../game.js", "bad\nname.js", ""):
                with self.subTest(path=path):
                    self.assertIsNotNone(validate_artifact_path(path, workdir=root))

    def test_complete_generation_atomically_creates_file_and_exact_parents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = "console.log('ready');\n"
            backend = _Backend(
                _result(content),
                during_call=lambda: self.assertFalse((root / "samples").exists()),
            )

            generation, publication = self._generate(
                root,
                backend,
                path="samples/games/snake.js",
            )

            target = root / "samples/games/snake.js"
            self.assertEqual(generation.finish_reason, "stop")
            self.assertEqual(target.read_text(encoding="utf-8"), content)
            self.assertEqual(publication.path, "samples/games/snake.js")
            self.assertEqual(publication.bytes_written, len(content.encode()))
            self.assertEqual(publication.sha256, hashlib.sha256(content.encode()).hexdigest())
            self.assertEqual(publication.created_parent_directories, ("samples", "samples/games"))
            self.assertEqual(os.stat(root / "samples").st_mode & 0o777, 0o755)
            self.assertEqual(list(root.glob(".orbit-artifact-*")), [])
            self.assertEqual(backend.calls[0][1]["max_tokens"], ARTIFACT_CONTENT_MAX_TOKENS)
            self.assertNotIn("tools", backend.calls[0][1])

    def test_length_cancel_and_timeout_publish_nothing_and_create_no_parents(self) -> None:
        for finish_reason in ("length", "cancelled", "timeout"):
            with self.subTest(finish_reason=finish_reason), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                backend = _Backend(_result("partial", finish_reason=finish_reason))
                with self.assertRaisesRegex(RuntimeError, f"finish_reason={finish_reason}"):
                    self._generate(root, backend)
                self.assertFalse((root / "samples").exists())
                self.assertEqual(list(root.glob(".orbit-artifact-*")), [])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            class TimeoutBackend:
                def artifact_content_stream(self, *args, **kwargs):
                    raise TimeoutError("timed out")

            with self.assertRaises(TimeoutError):
                self._generate(root, TimeoutBackend())
            self.assertFalse((root / "samples").exists())

    def test_existing_file_is_rejected_before_generation_unless_overwrite_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "samples").mkdir()
            target = root / "samples/game.js"
            target.write_text("old", encoding="utf-8")
            denied = _Backend(_result("new"))

            with self.assertRaisesRegex(ValueError, "already exists"):
                self._generate(root, denied, create_parents=False)
            self.assertEqual(denied.calls, [])
            self.assertEqual(target.read_text(encoding="utf-8"), "old")

            allowed = _Backend(_result("new"))
            _, publication = self._generate(
                root,
                allowed,
                overwrite=True,
                create_parents=False,
            )
            self.assertTrue(publication.overwrite)
            self.assertEqual(target.read_text(encoding="utf-8"), "new")

    def test_overwrite_preserves_existing_permission_bits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "samples").mkdir()
            target = root / "samples/game.js"
            target.write_text("old", encoding="utf-8")
            target.chmod(0o755)

            self._generate(
                root,
                _Backend(_result("new")),
                overwrite=True,
                create_parents=False,
            )

            self.assertEqual(target.read_text(encoding="utf-8"), "new")
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o755)

    def test_overwrite_authorization_on_absent_target_creates_without_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "samples").mkdir()

            _, publication = self._generate(
                root,
                _Backend(_result("new")),
                overwrite=True,
                create_parents=False,
            )

            self.assertEqual((root / "samples/game.js").read_text(), "new")
            self.assertFalse(publication.overwrite)
            self.assertEqual(list((root / "samples").glob(".orbit-artifact-*")), [])

    def test_existing_parent_falls_back_when_filesystem_rejects_o_tmpfile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "samples").mkdir()
            content = "console.log('portable');\n"
            real_open = os.open
            tmpfile = getattr(os, "O_TMPFILE", 0)
            anonymous_attempts = 0

            def reject_anonymous(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal anonymous_attempts
                if tmpfile and (flags & tmpfile) == tmpfile:
                    anonymous_attempts += 1
                    raise OSError(errno.EOPNOTSUPP, "Operation not supported", path)
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with mock.patch("orbit.runtime.artifacts.os.open", side_effect=reject_anonymous):
                _, publication = self._generate(
                    root,
                    _Backend(_result(content)),
                    create_parents=False,
                )

            self.assertEqual(anonymous_attempts, 1)
            self.assertEqual((root / "samples/game.js").read_text(encoding="utf-8"), content)
            self.assertEqual(publication.sha256, hashlib.sha256(content.encode()).hexdigest())
            self.assertEqual(list((root / "samples").glob(".orbit-artifact-*")), [])

    def test_existing_parent_falls_back_when_anonymous_temp_link_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            real_link = os.link

            def reject_proc_fd_link(src, dst, **kwargs):
                if str(src).startswith("/proc/self/fd/"):
                    raise OSError(errno.EINVAL, "anonymous link unsupported")
                return real_link(src, dst, **kwargs)

            with mock.patch(
                "orbit.runtime.artifacts.os.link",
                side_effect=reject_proc_fd_link,
            ):
                _, publication = self._generate(
                    root,
                    _Backend(_result("content")),
                    create_parents=False,
                )

            self.assertEqual(publication.bytes_written, 7)
            self.assertEqual((parent / "game.js").read_text(), "content")
            self.assertEqual(list(parent.glob(".orbit-artifact-*")), [])

    def test_existing_parent_falls_back_when_proc_fd_link_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            real_link = os.link

            def reject_proc_fd_link(src, dst, **kwargs):
                if str(src).startswith("/proc/self/fd/"):
                    raise OSError(errno.ENOENT, "proc is unavailable")
                return real_link(src, dst, **kwargs)

            with mock.patch("orbit.runtime.artifacts.os.link", side_effect=reject_proc_fd_link):
                _, publication = self._generate(
                    root,
                    _Backend(_result("content")),
                    create_parents=False,
                )

            self.assertEqual(publication.bytes_written, 7)
            self.assertEqual((parent / "game.js").read_text(), "content")
            self.assertEqual(list(parent.glob(".orbit-artifact-*")), [])

    def test_named_file_commit_falls_back_when_rename_noreplace_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            with mock.patch(
                "orbit.runtime.artifacts._open_private_artifact_temp",
                side_effect=lambda parent_fd: (*artifact_module._open_named_artifact_temp(parent_fd), False),
            ), mock.patch(
                "orbit.runtime.artifacts._rename_noreplace",
                side_effect=OSError(errno.EINVAL, "rename noreplace unsupported"),
            ):
                _, publication = self._generate(
                    root,
                    _Backend(_result("content")),
                    create_parents=False,
                )

            self.assertEqual(publication.bytes_written, 7)
            self.assertEqual((parent / "game.js").read_text(), "content")
            self.assertEqual(list(parent.glob(".orbit-artifact-*")), [])

    def test_directory_sync_failure_after_atomic_parent_publish_is_reported_honestly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("orbit.runtime.artifacts._try_fsync", return_value=False):
                _, publication = self._generate(root, _Backend(_result("complete")))

            self.assertEqual((root / "samples/game.js").read_text(encoding="utf-8"), "complete")
            self.assertFalse(publication.directory_sync_complete)
            self.assertIn("directory_sync: unconfirmed", publication.evidence())

    def test_generated_content_is_attested_before_parent_tree_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                mock.patch("orbit.runtime.artifacts._sha256_fd", return_value="wrong"),
                mock.patch("orbit.runtime.artifacts._rename_noreplace") as rename,
            ):
                with self.assertRaisesRegex(ValueError, "temporary content changed before publication"):
                    self._generate(root, _Backend(_result("complete")))

            rename.assert_not_called()
            self.assertFalse((root / "samples").exists())
            self.assertEqual(list(root.glob(".orbit-artifact-*")), [])

    def test_directory_sync_failure_after_new_file_commit_is_reported_honestly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "samples").mkdir()
            with mock.patch("orbit.runtime.artifacts._try_fsync", return_value=False):
                _, publication = self._generate(
                    root,
                    _Backend(_result("complete")),
                    create_parents=False,
                )

            self.assertEqual((root / "samples/game.js").read_text(), "complete")
            self.assertFalse(publication.directory_sync_complete)
            self.assertIn("directory_sync: unconfirmed", publication.evidence())
            self.assertEqual(list((root / "samples").glob(".orbit-artifact-*")), [])

    def test_post_commit_race_is_preserved_and_never_reported_as_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            real_try_fsync = artifact_module._try_fsync
            raced = False

            def race_during_directory_sync(fd: int) -> bool:
                nonlocal raced
                if not raced and stat.S_ISDIR(os.fstat(fd).st_mode):
                    target = parent / "game.js"
                    if target.exists():
                        target.write_text("concurr", encoding="utf-8")
                        raced = True
                        return False
                return real_try_fsync(fd)

            with mock.patch(
                "orbit.runtime.artifacts._try_fsync",
                side_effect=race_during_directory_sync,
            ):
                with self.assertRaisesRegex(ValueError, "changed after atomic publication"):
                    self._generate(
                        root,
                        _Backend(_result("complete")),
                        create_parents=False,
                    )

            self.assertTrue(raced)
            self.assertEqual((parent / "game.js").read_text(), "concurr")
            self.assertEqual(list(parent.glob(".orbit-artifact-*")), [])

    def test_same_size_post_commit_rewrite_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            real_try_fsync = artifact_module._try_fsync
            raced = False

            def rewrite_during_directory_sync(fd: int) -> bool:
                nonlocal raced
                target = parent / "game.js"
                if not raced and target.exists() and stat.S_ISDIR(os.fstat(fd).st_mode):
                    self.assertEqual(len(target.read_bytes()), len(b"generated"))
                    target.write_bytes(b"concurred")
                    raced = True
                    return False
                return real_try_fsync(fd)

            with mock.patch(
                "orbit.runtime.artifacts._try_fsync",
                side_effect=rewrite_during_directory_sync,
            ):
                with self.assertRaisesRegex(ValueError, "changed after atomic publication"):
                    self._generate(
                        root,
                        _Backend(_result("generated")),
                        create_parents=False,
                    )

            self.assertTrue(raced)
            self.assertEqual((parent / "game.js").read_bytes(), b"concurred")
            self.assertEqual(list(parent.glob(".orbit-artifact-*")), [])

    def test_hard_link_fallback_retracts_destination_when_temp_unlink_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            real_unlink = artifact_module.os.unlink
            unlink_calls = 0
            rename_calls = 0

            def fail_first_unlink(path, *args, **kwargs):
                nonlocal unlink_calls
                unlink_calls += 1
                if unlink_calls == 1:
                    raise OSError(errno.EIO, "simulated private unlink failure")
                return real_unlink(path, *args, **kwargs)

            def reject_initial_noreplace(**kwargs):
                nonlocal rename_calls
                rename_calls += 1
                if rename_calls == 1:
                    raise OSError(errno.ENOTSUP, "unsupported")
                return artifact_module._renameat2(flags=1, **kwargs)

            with mock.patch(
                "orbit.runtime.artifacts._open_private_artifact_temp",
                side_effect=lambda parent_fd: (
                    *artifact_module._open_named_artifact_temp(parent_fd),
                    False,
                ),
            ), mock.patch(
                "orbit.runtime.artifacts._rename_noreplace",
                side_effect=reject_initial_noreplace,
            ), mock.patch(
                "orbit.runtime.artifacts.os.unlink",
                side_effect=fail_first_unlink,
            ):
                with self.assertRaisesRegex(OSError, "private unlink failure"):
                    self._generate(
                        root,
                        _Backend(_result("generated")),
                        create_parents=False,
                    )

            self.assertFalse((parent / "game.js").exists())
            self.assertEqual(list(parent.glob(".orbit-artifact-*")), [])

    def test_post_commit_fifo_replacement_fails_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            real_commit = artifact_module._commit_named_file_noreplace

            def replace_with_fifo(**kwargs) -> None:
                real_commit(**kwargs)
                target = parent / "game.js"
                target.unlink()
                os.mkfifo(target)

            with mock.patch(
                "orbit.runtime.artifacts._open_private_artifact_temp",
                side_effect=lambda parent_fd: (*artifact_module._open_named_artifact_temp(parent_fd), False),
            ), mock.patch(
                "orbit.runtime.artifacts._commit_named_file_noreplace",
                side_effect=replace_with_fifo,
            ):
                with self.assertRaisesRegex(ValueError, "changed after atomic publication"):
                    self._generate(
                        root,
                        _Backend(_result("generated")),
                        create_parents=False,
                    )

            self.assertTrue(stat.S_ISFIFO(os.stat(parent / "game.js").st_mode))

    def test_symlink_parent_symlink_target_and_fifo_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            (root / "samples").symlink_to(Path(outside), target_is_directory=True)
            with self.assertRaises(OSError):
                prepare_artifact_target(
                    "samples/game.js", overwrite=False, create_parents=False, workdir=root
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "samples").mkdir()
            (root / "other.js").write_text("other", encoding="utf-8")
            (root / "samples/game.js").symlink_to(root / "other.js")
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                prepare_artifact_target(
                    "samples/game.js", overwrite=True, create_parents=False, workdir=root
                )

            fifo = root / "samples/pipe"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                prepare_artifact_target(
                    "samples/pipe", overwrite=True, create_parents=False, workdir=root
                )

    def test_fifo_replacement_between_stat_and_open_fails_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            target = parent / "game.js"
            target.write_text("old", encoding="utf-8")
            real_open = os.open
            replaced = False

            def replace_before_open(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal replaced
                if path == "game.js" and dir_fd is not None and not replaced:
                    replaced = True
                    target.unlink()
                    os.mkfifo(target)
                    self.assertTrue(flags & getattr(os, "O_NONBLOCK", 0))
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with mock.patch("orbit.runtime.artifacts.os.open", side_effect=replace_before_open):
                with self.assertRaisesRegex(ValueError, "not a regular file"):
                    prepare_artifact_target(
                        "samples/game.js",
                        overwrite=True,
                        create_parents=False,
                        workdir=root,
                    )

            self.assertTrue(replaced)
            self.assertTrue(stat.S_ISFIFO(target.stat().st_mode))

    def test_parent_and_destination_changes_during_generation_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()

            def replace_parent() -> None:
                parent.rename(root / "old-samples")
                parent.mkdir()

            backend = _Backend(_result("content"), during_call=replace_parent)
            with self.assertRaisesRegex(ValueError, "parent directory changed"):
                self._generate(root, backend, create_parents=False)
            self.assertFalse((parent / "game.js").exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            target = parent / "game.js"

            backend = _Backend(_result("content"), during_call=lambda: target.write_text("racer", encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "destination changed"):
                self._generate(root, backend, create_parents=False)
            self.assertEqual(target.read_text(encoding="utf-8"), "racer")

    def test_existing_parent_rename_during_commit_cannot_report_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            real_commit = artifact_module._commit_named_file_noreplace

            def move_parent_then_commit(**kwargs) -> None:
                parent.rename(root / "old-samples")
                parent.mkdir()
                real_commit(**kwargs)

            with mock.patch(
                "orbit.runtime.artifacts._open_private_artifact_temp",
                side_effect=lambda parent_fd: (*artifact_module._open_named_artifact_temp(parent_fd), False),
            ), mock.patch(
                "orbit.runtime.artifacts._commit_named_file_noreplace",
                side_effect=move_parent_then_commit,
            ):
                with self.assertRaisesRegex(ValueError, "parent directory changed"):
                    self._generate(
                        root,
                        _Backend(_result("generated")),
                        create_parents=False,
                    )

            self.assertFalse((root / "samples/game.js").exists())
            self.assertFalse((root / "old-samples/game.js").exists())

    def test_overwrite_parent_rename_during_commit_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            target = parent / "game.js"
            target.write_text("original", encoding="utf-8")
            real_exchange = artifact_module._rename_exchange
            calls = 0

            def move_parent_then_exchange(**kwargs) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    parent.rename(root / "old-samples")
                    parent.mkdir()
                real_exchange(**kwargs)

            with mock.patch(
                "orbit.runtime.artifacts._rename_exchange",
                side_effect=move_parent_then_exchange,
            ):
                with self.assertRaisesRegex(ValueError, "parent directory changed"):
                    self._generate(
                        root,
                        _Backend(_result("generated")),
                        overwrite=True,
                        create_parents=False,
                    )

            self.assertFalse((root / "samples/game.js").exists())
            self.assertEqual((root / "old-samples/game.js").read_text(), "original")

    def test_overwrite_attestation_exception_restores_displaced_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            target = parent / "game.js"
            target.write_text("original", encoding="utf-8")
            real_exchange = artifact_module._rename_exchange
            calls = 0

            def grow_displaced_file_after_exchange(**kwargs) -> None:
                nonlocal calls
                calls += 1
                real_exchange(**kwargs)
                if calls == 1:
                    fd = os.open(
                        kwargs["old_name"],
                        os.O_WRONLY | os.O_APPEND,
                        dir_fd=kwargs["old_dir_fd"],
                    )
                    try:
                        os.write(fd, b"x" * (MAX_ARTIFACT_BYTES + 1))
                    finally:
                        os.close(fd)

            with mock.patch(
                "orbit.runtime.artifacts._rename_exchange",
                side_effect=grow_displaced_file_after_exchange,
            ):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    self._generate(
                        root,
                        _Backend(_result("generated")),
                        overwrite=True,
                        create_parents=False,
                    )

            self.assertTrue(target.read_bytes().startswith(b"original"))
            self.assertNotEqual(target.read_text(encoding="utf-8"), "generated")

    def test_parent_permission_revoked_during_generation_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()

            def revoke_permission() -> None:
                parent.chmod(0o500)

            try:
                with self.assertRaises(PermissionError):
                    self._generate(
                        root,
                        _Backend(_result("content"), during_call=revoke_permission),
                        create_parents=False,
                    )
            finally:
                parent.chmod(0o700)

            self.assertFalse((parent / "game.js").exists())
            self.assertEqual(list(parent.glob(".orbit-artifact-*")), [])

    def test_concurrent_parent_wins_and_is_never_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = _Backend(_result("content"))
            real_mkdir = os.mkdir

            def concurrent_parent(path, mode=0o777, *, dir_fd=None) -> None:
                real_mkdir(path, mode, dir_fd=dir_fd)
                fd = os.open(
                    f"{path}/concurrent.txt",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=dir_fd,
                )
                os.write(fd, b"concurrent")
                os.close(fd)
                raise FileExistsError(errno.EEXIST, "exists")

            with mock.patch("orbit.runtime.artifacts.os.mkdir", side_effect=concurrent_parent):
                with self.assertRaisesRegex(ValueError, "parent path changed"):
                    self._generate(root, backend)

            self.assertEqual((root / "samples/concurrent.txt").read_text(), "concurrent")
            self.assertFalse((root / "samples/game.js").exists())
            self.assertEqual(list(root.glob(".orbit-artifact-*")), [])

    def test_intermediate_parent_setup_failure_removes_private_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch(
                "orbit.runtime.artifacts.os.fchmod",
                side_effect=OSError(errno.EIO, "injected parent setup failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected parent setup failure"):
                    begin_artifact_generation(
                        backend=_Backend(_result("content")),
                        user_request="create text",
                        path="one/two/note.txt",
                        overwrite=False,
                        create_parents=True,
                        workdir=root,
                        temperature=0,
                        on_progress=None,
                    )

            self.assertFalse((root / "one").exists())
            self.assertEqual(list(root.glob(".orbit-artifact-*")), [])

    def test_new_parent_post_commit_replacement_is_reported_without_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_attest = artifact_module._attest_published_artifact
            replaced = False

            def replace_after_commit(parent_fd, basename, **kwargs) -> None:
                nonlocal replaced
                if basename == "game.js" and not replaced:
                    replaced = True
                    target = root / "samples/game.js"
                    target.write_text("different", encoding="utf-8")
                return real_attest(parent_fd, basename, **kwargs)

            with mock.patch(
                "orbit.runtime.artifacts._attest_published_artifact",
                side_effect=replace_after_commit,
            ):
                with self.assertRaisesRegex(ValueError, "changed after atomic publication"):
                    self._generate(root, _Backend(_result("generated")))

            self.assertEqual((root / "samples/game.js").read_text(), "different")
            self.assertEqual(list(root.glob(".orbit-artifact-*")), [])

    def test_existing_ancestor_rename_during_new_parent_commit_cannot_report_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            real_attest = artifact_module._attest_created_directories
            calls = 0

            def move_ancestor_after_commit(created) -> None:
                nonlocal calls
                calls += 1
                if calls == 3:
                    base.rename(root / "old-base")
                    base.mkdir()
                real_attest(created)

            with mock.patch(
                "orbit.runtime.artifacts._attest_created_directories",
                side_effect=move_ancestor_after_commit,
            ):
                with self.assertRaisesRegex(ValueError, "parent directory changed"):
                    self._generate(
                        root,
                        _Backend(_result("generated")),
                        path="base/nested/game.js",
                        create_parents=True,
                    )

            self.assertFalse((root / "base/nested/game.js").exists())
            self.assertFalse((root / "old-base/nested/game.js").exists())

    def test_intermediate_parent_symlink_swap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            nested = root / "one/two"
            nested.mkdir(parents=True)
            pending = begin_artifact_generation(
                backend=_Backend(_result("generated")),
                user_request="create text",
                path="one/two/game.js",
                overwrite=False,
                create_parents=False,
                workdir=root,
                temperature=0,
                on_progress=None,
            )
            moved = Path(outside) / "one"
            (root / "one").rename(moved)
            (root / "one").symlink_to(moved, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "unavailable for verification"):
                pending.verify({"check": "text_integrity"})

            self.assertTrue((moved / "two/game.js").is_file())
            self.assertTrue((root / "one").is_symlink())

    def test_concurrent_overwrite_replacement_is_rolled_back_without_data_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            target = parent / "game.js"
            target.write_text("original", encoding="utf-8")
            real_exchange = artifact_module._rename_exchange
            calls = 0

            def replace_then_exchange(**kwargs) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    target.rename(parent / "original.saved")
                    target.write_text("concurrent", encoding="utf-8")
                real_exchange(**kwargs)

            with mock.patch(
                "orbit.runtime.artifacts._rename_exchange",
                side_effect=replace_then_exchange,
            ):
                with self.assertRaisesRegex(ValueError, "changed during atomic overwrite"):
                    self._generate(
                        root,
                        _Backend(_result("generated")),
                        overwrite=True,
                        create_parents=False,
                    )

            self.assertEqual(target.read_text(encoding="utf-8"), "concurrent")
            self.assertEqual((parent / "original.saved").read_text(encoding="utf-8"), "original")
            self.assertEqual(list(parent.glob(".orbit-artifact-*")), [])

    def test_oversized_and_empty_content_are_not_published(self) -> None:
        for content, message in (("", "empty"), ("x" * (MAX_ARTIFACT_BYTES + 1), "exceeds")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.assertRaisesRegex(RuntimeError, message):
                    self._generate(root, _Backend(_result(content)))
                self.assertFalse((root / "samples").exists())

    def test_exact_byte_and_token_boundaries_publish_only_stopped_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exact = _result("x" * MAX_ARTIFACT_BYTES)
            exact = ChatResult(
                content=exact.content,
                model=exact.model,
                finish_reason="stop",
                tool_calls=exact.tool_calls,
                prompt_tokens=exact.prompt_tokens,
                completion_tokens=ARTIFACT_CONTENT_MAX_TOKENS,
                cached_tokens=exact.cached_tokens,
                prompt_tokens_per_second=exact.prompt_tokens_per_second,
                generation_tokens_per_second=exact.generation_tokens_per_second,
            )

            _, publication = self._generate(root, _Backend(exact))

            self.assertEqual(publication.bytes_written, MAX_ARTIFACT_BYTES)
            self.assertEqual((root / "samples/game.js").stat().st_size, MAX_ARTIFACT_BYTES)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "exceeds"):
                self._generate(
                    root,
                    _Backend(_result("x" * (MAX_ARTIFACT_BYTES + 1))),
                )
            self.assertFalse((root / "samples").exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            over = _result("content")
            over = ChatResult(
                content=over.content,
                model=over.model,
                finish_reason="stop",
                tool_calls=over.tool_calls,
                prompt_tokens=over.prompt_tokens,
                completion_tokens=ARTIFACT_CONTENT_MAX_TOKENS + 1,
                cached_tokens=over.cached_tokens,
                prompt_tokens_per_second=over.prompt_tokens_per_second,
                generation_tokens_per_second=over.generation_tokens_per_second,
            )
            with self.assertRaisesRegex(RuntimeError, "token limit exceeded"):
                self._generate(root, _Backend(over))
            self.assertFalse((root / "samples").exists())

    def test_structured_or_reasoning_output_is_not_published(self) -> None:
        base = _result("content")
        results = (
            ChatResult(
                content=base.content,
                model=base.model,
                finish_reason="stop",
                tool_calls=[{"function": {"name": "unexpected", "arguments": "{}"}}],
                prompt_tokens=base.prompt_tokens,
                completion_tokens=base.completion_tokens,
                cached_tokens=base.cached_tokens,
                prompt_tokens_per_second=base.prompt_tokens_per_second,
                generation_tokens_per_second=base.generation_tokens_per_second,
            ),
            ChatResult(
                content=base.content,
                model=base.model,
                finish_reason="stop",
                tool_calls=[],
                prompt_tokens=base.prompt_tokens,
                completion_tokens=base.completion_tokens,
                cached_tokens=base.cached_tokens,
                prompt_tokens_per_second=base.prompt_tokens_per_second,
                generation_tokens_per_second=base.generation_tokens_per_second,
                reasoning_content="unexpected reasoning",
            ),
        )
        for result in results:
            with self.subTest(result=result), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.assertRaisesRegex(RuntimeError, "unexpected structured"):
                    self._generate(root, _Backend(result))
                self.assertFalse((root / "samples").exists())

    def test_invalid_utf8_and_backend_error_leave_no_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "failed validation"):
                self._generate(root, _Backend(_result("\ud800")))
            self.assertFalse((root / "samples").exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            class FailedBackend:
                def artifact_content_stream(self, *args, **kwargs):
                    raise RuntimeError("synthetic backend failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic backend failure"):
                self._generate(root, FailedBackend())
            self.assertFalse((root / "samples").exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = prepare_artifact_target(
                "samples/game.js",
                overwrite=False,
                create_parents=True,
                workdir=root,
            )

            class InterruptedBackend:
                def artifact_content_stream(self, *args, **kwargs):
                    raise KeyboardInterrupt

            with mock.patch(
                "orbit.runtime.artifacts.prepare_artifact_target",
                return_value=target,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    begin_artifact_generation(
                        backend=InterruptedBackend(),
                        user_request="create text",
                        path="samples/game.js",
                        overwrite=False,
                        create_parents=True,
                        workdir=root,
                        temperature=0,
                        on_progress=None,
                    )

            self.assertEqual(target.parent_fd, -1)
            self.assertEqual(target.destination_fd, -1)
            self.assertFalse((root / "samples").exists())

    def test_unix_socket_target_is_rejected_as_special_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "samples").mkdir()
            target = root / "samples/game.js"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                server.bind(str(target))
                with self.assertRaisesRegex(ValueError, "not a regular file"):
                    prepare_artifact_target(
                        "samples/game.js",
                        overwrite=True,
                        create_parents=False,
                        workdir=root,
                    )
            finally:
                server.close()

    def test_pending_artifact_cannot_be_verified_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pending = begin_artifact_generation(
                backend=_Backend(_result("text")),
                user_request="create text",
                path="samples/note.txt",
                overwrite=False,
                create_parents=True,
                workdir=root,
                temperature=0,
                on_progress=None,
            )
            pending.verify({"check": "text_integrity"})

            with self.assertRaisesRegex(ValueError, "no longer pending"):
                pending.verify({"check": "text_integrity"})

    def test_verification_capability_is_intrinsically_target_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pending = begin_artifact_generation(
                backend=_Backend(_result("bounded content\n")),
                user_request="create one note",
                path="samples/note.txt",
                overwrite=False,
                create_parents=True,
                workdir=root,
                temperature=0,
                on_progress=None,
            )

            evidence = pending.verify({"check": "text_integrity"})

            self.assertIn("artifact_verification: complete", evidence)
            self.assertIn("path: samples/note.txt", evidence)

    def test_verification_rejects_path_substitution_as_an_extra_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pending = begin_artifact_generation(
                backend=_Backend(_result("bounded content\n")),
                user_request="create one note",
                path="samples/note.txt",
                overwrite=False,
                create_parents=True,
                workdir=root,
                temperature=0,
                on_progress=None,
            )

            with self.assertRaisesRegex(ValueError, "arguments are invalid"):
                pending.verify(
                    {"path": "samples/other.txt", "check": "text_integrity"}
                )

            self.assertTrue((root / "samples/note.txt").is_file())
            self.assertFalse((root / "samples/other.txt").exists())

    def test_tool_like_text_is_written_verbatim_not_parsed(self) -> None:
        content = '<tool_call>{"command":"rm -rf /"}</tool_call>\n```sh\ncat <<EOF\n```\n'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._generate(root, _Backend(_result(content)))
            self.assertEqual((root / "samples/game.js").read_text(), content)

    def test_text_artifact_publication_is_content_type_neutral(self) -> None:
        fixtures = {
            "site/index.html": "<!doctype html><style>body{color:#123}</style><script>const ready=true;</script>\n",
            "src/module.js": "export const value = {enabled: true};\n",
            "src/tool.py": "def value():\n    return {'enabled': True}\n",
            "docs/note.md": "# Note\n\n- bounded\n- verified\n",
            "config/service.yaml": "service:\n  host: localhost\n  port: 8080\n",
            "config/service.toml": '[service]\nhost = "localhost"\nport = 8080\n',
            "scripts/check.sh": "#!/bin/sh\nprintf '%s\\n' 'inert content only'\n",
            "config/app.ini": "[app]\nenabled = true\n",
            "notes/plain.txt": "bounded plain text\nsecond line\n",
            "fixtures/mixed.txt": (
                "JSON: {\"quoted\": \"value\"}\n"
                "XML: <item key='value'>text</item>\n"
                "shell-like: cat <<'EOF'\nnot executed\nEOF\n"
            ),
        }
        for path, content in fixtures.items():
            with self.subTest(path=path), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._generate(
                    root,
                    _Backend(_result(content)),
                    path=path,
                    check="text_integrity",
                )

                self.assertEqual((root / path).read_text(encoding="utf-8"), content)

    def test_recovery_preserves_private_entry_owned_by_live_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                temp_fd, temp_name = artifact_module._open_named_artifact_temp(parent_fd)
                assert temp_name is not None
                lease = artifact_module._register_private_entry(
                    workdir=root,
                    parent_relative=Path("samples"),
                    temp_name=temp_name,
                    temp_fd=temp_fd,
                )
                os.close(temp_fd)
            finally:
                os.close(parent_fd)

            result = cleanup_stale_artifact_entries(root, stale_seconds=0)

            self.assertEqual(result.preserved, 1)
            self.assertTrue((parent / temp_name).is_file())
            (parent / temp_name).unlink()
            lease.release()
            self.assertFalse((root / ".orbit-artifact-state").exists())

    def test_restart_cleanup_removes_exact_private_entry_from_dead_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                temp_fd, temp_name = artifact_module._open_named_artifact_temp(parent_fd)
                assert temp_name is not None
                lease = artifact_module._register_private_entry(
                    workdir=root,
                    parent_relative=Path("samples"),
                    temp_name=temp_name,
                    temp_fd=temp_fd,
                )
                os.close(temp_fd)
            finally:
                os.close(parent_fd)
            manifest = root / ".orbit-artifact-state" / lease.manifest_name
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["pid"] = 2**30
            manifest.write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            manifest.chmod(0o600)

            result = cleanup_stale_artifact_entries(root)

            self.assertEqual(result.removed, 1)
            self.assertFalse((parent / temp_name).exists())
            self.assertFalse((root / ".orbit-artifact-state").exists())

    def test_recovery_preserves_ambiguous_symlink_and_similar_user_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "samples"
            parent.mkdir()
            user_file = parent / ".orbit-artifact-user.tmp"
            user_file.write_text("user", encoding="utf-8")
            outside = root / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                temp_fd, temp_name = artifact_module._open_named_artifact_temp(parent_fd)
                assert temp_name is not None
                lease = artifact_module._register_private_entry(
                    workdir=root,
                    parent_relative=Path("samples"),
                    temp_name=temp_name,
                    temp_fd=temp_fd,
                )
                os.close(temp_fd)
            finally:
                os.close(parent_fd)
            (parent / temp_name).unlink()
            (parent / temp_name).symlink_to(outside)
            manifest = root / ".orbit-artifact-state" / lease.manifest_name
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["pid"] = 2**30
            manifest.write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            manifest.chmod(0o600)
            malformed = root / ".orbit-artifact-state" / ("f" * 32 + ".json")
            malformed.write_text("{}", encoding="utf-8")
            malformed.chmod(0o600)

            result = cleanup_stale_artifact_entries(root)

            self.assertGreaterEqual(result.preserved, 2)
            self.assertTrue((parent / temp_name).is_symlink())
            self.assertEqual(outside.read_text(encoding="utf-8"), "outside")
            self.assertEqual(user_file.read_text(encoding="utf-8"), "user")
            self.assertTrue(manifest.is_file())
            self.assertTrue(malformed.is_file())

    def test_recovery_scan_failure_is_bounded_and_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / ".orbit-artifact-state"
            state.mkdir(mode=0o700)

            with mock.patch(
                "orbit.runtime.artifacts.os.listdir",
                side_effect=PermissionError(errno.EACCES, "denied"),
            ):
                result = cleanup_stale_artifact_entries(root)

            self.assertEqual(result.errors, 1)
            self.assertEqual(result.preserved, 1)
            self.assertTrue(state.is_dir())

if __name__ == "__main__":
    unittest.main()
