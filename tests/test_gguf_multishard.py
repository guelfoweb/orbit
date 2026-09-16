"""GGUF-MULTISHARD-DOWNLOAD-21: split GGUF sets are downloaded, resumed,
validated and discovered as one artifact named by their first shard.

Every test is offline: shards are tiny synthetic GGUF files and HTTP is a fake
Range-aware opener that can be told to fail mid-transfer.
"""
from __future__ import annotations

import io
import os
import pathlib
import struct
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_llama import download_cli  # noqa: E402
from orbit.native_llama.gguf_split import (  # noqa: E402
    is_secondary_shard,
    parse_split_name,
    read_gguf_header,
    split_problems,
    split_sibling_names,
    validate_split_set,
)
from orbit.native_llama.model_discovery import discover_models, paint_model_status  # noqa: E402
from orbit.native_llama.model_download import DownloadIncomplete, download_model, fetch_resumable  # noqa: E402
from orbit.native_llama.model_profiles import detect_native_model_profile  # noqa: E402
from orbit.native_llama.model_registry import MODELS_DIR_ENV, get_manifest, local_model_path, resolve_model  # noqa: E402

QWEN_REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
QWEN_FIRST = "Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf"
QWEN_ID = "qwen38-flash-next-ud-iq1-m"
HF = "https://huggingface.co"


# --- synthetic GGUF shards --------------------------------------------------


def _kv_string(key: str) -> bytes:
    raw = key.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf_bytes(*, count: int, number: int, tensors: int = 1224, payload: bytes = b"P" * 64, extra: dict | None = None,
               version: int = 3, magic: int = 0x46554747) -> bytes:
    """A minimal GGUF header carrying the gguf-split keys (u16/u16/i32 as
    gguf-split writes them) plus optional string metadata, then some payload."""
    kv = []
    for key, value in (extra or {}).items():
        kv.append(_kv_string(key) + struct.pack("<I", 8) + _kv_string(value))
    kv.append(_kv_string("split.no") + struct.pack("<I", 2) + struct.pack("<H", number))
    kv.append(_kv_string("split.count") + struct.pack("<I", 2) + struct.pack("<H", count))
    kv.append(_kv_string("split.tensors.count") + struct.pack("<I", 5) + struct.pack("<i", tensors))
    header = struct.pack("<II", magic, version) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv)) + b"".join(kv)
    return header + payload


def shard_set(count: int = 3, base: str = "Qwen3.8-Flash-Next-UD-IQ1_M", tensors: int = 1224) -> dict[str, bytes]:
    return {
        f"{base}-{i:05d}-of-{count:05d}.gguf": gguf_bytes(count=count, number=i - 1, tensors=tensors,
                                                          payload=bytes([i]) * (32 + 7 * i),
                                                          extra={"general.architecture": "qwen4exp"} if i == 1 else None)
        for i in range(1, count + 1)
    }


class FakeHTTP:
    """A Range-aware fake for urllib's opener. `fail_after` = {url: n} makes a
    transfer stop after n payload bytes (simulating a dropped connection);
    `ignore_range` makes the server answer 200 to a ranged request."""

    def __init__(self, files: dict[str, bytes], *, fail_after: dict[str, int] | None = None, ignore_range: bool = False):
        self.files = files
        self.fail_after = dict(fail_after or {})
        self.ignore_range = ignore_range
        self.requests: list[tuple[str, str | None]] = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        rng = request.headers.get("Range")
        self.requests.append((url, rng))
        if url not in self.files:
            raise OSError(f"404 {url}")
        body = self.files[url]
        start = 0
        status = 200
        headers = {"Content-Length": str(len(body))}
        if rng and not self.ignore_range:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= len(body):
                status = 416
                headers = {"Content-Range": f"bytes */{len(body)}"}
                body = b""
            else:
                status = 206
                headers = {"Content-Range": f"bytes {start}-{len(body)-1}/{len(body)}",
                           "Content-Length": str(len(body) - start)}
                body = body[start:]
        limit = self.fail_after.pop(url, None)
        return _FakeResponse(status, headers, body, limit)


class _FakeResponse:
    def __init__(self, status, headers, body, limit):
        self.status = status
        self.headers = headers
        self._buf = io.BytesIO(body)
        self._limit = limit
        self._sent = 0

    def read(self, size=-1):
        if self._limit is not None and self._sent >= self._limit:
            raise ConnectionResetError("connection dropped")
        if self._limit is not None:
            size = min(size, self._limit - self._sent) if size > 0 else self._limit - self._sent
        data = self._buf.read(size)
        self._sent += len(data)
        return data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _urls(repo: str, files: dict[str, bytes]) -> dict[str, bytes]:
    return {f"{HF}/{repo}/resolve/main/{name}": data for name, data in files.items()}


def _verified_qwen_profile():
    import hashlib
    metadata = {
        "general.architecture": "qwen4exp", "general.name": "Qwen3.8 Flash Next", "general.file_type": "31",
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.bos_token_id": "248044", "tokenizer.ggml.eos_token_id": "248046",
        "qwen4exp.context_length": "262144", "qwen4exp.block_count": "48",
        "qwen4exp.expert_count": "512", "qwen4exp.expert_used_count": "10",
    }
    template = "t"
    with mock.patch("orbit.native_llama.model_profiles.QWEN38_OFFICIAL_TEMPLATE_SHA256", hashlib.sha256(template.encode()).hexdigest()):
        return detect_native_model_profile(metadata, template)


def _inspector(first: Path):
    def inspect(path: Path):
        if path.resolve() == first.resolve():
            return _verified_qwen_profile()
        raise ValueError("unsupported")
    return inspect


# --- tests -------------------------------------------------------------------


class NamingTests(unittest.TestCase):
    def test_A_single_gguf_is_not_a_split(self) -> None:
        self.assertIsNone(parse_split_name("Ornith-1.5-35B-Q4_K_M.gguf"))
        self.assertEqual(split_sibling_names("dir/model.gguf"), ("model.gguf",))
        self.assertFalse(is_secondary_shard("model-2-of-3.gguf"))  # not the canonical padded pattern
        self.assertEqual(split_problems(Path("/nonexistent/model.gguf")), ())  # single files: contract unchanged

    def test_B_three_shard_expansion_preserves_padding(self) -> None:
        self.assertEqual(split_sibling_names(QWEN_FIRST), (
            "Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf",
            "Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf",
            "Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf",
        ))
        split = parse_split_name("sub/dir/" + QWEN_FIRST)
        self.assertEqual((split.index, split.count, split.width, split.first), (1, 3, 5, QWEN_FIRST))
        self.assertTrue(is_secondary_shard("Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf"))

    def test_C_arbitrary_shard_counts_and_any_requested_index_expand_to_the_whole_set(self) -> None:
        names = split_sibling_names("big-00007-of-00012.gguf")
        self.assertEqual(len(names), 12)
        self.assertEqual(names[0], "big-00001-of-00012.gguf")
        self.assertEqual(names[-1], "big-00012-of-00012.gguf")
        self.assertEqual(split_sibling_names("x-00001-of-00001.gguf"), ("x-00001-of-00001.gguf",))
        for bad in ("m-00000-of-00003.gguf", "m-00004-of-00003.gguf", "m-00001-of-00000.gguf"):
            with self.assertRaisesRegex(ValueError, "malformed split GGUF name"):
                parse_split_name(bad)
        self.assertFalse(is_secondary_shard("m-00004-of-00003.gguf"))  # malformed is never silently a shard


class ValidationTests(unittest.TestCase):
    def test_D_complete_consistent_set_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in shard_set().items():
                (Path(tmp) / name).write_bytes(data)
            validation = validate_split_set(Path(tmp) / QWEN_FIRST)
            self.assertTrue(validation.complete)
            self.assertEqual(validation.count, 3)
            self.assertEqual([s.name for s in validation.shards], list(shard_set()))
            header = read_gguf_header(Path(tmp) / QWEN_FIRST)
            self.assertEqual(header.kv["general.architecture"], "qwen4exp")
            # asking for the set through shard 2 validates the same set
            self.assertTrue(validate_split_set(Path(tmp) / "Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf").complete)

    def test_E_a_missing_or_empty_shard_makes_the_set_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = shard_set()
            for name, data in files.items():
                (Path(tmp) / name).write_bytes(data)
            (Path(tmp) / "Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf").unlink()
            problems = split_problems(Path(tmp) / QWEN_FIRST)
            self.assertEqual(problems, ("Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf: missing",))
            (Path(tmp) / "Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf").write_bytes(b"")
            self.assertIn("empty file", split_problems(Path(tmp) / QWEN_FIRST)[0])

    def test_F_corrupt_or_inconsistent_shards_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = shard_set()
            for name, data in files.items():
                (root / name).write_bytes(data)
            second = root / "Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf"
            second.write_bytes(b"not a gguf at all")
            self.assertIn("bad magic", split_problems(root / QWEN_FIRST)[0])
            second.write_bytes(files[second.name][:20])  # truncated header
            self.assertIn("truncated or malformed", split_problems(root / QWEN_FIRST)[0])
            second.write_bytes(gguf_bytes(count=4, number=1))   # split.count disagrees with the name
            self.assertIn("split.count is 4", split_problems(root / QWEN_FIRST)[0])
            second.write_bytes(gguf_bytes(count=3, number=2))   # wrong position
            self.assertIn("split.no is 2, expected 1", split_problems(root / QWEN_FIRST)[0])
            second.write_bytes(gguf_bytes(count=3, number=1, tensors=999))  # tensor count disagrees
            self.assertEqual(split_problems(root / QWEN_FIRST), ("Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf: split.tensors.count is 999, disagrees with the first shard (1224)",))
            second.write_bytes(gguf_bytes(count=3, number=1, version=9))
            self.assertIn("unsupported GGUF version", split_problems(root / QWEN_FIRST)[0])
            self.assertFalse(validate_split_set(root / "m-00004-of-00003.gguf").complete)  # malformed name


class DownloadTests(unittest.TestCase):
    def test_J_requesting_the_first_shard_downloads_the_whole_set(self) -> None:
        files = shard_set()
        http = FakeHTTP(_urls(QWEN_REPO, files))
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            result = download_model(f"{QWEN_REPO}/{QWEN_FIRST}", models_dir=Path(tmp), opener=http,
                                    on_shard=lambda i, n, name, action: events.append((i, n, action)))
            store = Path(tmp) / "unsloth--Qwen3.8-Flash-Next-GGUF"
            self.assertEqual(result.path, store / QWEN_FIRST)
            self.assertTrue(result.downloaded)
            self.assertEqual([s.name for s in result.shards], list(files))
            for name, data in files.items():
                self.assertEqual((store / name).read_bytes(), data)
            self.assertEqual(events, [(1, 3, "download"), (2, 3, "download"), (3, 3, "download")])
            self.assertEqual([r[0].rsplit("/", 1)[1] for r in http.requests], list(files))
            self.assertEqual(sorted(p.name for p in store.iterdir()), sorted(files))  # no .part/.lock leftovers
            self.assertTrue(validate_split_set(store / QWEN_FIRST).complete)
            # a second run reuses every shard without any request
            http.requests.clear(); events.clear()
            again = download_model(f"{QWEN_REPO}/{QWEN_FIRST}", models_dir=Path(tmp), opener=http,
                                   on_shard=lambda i, n, name, action: events.append(action))
            self.assertFalse(again.downloaded)
            self.assertEqual(http.requests, [])
            self.assertEqual(events, ["present"] * 3)

    def test_A_single_file_download_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def retrieve(url, dest):
                calls.append(url); Path(dest).write_bytes(b"single")

            http = FakeHTTP({})
            result = download_model("owner/repo/model.gguf", models_dir=Path(tmp), retrieve=retrieve, opener=http)
            self.assertEqual(calls, [f"{HF}/owner/repo/resolve/main/model.gguf"])
            self.assertEqual(http.requests, [])  # the resumable path is split-only
            self.assertEqual(result.shards, (result.path,))
            self.assertEqual(result.path.read_bytes(), b"single")
            self.assertEqual(sorted(p.name for p in result.path.parent.iterdir()), ["model.gguf"])

    def test_G_a_dropped_shard_2_keeps_shards_1_and_partial_2_and_resumes(self) -> None:
        files = shard_set()
        urls = _urls(QWEN_REPO, files)
        second_url = f"{HF}/{QWEN_REPO}/resolve/main/Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf"
        http = FakeHTTP(urls, fail_after={second_url: 20})
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "unsloth--Qwen3.8-Flash-Next-GGUF"
            with self.assertRaises(ConnectionResetError):
                download_model(f"{QWEN_REPO}/{QWEN_FIRST}", models_dir=Path(tmp), opener=http)
            self.assertTrue((store / QWEN_FIRST).exists())                                  # shard 1 kept
            second = store / "Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf"
            self.assertFalse(second.exists())                                                # never looks complete
            part = store / "Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf.part"
            self.assertEqual(part.stat().st_size, 20)                                        # partial kept for resume
            self.assertFalse(part.with_name(part.name[:-5] + ".lock").exists())               # lock never lingers
            self.assertFalse(validate_split_set(store / QWEN_FIRST).complete)
            # resume: shard 1 present, shard 2 continued from byte 20, shard 3 fetched
            http.requests.clear(); events = []
            result = download_model(f"{QWEN_REPO}/{QWEN_FIRST}", models_dir=Path(tmp), opener=http,
                                    on_shard=lambda i, n, name, action: events.append((i, action)))
            self.assertEqual(events, [(1, "present"), (2, "resume"), (3, "download")])
            self.assertEqual(http.requests[0], (second_url, "bytes=20-"))
            self.assertEqual(len(http.requests), 2)                                          # shard 1 not refetched
            self.assertEqual(second.read_bytes(), files[second.name])
            self.assertTrue(result.downloaded)
            self.assertTrue(validate_split_set(store / QWEN_FIRST).complete)
            self.assertFalse(part.exists())

    def test_G_existing_shard_1_with_missing_2_and_3_downloads_only_the_missing_ones(self) -> None:
        files = shard_set()
        http = FakeHTTP(_urls(QWEN_REPO, files))
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "unsloth--Qwen3.8-Flash-Next-GGUF"; store.mkdir(parents=True)
            (store / QWEN_FIRST).write_bytes(files[QWEN_FIRST])
            result = download_model(f"{QWEN_REPO}/{QWEN_FIRST}", models_dir=Path(tmp), opener=http)
            self.assertEqual([r[0].rsplit("/", 1)[1] for r in http.requests], list(files)[1:])
            self.assertTrue(result.downloaded)
            self.assertTrue(validate_split_set(store / QWEN_FIRST).complete)

    def test_a_server_that_ignores_range_restarts_the_shard_cleanly(self) -> None:
        files = shard_set()
        http = FakeHTTP(_urls(QWEN_REPO, files), ignore_range=True)
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "unsloth--Qwen3.8-Flash-Next-GGUF"; store.mkdir(parents=True)
            part = store / "Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf.part"
            part.write_bytes(b"stale partial")
            download_model(f"{QWEN_REPO}/{QWEN_FIRST}", models_dir=Path(tmp), opener=http)
            second = store / "Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf"
            self.assertEqual(second.read_bytes(), files[second.name])

    def test_a_short_transfer_never_finalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            url = "https://example.invalid/x.gguf"
            body = b"Y" * 100

            class Short(FakeHTTP):
                def __call__(self, request, timeout=None):
                    resp = super().__call__(request, timeout)
                    resp._buf = io.BytesIO(body[:40])  # server closes early but claimed 100
                    return resp

            with self.assertRaisesRegex(DownloadIncomplete, "received 40 of 100"):
                fetch_resumable(url, Path(tmp) / "x.gguf", opener=Short({url: body}))
            self.assertFalse((Path(tmp) / "x.gguf").exists())
            self.assertEqual((Path(tmp) / "x.gguf.part").stat().st_size, 40)

    def test_F_a_corrupt_downloaded_shard_is_discarded_and_reported(self) -> None:
        files = shard_set()
        files["Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf"] = gguf_bytes(count=3, number=2, tensors=5)  # inconsistent
        http = FakeHTTP(_urls(QWEN_REPO, files))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "not consistent.*00003-of-00003.gguf: split.tensors.count is 5"):
                download_model(f"{QWEN_REPO}/{QWEN_FIRST}", models_dir=Path(tmp), opener=http)
            store = Path(tmp) / "unsloth--Qwen3.8-Flash-Next-GGUF"
            self.assertTrue((store / QWEN_FIRST).exists())
            # the set is not complete; shards we fetched that fail their own
            # check are removed, so nothing partial masquerades as complete
            self.assertFalse(validate_split_set(store / QWEN_FIRST).complete)

    def test_F_a_present_but_wrong_shard_is_reported_not_overwritten(self) -> None:
        files = shard_set()
        http = FakeHTTP(_urls(QWEN_REPO, files))
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "unsloth--Qwen3.8-Flash-Next-GGUF"; store.mkdir(parents=True)
            wrong = store / "Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf"
            wrong.write_bytes(b"user placed junk")
            with self.assertRaisesRegex(ValueError, "not a valid shard 2 of 3"):
                download_model(f"{QWEN_REPO}/{QWEN_FIRST}", models_dir=Path(tmp), opener=http)
            self.assertEqual(wrong.read_bytes(), b"user placed junk")

    def test_F_a_malformed_split_name_is_refused_before_any_request(self) -> None:
        http = FakeHTTP({})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "malformed split GGUF name"):
                download_model("owner/repo/m-00004-of-00003.gguf", models_dir=Path(tmp), opener=http)
        self.assertEqual(http.requests, [])

    def test_one_writer_per_destination(self) -> None:
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "x.gguf"
            lock = dest.with_name("x.gguf.lock")
            with lock.open("a+b") as held:
                fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "another download is already writing"):
                    fetch_resumable("https://example.invalid/x.gguf", dest, opener=FakeHTTP({}))

    def test_H_the_canonical_models_dir_is_used(self) -> None:
        files = shard_set()
        http = FakeHTTP(_urls(QWEN_REPO, files))
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "configured"
            with mock.patch.dict(os.environ, {MODELS_DIR_ENV: str(store)}, clear=False):
                result = download_model(f"{QWEN_REPO}/{QWEN_FIRST}", opener=http)
            self.assertEqual(result.path, store / "unsloth--Qwen3.8-Flash-Next-GGUF" / QWEN_FIRST)
            self.assertTrue(result.path.exists())


class DiscoveryTests(unittest.TestCase):
    def _store(self, tmp: Path, files: dict[str, bytes]) -> Path:
        store = tmp / "models" / "unsloth--Qwen3.8-Flash-Next-GGUF"; store.mkdir(parents=True)
        for name, data in files.items():
            (store / name).write_bytes(data)
        return store

    def test_D_complete_set_is_available_by_its_first_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp), shard_set())
            result = discover_models(models_dir=Path(tmp) / "models", hf_cache=Path(tmp) / "hf",
                                     inspector=_inspector(store / QWEN_FIRST))
            row = {r.model: r for r in result.rows}["Qwen 3.8 Flash Next"]
            self.assertEqual((row.local, row.support), ("AVAILABLE", "VERIFIED"))
            self.assertEqual(Path(row.path_or_action).name, QWEN_FIRST)          # I: first shard only
            self.assertEqual(len([r for r in result.rows if "IQ1_M" in r.model]), 0)  # shards 2/3 never rows

    def test_E_missing_shard_is_incomplete_never_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = shard_set(); del files["Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf"]
            store = self._store(Path(tmp), files)
            inspector = mock.Mock(side_effect=AssertionError("an incomplete set must not be inspected natively"))
            result = discover_models(models_dir=Path(tmp) / "models", hf_cache=Path(tmp) / "hf", inspector=inspector)
            row = {r.model: r for r in result.rows}["Qwen 3.8 Flash Next"]
            self.assertEqual((row.local, row.support), ("INCOMPLETE", "VERIFIED"))
            self.assertEqual(row.path_or_action, f"orbit download {QWEN_REPO}/{QWEN_FIRST}")
            self.assertEqual(row.model_id, QWEN_ID)
            self.assertEqual(sum(1 for r in result.rows if r.model == "Qwen 3.8 Flash Next"), 1)  # no extra MISSING row
            inspector.assert_not_called()
            self.assertEqual(paint_model_status("INCOMPLETE", color=False), "INCOMPLETE")
            self.assertIn("INCOMPLETE", paint_model_status("INCOMPLETE", color=True))
            # I: the server-side availability check refuses the set with a resume hint
            manifest = get_manifest(QWEN_ID)
            with self.assertRaisesRegex(FileNotFoundError, "incomplete.*00003-of-00003.gguf: missing.*orbit download"):
                resolve_model(manifest, models_dir=Path(tmp) / "models", hf_cache=Path(tmp) / "hf")

    def test_F_inconsistent_set_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = shard_set()
            files["Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf"] = gguf_bytes(count=3, number=0)  # wrong split.no
            self._store(Path(tmp), files)
            result = discover_models(models_dir=Path(tmp) / "models", hf_cache=Path(tmp) / "hf",
                                     inspector=mock.Mock(side_effect=AssertionError("not inspected")))
            row = {r.model: r for r in result.rows}["Qwen 3.8 Flash Next"]
            self.assertEqual(row.local, "INCOMPLETE")

    def test_I_server_resolution_hands_the_backend_the_first_shard_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp), shard_set())
            resolved = resolve_model(get_manifest(QWEN_ID), models_dir=Path(tmp) / "models", hf_cache=Path(tmp) / "hf")
            self.assertEqual(resolved.target_path, store / QWEN_FIRST)
            self.assertEqual(local_model_path(get_manifest(QWEN_ID).target, models_dir=Path(tmp) / "models"), store / QWEN_FIRST)

    def test_an_unregistered_incomplete_set_is_shown_incomplete_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "models" / "other--repo"; store.mkdir(parents=True)
            (store / "thing-00001-of-00002.gguf").write_bytes(gguf_bytes(count=2, number=0))
            result = discover_models(models_dir=Path(tmp) / "models", hf_cache=Path(tmp) / "hf",
                                     inspector=mock.Mock(side_effect=AssertionError("not inspected")))
            row = next(r for r in result.rows if r.model == "thing-00001-of-00002.gguf")
            self.assertEqual((row.local, row.support), ("INCOMPLETE", "UNVERIFIED"))


class CliTests(unittest.TestCase):
    def test_J_orbit_download_prints_one_line_per_shard(self) -> None:
        files = shard_set()
        http = FakeHTTP(_urls(QWEN_REPO, files))
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(spec=f"{QWEN_REPO}/{QWEN_FIRST}", all=False, mmproj=False, models_dir=str(Path(tmp)))
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.object(download_cli, "download_advisory", return_value=None), \
                 mock.patch("orbit.native_llama.model_download.urlopen", http), \
                 redirect_stdout(out), redirect_stderr(err):
                code = download_cli._download(args)
            self.assertEqual(code, 0, err.getvalue())
            text = out.getvalue()
            for line in (f"shard 1/3: {QWEN_FIRST} (downloading)",
                         "shard 2/3: Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf (downloading)",
                         "shard 3/3: Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf (downloading)"):
                self.assertIn(line, text)
            self.assertIn(f"downloaded: {Path(tmp) / 'unsloth--Qwen3.8-Flash-Next-GGUF' / QWEN_FIRST}", text)
            # a re-run reports every shard as present
            out = io.StringIO()
            with mock.patch.object(download_cli, "download_advisory", return_value=None), \
                 mock.patch("orbit.native_llama.model_download.urlopen", http), redirect_stdout(out), redirect_stderr(io.StringIO()):
                self.assertEqual(download_cli._download(args), 0)
            self.assertIn("shard 3/3: Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf (already present)", out.getvalue())
            self.assertIn("already present:", out.getvalue())


if __name__ == "__main__":
    unittest.main()
