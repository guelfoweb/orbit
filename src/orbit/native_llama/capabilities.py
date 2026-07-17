from __future__ import annotations

from ctypes import c_char_p, c_int
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Protocol

from .bindings import load_native_cdll, native_cdll_flags
from .chat_template import render_gemma4_chat, render_gemma4_route_prompt_segments
from .native_names import runtime_library_filename


CAPABILITY_SCHEMA_VERSION = 1
GEMMA4_PROFILE_ID = "orbit-gemma4-native-v1"
GEMMA4_RENDERER_FIXTURE_SUITE_VERSION = 1
GEMMA4_TOOL_PROTOCOL_TEXT_HASH = "ab72a1f741975b8b541ae3f3842a2ac0593dec6477dd41f1ea105410077eee1c"
GEMMA4_RENDERER_FIXTURE_HASHES = {
    "argument_shapes": "743b4db5d3fe5b509362ab324abd8f32c1e113fd5ebc9a2031465071ac6cdbca",
    "tool_error_response": "01dbff3f4bcaea4a8db60b3dc116394985e8ab3675e3619dc1a3238476818bcf",
    "tool_generation": "b56234bba876637765951d69f6d757306a49cab7fbb3e2760dd0b593bbcd8d34",
    "tool_round_trip": GEMMA4_TOOL_PROTOCOL_TEXT_HASH,
}
GEMMA4_RENDERER_FIXTURE_SUITE_HASH = "b5148f35dd93d584e439dfefc38674824d4e645b42b84988c70964ef00023146"
GEMMA4_FINAL_PREFIX_TEXT_HASH = "c3b8e45ac695a87e60146bb8017a98f1b41fc13a708565b58160cce6d419c6f3"
GEMMA4_FINAL_PREFIX_TOKEN_HASH = "398338fd38a9c80d54b269e09ae70077ab7323ec1a47920879a24896928cdfc5"
GEMMA4_FINAL_PREFIX_TOKEN_COUNT = 64
GEMMA4_NEXT_DYNAMIC_TOKEN = 105
VERIFIED_LLAMA_CPP_COMMITS = frozenset({"6f79e02"})


class _TokenizingClient(Protocol):
    paths: object

    def tokenize(self, prompt: str) -> list[int]: ...


@dataclass(frozen=True)
class LlamaCppBuildInfo:
    build_number: int | None
    commit: str | None
    target: str | None
    compiler: str | None
    library_hash: str | None
    source: str
    error: str | None = None


def safe_gemma4_capability_manifest(client: _TokenizingClient, *, final_system_prompt: str) -> dict[str, object]:
    try:
        return build_gemma4_capability_manifest(client, final_system_prompt=final_system_prompt)
    except Exception as exc:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION,
            "profile_id": GEMMA4_PROFILE_ID,
            "status": "manifest_unavailable",
            "verification_scope": "backend_identity_and_prompt_conformance",
            "behavior_enforced": False,
            "error": f"manifest_failed:{type(exc).__name__}",
        }


def build_gemma4_capability_manifest(client: _TokenizingClient, *, final_system_prompt: str) -> dict[str, object]:
    """Return bounded, observational native-backend compatibility metadata."""

    build_bin = _client_build_bin(client)
    build = read_llama_cpp_build_info(build_bin) if build_bin is not None else _unavailable_build("build_bin_unavailable")
    renderer_fixture_hashes = {
        name: _sha256_text(rendered)
        for name, rendered in _render_renderer_fixtures().items()
    }
    renderer_suite_hash = _hash_named_hashes(renderer_fixture_hashes)
    tool_protocol_hash = renderer_fixture_hashes["tool_round_trip"]
    renderer_conformant = (
        renderer_fixture_hashes == GEMMA4_RENDERER_FIXTURE_HASHES
        and renderer_suite_hash == GEMMA4_RENDERER_FIXTURE_SUITE_HASH
    )

    segments = render_gemma4_route_prompt_segments(
        [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": "capability request"},
            {"role": "system", "content": "bounded capability evidence"},
        ],
        thinking=False,
    )
    prefix_text_hash = _sha256_text(segments.stable_prefix_text)
    tokenizer = _tokenizer_conformance(client, segments.stable_prefix_text, segments.full_prompt_text)
    build_verified = build.commit in VERIFIED_LLAMA_CPP_COMMITS

    if not renderer_conformant or prefix_text_hash != GEMMA4_FINAL_PREFIX_TEXT_HASH:
        status = "renderer_mismatch"
    elif tokenizer["status"] == "mismatch":
        status = "tokenizer_mismatch"
    elif tokenizer["status"] == "unavailable":
        status = "tokenizer_unavailable"
    elif not build_verified:
        status = "backend_unverified"
    else:
        status = "verified"

    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "profile_id": GEMMA4_PROFILE_ID,
        "status": status,
        "verification_scope": "backend_identity_and_prompt_conformance",
        "behavior_enforced": False,
        "backend": {
            "engine": "llama.cpp",
            "build_number": build.build_number,
            "commit": build.commit,
            "target": build.target,
            "compiler": build.compiler,
            "library_hash": build.library_hash,
            "source": build.source,
            "error": build.error,
            "verified_commit": build_verified,
        },
        "renderer": {
            "implementation": "orbit-gemma4",
            "fixture_suite_version": GEMMA4_RENDERER_FIXTURE_SUITE_VERSION,
            "fixture_hashes": renderer_fixture_hashes,
            "expected_fixture_hashes": GEMMA4_RENDERER_FIXTURE_HASHES,
            "fixture_suite_hash": renderer_suite_hash,
            "expected_fixture_suite_hash": GEMMA4_RENDERER_FIXTURE_SUITE_HASH,
            "tool_protocol_text_hash": tool_protocol_hash,
            "expected_tool_protocol_text_hash": GEMMA4_TOOL_PROTOCOL_TEXT_HASH,
            "final_prefix_text_hash": prefix_text_hash,
            "expected_final_prefix_text_hash": GEMMA4_FINAL_PREFIX_TEXT_HASH,
            "conformant": renderer_conformant and prefix_text_hash == GEMMA4_FINAL_PREFIX_TEXT_HASH,
        },
        "tokenizer": tokenizer,
        "requirements": {
            "bos_token": 2,
            "turn_token": GEMMA4_NEXT_DYNAMIC_TOKEN,
            "stable_prefix_tokens": GEMMA4_FINAL_PREFIX_TOKEN_COUNT,
            "production_prefill_alignment": 64,
        },
    }


def read_llama_cpp_build_info(build_bin: Path) -> LlamaCppBuildInfo:
    common_path = build_bin / runtime_library_filename("llama-common")
    llama_path = build_bin / runtime_library_filename("llama")
    if not common_path.is_file():
        return _unavailable_build("llama_common_unavailable")
    try:
        common = load_native_cdll(common_path, mode=native_cdll_flags())
        return LlamaCppBuildInfo(
            build_number=_read_int_symbol(common, "LLAMA_BUILD_NUMBER"),
            commit=_read_text_symbol(common, "LLAMA_COMMIT", limit=64),
            target=_read_text_symbol(common, "LLAMA_BUILD_TARGET", limit=128),
            compiler=_read_text_symbol(common, "LLAMA_COMPILER", limit=128),
            library_hash=_sha256_file(llama_path) if llama_path.is_file() else None,
            source="runtime_symbols",
        )
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        return _unavailable_build(f"build_info_unavailable:{type(exc).__name__}")


def _render_tool_protocol_fixture() -> str:
    tools = [_renderer_fixture_tool()]
    messages = [
        {"role": "system", "content": "Use one available tool when required."},
        {"role": "user", "content": "Read the synthetic fixture."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "fixture.txt"}},
                }
            ],
        },
        {"role": "tool", "name": "read_file", "content": "alpha=1"},
    ]
    return render_gemma4_chat(messages, tools=tools, thinking=False)


def _render_renderer_fixtures() -> dict[str, str]:
    tool = _renderer_fixture_tool()
    return {
        "argument_shapes": render_gemma4_chat(
            [
                {"role": "system", "content": "Use the provided tool."},
                {"role": "user", "content": "Use structured arguments."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "synthetic_tool",
                                "arguments": {
                                    "enabled": True,
                                    "items": ["a", "quoted \"value\""],
                                    "nested": {"count": 2, "empty": None},
                                },
                            },
                        }
                    ],
                },
            ],
            thinking=False,
        ),
        "tool_error_response": render_gemma4_chat(
            [
                {"role": "system", "content": "Answer from tool evidence."},
                {"role": "user", "content": "Run the synthetic operation."},
                {"role": "tool", "name": "synthetic_tool", "content": "error: denied\ncode=7"},
            ],
            thinking=False,
        ),
        "tool_generation": render_gemma4_chat(
            [
                {"role": "system", "content": "Use one available tool when required."},
                {"role": "user", "content": "Read the synthetic fixture."},
            ],
            tools=[tool],
            thinking=False,
        ),
        "tool_round_trip": _render_tool_protocol_fixture(),
    }


def _renderer_fixture_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path."}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


def _hash_named_hashes(values: dict[str, str]) -> str:
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return _sha256_text(payload)


def _tokenizer_conformance(client: _TokenizingClient, stable_prefix: str, full_prompt: str) -> dict[str, object]:
    try:
        prefix_tokens = client.tokenize(stable_prefix)
        full_tokens = client.tokenize(full_prompt)
        if not _valid_token_ids(prefix_tokens) or not _valid_token_ids(full_tokens):
            return _unavailable_tokenizer("invalid_token_ids")
        next_dynamic_token = full_tokens[len(prefix_tokens)] if len(full_tokens) > len(prefix_tokens) else None
        token_hash = _hash_token_ids(prefix_tokens)
    except (AttributeError, OverflowError, RuntimeError, struct.error, TypeError, ValueError):
        return _unavailable_tokenizer("tokenizer_probe_unavailable")

    conformant = (
        len(prefix_tokens) == GEMMA4_FINAL_PREFIX_TOKEN_COUNT
        and token_hash == GEMMA4_FINAL_PREFIX_TOKEN_HASH
        and next_dynamic_token == GEMMA4_NEXT_DYNAMIC_TOKEN
    )
    return {
        "status": "verified" if conformant else "mismatch",
        "prefix_tokens": len(prefix_tokens),
        "prefix_token_hash": token_hash,
        "expected_prefix_token_hash": GEMMA4_FINAL_PREFIX_TOKEN_HASH,
        "next_dynamic_token": next_dynamic_token,
        "expected_next_dynamic_token": GEMMA4_NEXT_DYNAMIC_TOKEN,
        "conformant": conformant,
        "error": None,
    }


def _unavailable_tokenizer(error: str) -> dict[str, object]:
    return {
        "status": "unavailable",
        "prefix_tokens": None,
        "prefix_token_hash": None,
        "expected_prefix_token_hash": GEMMA4_FINAL_PREFIX_TOKEN_HASH,
        "next_dynamic_token": None,
        "expected_next_dynamic_token": GEMMA4_NEXT_DYNAMIC_TOKEN,
        "conformant": False,
        "error": error,
    }


def _client_build_bin(client: _TokenizingClient) -> Path | None:
    paths = getattr(client, "paths", None)
    value = getattr(paths, "build_bin", None)
    return value if isinstance(value, Path) else None


def _unavailable_build(error: str) -> LlamaCppBuildInfo:
    return LlamaCppBuildInfo(None, None, None, None, None, "unavailable", error)


def _read_int_symbol(library: object, name: str) -> int:
    return int(c_int.in_dll(library, name).value)


def _read_text_symbol(library: object, name: str, *, limit: int) -> str | None:
    raw = c_char_p.in_dll(library, name).value
    if raw is None:
        return None
    value = raw.decode("utf-8", errors="replace")
    if not value.isprintable():
        return None
    return value[:limit]


def _hash_token_ids(tokens: list[int]) -> str:
    payload = b"".join(struct.pack("<i", token) for token in tokens)
    return hashlib.sha256(payload).hexdigest()


def _valid_token_ids(tokens: object) -> bool:
    return isinstance(tokens, list) and all(isinstance(token, int) and -(2**31) <= token < 2**31 for token in tokens)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
