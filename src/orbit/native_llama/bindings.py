from __future__ import annotations

from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    Structure,
    c_bool,
    c_char,
    c_char_p,
    c_float,
    c_int,
    c_int8,
    c_int32,
    c_size_t,
    c_ubyte,
    c_uint32,
    c_void_p,
    cast,
)
from pathlib import Path
import ctypes
import json
import os

from .native_names import platform_runtime_libs, runtime_library_filename
from .mtmd_bridge import validate_mtmd_bridge_artifact


llama_token = c_int32
llama_pos = c_int32
llama_seq_id = c_int32


_CDLL_CACHE: dict[tuple[str, int], CDLL] = {}


def native_cdll_flags() -> int:
    return (
        getattr(os, "RTLD_GLOBAL", 0)
        | getattr(os, "RTLD_NOW", 0)
        | getattr(os, "RTLD_NODELETE", 0)
    )


def load_native_cdll(path: Path, *, mode: int) -> CDLL:
    key = (str(path.resolve()), mode)
    lib = _CDLL_CACHE.get(key)
    if lib is None:
        lib = ctypes.CDLL(str(path), mode=mode)
        _CDLL_CACHE[key] = lib
    return lib

LlamaProgressCallback = CFUNCTYPE(c_bool, c_float, c_void_p)
GgmlAbortCallback = CFUNCTYPE(c_bool, c_void_p)
GgmlLogCallback = CFUNCTYPE(None, c_int, c_char_p, c_void_p)


class LlamaBatch(Structure):
    _fields_ = [
        ("n_tokens", c_int32),
        ("token", POINTER(llama_token)),
        ("embd", POINTER(c_float)),
        ("pos", POINTER(llama_pos)),
        ("n_seq_id", POINTER(c_int32)),
        ("seq_id", POINTER(POINTER(llama_seq_id))),
        ("logits", POINTER(c_int8)),
    ]


class LlamaModelParams(Structure):
    _fields_ = [
        ("devices", c_void_p),
        ("tensor_buft_overrides", c_void_p),
        ("n_gpu_layers", c_int32),
        ("split_mode", c_int),
        ("main_gpu", c_int32),
        ("tensor_split", POINTER(c_float)),
        ("progress_callback", LlamaProgressCallback),
        ("progress_callback_user_data", c_void_p),
        ("kv_overrides", c_void_p),
        ("vocab_only", c_bool),
        ("use_mmap", c_bool),
        ("use_direct_io", c_bool),
        ("use_mlock", c_bool),
        ("check_tensors", c_bool),
        ("use_extra_bufts", c_bool),
        ("no_host", c_bool),
        ("no_alloc", c_bool),
    ]


class LlamaContextParams(Structure):
    _fields_ = [
        ("n_ctx", c_uint32),
        ("n_batch", c_uint32),
        ("n_ubatch", c_uint32),
        ("n_seq_max", c_uint32),
        ("n_rs_seq", c_uint32),
        ("n_outputs_max", c_uint32),
        ("n_threads", c_int32),
        ("n_threads_batch", c_int32),
        ("ctx_type", c_int),
        ("rope_scaling_type", c_int),
        ("pooling_type", c_int),
        ("attention_type", c_int),
        ("flash_attn_type", c_int),
        ("rope_freq_base", c_float),
        ("rope_freq_scale", c_float),
        ("yarn_ext_factor", c_float),
        ("yarn_attn_factor", c_float),
        ("yarn_beta_fast", c_float),
        ("yarn_beta_slow", c_float),
        ("yarn_orig_ctx", c_uint32),
        ("defrag_thold", c_float),
        ("cb_eval", c_void_p),
        ("cb_eval_user_data", c_void_p),
        ("type_k", c_int),
        ("type_v", c_int),
        ("abort_callback", GgmlAbortCallback),
        ("abort_callback_data", c_void_p),
        ("embeddings", c_bool),
        ("offload_kqv", c_bool),
        ("no_perf", c_bool),
        ("op_offload", c_bool),
        ("swa_full", c_bool),
        ("kv_unified", c_bool),
        ("samplers", c_void_p),
        ("n_samplers", c_size_t),
        ("ctx_other", c_void_p),
    ]


class LlamaSamplerChainParams(Structure):
    _fields_ = [("no_perf", c_bool)]


class LlamaChatMessage(Structure):
    _fields_ = [
        ("role", c_char_p),
        ("content", c_char_p),
    ]


class LlamaLibrary:
    def __init__(self, build_bin: Path) -> None:
        self.build_bin = build_bin
        self._handles: list[CDLL] = []
        self.lib = self._load_library(runtime_library_filename("llama"))
        self._configure_api()

    def _load_library(self, name: str) -> CDLL:
        flags = native_cdll_flags()
        # Load dependencies explicitly because LD_LIBRARY_PATH cannot be changed
        # reliably after Python startup.
        for dep in platform_runtime_libs():
            if dep == name:
                continue
            path = self.build_bin / dep
            if path.exists():
                try:
                    self._handles.append(load_native_cdll(path, mode=flags))
                except OSError:
                    pass
        return load_native_cdll(self.build_bin / name, mode=flags)

    def _configure_api(self) -> None:
        lib = self.lib
        lib.ggml_backend_load_all.argtypes = []
        lib.ggml_backend_load_all.restype = None
        lib.llama_backend_free.argtypes = []
        lib.llama_backend_free.restype = None
        lib.llama_log_set.argtypes = [GgmlLogCallback, c_void_p]
        lib.llama_log_set.restype = None

        lib.llama_model_default_params.argtypes = []
        lib.llama_model_default_params.restype = LlamaModelParams
        lib.llama_context_default_params.argtypes = []
        lib.llama_context_default_params.restype = LlamaContextParams
        lib.llama_sampler_chain_default_params.argtypes = []
        lib.llama_sampler_chain_default_params.restype = LlamaSamplerChainParams

        lib.llama_model_load_from_file.argtypes = [c_char_p, LlamaModelParams]
        lib.llama_model_load_from_file.restype = c_void_p
        lib.llama_model_free.argtypes = [c_void_p]
        lib.llama_model_free.restype = None
        lib.llama_init_from_model.argtypes = [c_void_p, LlamaContextParams]
        lib.llama_init_from_model.restype = c_void_p
        lib.llama_free.argtypes = [c_void_p]
        lib.llama_free.restype = None
        lib.llama_get_memory.argtypes = [c_void_p]
        lib.llama_get_memory.restype = c_void_p
        lib.llama_memory_clear.argtypes = [c_void_p, c_bool]
        lib.llama_memory_clear.restype = None
        lib.llama_memory_seq_cp.argtypes = [c_void_p, c_int32, c_int32, c_int32, c_int32]
        lib.llama_memory_seq_cp.restype = None
        lib.llama_memory_seq_keep.argtypes = [c_void_p, c_int32]
        lib.llama_memory_seq_keep.restype = None
        lib.llama_memory_seq_rm.argtypes = [c_void_p, c_int32, c_int32, c_int32]
        lib.llama_memory_seq_rm.restype = c_bool
        lib.llama_state_get_size.argtypes = [c_void_p]
        lib.llama_state_get_size.restype = c_size_t
        lib.llama_state_get_data.argtypes = [c_void_p, POINTER(c_ubyte), c_size_t]
        lib.llama_state_get_data.restype = c_size_t
        lib.llama_state_set_data.argtypes = [c_void_p, POINTER(c_ubyte), c_size_t]
        lib.llama_state_set_data.restype = c_size_t
        lib.llama_state_seq_get_size.argtypes = [c_void_p, c_int32]
        lib.llama_state_seq_get_size.restype = c_size_t
        lib.llama_state_seq_get_data.argtypes = [c_void_p, POINTER(c_ubyte), c_size_t, c_int32]
        lib.llama_state_seq_get_data.restype = c_size_t
        lib.llama_state_seq_set_data.argtypes = [c_void_p, POINTER(c_ubyte), c_size_t, c_int32]
        lib.llama_state_seq_set_data.restype = c_size_t
        lib.llama_get_memory.argtypes = [c_void_p]
        lib.llama_get_memory.restype = c_void_p
        if hasattr(lib, "llama_memory_seq_pos_min"):
            lib.llama_memory_seq_pos_min.argtypes = [c_void_p, llama_seq_id]
            lib.llama_memory_seq_pos_min.restype = llama_pos
        if hasattr(lib, "llama_memory_seq_pos_max"):
            lib.llama_memory_seq_pos_max.argtypes = [c_void_p, llama_seq_id]
            lib.llama_memory_seq_pos_max.restype = llama_pos

        lib.llama_model_get_vocab.argtypes = [c_void_p]
        lib.llama_model_get_vocab.restype = c_void_p
        lib.llama_vocab_n_tokens.argtypes = [c_void_p]
        lib.llama_vocab_n_tokens.restype = c_int32
        lib.llama_model_chat_template.argtypes = [c_void_p, c_char_p]
        lib.llama_model_chat_template.restype = c_char_p
        lib.llama_chat_apply_template.argtypes = [
            c_char_p,
            POINTER(LlamaChatMessage),
            c_size_t,
            c_bool,
            POINTER(c_char),
            c_int32,
        ]
        lib.llama_chat_apply_template.restype = c_int32
        lib.llama_tokenize.argtypes = [c_void_p, c_char_p, c_int32, POINTER(llama_token), c_int32, c_bool, c_bool]
        lib.llama_tokenize.restype = c_int32
        lib.llama_token_to_piece.argtypes = [c_void_p, llama_token, POINTER(c_char), c_int32, c_int32, c_bool]
        lib.llama_token_to_piece.restype = c_int32
        lib.llama_vocab_is_eog.argtypes = [c_void_p, llama_token]
        lib.llama_vocab_is_eog.restype = c_bool

        lib.llama_batch_get_one.argtypes = [POINTER(llama_token), c_int32]
        lib.llama_batch_get_one.restype = LlamaBatch
        lib.llama_batch_init.argtypes = [c_int32, c_int32, c_int32]
        lib.llama_batch_init.restype = LlamaBatch
        lib.llama_batch_free.argtypes = [LlamaBatch]
        lib.llama_batch_free.restype = None
        lib.llama_decode.argtypes = [c_void_p, LlamaBatch]
        lib.llama_decode.restype = c_int32
        lib.llama_synchronize.argtypes = [c_void_p]
        lib.llama_synchronize.restype = None
        lib.llama_time_us.argtypes = []
        lib.llama_time_us.restype = ctypes.c_int64
        lib.llama_get_logits_ith.argtypes = [c_void_p, c_int32]
        lib.llama_get_logits_ith.restype = POINTER(c_float)

        lib.llama_sampler_chain_init.argtypes = [LlamaSamplerChainParams]
        lib.llama_sampler_chain_init.restype = c_void_p
        lib.llama_sampler_chain_add.argtypes = [c_void_p, c_void_p]
        lib.llama_sampler_chain_add.restype = None
        lib.llama_sampler_init_greedy.argtypes = []
        lib.llama_sampler_init_greedy.restype = c_void_p
        lib.llama_sampler_sample.argtypes = [c_void_p, c_void_p, c_int32]
        lib.llama_sampler_sample.restype = llama_token
        lib.llama_sampler_accept.argtypes = [c_void_p, llama_token]
        lib.llama_sampler_accept.restype = None
        lib.llama_sampler_reset.argtypes = [c_void_p]
        lib.llama_sampler_reset.restype = None
        lib.llama_sampler_free.argtypes = [c_void_p]
        lib.llama_sampler_free.restype = None


class MtmdLibrary:
    def __init__(self, build_bin: Path, bridge_path: Path | None = None) -> None:
        resolved_bridge = bridge_path or build_bin / runtime_library_filename("orbit-mtmd-bridge")
        self.build_identity = validate_mtmd_bridge_artifact(build_bin, resolved_bridge)
        flags = native_cdll_flags()
        self.lib = load_native_cdll(
            resolved_bridge,
            mode=flags,
        )
        self._configure_api()
        if self.lib.orbit_mtmd_bridge_api_version() != 1:
            raise RuntimeError("unsupported Orbit mtmd bridge API")
        if not self.lib.orbit_mtmd_bridge_abi_supported():
            error = self.last_error() or "unsupported mtmd ABI profile"
            raise RuntimeError(error)
        raw_manifest = self.lib.orbit_mtmd_bridge_manifest_json()
        if not raw_manifest:
            raise RuntimeError("missing Orbit mtmd bridge ABI manifest")
        try:
            self.manifest = json.loads(raw_manifest)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("invalid Orbit mtmd bridge ABI manifest") from exc
        for manifest_key, identity_key in (
            ("upstream_commit", "upstream_commit"),
            ("upstream_tag", "upstream_tag"),
            ("source_tree_hash", "source_tree_sha256"),
            ("patchset_hash", "patchset_sha256"),
        ):
            if self.manifest.get(manifest_key) != self.build_identity.get(identity_key):
                raise RuntimeError("Orbit mtmd bridge provenance mismatch")
        verify_core_abi_layouts(self.manifest)

    def _configure_api(self) -> None:
        lib = self.lib
        lib.orbit_mtmd_bridge_api_version.argtypes = []
        lib.orbit_mtmd_bridge_api_version.restype = c_uint32
        lib.orbit_mtmd_bridge_abi_supported.argtypes = []
        lib.orbit_mtmd_bridge_abi_supported.restype = c_bool
        lib.orbit_mtmd_bridge_last_error.argtypes = []
        lib.orbit_mtmd_bridge_last_error.restype = c_char_p
        lib.orbit_mtmd_bridge_manifest_json.argtypes = []
        lib.orbit_mtmd_bridge_manifest_json.restype = c_char_p
        lib.orbit_mtmd_default_marker.argtypes = []
        lib.orbit_mtmd_default_marker.restype = c_char_p
        lib.orbit_mtmd_context_create.argtypes = [
            c_char_p,
            c_void_p,
            c_bool,
            c_bool,
            c_int32,
            c_char_p,
        ]
        lib.orbit_mtmd_context_create.restype = c_void_p
        lib.orbit_mtmd_context_free.argtypes = [c_void_p]
        lib.orbit_mtmd_context_free.restype = None
        lib.orbit_mtmd_support_vision.argtypes = [c_void_p]
        lib.orbit_mtmd_support_vision.restype = c_bool
        lib.orbit_mtmd_support_audio.argtypes = [c_void_p]
        lib.orbit_mtmd_support_audio.restype = c_bool
        lib.orbit_mtmd_get_cap_from_file.argtypes = [
            c_char_p,
            POINTER(c_bool),
            POINTER(c_bool),
        ]
        lib.orbit_mtmd_get_cap_from_file.restype = c_bool
        lib.orbit_mtmd_bitmap_init_from_buf.argtypes = [
            c_void_p,
            POINTER(c_ubyte),
            c_size_t,
            c_bool,
        ]
        lib.orbit_mtmd_bitmap_init_from_buf.restype = c_void_p
        lib.orbit_mtmd_bitmap_free.argtypes = [c_void_p]
        lib.orbit_mtmd_bitmap_free.restype = None
        lib.orbit_mtmd_chunks_create.argtypes = []
        lib.orbit_mtmd_chunks_create.restype = c_void_p
        lib.orbit_mtmd_chunks_free.argtypes = [c_void_p]
        lib.orbit_mtmd_chunks_free.restype = None
        lib.orbit_mtmd_chunks_size.argtypes = [c_void_p]
        lib.orbit_mtmd_chunks_size.restype = c_size_t
        lib.orbit_mtmd_chunks_get.argtypes = [c_void_p, c_size_t]
        lib.orbit_mtmd_chunks_get.restype = c_void_p
        lib.orbit_mtmd_chunk_token_count.argtypes = [c_void_p]
        lib.orbit_mtmd_chunk_token_count.restype = c_size_t
        lib.orbit_mtmd_chunks_token_count.argtypes = [c_void_p]
        lib.orbit_mtmd_chunks_token_count.restype = c_size_t
        lib.orbit_mtmd_tokenize.argtypes = [
            c_void_p,
            c_void_p,
            c_char_p,
            c_size_t,
            c_bool,
            c_bool,
            POINTER(c_void_p),
            c_size_t,
        ]
        lib.orbit_mtmd_tokenize.restype = c_int32
        lib.orbit_mtmd_eval_chunk.argtypes = [
            c_void_p,
            c_void_p,
            c_void_p,
            llama_pos,
            llama_seq_id,
            c_int32,
            c_bool,
            POINTER(llama_pos),
        ]
        lib.orbit_mtmd_eval_chunk.restype = c_int32

    def last_error(self) -> str | None:
        value = self.lib.orbit_mtmd_bridge_last_error()
        if not value:
            return None
        return value.decode("utf-8", errors="replace") or None


def verify_core_abi_layouts(manifest: object) -> None:
    if not isinstance(manifest, dict):
        raise RuntimeError("invalid Orbit mtmd bridge ABI manifest")
    structures = {
        "llama_batch": (LlamaBatch, tuple(name for name, _ctype in LlamaBatch._fields_)),
        "llama_model_params": (
            LlamaModelParams,
            (
                "n_gpu_layers",
                "progress_callback",
                "progress_callback_user_data",
                "kv_overrides",
                "vocab_only",
                "no_alloc",
            ),
        ),
        "llama_context_params": (
            LlamaContextParams,
            (
                "n_ctx",
                "n_outputs_max",
                "n_threads",
                "flash_attn_type",
                "defrag_thold",
                "cb_eval",
                "type_k",
                "abort_callback",
                "embeddings",
                "samplers",
                "ctx_other",
            ),
        ),
        "llama_sampler_chain_params": (LlamaSamplerChainParams, ("no_perf",)),
        "llama_chat_message": (LlamaChatMessage, ("role", "content")),
    }
    for name, (structure, fields) in structures.items():
        actual = manifest.get(name)
        if not isinstance(actual, dict):
            raise RuntimeError(f"missing native ABI layout: {name}")
        expected = {
            "size": ctypes.sizeof(structure),
            "align": ctypes.alignment(structure),
            **{field: getattr(structure, field).offset for field in fields},
        }
        if any(actual.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"native ABI layout mismatch: {name}")
