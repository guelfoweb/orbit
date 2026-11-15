"""Explicit, process-local admission for a hash-pinned qualification run.

Nothing here registers a model or marks it production-verified. The normal
server has no CLI/environment/config switch for this admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .gguf_split import read_gguf_header, parse_split_name
from .model_contract import ReasoningBehavior
from .model_profiles import NativeModelProfile


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("research identity requires lowercase SHA-256 digests")
    return value


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate research manifest key")
        result[key] = value
    return result


def _stat_identity(stat) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True)
class ResearchAdmission:
    artifact_sha256: str
    artifact_size: int
    template_sha256: str
    model_name: str
    architecture: str
    metadata: tuple[tuple[str, str], ...]
    reasoning_behavior: ReasoningBehavior
    separated_reasoning: bool

    @classmethod
    def read(cls, path: Path) -> "ResearchAdmission":
        with path.open("rb") as handle:
            raw = handle.read(16385)
        if len(raw) > 16384:
            raise ValueError("research manifest exceeds 16 KiB")
        value = json.loads(raw, object_pairs_hook=_unique)
        expected = {"version", "artifact_sha256", "artifact_size", "template_sha256",
                    "model_name", "architecture", "metadata", "reasoning_behavior",
                    "separated_reasoning"}
        if not isinstance(value, dict) or set(value) != expected or type(value["version"]) is not int or value["version"] != 1:
            raise ValueError("unsupported research manifest shape/version")
        metadata = value["metadata"]
        if not isinstance(metadata, dict) or not metadata or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in metadata.items()
        ):
            raise ValueError("research metadata must contain exact string pairs")
        return cls(value["artifact_sha256"], value["artifact_size"], value["template_sha256"],
                   value["model_name"], value["architecture"], tuple(sorted(metadata.items())),
                   ReasoningBehavior(value["reasoning_behavior"]), value["separated_reasoning"])

    def __post_init__(self) -> None:
        _digest(self.artifact_sha256)
        _digest(self.template_sha256)
        if type(self.artifact_size) is not int or self.artifact_size <= 0:
            raise ValueError("research artifact size must be positive")
        if any(not isinstance(s, str) or not s.strip() for s in (self.model_name, self.architecture)):
            raise ValueError("research model name and architecture are required")
        if not isinstance(self.reasoning_behavior, ReasoningBehavior) or type(self.separated_reasoning) is not bool:
            raise ValueError("invalid research reasoning contract")
        if (not isinstance(self.metadata, tuple) or not 1 <= len(self.metadata) <= 64
                or any(not isinstance(pair, tuple) or len(pair) != 2
                       or any(not isinstance(s, str) or not s for s in pair)
                       for pair in self.metadata)):
            raise ValueError("research metadata must contain bounded exact string pairs")
        metadata = dict(self.metadata)
        required = {"general.name": self.model_name, "general.architecture": self.architecture}
        if len(metadata) != len(self.metadata) or any(metadata.get(k) != v for k, v in required.items()):
            raise ValueError("research metadata must bind model name and architecture")

    @property
    def profile_id(self) -> str:
        return "research-" + self.artifact_sha256

    def attest(self, path: Path) -> tuple[int, int, int, int, int]:
        if parse_split_name(path.name) is not None:
            raise ValueError("research admission currently requires a single-file GGUF")
        with path.open("rb") as handle:
            import os
            before = _stat_identity(os.fstat(handle.fileno()))
            if before[2] != self.artifact_size:
                raise ValueError("research artifact size mismatch")
            digest = hashlib.sha256()
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
            after = _stat_identity(os.fstat(handle.fileno()))
        if before != after or _stat_identity(path.stat()) != after:
            raise ValueError("research artifact changed during attestation")
        if digest.hexdigest() != self.artifact_sha256:
            raise ValueError("research artifact SHA-256 mismatch")
        keys = frozenset(dict(self.metadata)) | {"tokenizer.chat_template"}
        header = read_gguf_header(path, keys=keys)
        if any(k not in header.kv or str(header.kv[k]) != v for k, v in self.metadata):
            raise ValueError("research GGUF metadata mismatch")
        template = header.kv.get("tokenizer.chat_template")
        if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != self.template_sha256:
            raise ValueError("research GGUF template mismatch")
        if _stat_identity(path.stat()) != after:
            raise ValueError("research artifact changed during metadata inspection")
        return after

    def profile(self, path: Path, template: str, attestation) -> NativeModelProfile:
        if attestation is None or _stat_identity(path.stat()) != attestation:
            raise ValueError("research artifact changed before native admission")
        if hashlib.sha256(template.encode()).hexdigest() != self.template_sha256:
            raise ValueError("native research template mismatch")
        return NativeModelProfile(
            profile_id=self.profile_id, family=self.architecture, model_name=self.model_name,
            architecture=self.architecture, renderer="llama.cpp-jinja",
            reasoning_protocol="research-jinja", tool_call_protocol="research-jinja",
            history_serialization="preserve", verified=False,
            failure_reason="research_only_not_production_qualified",
            template_source="hash-pinned-research", template_sha256=self.template_sha256,
            thinking_supported=self.reasoning_behavior != ReasoningBehavior.NONE,
            mtp_supported=False, gemma_prefix_reuse_supported=False,
            supported_capabilities=frozenset(), reasoning_behavior=self.reasoning_behavior,
            separated_reasoning=self.separated_reasoning,
        )
