from __future__ import annotations

import re
import shlex
from typing import Any, Callable

from .binary_guardrails import ARCHIVE_CONTAINER_EXTENSIONS
from .intent_signals import LEARNING_MARKERS, contains_phrase, looks_like_discursive_security_text
from .intent_router import is_binary_or_pdf_analysis_intent
from .message_ops import (
    has_recent_tool_result,
    likely_binary_candidates_from_recent_listing,
    successful_bash_results_in_current_turn,
)


MAX_BINARY_SEED_CANDIDATES = 3
SCRIPT_STATIC_SAMPLE_EXTENSIONS = (
    ".js",
    ".jse",
    ".vbs",
    ".vbe",
    ".wsf",
    ".hta",
    ".ps1",
    ".bat",
    ".cmd",
    ".sh",
    ".py",
    ".php",
    ".pl",
    ".rb",
    ".lua",
    ".html",
    ".htm",
    ".mhtml",
    ".svg",
    ".xml",
    ".xsl",
)
STATIC_SAMPLE_EXTENSIONS_RE = (
    "apk|zip|jar|aar|ipa|dex|so|dll|exe|bin|pdf|"
    "doc|docm|docx|xls|xlsm|xlsx|ppt|pptm|pptx|rtf|"
    "js|jse|vbs|vbe|wsf|hta|ps1|bat|cmd|sh|py|php|pl|rb|lua|"
    "html|htm|mhtml|svg|xml|xsl"
)
EXPLICIT_BINARY_PATH_RE = re.compile(
    rf"""(?:"(?P<double>[^"]+\.(?:{STATIC_SAMPLE_EXTENSIONS_RE}))"|'(?P<single>[^']+\.(?:{STATIC_SAMPLE_EXTENSIONS_RE}))'|(?P<bare>[^\s"'`]+\.(?:{STATIC_SAMPLE_EXTENSIONS_RE})))""",
    re.IGNORECASE,
)


def seed_binary_discovery_impl(
    *,
    user_input: str = "",
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
    run_guardrail_tool: Callable[..., dict[str, Any]],
    binary_listing_guidance: Callable[[list[dict[str, Any]]], str],
    has_pdf_text_extract_in_current_turn: Callable[[list[dict[str, Any]]], bool],
) -> None:
    if not is_binary_or_pdf_analysis_intent(route.intent):
        return
    if not _is_operational_static_analysis_request(user_input):
        return
    if has_pdf_text_extract_in_current_turn(messages):
        return
    explicit_path = _extract_explicit_binary_path(user_input)
    if explicit_path and not has_recent_tool_result(messages, "bash"):
        _seed_binary_path_probe(
            path=explicit_path,
            deep_static=_asks_for_static_reverse_analysis(user_input),
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            run_guardrail_tool=run_guardrail_tool,
        )
        return
    if has_recent_tool_result(messages, "list_files"):
        return
    all_samples = _asks_for_multiple_binary_samples(user_input)
    discovery_path = _extract_explicit_sample_directory(user_input) or "."
    run_guardrail_tool(
        name="list_files",
        arguments={"path": discovery_path, "recursive": all_samples, "max_entries": 80 if all_samples else 12},
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        emit_route=True,
    )
    messages.append({"role": "system", "content": binary_listing_guidance(messages)})
    candidates = likely_binary_candidates_from_recent_listing(
        messages,
        limit=MAX_BINARY_SEED_CANDIDATES if all_samples else 1,
    )
    if not candidates or has_recent_tool_result(messages, "bash"):
        return
    for candidate in candidates:
        _seed_binary_path_probe(
            path=candidate,
            deep_static=_asks_for_static_reverse_analysis(user_input),
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            run_guardrail_tool=run_guardrail_tool,
        )


def local_static_sample_evidence_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    if not is_binary_or_pdf_analysis_intent(intent):
        return None
    lowered = user_input.lower()
    evidence_hints = ("initial evidence", "hash", "hashes", "file type", "md5", "sha1", "sha256", "evidenze")
    if not any(hint in lowered for hint in evidence_hints):
        return None
    if not _asks_for_only_initial_static_evidence(lowered):
        return None
    samples = _static_sample_evidence_from_bash(messages)
    if not samples:
        return None
    lines = [f"Initial static evidence collected for {len(samples)} sample(s):"]
    for path, evidence in samples.items():
        lines.append(f"- `{path}`")
        file_type = evidence.get("file")
        if file_type:
            lines.append(f"  - type: {file_type}")
        for key in ("md5", "sha1", "sha256"):
            value = evidence.get(key)
            if value:
                lines.append(f"  - {key}: {value}")
        if evidence.get("container_listing"):
            lines.append("  - container listing: collected with bounded `unzip -l | head`")
    lines.append("Next steps: inspect manifest/resources, code, scripts, assets, native libraries, configs, URLs, and suspicious strings according to the active static-analysis skill.")
    return "\n".join(lines)


def local_static_reverse_inspection_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    if not is_binary_or_pdf_analysis_intent(intent):
        return None
    lowered = user_input.lower()
    if not _asks_for_static_reverse_analysis(user_input):
        return None
    if any(hint in lowered for hint in ("report.md", "agents.md", "case report", "update report", "aggiorna report")):
        return None
    samples = _static_sample_evidence_from_bash(messages)
    if not samples:
        return None
    inspection = _static_reverse_inspection_from_bash(messages)
    if not inspection:
        return None
    lines = [f"Bounded static reverse inspection completed for {len(samples)} sample(s)."]
    for path, evidence in samples.items():
        lines.append(f"- `{path}`")
        file_type = evidence.get("file")
        if file_type:
            lines.append(f"  - type: {file_type}")
        sha256 = evidence.get("sha256")
        if sha256:
            lines.append(f"  - sha256: {sha256}")
        details = inspection.get(path, {})
        archive_focus = details.get("archive_focus")
        if archive_focus:
            lines.append("  - APK/container focus:")
            lines.extend(f"    - {line}" for line in archive_focus[:8])
        script_preview = details.get("script_preview")
        if script_preview:
            lines.append("  - script/code preview:")
            lines.extend(f"    - {line}" for line in script_preview[:8])
        indicators = details.get("indicators")
        if indicators:
            lines.append("  - suspicious string/config/URL hits:")
            lines.extend(f"    - {line}" for line in indicators[:10])
        else:
            lines.append("  - suspicious string/config/URL hits: none found in the bounded initial scan")
    lines.append("Limits: this is bounded static inspection, not full decompilation or dynamic execution.")
    lines.append("Next deeper steps: decompile/disassemble the relevant stage, inspect decoded configs, extract embedded payloads, and update the case report if requested.")
    return "\n".join(lines)


def _extract_explicit_binary_path(user_input: str) -> str | None:
    if not isinstance(user_input, str) or not user_input.strip():
        return None
    for match in EXPLICIT_BINARY_PATH_RE.finditer(user_input):
        candidate = (match.group("double") or match.group("single") or match.group("bare") or "").strip()
        if candidate:
            return candidate
    return None


def _asks_for_multiple_binary_samples(user_input: str) -> bool:
    lowered = user_input.lower()
    all_terms = ("all", "every", "each", "tutti", "tutte", "ogni")
    sample_terms = ("sample", "samples", "malware", "case", "campione", "campioni", "caso")
    return any(term in lowered for term in all_terms) and any(term in lowered for term in sample_terms)


def _asks_for_static_reverse_analysis(user_input: str) -> bool:
    lowered = user_input.lower()
    return bool(
        re.search(
            r"\b(static analysis|static reverse|reverse engineering|malware analysis|analyze|analyse|analisi statica|reverse|analizza|triage|inspect|inspection|ispeziona|ispezionare|try|prova)\b",
            lowered,
        )
    )


def _is_operational_static_analysis_request(user_input: str) -> bool:
    lowered = user_input.lower()
    if _looks_like_discursive_static_statement(lowered):
        return False
    evidence_hints = ("initial evidence", "hash", "hashes", "file type", "md5", "sha1", "sha256", "evidenze")
    return _asks_for_static_reverse_analysis(user_input) or any(hint in lowered for hint in evidence_hints)


def _looks_like_discursive_static_statement(lowered_input: str) -> bool:
    if contains_phrase(lowered_input, LEARNING_MARKERS):
        return True
    return looks_like_discursive_security_text(lowered_input)


def _extract_explicit_sample_directory(user_input: str) -> str | None:
    lowered = user_input.lower()
    if re.search(r"\b(?:under|inside|in|directory|folder|dir|case)\s+[`'\"]?malware[`'\"]?\b", lowered):
        return "malware"
    if re.search(r"\bmalware\s+(?:directory|folder|dir|case)\b", lowered):
        return "malware"
    return None


def _seed_binary_path_probe(
    *,
    path: str,
    deep_static: bool = False,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
    run_guardrail_tool: Callable[..., dict[str, Any]],
) -> None:
    quoted = shlex.quote(path)
    commands = [
        f"file {quoted}",
        f"md5sum {quoted}",
        f"sha1sum {quoted}",
        f"sha256sum {quoted}",
    ]
    if any(path.lower().endswith(ext) for ext in ARCHIVE_CONTAINER_EXTENSIONS):
        commands.append(f"unzip -l {quoted} | head -n 20")
    if deep_static:
        commands.extend(_bounded_static_inspection_commands(path, quoted))
    for command in commands:
        run_guardrail_tool(
            name="bash",
            arguments={"command": command},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
        )
    messages.append(
        {
            "role": "system",
            "content": (
                "Initial static evidence has been collected for this sample. "
                "If the active skill or user request asks for malware/static analysis, do not stop at hashes. "
                "Continue with bounded reverse-engineering steps appropriate to the file type: manifests, resources, strings, "
                "embedded stages, scripts, URLs, configs, native libraries, and report/memory updates when requested by the skill. "
                "Do not execute the sample."
            ),
        }
    )


def _bounded_static_inspection_commands(path: str, quoted: str) -> list[str]:
    lowered = path.lower()
    if lowered.endswith(".apk"):
        return [
            (
                f"unzip -l {quoted} | grep -Ei "
                "'AndroidManifest|classes[0-9]*\\.dex|\\.so|assets/|res/raw|META-INF' | head -n 80"
            ),
            (
                f"strings {quoted} | grep -Ei "
                "'https?://|telegram|api|token|secret|password|firebase|bnl|pagopa|host|endpoint|url' | head -n 80"
            ),
        ]
    if any(lowered.endswith(ext) for ext in SCRIPT_STATIC_SAMPLE_EXTENSIONS):
        return [
            f"head -n 160 {quoted}",
            (
                f"grep -Ein "
                "'https?://|download|invoke|iex|powershell|cmd\\.exe|wscript|cscript|eval|base64|fromcharcode|token|secret|password' "
                f"{quoted} | head -n 80"
            ),
        ]
    if any(lowered.endswith(ext) for ext in (".exe", ".dll", ".so", ".dex", ".bin", ".jar", ".pdf")):
        return [
            (
                f"strings {quoted} | grep -Ei "
                "'https?://|[[:alnum:]._-]+\\.[a-z]{2,}|cmd\\.exe|powershell|/bin/sh|token|secret|password|mutex|user-agent' | head -n 80"
            )
        ]
    return []


def _static_sample_evidence_from_bash(messages: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    samples: dict[str, dict[str, str]] = {}
    for result in successful_bash_results_in_current_turn(messages):
        command = result.get("command")
        stdout = result.get("stdout")
        if not isinstance(command, str) or not isinstance(stdout, str):
            continue
        parsed = _parse_static_evidence_command(command)
        if parsed is None:
            continue
        kind, path = parsed
        evidence = samples.setdefault(path, {})
        if kind == "file":
            evidence["file"] = _clean_file_command_output(stdout, path)
        elif kind in {"md5", "sha1", "sha256"}:
            digest = stdout.strip().split(maxsplit=1)[0] if stdout.strip() else ""
            if digest:
                evidence[kind] = digest
        elif kind == "container_listing":
            evidence["container_listing"] = "yes"
    return {path: evidence for path, evidence in samples.items() if evidence}


def _parse_static_evidence_command(command: str) -> tuple[str, str] | None:
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if not parts:
        return None
    name = parts[0]
    if name == "file" and len(parts) >= 2:
        return "file", parts[1]
    if name in {"md5sum", "sha1sum", "sha256sum"} and len(parts) >= 2:
        return name.removesuffix("sum"), parts[1]
    if name == "unzip" and len(parts) >= 3 and parts[1] == "-l":
        return "container_listing", parts[2]
    return None


def _clean_file_command_output(stdout: str, path: str) -> str:
    text = " ".join(stdout.strip().split())
    prefix = f"{path}:"
    if text.startswith(prefix):
        text = text[len(prefix) :].strip()
    return text[:240]


def _static_reverse_inspection_from_bash(messages: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    by_path: dict[str, dict[str, list[str]]] = {}
    for result in successful_bash_results_in_current_turn(messages):
        command = result.get("command")
        stdout = result.get("stdout")
        if not isinstance(command, str) or not isinstance(stdout, str) or not stdout.strip():
            continue
        parsed = _parse_static_inspection_command(command)
        if parsed is None:
            continue
        kind, path = parsed
        lines = [" ".join(line.split())[:240] for line in stdout.splitlines() if line.strip()]
        if not lines:
            continue
        by_path.setdefault(path, {}).setdefault(kind, []).extend(lines[:20])
    return by_path


def _parse_static_inspection_command(command: str) -> tuple[str, str] | None:
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if not parts:
        return None
    if parts[0] == "unzip" and len(parts) >= 3 and parts[1] == "-l" and "AndroidManifest" in command:
        return "archive_focus", parts[2]
    if parts[0] == "strings" and len(parts) >= 2 and "grep" in parts:
        return "indicators", parts[1]
    if parts[0] == "head" and len(parts) >= 4:
        return "script_preview", parts[-1]
    if parts[0] == "grep" and len(parts) >= 3:
        return "indicators", parts[-1]
    return None


def _asks_for_only_initial_static_evidence(lowered_input: str) -> bool:
    only_hints = (
        "only initial evidence",
        "initial evidence only",
        "hashes only",
        "hash only",
        "file type only",
        "solo evidenze iniziali",
        "soltanto evidenze iniziali",
        "solo hash",
        "soltanto hash",
    )
    if any(hint in lowered_input for hint in only_hints):
        return True
    analysis_hints = (
        "static analysis",
        "reverse",
        "malware analysis",
        "analyze",
        "analyse",
        "analizza",
        "analisi statica",
        "reverse engineering",
        "inspect",
        "inspection",
        "ispeziona",
        "ispezionare",
    )
    return not any(hint in lowered_input for hint in analysis_hints)
