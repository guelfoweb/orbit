from __future__ import annotations

from typing import Any, Callable

from .intent_router import is_binary_or_pdf_analysis_intent
from .turn_policy import TurnPolicyState

LIKELY_BINARY_EXTENSIONS = {".pdf", ".apk", ".dex", ".so", ".dll", ".exe", ".bin", ".dylib", ".a", ".o"}
ARCHIVE_CONTAINER_EXTENSIONS = {".apk", ".zip", ".jar", ".aar", ".ipa"}
LOW_PRIORITY_TEXT_NAMES = {
    "readme.md",
    "agents.md",
    "defects.md",
    ".gitignore",
    "pyproject.toml",
    "package.json",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    ".env",
    ".gitmodules",
    "workspace.xml",
    "settings.json",
    "tasks.json",
    "config",
}


def binary_analysis_guard_prompt(
    *,
    intent: str | None,
    tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    parse_arguments: Any,
    normalize_relative_path: Callable[[str], str],
    recent_archive_container_for_member: Callable[[list[dict[str, Any]], str], str | None],
    was_listed_by_list_files: Callable[[list[dict[str, Any]], str], bool],
) -> str | None:
    if not is_binary_or_pdf_analysis_intent(intent) or not tool_calls:
        return None
    if len(tool_calls) != 1:
        return None
    function = tool_calls[0].get("function", {}) or {}
    if function.get("name") != "read_file":
        return None
    arguments = parse_arguments(function.get("arguments"))
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        return (
            "For binary or PDF analysis, do not guess a file name. "
            "First call list_files on the relevant subtree to discover a real candidate path, "
            "then inspect that candidate with a binary-aware command such as file, strings, or pdftotext."
        )
    normalized = normalize_relative_path(path)
    name = normalized.rsplit("/", 1)[-1].lower()
    if name in LOW_PRIORITY_TEXT_NAMES or name.startswith("."):
        return (
            "For binary or PDF analysis, do not start from hidden files, config files, or metadata documents. "
            "First call list_files to discover the actual binary or PDF candidate, then inspect that candidate with bash using file, strings, pdftotext, or another bounded binary-aware command."
        )
    if any(normalized.lower().endswith(ext) for ext in ARCHIVE_CONTAINER_EXTENSIONS):
        return (
            "This looks like an archive/container format such as APK, ZIP, JAR, AAR, or IPA. "
            "Do not use read_file on the raw container. "
            "Inspect it first with a bounded archive-aware command such as unzip -l, zipinfo -1, or another listing command, "
            "then choose specific embedded files like AndroidManifest.xml, classes.dex, resources, or native libraries for deeper analysis."
        )
    archive_container = recent_archive_container_for_member(messages, normalized)
    if archive_container is not None:
        return (
            f"{normalized} looks like a file inside the archive/container {archive_container}, not a real file in the workdir. "
            "Do not use read_file on an embedded member that has not been extracted. "
            "Use an archive-aware command to inspect or extract that member first, for example unzip -p on the container, "
            "or extract the member to a real path and then analyze that extracted file."
        )
    if was_listed_by_list_files(messages, normalized):
        return None
    if any(normalized.lower().endswith(ext) for ext in LIKELY_BINARY_EXTENSIONS):
        return None
    return (
        "For binary or PDF analysis, do not guess a read_file path. "
        "First call list_files on the relevant subtree to discover a real candidate path, "
        "then inspect that candidate with bash using file, strings, pdftotext, or another bounded binary-aware command. "
        "Use read_file only for extracted text or manifest-like text files."
    )


def binary_tool_strategy_prompt(
    *,
    intent: str | None,
    tool_calls: list[dict[str, Any]],
    parse_arguments: Any,
) -> str | None:
    if not is_binary_or_pdf_analysis_intent(intent) or len(tool_calls) != 1:
        return None
    function = tool_calls[0].get("function", {}) or {}
    if function.get("name") != "bash":
        return None
    arguments = parse_arguments(function.get("arguments"))
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    lowered = command.lower()
    if "strings " not in lowered:
        return None
    if not any(ext in lowered for ext in ARCHIVE_CONTAINER_EXTENSIONS):
        return None
    return (
        "Do not start archive/container analysis with strings on the whole APK/ZIP/JAR/AAR/IPA. "
        "First inspect the container structure with unzip -l, zipinfo -1, or another bounded archive listing command. "
        "Then target specific embedded files such as AndroidManifest.xml, classes.dex, resources, or native libraries."
    )


def binary_listing_retry_prompt(
    *,
    intent: str | None,
    tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    parse_arguments: Any,
    normalize_relative_path: Callable[[str], str],
    has_recent_tool_result: Callable[[list[dict[str, Any]], str], bool],
    likely_binary_candidates_from_recent_listing: Callable[[list[dict[str, Any]]], list[str]],
) -> str | None:
    if not is_binary_or_pdf_analysis_intent(intent) or len(tool_calls) != 1:
        return None
    function = tool_calls[0].get("function", {}) or {}
    if function.get("name") != "list_files":
        return None
    arguments = parse_arguments(function.get("arguments"))
    path = arguments.get("path")
    if not isinstance(path, str) or normalize_relative_path(path) not in {"", "."}:
        return None
    if not has_recent_tool_result(messages, "list_files"):
        return None
    return binary_listing_guidance(messages, likely_binary_candidates_from_recent_listing=likely_binary_candidates_from_recent_listing)


def binary_text_reply_handling(
    *,
    intent: str | None,
    content: str,
    policy_state: TurnPolicyState,
) -> tuple[str, str] | None:
    if not is_binary_or_pdf_analysis_intent(intent):
        return None
    if policy_state.tool_history:
        return None
    lowered = content.lower()
    suspicious = (
        "strings ./" in lowered
        or "file ./" in lowered
        or "pdftotext ./" in lowered
        or "not found" in lowered
        or "non è stato trovato" in lowered
    )
    if not suspicious:
        return None
    if policy_state.synthesis_retries >= 1:
        return (
            "final",
            "I could not identify a real binary or PDF candidate because the model kept guessing a missing path. "
            "List files in the target subtree first or specify the filename explicitly, then inspect that discovered path with a bounded binary-aware command.",
        )
    policy_state.synthesis_retries += 1
    return (
        "retry",
        "Do not answer with a guessed binary path or a suggested command on a missing file. "
        "First call list_files on the relevant subtree to discover a real candidate binary or PDF. "
        "Then inspect that discovered path with one bounded binary-aware command such as file, strings, or pdftotext."
    )


def binary_seeded_summary(
    messages: list[dict[str, Any]],
    *,
    likely_binary_candidates_from_recent_listing: Callable[[list[dict[str, Any]], int], list[str]],
) -> str | None:
    candidates = likely_binary_candidates_from_recent_listing(messages, limit=1)
    if not candidates:
        return None
    return binary_seeded_summary_for_candidate(candidates[0], normalize_relative_path=lambda value: value)


def binary_seeded_summary_for_candidate(
    candidate: str,
    *,
    normalize_relative_path: Callable[[str], str],
) -> str:
    normalized = normalize_relative_path(candidate)
    lowered = normalized.lower()
    if lowered.endswith(".apk"):
        return (
            f"Initial APK triage completed for `{normalized}`. "
            "The runtime collected a real candidate path, a file-type probe, and an archive listing. "
            "Next bounded steps should inspect AndroidManifest.xml, classes.dex, assets, and native libraries."
        )
    if lowered.endswith(".pdf"):
        return (
            f"Initial PDF triage completed for `{normalized}`. "
            "The runtime collected a real candidate path and a file-type probe. "
            "Next bounded steps should extract text with pdftotext, inspect strings or metadata, and check for embedded attachments."
        )
    if any(lowered.endswith(ext) for ext in ARCHIVE_CONTAINER_EXTENSIONS):
        return (
            f"Initial archive triage completed for `{normalized}`. "
            "The runtime collected a real candidate path, a file-type probe, and an archive listing. "
            "Next bounded steps should inspect the container members with unzip -l, zipinfo -1, or another bounded archive-aware command."
        )
    return (
        f"Initial binary triage completed for `{normalized}`. "
        "The runtime collected a real candidate path and a file-type probe. "
        "Next bounded steps should inspect the file type and continue with bounded binary-aware commands."
    )


def binary_listing_guidance(
    messages: list[dict[str, Any]],
    *,
    likely_binary_candidates_from_recent_listing: Callable[[list[dict[str, Any]]], list[str]],
) -> str:
    candidates = likely_binary_candidates_from_recent_listing(messages)
    if candidates:
        preview = ", ".join(candidates)
        return (
            "A top-level file listing has already been collected for this binary/PDF task. "
            f"Likely candidate paths from that listing: {preview}. "
            "Do not call list_files again for the same location. "
            "Choose one real candidate path from the listing, then inspect it with one bounded binary-aware or archive-aware command. "
            "Do not guess placeholder names."
        )
    return (
        "A top-level file listing has already been collected for this binary/PDF task. "
        "Do not call list_files again for the same location. "
        "Choose one real candidate path from that listing before attempting any binary-aware command. "
        "Do not guess placeholder names."
    )
