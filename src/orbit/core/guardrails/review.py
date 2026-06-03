from __future__ import annotations

import json
from typing import Any, Callable
import re
from dataclasses import dataclass

from .review_signals import (
    CODE_FILE_EXTENSIONS,
    CODE_REVIEW_OUTPUT_HINTS,
    CODE_REVIEW_REQUEST_HINTS,
    has_code_file_extension,
    has_code_language_hint,
)
from ..intent.router import INTENT_CODEBASE_INSPECTION
from ..messages import (
    has_recent_tool_result,
    last_read_file_result,
    merged_read_file_result_in_current_turn,
    normalize_relative_path,
    recent_listed_file_paths,
    successful_read_results_in_current_turn,
)


IMPORTANT_FILES_HINTS = (
    "most important files",
    "file più importanti",
    "file piu importanti",
    "important files to read",
    "files to inspect first",
)
CODE_REVIEW_HINTS = (
    "code review",
    "review this codebase",
    "review the code",
    "review this project",
    "fai code review",
    "rivedi il codice",
    "review",
    "findings",
    "finding",
    "risks",
    "risk",
    "rischi",
    "weakness",
    "weaknesses",
    "bug",
    "bugs",
    "issues",
    "issue",
    "problemi",
    "debolezze",
    "regression",
    "regressions",
    "vulnerability",
    "vulnerabilities",
    "vuln",
    "vulns",
    "security",
    "insecure",
    "exploit",
    "exploitable",
    "cve",
    "injection",
    "xss",
    "csrf",
    "rce",
    "traversal",
    "auth bypass",
    "vulnerabilità",
    "vulnerabilita",
    "vulnerabile",
    "sicurezza",
    "insicuro",
    "falla",
    "falle",
)
ARCHITECTURE_SUMMARY_HINTS = (
    "architecture",
    "architettura",
    "summarize",
    "riassumi",
)
HOTSPOT_HINTS = (
    "stability",
    "stabilita",
    "stabilità",
    "maintainability",
    "maintenance",
    "manutenzione",
    "attention",
    "attenzione",
)

CODE_REVIEW_READ_MAX_FILES = 3
CODE_REVIEW_READ_MAX_LINES = 220
CODE_REVIEW_READ_MAX_CHARS = 9000
CODE_REVIEW_READ_MAX_CHUNKS = 2


@dataclass(frozen=True)
class SecurityDetector:
    pattern: re.Pattern[str]
    english_template: str
    italian_template: str


SECURITY_DETECTORS: tuple[SecurityDetector, ...] = (
    SecurityDetector(
        re.compile(r"\bshell\s*=\s*True\b"),
        "High: `{path}:{line}` enables `shell=True`, which expands command-injection impact if any argument is user-controlled.",
        "Alta: `{path}:{line}` abilita `shell=True`, aumentando l'impatto di command injection se un argomento e` controllabile dall'utente.",
    ),
    SecurityDetector(
        re.compile(r"\b(?:os\.system|os\.popen|commands\.getoutput)\s*\("),
        "High: `{path}:{line}` executes shell commands through a broad shell API; prefer bounded subprocess calls with explicit arguments.",
        "Alta: `{path}:{line}` esegue comandi tramite API shell ampia; meglio subprocess bounded con argomenti espliciti.",
    ),
    SecurityDetector(
        re.compile(r"\b(?:eval|exec)\s*\("),
        "High: `{path}:{line}` uses dynamic code execution; verify that no untrusted input can reach this call.",
        "Alta: `{path}:{line}` usa esecuzione dinamica di codice; verifica che input non fidato non possa raggiungerla.",
    ),
    SecurityDetector(
        re.compile(r"\bpickle\.(?:load|loads)\s*\("),
        "High: `{path}:{line}` deserializes pickle data, which is unsafe for untrusted content.",
        "Alta: `{path}:{line}` deserializza dati pickle, non sicuri con contenuti non fidati.",
    ),
    SecurityDetector(
        re.compile(r"\byaml\.load\s*\((?![^)]*SafeLoader)"),
        "High: `{path}:{line}` calls `yaml.load` without an obvious `SafeLoader`.",
        "Alta: `{path}:{line}` chiama `yaml.load` senza un `SafeLoader` evidente.",
    ),
    SecurityDetector(
        re.compile(r"\bverify\s*=\s*False\b"),
        "Medium: `{path}:{line}` disables TLS certificate verification.",
        "Media: `{path}:{line}` disabilita la verifica dei certificati TLS.",
    ),
    SecurityDetector(
        re.compile(r"\btempfile\.mktemp\s*\("),
        "Medium: `{path}:{line}` uses `tempfile.mktemp`, which can introduce race conditions.",
        "Media: `{path}:{line}` usa `tempfile.mktemp`, che puo` introdurre race condition.",
    ),
    SecurityDetector(
        re.compile(r"\b(?:password|passwd|secret|api[_-]?key|token)\s*=\s*['\"][^'\"]{8,}['\"]", re.IGNORECASE),
        "Medium: `{path}:{line}` appears to contain a hardcoded secret-like value.",
        "Media: `{path}:{line}` sembra contenere un valore hardcoded simile a un segreto.",
    ),
)

UNANCHORED_SECURITY_REVIEW_HINTS = (
    "prompt injection risk",
    "tool execution security",
    "input sanitization",
    "error handling and state management",
    "command injection or arbitrary file system access",
    "if the inputs",
    "if any argument",
    "could be a risk",
    "could lead to",
)
CODE_REVIEW_UNCERTAINTY_HINTS = (
    "cannot definitively",
    "without executing the code",
    "having more context",
    "need the actual implementation",
    "need more context",
    "surrounding modules",
    "other modules",
    "does not contain executable logic",
    "only contains imports",
    "primarily defines the dependencies",
    "large import block",
    "non posso determinare",
    "senza eseguire il codice",
    "servirebbe più contesto",
    "mi servirebbe più contesto",
    "moduli circostanti",
    "altri moduli",
)
CODE_REVIEW_IMPORT_ONLY_HINTS = (
    "import block",
    "only contains imports",
    "does not contain executable logic",
    "primarily defines the dependencies",
)
CODE_REVIEW_SECTION_LIMITED_HINTS = (
    "specific section",
    "code snippet",
    "provided code snippet",
    "this section",
    "this snippet",
    "questa sezione",
    "questo snippet",
)
CODE_REVIEW_GENERIC_SUMMARY_HINTS = (
    "the file",
    "this file",
    "contains",
    "defines",
    "orchestrates",
    "handles",
    "includes",
    "the module",
    "il file",
    "questo file",
    "contiene",
    "definisce",
    "gestisce",
    "include",
    "il modulo",
)


def seed_codebase_listing_impl(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
    run_guardrail_tool: Callable[..., dict[str, Any]],
) -> None:
    if route.intent != INTENT_CODEBASE_INSPECTION:
        return
    lowered = user_input.lower()
    if not any(hint in lowered for hint in IMPORTANT_FILES_HINTS + ARCHITECTURE_SUMMARY_HINTS + HOTSPOT_HINTS + CODE_REVIEW_HINTS):
        return
    if any("/" in path or "\\" in path for path in explicit_code_review_paths(user_input)):
        return
    if has_recent_tool_result(messages, "list_files"):
        return
    run_guardrail_tool(
        name="list_files",
        arguments={"path": ".", "recursive": True, "max_entries": 80},
        route=route,
        registry=registry,
        messages=messages,
        metrics=metrics,
        policy_state=policy_state,
        on_event=on_event,
        emit_route=True,
    )


def codebase_redundant_listing_prompt(
    *,
    intent: str | None,
    tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    parse_arguments: Any,
) -> str | None:
    if intent != INTENT_CODEBASE_INSPECTION:
        return None
    if not _has_recursive_listing_for_path(messages, "."):
        return None
    for tool_call in tool_calls:
        function = tool_call.get("function", {}) or {}
        if function.get("name") != "list_files":
            continue
        arguments = parse_arguments(function.get("arguments"))
        path = str(arguments.get("path") or ".").strip() or "."
        if path in {".", "./"}:
            return (
                "You already have a recursive list_files result for the current workspace. "
                "Do not call list_files again for this location. Use the existing paths to choose the most relevant file, "
                "then call read_file on one concrete returned file path or answer from the evidence already available."
            )
    return None


def seed_codebase_review_reads_impl(
    *,
    user_input: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
    run_guardrail_tool: Callable[..., dict[str, Any]],
    max_chunks_per_file: int = CODE_REVIEW_READ_MAX_CHUNKS,
) -> None:
    if route.intent != INTENT_CODEBASE_INSPECTION:
        return
    lowered = user_input.lower()
    if not any(hint in lowered for hint in CODE_REVIEW_HINTS):
        return
    explicit_paths = [path for path in explicit_code_review_paths(user_input) if "/" in path or "\\" in path]
    if explicit_paths:
        already_read = {
            normalize_relative_path(str(item.get("path", "")))
            for item in successful_read_results_in_current_turn(messages)
            if isinstance(item.get("path"), str)
        }
        for path in explicit_paths[:CODE_REVIEW_READ_MAX_FILES]:
            normalized = normalize_relative_path(path)
            if normalized in already_read:
                continue
            seed_progressive_code_review_reads(
                path=normalized,
                route=route,
                registry=registry,
                messages=messages,
                metrics=metrics,
                policy_state=policy_state,
                on_event=on_event,
                run_guardrail_tool=run_guardrail_tool,
                max_chunks=max_chunks_per_file,
            )
        return
    listed = recent_listed_file_paths(messages, limit=40)
    if not listed:
        return
    already_read = {
        normalize_relative_path(str(item.get("path", "")))
        for item in successful_read_results_in_current_turn(messages)
        if isinstance(item.get("path"), str)
    }
    candidates = code_review_target_paths(listed)
    for path in candidates[:CODE_REVIEW_READ_MAX_FILES]:
        normalized = normalize_relative_path(path)
        if normalized not in already_read:
            seed_progressive_code_review_reads(
                path=normalized,
                route=route,
                registry=registry,
                messages=messages,
                metrics=metrics,
                policy_state=policy_state,
                on_event=on_event,
                run_guardrail_tool=run_guardrail_tool,
                max_chunks=max_chunks_per_file,
            )


def _has_recursive_listing_for_path(messages: list[dict[str, Any]], path: str) -> bool:
    normalized = path.rstrip("/") or "."
    for message in reversed(messages):
        if message.get("role") == "user":
            return False
        if message.get("role") != "tool" or message.get("tool_name") != "list_files":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            continue
        listed_path = str(payload.get("path") or ".").rstrip("/") or "."
        if listed_path == normalized and payload.get("recursive") is True:
            return True
    return False


def local_codebase_priority_files_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
) -> str | None:
    if intent != INTENT_CODEBASE_INSPECTION:
        return None
    lowered = user_input.lower()
    if not any(hint in lowered for hint in IMPORTANT_FILES_HINTS):
        return None
    listed = recent_listed_file_paths(messages, limit=5)
    if not listed:
        return None
    return "\n".join(f"- `{path}`" for path in listed[:5])


def local_codebase_architecture_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
    prefers_english_output: Callable[[str], bool],
) -> str | None:
    if intent != INTENT_CODEBASE_INSPECTION:
        return None
    lowered = user_input.lower()
    if not any(hint in lowered for hint in ARCHITECTURE_SUMMARY_HINTS):
        return None
    listed = recent_listed_file_paths(messages, limit=40)
    if not listed:
        return None
    english = prefers_english_output(lowered)
    bullets: list[str] = []
    if any(path.startswith("src/orbit/core/") for path in listed):
        bullets.append("- `src/orbit/core/` contains the agent loop, runtime, policy, and guardrails." if english else "- `src/orbit/core/` contiene loop agente, runtime, policy e guardrail.")
    if any(path.startswith("src/orbit/tooling/") for path in listed):
        bullets.append("- `src/orbit/tooling/` separates local tools by domain such as filesystem, shell, and web." if english else "- `src/orbit/tooling/` separa le tool locali per dominio come filesystem, shell e web.")
    if any(path.startswith("src/orbit/terminal/") for path in listed):
        bullets.append("- `src/orbit/terminal/` contains the CLI, config, history, and text rendering." if english else "- `src/orbit/terminal/` contiene CLI, config, history e rendering testuale.")
    if not bullets:
        top_paths = listed[:3]
        bullets = [f"- The main paths that stand out are: {', '.join(top_paths)}." if english else f"- I path principali emersi sono: {', '.join(top_paths)}."]
    return "\n".join(bullets[:3])


def local_codebase_hotspot_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
    prefers_english_output: Callable[[str], bool],
) -> str | None:
    if intent != INTENT_CODEBASE_INSPECTION:
        return None
    lowered = user_input.lower()
    if not any(hint in lowered for hint in HOTSPOT_HINTS):
        return None
    listed = recent_listed_file_paths(messages, limit=60)
    if not listed:
        return None
    english = prefers_english_output(lowered)
    bullets: list[str] = []
    if "src/orbit/core/agent.py" in listed:
        bullets.append("- `src/orbit/core/agent.py`: it is the most sensitive point for loop stability, tool routing, and stop conditions." if english else "- `src/orbit/core/agent.py`: e` il punto piu` sensibile per stabilita` del loop, tool routing e stop conditions.")
    if "src/orbit/core/runtime.py" in listed:
        bullets.append("- `src/orbit/core/runtime.py`: it concentrates bootstrap, sessions, and startup policy, so changes there have broad impact." if english else "- `src/orbit/core/runtime.py`: concentra bootstrap, sessioni e policy di avvio, quindi ha impatto trasversale.")
    if "src/orbit/core/tools/guardrails.py" in listed:
        bullets.append("- `src/orbit/core/tools/guardrails.py`: it concentrates many runtime heuristics, so it needs close control to avoid drift and regressions." if english else "- `src/orbit/core/tools/guardrails.py`: raccoglie molte euristiche runtime; va tenuto sotto controllo per evitare drift e regressioni.")
    elif "src/orbit/terminal/cli.py" in listed:
        bullets.append("- `src/orbit/terminal/cli.py`: it deserves attention for REPL UX and safe slash-command handling." if english else "- `src/orbit/terminal/cli.py`: merita attenzione per UX REPL e gestione sicura dei comandi slash.")
    if not bullets:
        bullets = [f"- `{path}`: this file stands out as a central node in the current structure." if english else f"- `{path}`: file emerso come nodo centrale della struttura corrente." for path in listed[:3]]
    return "\n".join(bullets[:3])


def local_codebase_review_result(
    *,
    intent: str | None,
    user_input: str,
    messages: list[dict[str, Any]],
    prefers_english_output: Callable[[str], bool],
) -> str | None:
    if intent != INTENT_CODEBASE_INSPECTION:
        return None
    lowered = user_input.lower()
    if _asks_for_explanatory_codebase_bug_analysis(user_input):
        return None
    if not any(hint in lowered for hint in CODE_REVIEW_HINTS):
        return None
    read_results = successful_read_results_in_current_turn(messages)
    if not read_results:
        return None
    english = prefers_english_output(lowered)
    review_targets = code_review_target_paths(recent_listed_file_paths(messages, limit=40))
    if not review_targets:
        review_targets = code_review_target_paths(
            [str(item.get("path")) for item in read_results if isinstance(item.get("path"), str)]
        )
    findings: list[str] = []
    merged_results: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for path in review_targets[:CODE_REVIEW_READ_MAX_FILES]:
        result = merged_read_file_result_in_current_turn(messages, path) or last_read_file_result(messages, path)
        if result is None:
            continue
        normalized = normalize_relative_path(str(result.get("path", path)))
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        merged_results.append(result)
        for finding in infer_security_review_findings(result, english=english):
            if finding in findings:
                continue
            findings.append(finding)
            if len(findings) >= 3:
                break
        if len(findings) >= 3:
            break
        finding = infer_code_review_finding(result, english=english)
        if finding is not None:
            findings.append(finding)
        if len(findings) >= 3:
            break
    if len(findings) < 3:
        for finding in infer_cross_file_review_findings(merged_results, english=english):
            if finding in findings:
                continue
            findings.append(finding)
            if len(findings) >= 3:
                break
    if not findings:
        if _asks_for_explanatory_codebase_bug_analysis(user_input):
            return None
        targets = [item.get("path") for item in read_results if isinstance(item.get("path"), str)]
        targets = [path for path in targets if isinstance(path, str)][:3]
        if not targets:
            return None
        if english:
            return "\n".join(f"- Preliminary review target: `{path}` should be inspected before claiming bug findings." for path in targets)
        return "\n".join(f"- Target preliminare di review: `{path}` va ispezionato prima di affermare bug concreti." for path in targets)
    return "\n".join(findings[:3])


def codebase_review_reply_handling(
    *,
    intent: str | None,
    user_input: str,
    content: str,
    messages: list[dict[str, Any]],
    policy_state: Any,
    prefers_english_output: Callable[[str], bool],
) -> tuple[str, str] | None:
    if intent != INTENT_CODEBASE_INSPECTION:
        return None
    if policy_state.synthesis_retries >= 2:
        return None
    read_results = successful_read_results_in_current_turn(messages)
    if not read_results:
        return _codebase_retry_after_listing_only(
            user_input=user_input,
            messages=messages,
            policy_state=policy_state,
        )
    if not _looks_like_explicit_file_review_request(user_input) and not _looks_like_codebase_request_that_needs_read(user_input):
        return None
    latest = read_results[-1]
    path = latest.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    lowered = content.lower()
    retry = _codebase_retry_after_incomplete_chunk(
        latest=latest,
        path=path,
        lowered_content=lowered,
        policy_state=policy_state,
    )
    if retry is not None:
        return retry
    retry = _codebase_retry_after_generic_summary(
        path=path,
        lowered_content=lowered,
        policy_state=policy_state,
    )
    if retry is not None:
        return retry
    retry = _codebase_retry_after_file_list_only_reply(
        user_input=user_input,
        content=content,
        read_results=read_results,
        policy_state=policy_state,
    )
    if retry is not None:
        return retry
    final = _codebase_final_from_unanchored_security_review(
        intent=intent,
        user_input=user_input,
        content=content,
        messages=messages,
        prefers_english_output=prefers_english_output,
    )
    if final is not None:
        return final
    return _codebase_retry_after_generic_uncertainty(
        path=path,
        lowered_content=lowered,
        policy_state=policy_state,
    )


def _codebase_retry_after_listing_only(
    *,
    user_input: str,
    messages: list[dict[str, Any]],
    policy_state: Any,
) -> tuple[str, str] | None:
    if not _looks_like_codebase_request_that_needs_read(user_input):
        return None
    listed = recent_listed_file_paths(messages, limit=80)
    if not listed:
        return None
    candidates = topic_aware_code_review_target_paths(listed, user_input)
    if not candidates:
        return None
    policy_state.synthesis_retries += 1
    return (
        "retry",
        "You only listed workspace paths, but the user asked for code inspection or bug analysis. "
        f"Call read_file now on `{candidates[0]}`, then answer from that concrete evidence. "
        "Do not finalize from a file list alone.",
    )


def _codebase_retry_after_incomplete_chunk(
    *,
    latest: dict[str, Any],
    path: str,
    lowered_content: str,
    policy_state: Any,
) -> tuple[str, str] | None:
    if not (latest.get("has_more") or latest.get("truncated")):
        return None
    next_start_line = latest.get("next_start_line")
    if policy_state.synthesis_retries >= 1 or not isinstance(next_start_line, int) or next_start_line <= 1:
        return None
    if not any(hint in lowered_content for hint in CODE_REVIEW_UNCERTAINTY_HINTS + CODE_REVIEW_IMPORT_ONLY_HINTS + CODE_REVIEW_SECTION_LIMITED_HINTS):
        return None
    policy_state.synthesis_retries += 1
    return (
        "retry",
        f"You only saw an incomplete chunk of `{path}`. "
        f"Continue reading the same file with read_file using start_line={next_start_line}, then continue the review from the combined file evidence. "
        "Do not conclude from the first import or header block alone.",
    )


def _codebase_retry_after_generic_summary(
    *,
    path: str,
    lowered_content: str,
    policy_state: Any,
) -> tuple[str, str] | None:
    if not any(hint in lowered_content for hint in CODE_REVIEW_GENERIC_SUMMARY_HINTS):
        return None
    if any(hint in lowered_content for hint in CODE_REVIEW_OUTPUT_HINTS):
        return None
    policy_state.synthesis_retries += 1
    return (
        "retry",
        f"You already read concrete code from `{path}`. "
        "Do not summarize the file structure. "
        "Perform a bug and risk review of the inspected code and report up to 3 concrete findings. "
        "If no clear bug is visible in the inspected portions, say `No concrete bug found in the inspected portions.` and give at most 1 precise remaining uncertainty.",
    )


def _codebase_retry_after_file_list_only_reply(
    *,
    user_input: str,
    content: str,
    read_results: list[dict[str, Any]],
    policy_state: Any,
) -> tuple[str, str] | None:
    if not (_asks_for_explanatory_codebase_bug_analysis(user_input) and _looks_like_file_list_only_reply(content)):
        return None
    paths = [str(item.get("path")) for item in read_results if isinstance(item.get("path"), str)]
    policy_state.synthesis_retries += 1
    return (
        "retry",
        "You already read concrete code. Do not answer with only a list of files. "
        f"Use the evidence from {', '.join(paths[:3])} to explain one possible bug or routing risk and a minimal fix. "
        "If no concrete bug is visible, say that explicitly and give one precise remaining uncertainty.",
    )


def _codebase_final_from_unanchored_security_review(
    *,
    intent: str,
    user_input: str,
    content: str,
    messages: list[dict[str, Any]],
    prefers_english_output: Callable[[str], bool],
) -> tuple[str, str] | None:
    if not _looks_like_unanchored_generic_security_review(content):
        return None
    local_review = local_codebase_review_result(
        intent=intent,
        user_input=user_input,
        messages=messages,
        prefers_english_output=prefers_english_output,
    )
    if local_review is None:
        return None
    return ("final", local_review)


def _codebase_retry_after_generic_uncertainty(
    *,
    path: str,
    lowered_content: str,
    policy_state: Any,
) -> tuple[str, str] | None:
    if not any(hint in lowered_content for hint in CODE_REVIEW_UNCERTAINTY_HINTS):
        return None
    policy_state.synthesis_retries += 1
    return (
        "retry",
        f"You already read the concrete source file `{path}`. "
        "Do not stop at generic uncertainty about surrounding modules. "
        "Review this file directly and report up to 3 concrete bug or risk findings visible in the file itself. "
        "If no clear bug is visible in this file, say that explicitly and give at most 1 precise remaining uncertainty.",
    )


def _looks_like_unanchored_generic_security_review(content: str) -> bool:
    lowered = content.lower()
    if not any(hint in lowered for hint in UNANCHORED_SECURITY_REVIEW_HINTS):
        return False
    if re.search(r"`[^`]+:\d+`", content):
        return False
    if re.search(r"\bline\s+\d+\b", lowered):
        return False
    if "no concrete bug" in lowered or "no concrete vulnerability" in lowered:
        return False
    return True


def _looks_like_file_list_only_reply(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False
    path_like_lines = 0
    for line in lines:
        if re.search(r"`[^`]+\.(?:py|js|ts|tsx|jsx|go|rs|java|c|cpp|h|hpp|ps1|sh|md|toml|json|yaml|yml)`", line):
            path_like_lines += 1
        elif re.search(r"\b[\w./-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|c|cpp|h|hpp|ps1|sh|md|toml|json|yaml|yml)\b", line):
            path_like_lines += 1
    if path_like_lines == 0:
        return False
    explanatory_terms = ("bug", "risk", "fix", "because", "therefore", "issue", "vulnerability", "exploit", "rischio", "problema", "correzione")
    if any(term in stripped.lower() for term in explanatory_terms):
        return False
    return path_like_lines >= max(1, len(lines) - 1)


def _asks_for_explanatory_codebase_bug_analysis(user_input: str) -> bool:
    lowered = user_input.lower()
    explanation_hints = ("explain", "propose", "minimal fix", "fix", "spiega", "proponi", "correzione")
    bug_hints = ("bug", "bugs", "issue", "issues", "risk", "risks", "problema", "problemi", "rischio", "rischi")
    return any(hint in lowered for hint in explanation_hints) and any(hint in lowered for hint in bug_hints)


def code_review_target_paths(paths: list[str]) -> list[str]:
    scored: list[tuple[int, str]] = []
    for path in paths:
        normalized = normalize_relative_path(path)
        lowered = normalized.lower()
        if lowered.startswith(".") or not any(lowered.endswith(ext) for ext in CODE_FILE_EXTENSIONS):
            continue
        score = 0
        if "/core/" in lowered:
            score += 8
        if any(token in lowered for token in ("/agent", "/runtime", "guardrails", "router", "registry", "/cli", "main", "app")):
            score += 6
        if "/tooling/" in lowered or "/terminal/" in lowered:
            score += 3
        if lowered.endswith(("/test.py", "_test.py")) or "/tests/" in lowered:
            score -= 4
        score += max(0, 4 - lowered.count("/"))
        scored.append((score, normalized))
    scored.sort(key=lambda item: (-item[0], item[1]))
    out: list[str] = []
    seen: set[str] = set()
    for _, path in scored:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def topic_aware_code_review_target_paths(paths: list[str], user_input: str) -> list[str]:
    candidates = code_review_target_paths(paths)
    if not candidates:
        return []
    lowered = user_input.lower()
    topic_preferences: tuple[str, ...] = ()
    if any(token in lowered for token in ("routing", "router", "intent", "tool routing", "tool-call", "tool call", "tool calls")):
        topic_preferences = (
            "src/orbit/core/intent/router.py",
            "src/orbit/core/tools/router.py",
            "src/orbit/core/tools/guardrails.py",
            "src/orbit/core/agent.py",
        )
    elif any(token in lowered for token in ("tool", "tools", "strumento", "strumenti")):
        topic_preferences = (
            "src/orbit/core/tools/guardrails.py",
            "src/orbit/core/tools/router.py",
            "src/orbit/tooling/registry.py",
            "src/orbit/core/agent.py",
        )
    if not topic_preferences:
        return candidates
    ranked = sorted(
        candidates,
        key=lambda path: (
            next((index for index, preferred in enumerate(topic_preferences) if path == preferred or path.endswith(preferred)), len(topic_preferences)),
            candidates.index(path),
        ),
    )
    return ranked


def _looks_like_codebase_request_that_needs_read(user_input: str) -> bool:
    lowered = user_input.lower()
    if "most relevant files" in lowered or "most important files" in lowered:
        return False
    required_action_hints = (
        "inspect",
        "read",
        "review",
        "analyze",
        "analyse",
        "bug",
        "bugs",
        "issue",
        "issues",
        "risk",
        "risks",
        "finding",
        "findings",
        "vulnerability",
        "vulnerabilities",
        "ispeziona",
        "leggi",
        "rivedi",
        "analizza",
        "problema",
        "problemi",
        "rischio",
        "rischi",
        "vulnerabilità",
        "vulnerabilita",
    )
    return any(hint in lowered for hint in required_action_hints)


def explicit_code_review_paths(user_input: str) -> list[str]:
    extension_pattern = "|".join(re.escape(ext) for ext in sorted(CODE_FILE_EXTENSIONS, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\w.-])([A-Za-z0-9_./\\-]+(?:{extension_pattern}))(?![\w.-])", re.IGNORECASE)
    paths: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(user_input):
        path = normalize_relative_path(match.group(1).strip("`'\".,:;()[]{}"))
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _looks_like_explicit_file_review_request(user_input: str) -> bool:
    lowered = user_input.lower()
    if not any(hint in lowered for hint in CODE_REVIEW_REQUEST_HINTS):
        return False
    if not _looks_like_text_path_request_for_review(lowered):
        return False
    return has_code_file_extension(lowered) or has_code_language_hint(lowered) or "this file" in lowered or "questo file" in lowered


def _looks_like_text_path_request_for_review(lowered: str) -> bool:
    return has_code_file_extension(lowered) or "/" in lowered or "\\" in lowered


def seed_progressive_code_review_reads(
    *,
    path: str,
    route: Any,
    registry: Any,
    messages: list[dict[str, Any]],
    metrics: Any,
    policy_state: Any,
    on_event: Any,
    run_guardrail_tool: Callable[..., dict[str, Any]],
    max_chunks: int = CODE_REVIEW_READ_MAX_CHUNKS,
) -> None:
    existing = [item for item in successful_read_results_in_current_turn(messages) if normalize_relative_path(str(item.get("path", ""))) == path]
    latest = existing[-1] if existing else None
    chunks_read = len(existing)
    if latest is None:
        latest = run_guardrail_tool(
            name="read_file",
            arguments={"path": path, "start_line": 1, "max_lines": CODE_REVIEW_READ_MAX_LINES, "max_chars": CODE_REVIEW_READ_MAX_CHARS},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=False,
        )
        if not latest.get("ok"):
            return
        chunks_read += 1
    while chunks_read < max_chunks and latest.get("ok") and (latest.get("has_more") or latest.get("truncated")):
        next_start_line = latest.get("next_start_line")
        if not isinstance(next_start_line, int) or next_start_line <= 1:
            break
        latest = run_guardrail_tool(
            name="read_file",
            arguments={"path": path, "start_line": next_start_line, "max_lines": CODE_REVIEW_READ_MAX_LINES, "max_chars": CODE_REVIEW_READ_MAX_CHARS},
            route=route,
            registry=registry,
            messages=messages,
            metrics=metrics,
            policy_state=policy_state,
            on_event=on_event,
            emit_route=False,
        )
        if not latest.get("ok"):
            break
        chunks_read += 1


def infer_code_review_finding(result: dict[str, Any], *, english: bool) -> str | None:
    path = result.get("path")
    content = result.get("content")
    if not isinstance(path, str) or not isinstance(content, str):
        return None
    total_lines = result.get("total_lines")
    line_count = total_lines if isinstance(total_lines, int) and total_lines > 0 else len(content.splitlines())
    lowered_path = normalize_relative_path(path).lower()
    branch_markers = sum(content.count(token) for token in ("if ", "elif ", "for ", "while ", "try:", "except", "continue", "break"))
    heuristic_markers = content.count("re.compile") + content.count("_HINTS") + content.count("startswith(")
    pass_markers = len(re.findall(r"(?m)^\s*pass\s*$", content))
    broad_except_markers = len(re.findall(r"(?m)^\s*except\s*:\s*$", content))
    todo_markers = content.lower().count("todo") + content.lower().count("fixme")
    placeholder_markers = content.count("NotImplementedError") + content.count("raise NotImplemented")
    if broad_except_markers >= 1:
        return f"- High: `{path}` contains a bare `except:`, which can hide real failures and makes review of error handling a priority." if english else f"- Alta: `{path}` contiene un `except:` nudo, che puo` nascondere errori reali e rende prioritaria la review della gestione errori."
    if pass_markers >= 3 or placeholder_markers >= 2:
        return f"- Medium: `{path}` still contains placeholder control paths (`pass` or not-implemented stubs), so behavior gaps should be checked before trusting changes there." if english else f"- Media: `{path}` contiene ancora percorsi placeholder (`pass` o stub non implementati), quindi vanno verificati gap di comportamento prima di fidarsi delle modifiche."
    if todo_markers >= 2:
        return f"- Medium: `{path}` includes multiple TODO/FIXME markers, which suggests known debt or unfinished behavior worth checking during review." if english else f"- Media: `{path}` include piu` marker TODO/FIXME, segnale di debito noto o comportamento incompleto da controllare in review."
    if line_count >= 180 and any(token in lowered_path for token in ("/core/", "agent.py", "runtime.py", "guardrails.py", "router.py", "cli.py")):
        return f"- High: `{path}` is a central module with about {line_count} lines, so regressions there can affect broad behavior and should be reviewed first." if english else f"- Alta: `{path}` e` un modulo centrale con circa {line_count} righe, quindi una regressione qui puo` alterare comportamento trasversale e va rivisto per primo."
    if heuristic_markers >= 4:
        return f"- Medium: `{path}` concentrates many routing or heuristic rules, so it deserves tight regression coverage and careful review for drift." if english else f"- Media: `{path}` concentra molte regole euristiche o di routing, quindi richiede review attenta e copertura regressiva stretta per evitare drift."
    if branch_markers >= 12:
        return f"- Medium: `{path}` has dense control flow, which increases the chance of edge-case regressions around retries, stops, or fallbacks." if english else f"- Media: `{path}` ha un flusso di controllo denso, che aumenta il rischio di regressioni sugli edge case di retry, stop o fallback."
    if any(token in lowered_path for token in ("/registry.py", "/router.py", "/cli.py", "/runtime.py")):
        return f"- Medium: `{path}` is an integration surface, so small changes there can propagate widely even when the diff looks localized." if english else f"- Media: `{path}` e` una superficie di integrazione, quindi anche modifiche piccole possono propagare effetti ampi pur sembrando locali."
    return None


def infer_security_review_findings(result: dict[str, Any], *, english: bool) -> list[str]:
    path = result.get("path")
    content = result.get("content")
    if not isinstance(path, str) or not isinstance(content, str):
        return []
    start_line = result.get("start_line")
    base_line = start_line if isinstance(start_line, int) and start_line > 0 else 1
    findings: list[str] = []
    for line_number, detector in _security_detector_hits(content, base_line=base_line):
        findings.append(_format_security_finding(detector, path=path, line=line_number, english=english))
        if len(findings) >= 3:
            break
    return findings


def _security_detector_hits(content: str, *, base_line: int) -> list[tuple[int, SecurityDetector]]:
    hits: list[tuple[int, SecurityDetector]] = []
    for offset, line in enumerate(content.splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for detector in SECURITY_DETECTORS:
            if not detector.pattern.search(stripped):
                continue
            hits.append((base_line + offset, detector))
            break
    return hits


def _format_security_finding(detector: SecurityDetector, *, path: str, line: int, english: bool) -> str:
    template = detector.english_template if english else detector.italian_template
    return template.format(path=path, line=line)


def infer_cross_file_review_findings(results: list[dict[str, Any]], *, english: bool) -> list[str]:
    normalized_paths = {normalize_relative_path(str(item.get("path", ""))) for item in results if isinstance(item.get("path"), str)}
    findings: list[str] = []
    if {"src/orbit/core/agent.py", "src/orbit/core/runtime.py"} <= normalized_paths:
        findings.append("- High: `src/orbit/core/agent.py` and `src/orbit/core/runtime.py` form a tight execution boundary, so changes usually need regression checks across loop behavior, sessions, and startup policy." if english else "- Alta: `src/orbit/core/agent.py` e `src/orbit/core/runtime.py` formano un confine esecutivo stretto, quindi le modifiche richiedono di solito regressioni su loop, sessioni e policy di avvio.")
    if {"src/orbit/core/agent.py", "src/orbit/core/tools/guardrails.py"} <= normalized_paths:
        findings.append("- Medium: `src/orbit/core/agent.py` and `src/orbit/core/tools/guardrails.py` are coupled through routing and stop/fallback behavior, so partial fixes can drift unless both sides are reviewed together." if english else "- Media: `src/orbit/core/agent.py` e `src/orbit/core/tools/guardrails.py` sono accoppiati tramite routing e fallback/stop, quindi fix parziali rischiano drift se non vengono rivisti insieme.")
    if {"src/orbit/core/tools/guardrails.py", "src/orbit/core/intent/router.py"} <= normalized_paths or {"src/orbit/core/tools/guardrails.py", "src/orbit/core/tools/router.py"} <= normalized_paths:
        findings.append("- Medium: routing rules and guardrails are split across multiple modules, so review should check that intent classification and runtime enforcement stay aligned." if english else "- Media: regole di routing e guardrail sono distribuiti su piu` moduli, quindi la review deve verificare che classificazione intent ed enforcement runtime restino allineati.")
    if {"src/orbit/tooling/registry.py", "src/orbit/core/tools/guardrails.py"} <= normalized_paths:
        findings.append("- Medium: tool exposure and runtime assumptions are split between `registry` and guardrails, so adding or changing tools should be reviewed across both layers." if english else "- Media: esposizione delle tool e assunzioni runtime sono divise tra `registry` e guardrail, quindi aggiunte o cambi alle tool vanno riviste su entrambi i livelli.")
    return findings
