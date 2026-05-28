from __future__ import annotations

from typing import Any, Callable
import re

from .code_review_signals import (
    CODE_FILE_EXTENSIONS,
    CODE_REVIEW_OUTPUT_HINTS,
    CODE_REVIEW_REQUEST_HINTS,
    has_code_file_extension,
    has_code_language_hint,
)
from .intent_router import INTENT_CODEBASE_INSPECTION
from .message_ops import (
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
    if "src/orbit/core/tool_guardrails.py" in listed:
        bullets.append("- `src/orbit/core/tool_guardrails.py`: it concentrates many runtime heuristics, so it needs close control to avoid drift and regressions." if english else "- `src/orbit/core/tool_guardrails.py`: raccoglie molte euristiche runtime; va tenuto sotto controllo per evitare drift e regressioni.")
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
    if not any(hint in lowered for hint in CODE_REVIEW_HINTS):
        return None
    read_results = successful_read_results_in_current_turn(messages)
    if not read_results:
        return None
    english = prefers_english_output(lowered)
    review_targets = code_review_target_paths(recent_listed_file_paths(messages, limit=40))
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
) -> tuple[str, str] | None:
    if intent != INTENT_CODEBASE_INSPECTION:
        return None
    if policy_state.synthesis_retries >= 2:
        return None
    if not _looks_like_explicit_file_review_request(user_input):
        return None
    read_results = successful_read_results_in_current_turn(messages)
    if not read_results:
        return None
    latest = read_results[-1]
    path = latest.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    lowered = content.lower()
    uncertainty_hints = (
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
    import_only_hints = (
        "import block",
        "only contains imports",
        "does not contain executable logic",
        "primarily defines the dependencies",
    )
    section_limited_hints = (
        "specific section",
        "code snippet",
        "provided code snippet",
        "this section",
        "this snippet",
        "questa sezione",
        "questo snippet",
    )
    review_output_hints = CODE_REVIEW_OUTPUT_HINTS
    generic_summary_hints = (
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
    if latest.get("has_more") or latest.get("truncated"):
        next_start_line = latest.get("next_start_line")
        if policy_state.synthesis_retries < 1 and isinstance(next_start_line, int) and next_start_line > 1:
            if any(hint in lowered for hint in uncertainty_hints + import_only_hints + section_limited_hints):
                policy_state.synthesis_retries += 1
                return (
                    "retry",
                    f"You only saw an incomplete chunk of `{path}`. "
                    f"Continue reading the same file with read_file using start_line={next_start_line}, then continue the review from the combined file evidence. "
                    "Do not conclude from the first import or header block alone.",
                )
    if any(hint in lowered for hint in generic_summary_hints) and not any(hint in lowered for hint in review_output_hints):
        policy_state.synthesis_retries += 1
        return (
            "retry",
            f"You already read concrete code from `{path}`. "
            "Do not summarize the file structure. "
            "Perform a bug and risk review of the inspected code and report up to 3 concrete findings. "
            "If no clear bug is visible in the inspected portions, say `No concrete bug found in the inspected portions.` and give at most 1 precise remaining uncertainty.",
        )
    if not any(hint in lowered for hint in uncertainty_hints):
        return None
    policy_state.synthesis_retries += 1
    return (
        "retry",
        f"You already read the concrete source file `{path}`. "
        "Do not stop at generic uncertainty about surrounding modules. "
        "Review this file directly and report up to 3 concrete bug or risk findings visible in the file itself. "
        "If no clear bug is visible in this file, say that explicitly and give at most 1 precise remaining uncertainty.",
    )


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


def infer_cross_file_review_findings(results: list[dict[str, Any]], *, english: bool) -> list[str]:
    normalized_paths = {normalize_relative_path(str(item.get("path", ""))) for item in results if isinstance(item.get("path"), str)}
    findings: list[str] = []
    if {"src/orbit/core/agent.py", "src/orbit/core/runtime.py"} <= normalized_paths:
        findings.append("- High: `src/orbit/core/agent.py` and `src/orbit/core/runtime.py` form a tight execution boundary, so changes usually need regression checks across loop behavior, sessions, and startup policy." if english else "- Alta: `src/orbit/core/agent.py` e `src/orbit/core/runtime.py` formano un confine esecutivo stretto, quindi le modifiche richiedono di solito regressioni su loop, sessioni e policy di avvio.")
    if {"src/orbit/core/agent.py", "src/orbit/core/tool_guardrails.py"} <= normalized_paths:
        findings.append("- Medium: `src/orbit/core/agent.py` and `src/orbit/core/tool_guardrails.py` are coupled through routing and stop/fallback behavior, so partial fixes can drift unless both sides are reviewed together." if english else "- Media: `src/orbit/core/agent.py` e `src/orbit/core/tool_guardrails.py` sono accoppiati tramite routing e fallback/stop, quindi fix parziali rischiano drift se non vengono rivisti insieme.")
    if {"src/orbit/core/tool_guardrails.py", "src/orbit/core/intent_router.py"} <= normalized_paths or {"src/orbit/core/tool_guardrails.py", "src/orbit/core/tool_router.py"} <= normalized_paths:
        findings.append("- Medium: routing rules and guardrails are split across multiple modules, so review should check that intent classification and runtime enforcement stay aligned." if english else "- Media: regole di routing e guardrail sono distribuiti su piu` moduli, quindi la review deve verificare che classificazione intent ed enforcement runtime restino allineati.")
    if {"src/orbit/tooling/registry.py", "src/orbit/core/tool_guardrails.py"} <= normalized_paths:
        findings.append("- Medium: tool exposure and runtime assumptions are split between `registry` and guardrails, so adding or changing tools should be reviewed across both layers." if english else "- Media: esposizione delle tool e assunzioni runtime sono divise tra `registry` e guardrail, quindi aggiunte o cambi alle tool vanno riviste su entrambi i livelli.")
    return findings
