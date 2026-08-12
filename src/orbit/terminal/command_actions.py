from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shlex
from urllib.parse import urlparse

from orbit.native_llama.model_discovery import discover_models, format_model_discovery
from orbit.native_llama.paths import resolve_native_build_bin
from orbit.runtime.command_evidence import AcquiredEvidence
from orbit.runtime.directory_listing import execute_list_directory
from orbit.runtime.document_search import literal_document_search_answer, search_document_snapshot
from orbit.runtime.document_tool import execute_read_file
from orbit.runtime.file_tools import read_full_document_snapshot, read_pdf
from orbit.runtime.full_document import FullDocumentSnapshot, attest_full_document_snapshot, parse_full_document_snapshot
from orbit.runtime.shell_guardrails import execute_exec_shell_full_command
from orbit.runtime.web import execute_fetch_url, fetch_url_result_status, fetch_url_result_text, search_web


@dataclass(frozen=True)
class CommandAction:
    output: str
    prompt: str | None = None
    evidence: AcquiredEvidence | None = None
    full_document_path: str | None = None

    @property
    def needs_model(self) -> bool:
        return self.prompt is not None and (self.evidence is not None or self.full_document_path is not None)


def build_read_action(arguments: str, *, workdir: Path) -> CommandAction:
    values = _split(arguments, usage="/read <source> [prompt]")
    if isinstance(values, CommandAction):
        return values
    if not values:
        return _usage_error("source is required", "/read <source> [prompt]")
    source = values[0]
    prompt = " ".join(values[1:]).strip() or None
    url_state = _url_state(source)
    if url_state == "invalid":
        return _usage_error("source is not a valid HTTP(S) URL", "/read <source> [prompt]")
    if url_state == "valid":
        content = execute_fetch_url({"url": source})
        if fetch_url_result_status(content) != "ok":
            return CommandAction(content)
        return _with_optional_prompt(
            content,
            prompt=prompt,
            evidence=AcquiredEvidence("fetch_url", {"url": source}, content, source),
        )
    if Path(source).suffix.lower() == ".pdf":
        content = read_pdf(source, arguments={}, workdir=workdir)
        if content.startswith("error:") or prompt is None:
            return CommandAction(content)
        evidence_content = "shell_output_pdf_text: true\n" + content
        return CommandAction(
            content,
            prompt=prompt,
            evidence=AcquiredEvidence(
                "exec_shell_full_command",
                {"command": f"pdftotext -- {shlex.quote(source)} -"},
                evidence_content,
                source,
            ),
        )
    if prompt is not None:
        # Exact full-document admission, including context fit, remains a runtime decision.
        return CommandAction(f"read: {source}", prompt=prompt, full_document_path=source)
    return CommandAction(execute_read_file({"path": source}, workdir=workdir))


def build_search_action(arguments: str, *, workdir: Path) -> CommandAction:
    values = _split(arguments, usage="/search <query> [source] [prompt]")
    if isinstance(values, CommandAction):
        return values
    if not values:
        return _usage_error("query is required", "/search <query> [source] [prompt]")
    query = values[0].strip()
    if not query:
        return _usage_error("query must be non-empty", "/search <query> [source] [prompt]")
    source = values[1] if len(values) > 1 and _is_source_token(values[1], workdir=workdir) else None
    prompt_values = values[2:] if source is not None else values[1:]
    prompt = " ".join(prompt_values).strip() or None
    if source is None:
        content = search_web(query)
        if content.startswith("error:"):
            return CommandAction(content)
        return _with_optional_prompt(
            content,
            prompt=prompt,
            evidence=AcquiredEvidence(
                "exec_shell_full_command",
                {"command": f"orbit-web-search {shlex.quote(query)}"},
                content,
                "web",
            ),
        )
    url_state = _url_state(source)
    if url_state == "invalid":
        return _usage_error("source is not a valid HTTP(S) URL", "/search <query> [source] [prompt]")
    if url_state == "valid":
        fetched = execute_fetch_url({"url": source})
        if fetch_url_result_status(fetched) != "ok":
            return CommandAction(fetched)
        text = fetch_url_result_text(fetched)
        if text is None:
            return CommandAction("error: fetched URL did not provide searchable text")
        snapshot = _url_snapshot(source, text)
        try:
            result = search_document_snapshot(snapshot, mode="literal", terms=(query,), whole_word=False)
        except ValueError as exc:
            return CommandAction(f"error: {exc}")
        content = _url_search_answer(result, fetched=fetched)
        return _with_optional_prompt(
            content,
            prompt=prompt,
            evidence=AcquiredEvidence("fetch_url", {"url": source}, content, source),
        )
    if Path(source).suffix.lower() == ".pdf":
        command = (
            f"pdftotext {shlex.quote(source)} - | "
            f"rg -e {shlex.quote(re.escape(query))}"
        )
        content = execute_exec_shell_full_command(
            {"command": command},
            workdir=workdir,
            user_prompt=f"search exactly for {query!r} in {source}",
        )
        if content.startswith("error:"):
            return CommandAction(content)
        return _with_optional_prompt(
            content,
            prompt=prompt,
            evidence=AcquiredEvidence("exec_shell_full_command", {"command": command}, content, source),
        )
    raw = read_full_document_snapshot(source, workdir=workdir)
    if raw.startswith("error:"):
        return CommandAction(raw)
    snapshot = parse_full_document_snapshot(raw)
    if snapshot is None:
        return CommandAction("error: document snapshot integrity failure")
    source_error = attest_full_document_snapshot(snapshot, workdir=workdir)
    if source_error is not None:
        return CommandAction(f"error: document snapshot rejected: {source_error}")
    try:
        result = search_document_snapshot(snapshot, mode="literal", terms=(query,), whole_word=False)
    except ValueError as exc:
        return CommandAction(f"error: {exc}")
    source_error = attest_full_document_snapshot(snapshot, workdir=workdir)
    if source_error is not None:
        return CommandAction(f"error: document changed during search: {source_error}")
    content = literal_document_search_answer(result, whole_word=False)
    return _with_optional_prompt(
        content,
        prompt=prompt,
        evidence=AcquiredEvidence(
            "exec_shell_full_command",
            {"command": f"rg -n -F -- {shlex.quote(query)} {shlex.quote(source)}"},
            content,
            source,
        ),
    )


def build_list_action(arguments: str, *, workdir: Path) -> CommandAction:
    values = _split(arguments, usage="/ls [path]")
    if isinstance(values, CommandAction):
        return values
    if len(values) > 1:
        return _usage_error("expected at most one path", "/ls [path]")
    return CommandAction(execute_list_directory({"path": values[0] if values else "."}, workdir=workdir))


def build_models_action(arguments: str) -> CommandAction:
    values = _split(arguments, usage="/models")
    if isinstance(values, CommandAction):
        return values
    if values:
        return _usage_error("this command takes no arguments", "/models")
    try:
        result = discover_models(build_bin=resolve_native_build_bin())
    except (OSError, RuntimeError) as exc:
        return CommandAction(f"error: model discovery unavailable: {exc}")
    return CommandAction(format_model_discovery(result))


def _split(arguments: str, *, usage: str) -> list[str] | CommandAction:
    try:
        return shlex.split(arguments, posix=True)
    except ValueError as exc:
        return _usage_error(str(exc), usage)


def _usage_error(message: str, usage: str) -> CommandAction:
    return CommandAction(f"error: {message}\nusage: {usage}")


def _with_optional_prompt(content: str, *, prompt: str | None, evidence: AcquiredEvidence) -> CommandAction:
    if prompt is None:
        return CommandAction(content)
    return CommandAction(content, prompt=prompt, evidence=evidence)


def _url_state(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        return "valid"
    if "://" in value or parsed.scheme.lower() in {"http", "https"}:
        return "invalid"
    return "local"


def _is_source_token(value: str, *, workdir: Path) -> bool:
    if _url_state(value) != "local":
        return True
    candidate = Path(value)
    return candidate.is_absolute() or (workdir / candidate).exists() or "/" in value or bool(candidate.suffix)


def _url_snapshot(url: str, text: str) -> FullDocumentSnapshot:
    encoded = text.encode("utf-8")
    return FullDocumentSnapshot(
        path=url,
        byte_count=len(encoded),
        char_count=len(text),
        line_count=len(text.splitlines()),
        sha256=hashlib.sha256(encoded).hexdigest(),
        device=0,
        inode=0,
        mtime_ns=0,
        ctime_ns=0,
        content=text,
    )


def _url_search_answer(result, *, fetched: str) -> str:
    truncated = "text_truncated: true" in fetched
    coverage = "partial" if truncated else "complete"
    lines = [
        (
            "URL search coverage: "
            f"retrieval_coverage={coverage}; search_coverage=complete_retrieved_text; "
            f"url={result.path}; searched_term={result.searched_terms[0]!r}; "
            f"total_matches={result.total_matches}; results_truncated={str(result.results_truncated).lower()}."
        )
    ]
    if result.total_matches == 0:
        lines.append("The exact string does not occur in the retrieved page text.")
        return "\n".join(lines)
    lines.append("Exact evidence windows:")
    for window in result.windows:
        lines.extend((f"- lines {window.start_line}-{window.end_line}:", window.text))
    return "\n".join(lines)
