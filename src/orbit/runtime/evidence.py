from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
import shlex
import uuid
from pathlib import Path
from urllib.parse import urlparse

from orbit.runtime.sessions import DEFAULT_SESSION_ROOT, SessionStore


RAW_REF_PREFIX = "evidence:"
INDEX_FILENAME = "index.json"
HEAD_CHARS = 700
TAIL_CHARS = 300
COMPAT_INLINE_CHARS = 1200
RECENT_EVIDENCE_LIMIT = 2
RAW_EXCERPT_CHARS = 900
COMPACT_FINAL_RAW_EXCERPT_CHARS = 500
POST_TOOL_ROUTE_TEXT_CHARS = 180
POST_TOOL_ROUTE_OUTPUT_CHARS = 80
ROUTE_OUTPUT_EXCERPT_CHARS = 400
WEB_FINAL_SNIPPET_CHARS = 220


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    tool_name: str
    kind: str
    raw_ref: str
    raw_sha256: str
    raw_chars: int
    raw_lines: int
    status: str
    metadata: dict[str, object]
    route_card: str
    final_card: str


class EvidenceStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.records: dict[str, EvidenceRecord] = {}
        self.raw_cache: dict[str, str] = {}

    @classmethod
    def for_workdir(cls, workdir: Path, *, root: Path = DEFAULT_SESSION_ROOT) -> "EvidenceStore":
        session_path = SessionStore.for_workdir(workdir, root=root).path
        store = cls(session_path.parent / f"{session_path.stem}.evidence")
        store.load_index()
        return store

    def clear_memory(self) -> None:
        self.records.clear()
        self.raw_cache.clear()

    def load_index(self) -> None:
        self.records.clear()
        for evidence_id, value in self._load_index().items():
            if not isinstance(evidence_id, str) or not isinstance(value, dict):
                continue
            record = _record_from_index(value)
            if record is not None:
                self.records[record.evidence_id] = record

    def add(self, tool_name: str, content: str, *, metadata: dict[str, object] | None = None) -> EvidenceRecord:
        record = build_evidence_record(tool_name, content, metadata or {})
        self.records[record.evidence_id] = record
        self.raw_cache[record.evidence_id] = content
        try:
            self.save(record, content)
        except OSError:
            failed = _record_with_sidecar_status(record, "sidecar_write_failed")
            self.records[failed.evidence_id] = failed
            return failed
        return record

    def save(self, record: EvidenceRecord, content: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{record.evidence_id}.txt").write_text(content, encoding="utf-8")
        index = self._load_index()
        index[record.evidence_id] = asdict(record)
        tmp = self.root / f"{INDEX_FILENAME}.tmp"
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.root / INDEX_FILENAME)

    def load_raw(self, evidence_id: str) -> str:
        if evidence_id in self.raw_cache:
            return self.raw_cache[evidence_id]
        path = self.root / f"{evidence_id}.txt"
        if not path.exists():
            raise FileNotFoundError(f"missing evidence sidecar: {evidence_id}")
        content = path.read_text(encoding="utf-8")
        self.raw_cache[evidence_id] = content
        return content

    def recent_records(self, limit: int = RECENT_EVIDENCE_LIMIT) -> list[EvidenceRecord]:
        if limit <= 0:
            return []
        return list(self.records.values())[-limit:]

    def raw_excerpt(self, record: EvidenceRecord, *, max_chars: int = RAW_EXCERPT_CHARS) -> str:
        try:
            raw = self.load_raw(record.evidence_id)
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            return f"raw_evidence_unavailable: {record.raw_ref}"
        return _bounded_text(raw, max_chars)

    def _load_index(self) -> dict[str, object]:
        path = self.root / INDEX_FILENAME
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}


def build_evidence_record(tool_name: str, content: str, metadata: dict[str, object] | None = None) -> EvidenceRecord:
    metadata = dict(metadata or {})
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    evidence_id = f"ev_{uuid.uuid4().hex[:12]}_{digest[:16]}"
    kind = _classify_kind(tool_name, content, metadata)
    status = _status_for(kind, content, metadata)
    enriched = _enriched_metadata(kind, content, metadata)
    raw_ref = f"{RAW_REF_PREFIX}{evidence_id}"
    raw_lines = len(content.splitlines())
    base = EvidenceRecord(
        evidence_id=evidence_id,
        tool_name=tool_name,
        kind=kind,
        raw_ref=raw_ref,
        raw_sha256=digest,
        raw_chars=len(content),
        raw_lines=raw_lines,
        status=status,
        metadata=enriched,
        route_card="",
        final_card="",
    )
    return EvidenceRecord(
        **{**asdict(base), "route_card": route_card(base), "final_card": final_card(base)}
    )


def _record_from_index(value: dict[str, object]) -> EvidenceRecord | None:
    try:
        metadata = value.get("metadata")
        record = EvidenceRecord(
            evidence_id=str(value["evidence_id"]),
            tool_name=str(value["tool_name"]),
            kind=str(value["kind"]),
            raw_ref=str(value["raw_ref"]),
            raw_sha256=str(value["raw_sha256"]),
            raw_chars=int(value["raw_chars"]),
            raw_lines=int(value["raw_lines"]),
            status=str(value["status"]),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
            route_card="",
            final_card="",
        )
    except (KeyError, TypeError, ValueError):
        return None
    return EvidenceRecord(**{**asdict(record), "route_card": route_card(record), "final_card": final_card(record)})


def route_card(record: EvidenceRecord) -> str:
    lines = [
        "tool_evidence_card: true",
        f"tool: {record.tool_name}",
        f"kind: {record.kind}",
        f"status: {record.status}",
        f"raw_ref: {record.raw_ref}",
        f"sha256: {record.raw_sha256[:16]}",
        f"size: {record.raw_chars} chars, {record.raw_lines} lines",
    ]
    lines.extend(_card_metadata_lines(record, compact=True))
    return "\n".join(lines)


def final_card(record: EvidenceRecord) -> str:
    lines = [route_card(record), "evidence_excerpt:"]
    for excerpt in _excerpt_lines(record):
        lines.append(excerpt)
    return "\n".join(lines)


def tool_evidence_ref(record: EvidenceRecord) -> str:
    lines = [
        "tool_evidence_ref: true",
        f"evidence_id: {record.evidence_id}",
        f"raw_ref: {record.raw_ref}",
        f"tool: {record.tool_name}",
        f"kind: {record.kind}",
        f"status: {record.status}",
        f"size: {record.raw_chars} chars, {record.raw_lines} lines",
        f"sha256: {record.raw_sha256[:16]}",
    ]
    compat = _compat_excerpt(record)
    if compat:
        lines.append("compat_excerpt:")
        lines.append(compat)
    return "\n".join(lines)


def build_route_evidence_context(store: EvidenceStore | None, *, limit: int = RECENT_EVIDENCE_LIMIT) -> str | None:
    records = store.recent_records(limit) if store is not None else []
    if not records:
        return None
    parts = ["available_evidence:"]
    for index, record in enumerate(records, start=1):
        parts.append(f"- evidence {index}:")
        parts.append(record.route_card)
    return "\n".join(parts)


def build_post_tool_route_evidence_context(store: EvidenceStore | None, *, limit: int = RECENT_EVIDENCE_LIMIT) -> str | None:
    records = store.recent_records(limit) if store is not None else []
    if not records:
        return None
    parts = ["available_evidence:"]
    for record in records:
        parts.append(_post_tool_route_card(record))
    return "\n".join(parts)


def build_final_evidence_context(store: EvidenceStore | None, *, limit: int = RECENT_EVIDENCE_LIMIT) -> str | None:
    records = store.recent_records(limit) if store is not None else []
    if not records:
        return None
    parts = ["evidence_context:"]
    for index, record in enumerate(records, start=1):
        parts.append(f"- evidence {index}:")
        parts.append(record.final_card)
        if _final_context_needs_raw_excerpt(record):
            parts.append("bounded_raw_excerpt:")
            parts.append(store.raw_excerpt(record) if store is not None else f"raw_evidence_unavailable: {record.raw_ref}")
    return "\n".join(parts)


def build_compact_final_evidence_context(store: EvidenceStore | None, *, limit: int = 1) -> str | None:
    records = store.recent_records(limit) if store is not None else []
    if not records:
        return None
    parts = ["evidence_context:"]
    for index, record in enumerate(records, start=1):
        parts.append(f"- evidence {index}:")
        parts.append(_compact_final_card(record))
        if _compact_final_context_needs_raw_excerpt(record):
            parts.append("bounded_raw_excerpt:")
            parts.append(
                store.raw_excerpt(record, max_chars=COMPACT_FINAL_RAW_EXCERPT_CHARS)
                if store is not None
                else f"raw_evidence_unavailable: {record.raw_ref}"
            )
    return "\n".join(parts)


def build_web_final_evidence_context(store: EvidenceStore | None) -> str | None:
    record = next(iter(store.recent_records(1)), None) if store is not None else None
    if record is None or record.kind != "web_search":
        return None
    snippets = record.metadata.get("top_snippets")
    if not isinstance(snippets, list) or not snippets:
        return None
    parts = [
        "evidence_context:",
        "- evidence 1:",
        "tool_evidence_card: true",
        "web_search_evidence: true",
        f"tool: {record.tool_name}",
        f"kind: {record.kind}",
        f"status: {record.status}",
        f"raw_ref: {record.raw_ref}",
        f"sha256: {record.raw_sha256[:16]}",
        f"size: {record.raw_chars} chars, {record.raw_lines} lines",
    ]
    for key in ("query", "result_count", "top_domains", "top_titles"):
        value = record.metadata.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            parts.append(f"{key}: {'; '.join(str(item) for item in value[:3])}")
        else:
            parts.append(f"{key}: {value}")
    parts.append("top_snippets:")
    for item in snippets[:3]:
        parts.append(f"- {_bounded_text(str(item), WEB_FINAL_SNIPPET_CHARS)}")
    return "\n".join(parts)


def _record_with_sidecar_status(record: EvidenceRecord, status: str) -> EvidenceRecord:
    metadata = dict(record.metadata)
    metadata["sidecar_status"] = status
    updated = EvidenceRecord(
        evidence_id=record.evidence_id,
        tool_name=record.tool_name,
        kind=record.kind,
        raw_ref=record.raw_ref,
        raw_sha256=record.raw_sha256,
        raw_chars=record.raw_chars,
        raw_lines=record.raw_lines,
        status=record.status,
        metadata=metadata,
        route_card="",
        final_card="",
    )
    return EvidenceRecord(**{**asdict(updated), "route_card": route_card(updated), "final_card": final_card(updated)})


def _classify_kind(tool_name: str, content: str, metadata: dict[str, object]) -> str:
    command = str(metadata.get("command") or "")
    if "web_search_results: true" in content or "orbit-web-search" in command:
        return "web_search"
    if (
        tool_name == "fetch_url"
        or "shell_output_html_cleaned: true" in content
        or re.search(r"^status:\s*\w+", content, re.MULTILINE)
    ):
        return "fetch"
    if re.search(r"\b(?:rg|grep|find)\b", command):
        return "grep_search"
    if tool_name == "exec_shell_full_command":
        return "shell"
    return "unknown"


def _status_for(kind: str, content: str, metadata: dict[str, object]) -> str:
    if kind == "web_search":
        if "results: none" in content:
            return "none"
        if "web_search_results: true" in content:
            return "ok"
    if "shell_command_failed: true" in content:
        return "error"
    status = metadata.get("status")
    return str(status) if isinstance(status, str) and status else "ok"


def _enriched_metadata(kind: str, content: str, metadata: dict[str, object]) -> dict[str, object]:
    enriched = dict(metadata)
    if kind == "web_search":
        enriched.update(_web_metadata(content))
    elif kind in {"shell", "grep_search", "unknown"}:
        enriched.update(_shell_metadata(content))
    if kind == "grep_search":
        enriched.update(_grep_metadata(content, metadata))
    enriched = _enrich_excerpts(content, enriched)
    return enriched


def _web_metadata(content: str) -> dict[str, object]:
    query = _line_value(content, "query")
    titles = re.findall(r"^\d+\.\s+title:\s*(.+)$", content, re.MULTILINE)
    urls = re.findall(r"^\s+url:\s*(.+)$", content, re.MULTILINE)
    snippets = re.findall(r"^\s+snippet:\s*(.+)$", content, re.MULTILINE)
    domains = [_domain(url) for url in urls]
    return {
        "query": query or "",
        "result_count": len(titles),
        "top_titles": titles[:3],
        "top_domains": [domain for domain in domains[:3] if domain],
        "top_snippets": snippets[:3],
    }


def _shell_metadata(content: str) -> dict[str, object]:
    stdout = _section_text(content, "STDOUT:", "STDERR:")
    stderr = _section_text(content, "STDERR:", None)
    if stdout is None and stderr is None:
        stdout = content
    return {
        "exit_code": _line_value(content, "exit_code") or "",
        "stdout_chars": _section_chars(content, "STDOUT:", "STDERR:"),
        "stderr_chars": _section_chars(content, "STDERR:", None),
        "stdout_excerpt": _route_output_excerpt(stdout),
        "stderr_excerpt": _route_output_excerpt(stderr),
    }


def _grep_metadata(content: str, metadata: dict[str, object]) -> dict[str, object]:
    stdout = _section_text(content, "STDOUT:", "STDERR:")
    source = stdout if stdout is not None else content
    lines = [line.strip() for line in source.splitlines() if line.strip() and line.strip() != "(empty)"]
    matches: list[str] = []
    file_paths: list[str] = []
    for line in lines:
        parsed = _parse_grep_line(line)
        if parsed is None:
            if _looks_like_path(line):
                file_paths.append(line)
            continue
        path, line_number, excerpt = parsed
        file_paths.append(path)
        if line_number:
            matches.append(f"{path}:{line_number}: {excerpt}")
        else:
            matches.append(f"{path}: {excerpt}")
    unique_paths = _unique_preserving_order(file_paths)
    return {
        "query": _grep_query_from_command(str(metadata.get("command") or "")),
        "match_count": len(matches) if matches else "",
        "files_count": len(unique_paths) if unique_paths else "",
        "file_paths": unique_paths[:8],
        "first_matches": matches[:5],
    }


def _card_metadata_lines(record: EvidenceRecord, *, compact: bool) -> list[str]:
    keys = (
        "command",
        "query",
        "result_count",
        "top_domains",
        "top_titles",
        "file_paths",
        "exit_code",
        "stdout_chars",
        "stderr_chars",
        "stdout_excerpt",
        "stderr_excerpt",
        "match_count",
        "files_count",
        "first_matches",
        "sidecar_status",
    )
    lines: list[str] = []
    for key in keys:
        value = record.metadata.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            joined = "; ".join(str(item) for item in value[:2 if compact else 4])
            lines.append(f"{key}: {joined}")
        else:
            lines.append(f"{key}: {value}")
    return lines


def _compact_final_card(record: EvidenceRecord) -> str:
    lines = [
        "tool_evidence_card: true",
        f"tool: {record.tool_name}",
        f"kind: {record.kind}",
        f"status: {record.status}",
        f"raw_ref: {record.raw_ref}",
        f"sha256: {record.raw_sha256[:16]}",
        f"size: {record.raw_chars} chars, {record.raw_lines} lines",
    ]
    lines.extend(_card_metadata_lines(record, compact=True))
    return "\n".join(lines)


def _final_context_needs_raw_excerpt(record: EvidenceRecord) -> bool:
    if record.kind == "web_search" and record.metadata.get("top_snippets"):
        return False
    if record.kind == "grep_search" and (
        record.metadata.get("first_matches") or record.metadata.get("file_paths")
    ):
        return False
    return True


def _compact_final_context_needs_raw_excerpt(record: EvidenceRecord) -> bool:
    if record.status != "error" and record.kind in {"shell", "unknown"} and (
        record.metadata.get("stdout_excerpt") or record.metadata.get("stderr_excerpt")
    ):
        return False
    return _final_context_needs_raw_excerpt(record)


def _post_tool_route_card(record: EvidenceRecord) -> str:
    fields = [
        "tool_evidence_card=true",
        f"t={_short_tool_name(record.tool_name)}",
        f"k={record.kind}",
        f"st={record.status}",
        f"raw_ref={record.raw_ref}",
        f"hash={record.raw_sha256[:16]}",
        f"size={record.raw_chars}c/{record.raw_lines}l",
    ]
    if record.kind == "grep_search":
        keys = ("query", "files_count", "file_paths")
    elif record.kind == "web_search":
        keys = ("query", "result_count", "top_domains")
    elif record.kind in {"shell", "unknown"}:
        keys = ("exit_code", "stdout_chars", "stderr_chars", "stdout_excerpt", "stderr_excerpt")
    else:
        keys = ("query", "command", "result_count", "top_domains", "top_titles")
    for key in keys:
        value = record.metadata.get(key)
        if value in (None, "", [], {}):
            continue
        if key in {"stdout_chars", "stderr_chars"} and value == 0:
            continue
        max_chars = (
            POST_TOOL_ROUTE_OUTPUT_CHARS
            if key in {"stdout_excerpt", "stderr_excerpt"}
            else POST_TOOL_ROUTE_TEXT_CHARS
        )
        if isinstance(value, list):
            items = [_bounded_text(str(item), max_chars) for item in value[:1]]
            fields.append(f"{_route_key(key)}={' | '.join(items)}")
        else:
            fields.append(f"{_route_key(key)}={_bounded_text(str(value), max_chars)}")
    return "; ".join(fields)


def _short_tool_name(tool_name: str) -> str:
    return "shell" if tool_name == "exec_shell_full_command" else tool_name


def _route_key(key: str) -> str:
    return {
        "command": "cmd",
        "exit_code": "exit",
        "stdout_chars": "stdout",
        "stderr_chars": "stderr",
        "result_count": "results",
        "match_count": "matches",
        "files_count": "files",
        "file_paths": "paths",
        "top_domains": "domains",
        "top_titles": "titles",
        "top_snippets": "snippets",
    }.get(key, key)


def _excerpt_lines(record: EvidenceRecord) -> list[str]:
    snippets = record.metadata.get("top_snippets")
    if isinstance(snippets, list) and snippets:
        return [f"- {str(item)}" for item in snippets[:3]]
    matches = record.metadata.get("first_matches")
    if isinstance(matches, list) and matches:
        return [f"- {str(item)}" for item in matches[:5]]
    paths = record.metadata.get("file_paths")
    if isinstance(paths, list) and paths:
        return [f"- {str(item)}" for item in paths[:5]]
    head = record.metadata.get("head_excerpt")
    tail = record.metadata.get("tail_excerpt")
    lines = []
    if isinstance(head, str) and head:
        lines.append(head)
    if isinstance(tail, str) and tail and tail != head:
        lines.append("[tail]")
        lines.append(tail)
    return lines or ["[no textual excerpt]"]


def _line_value(content: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else None


def _section_chars(content: str, start: str, end: str | None) -> int:
    section = _section_text(content, start, end)
    if section is None:
        return 0
    stripped = section.strip()
    return 0 if stripped == "(empty)" else len(stripped)


def _section_text(content: str, start: str, end: str | None) -> str | None:
    if start not in content:
        return None
    section = content.split(start, 1)[1]
    if end and end in section:
        section = section.split(end, 1)[0]
    return section.strip()


def _parse_grep_line(line: str) -> tuple[str, str, str] | None:
    match = re.match(r"^(.+?):(\d+):(.*)$", line)
    if match and _looks_like_path(match.group(1)):
        return match.group(1), match.group(2), match.group(3).strip()
    match = re.match(r"^(.+?):(.*)$", line)
    if match and _looks_like_path(match.group(1)):
        return match.group(1), "", match.group(2).strip()
    return None


def _looks_like_path(value: str) -> bool:
    if not value or value.startswith(("shell_", "exit_code", "STDOUT", "STDERR")):
        return False
    if any(char.isspace() for char in value):
        return False
    return "/" in value or value.startswith(".") or "." in value


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _grep_query_from_command(command: str) -> str:
    if not command:
        return ""
    try:
        parts = shlex.split(command)
    except ValueError:
        return ""
    if not parts:
        return ""
    for index, part in enumerate(parts):
        if part in {"grep", "rg"} or part.endswith(("/grep", "/rg")):
            for candidate in parts[index + 1 :]:
                if candidate.startswith("-"):
                    continue
                return candidate
    return ""


def _domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or parsed.path.split("/", 1)[0]


def _text_excerpts(content: str) -> dict[str, str]:
    stripped = content.strip()
    if len(stripped) <= HEAD_CHARS + TAIL_CHARS + 20:
        return {"head_excerpt": stripped, "tail_excerpt": ""}
    return {
        "head_excerpt": stripped[:HEAD_CHARS].rstrip(),
        "tail_excerpt": stripped[-TAIL_CHARS:].lstrip(),
    }


def _bounded_text(content: str, max_chars: int) -> str:
    stripped = content.strip()
    if len(stripped) <= max_chars:
        return stripped
    marker = "\n[...bounded...]\n"
    if max_chars <= len(marker) + 20:
        return stripped[:max_chars].rstrip()
    remaining = max_chars - len(marker)
    tail_chars = min(TAIL_CHARS, max(10, remaining // 2))
    head_chars = max(0, remaining - tail_chars)
    return f"{stripped[:head_chars].rstrip()}{marker}{stripped[-tail_chars:].lstrip()}"


def _route_output_excerpt(content: str | None) -> str:
    if content is None:
        return ""
    stripped = content.strip()
    if not stripped or stripped == "(empty)":
        return ""
    return _bounded_text(stripped, ROUTE_OUTPUT_EXCERPT_CHARS).replace("\n", " | ")


def _compat_excerpt(record: EvidenceRecord) -> str:
    if record.raw_chars > COMPAT_INLINE_CHARS:
        return ""
    head = record.metadata.get("head_excerpt")
    return head if isinstance(head, str) else ""


def _enrich_excerpts(content: str, metadata: dict[str, object]) -> dict[str, object]:
    enriched = dict(metadata)
    enriched.update(_text_excerpts(content))
    return enriched
