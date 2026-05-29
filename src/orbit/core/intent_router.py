from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import re

from .code_review_signals import (
    CODE_FILE_EXTENSIONS,
    CODE_REVIEW_REQUEST_HINTS,
    has_code_file_extension,
    has_code_language_hint,
)
from .text_utils import word_tokens


INTENT_CLASS_CHAT_GENERAL = "chat_general"
INTENT_CLASS_WORKSPACE_DISCOVERY = "workspace_discovery"
INTENT_CLASS_FILE_READING = "file_reading"
INTENT_CLASS_FILE_EDITING = "file_editing"
INTENT_CLASS_MACHINE_INSPECTION = "machine_inspection"
INTENT_CLASS_WEB_LOOKUP = "web_lookup"
INTENT_CLASS_URL_INSPECTION = "url_inspection"
INTENT_CLASS_SHELL_TASK = "shell_task"
INTENT_CLASS_CODEBASE_INSPECTION = "codebase_inspection"
INTENT_CLASS_BINARY_OR_PDF_ANALYSIS = "binary_or_pdf_analysis"
INTENT_CLASS_KNOWLEDGE_QUESTION = "knowledge_question"
INTENT_CLASS_AMBIGUOUS = "ambiguous"

INTENT_CODEBASE_INSPECTION = "codebase_inspection"
INTENT_TEXT_DOCUMENT_ANALYSIS = "text_document_analysis"
INTENT_BINARY_OR_PDF_ANALYSIS = "binary_or_pdf_analysis"
INTENT_CURRENT_FACTUAL_LOOKUP = "current_factual_lookup"
INTENT_GENERAL_KNOWLEDGE = "general_knowledge"
INTENT_CHITCHAT = "chitchat"
INTENT_FILE_EDIT = "file_edit"
INTENT_BOUNDED_COMMAND = "bounded_command"
INTENT_AMBIGUOUS = "ambiguous"

_CODEBASE_HINTS = (
    "codebase",
    "project",
    "progetto",
    "repo",
    "repository",
    "architettura",
    "architecture",
    "review",
    "code review",
    "review the code",
    "review this codebase",
    "rivedi il codice",
    "fai code review",
    "analizza il codice",
    "analyze the code",
    "tests",
)
_TEXT_HINTS = (
    "file",
    "files",
    "directory",
    "folder",
    "read",
    "open",
    "leggi",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    "documento",
    "document",
    "riassumi",
    "summarize",
    "read file",
)
_BINARY_HINTS = (
    "binary file",
    "executable",
    "shared library",
    ".pdf",
    ".apk",
    ".zip",
    ".jar",
    ".aar",
    ".ipa",
    ".dex",
    ".so",
    ".bin",
    ".exe",
    ".dll",
    "strings",
    "pdftotext",
    "binary",
    "malware",
    "static triage",
    "static analysis",
    "suspicious file",
    "sample",
)
_BINARY_FORMAT_TOKENS = (
    "apk",
    "pdf",
    "zip",
    "jar",
    "aar",
    "ipa",
    "dex",
    "elf",
    "exe",
    "dll",
    "so",
    "bin",
)
_FACTUAL_HINTS = (
    "meteo",
    "weather",
    "tempo oggi",
    "previsioni",
    "forecast",
    "news",
    "notizie",
    "chi è",
    "who is",
    "informazioni su",
    "info su",
    "online",
    "web",
    "internet",
    "url",
    "link",
    "sito",
    "site",
)
_GENERAL_KNOWLEDGE_HINTS = (
    "why",
    "perchè",
    "perche",
    "how does",
    "how do",
    "spiega",
    "explain",
    "say only",
    "say exactly",
    "reply only",
    "reply exactly",
    "answer only",
    "answer exactly",
    "respond only",
    "respond exactly",
    "rispondi solo",
    "rispondi soltanto",
    "rispondi esattamente",
)
_CODE_GENERATION_HINTS = (
    "code",
    "codice",
    "function",
    "funzione",
    "script",
    "snippet",
    "program",
    "programma",
    "class",
    "method",
    "algorithm",
    "algoritmo",
    "python",
    "javascript",
    "typescript",
    "java",
    "c++",
    "cpp",
    "c#",
    "ps1",
    "powershell",
)
_CREATIVE_GENERATION_HINTS = (
    "story",
    "stories",
    "poem",
    "poetry",
    "phrase",
    "sentence",
    "joke",
    "racconto",
    "storia",
    "poesia",
    "poesie",
    "frase",
    "battuta",
)
_CHITCHAT_HINTS = (
    "ciao",
    "hello",
    "hi",
    "hey",
    "buongiorno",
    "buonasera",
    "thanks",
    "grazie",
)
_EDIT_HINTS = (
    "write",
    "append",
    "create",
    "edit",
    "modify",
    "update",
    "replace",
    "save",
    "delete",
    "remove",
    "mkdir",
    "rmdir",
    "rm",
    "scrivi",
    "crea",
    "modifica",
    "aggiorna",
    "aggiungi",
    "cancella",
    "rimuovi",
    "elimina",
)
_EDIT_OBJECT_HINTS = (
    "file",
    "files",
    "folder",
    "directory",
    "cartella",
    "cartelle",
    "dir",
    "path",
    "named",
    "name it",
    "called",
    "save",
    "salva",
    "inside",
    "into",
)
_COMMAND_HINTS = (
    "bash",
    "shell",
    "command",
    "comando",
    "terminal",
    "grep",
    "find",
    "ls",
    "pwd",
    "sed",
    "awk",
    "unzip",
    "python3 -c",
)
_SYSTEM_INFO_MACHINE_NOUNS = (
    "pc",
    "computer",
    "machine",
    "macchina",
    "laptop",
    "notebook",
    "portatile",
    "system",
    "sistema",
)
_SYSTEM_INFO_QUERY_HINTS = (
    "configurazione",
    "configuration",
    "spec",
    "specs",
    "hardware",
    "system info",
    "system information",
    "informazioni di sistema",
    "caratteristiche",
)


@dataclass(frozen=True)
class IntentRoute:
    intent: str
    intent_class: str
    reason: str


@dataclass(frozen=True)
class IntentRule:
    intent: str
    intent_class: str
    reason: str
    matcher: Callable[["_IntentText"], bool]


@dataclass(frozen=True)
class _IntentText:
    text: str
    tokens: tuple[str, ...]
    token_set: frozenset[str]


def route_intent(user_input: str) -> IntentRoute:
    lowered = user_input.strip().lower()
    if not lowered:
        return IntentRoute(intent=INTENT_AMBIGUOUS, intent_class=INTENT_CLASS_AMBIGUOUS, reason="empty input fallback")
    tokens = tuple(word_tokens(lowered))
    intent_text = _IntentText(text=lowered, tokens=tokens, token_set=frozenset(tokens))
    for rule in _INTENT_RULES:
        if rule.matcher(intent_text):
            return IntentRoute(intent=rule.intent, intent_class=rule.intent_class, reason=rule.reason)
    return IntentRoute(intent=INTENT_AMBIGUOUS, intent_class=INTENT_CLASS_AMBIGUOUS, reason="ambiguous request fallback")


def _intent_rules() -> tuple[IntentRule, ...]:
    return (
        IntentRule(
            intent=INTENT_CHITCHAT,
            intent_class=INTENT_CLASS_CHAT_GENERAL,
            reason="chitchat hints",
            matcher=lambda text: _looks_like_pure_chitchat(text),
        ),
        IntentRule(
            intent=INTENT_CHITCHAT,
            intent_class=INTENT_CLASS_CHAT_GENERAL,
            reason="assistant persona hints",
            matcher=lambda text: _looks_like_assistant_persona_request(text),
        ),
        IntentRule(
            intent=INTENT_CURRENT_FACTUAL_LOOKUP,
            intent_class=INTENT_CLASS_URL_INSPECTION,
            reason="explicit url inspection hints",
            matcher=lambda text: _has_explicit_web_url(text) and _looks_like_explicit_web_fetch_request(text),
        ),
        IntentRule(
            intent=INTENT_TEXT_DOCUMENT_ANALYSIS,
            intent_class=INTENT_CLASS_FILE_READING,
            reason="local filesystem metadata hints",
            matcher=lambda text: _looks_like_local_filesystem_metadata_request(text),
        ),
        IntentRule(
            intent=INTENT_BOUNDED_COMMAND,
            intent_class=INTENT_CLASS_MACHINE_INSPECTION,
            reason="storage inspection hints",
            matcher=lambda text: _looks_like_storage_inspection_request(text),
        ),
        IntentRule(
            intent=INTENT_BOUNDED_COMMAND,
            intent_class=INTENT_CLASS_SHELL_TASK,
            reason="base64 transform hints",
            matcher=lambda text: _looks_like_base64_transform_request(text),
        ),
        IntentRule(
            intent=INTENT_CODEBASE_INSPECTION,
            intent_class=INTENT_CLASS_CODEBASE_INSPECTION,
            reason="workspace security inspection hints",
            matcher=lambda text: _looks_like_workspace_security_search_request(text),
        ),
        IntentRule(
            intent=INTENT_CURRENT_FACTUAL_LOOKUP,
            intent_class=INTENT_CLASS_WEB_LOOKUP,
            reason="factual web lookup hints",
            matcher=lambda text: _looks_like_current_factual_lookup(text),
        ),
        IntentRule(
            intent=INTENT_GENERAL_KNOWLEDGE,
            intent_class=INTENT_CLASS_KNOWLEDGE_QUESTION,
            reason="general knowledge hints",
            matcher=lambda text: _looks_like_general_knowledge_request(text),
        ),
        IntentRule(
            intent=INTENT_CHITCHAT,
            intent_class=INTENT_CLASS_CHAT_GENERAL,
            reason="code generation hints",
            matcher=lambda text: _looks_like_code_generation_request(text),
        ),
        IntentRule(
            intent=INTENT_CHITCHAT,
            intent_class=INTENT_CLASS_CHAT_GENERAL,
            reason="creative generation hints",
            matcher=lambda text: _looks_like_creative_generation_request(text),
        ),
        IntentRule(
            intent=INTENT_FILE_EDIT,
            intent_class=INTENT_CLASS_FILE_EDITING,
            reason="file edit hints",
            matcher=lambda text: _looks_like_file_edit_request(text),
        ),
        IntentRule(
            intent=INTENT_BINARY_OR_PDF_ANALYSIS,
            intent_class=INTENT_CLASS_BINARY_OR_PDF_ANALYSIS,
            reason="binary/pdf hints",
            matcher=lambda text: (
                _matches_any(text, _BINARY_HINTS)
                or _looks_like_binary_analysis_request(text)
                or _mentions_binary_format_token(text)
            ),
        ),
        IntentRule(
            intent=INTENT_BOUNDED_COMMAND,
            intent_class=INTENT_CLASS_MACHINE_INSPECTION,
            reason="machine inspection hints",
            matcher=lambda text: _looks_like_system_info_request(text),
        ),
        IntentRule(
            intent=INTENT_BOUNDED_COMMAND,
            intent_class=INTENT_CLASS_SHELL_TASK,
            reason="shell task hints",
            matcher=lambda text: (
                not _looks_like_explicit_code_file_review_request(text)
                and (
                    _matches_any(text, _COMMAND_HINTS)
                    or _looks_like_shell_command(text)
                    or _looks_like_arithmetic_request(text)
                )
            ),
        ),
        IntentRule(
            intent=INTENT_CODEBASE_INSPECTION,
            intent_class=INTENT_CLASS_CODEBASE_INSPECTION,
            reason="codebase hints",
            matcher=lambda text: _looks_like_codebase_inspection_request(text) or _looks_like_explicit_code_file_review_request(text),
        ),
        IntentRule(
            intent=INTENT_TEXT_DOCUMENT_ANALYSIS,
            intent_class=INTENT_CLASS_WORKSPACE_DISCOVERY,
            reason="workspace discovery hints",
            matcher=lambda text: _looks_like_workspace_discovery_request(text),
        ),
        IntentRule(
            intent=INTENT_TEXT_DOCUMENT_ANALYSIS,
            intent_class=INTENT_CLASS_FILE_READING,
            reason="file reading hints",
            matcher=lambda text: _matches_any(text, _TEXT_HINTS) or _looks_like_text_path_request(text),
        ),
    )


def _matches_any(intent_text: _IntentText, hints: tuple[str, ...]) -> bool:
    text = intent_text.text
    literal_hints, word_patterns = _hint_match_plan(hints)
    for hint in literal_hints:
        if hint in text:
            return True
    for pattern in word_patterns:
        if pattern.search(text):
            return True
    return False


@lru_cache(maxsize=None)
def _hint_match_plan(hints: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[re.Pattern[str], ...]]:
    literal_hints: list[str] = []
    word_patterns: list[re.Pattern[str]] = []
    for hint in hints:
        if any(char.isspace() for char in hint) or any(char in "._/-\\" for char in hint):
            literal_hints.append(hint)
        else:
            word_patterns.append(re.compile(rf"\b{re.escape(hint)}\b"))
    return tuple(literal_hints), tuple(word_patterns)


def _looks_like_shell_command(intent_text: _IntentText) -> bool:
    return bool(re.search(r"\b(ls|find|grep|pwd|cat|strings|sed|awk|unzip|python3)\b", intent_text.text))


def _looks_like_system_info_request(intent_text: _IntentText) -> bool:
    token_set = intent_text.token_set
    machine_tokens = {"pc", "computer", "machine", "macchina", "laptop", "notebook", "portatile", "system", "sistema"}
    info_tokens = {"config", "configuration", "configurazione", "spec", "specs", "hardware", "info", "information", "informazioni"}
    own_tokens = {"this", "thise", "my", "questo", "questa", "mio", "mia"}
    if _matches_any(intent_text, ("system info", "system information", "informazioni di sistema")):
        return True
    if _matches_any(intent_text, _SYSTEM_INFO_QUERY_HINTS):
        return bool(token_set & machine_tokens) or bool(token_set & own_tokens)
    return bool(token_set & machine_tokens) and bool(token_set & (info_tokens | own_tokens))


def _looks_like_storage_inspection_request(intent_text: _IntentText) -> bool:
    text = intent_text.text
    token_set = intent_text.token_set
    storage_phrases = (
        "disk usage",
        "available storage",
        "free space",
        "storage space",
        "filesystem capacity",
        "spazio disponibile",
        "spazio libero",
        "uso disco",
    )
    if any(phrase in text for phrase in storage_phrases):
        return True
    storage_tokens = {"disk", "storage", "filesystem", "spazio", "disco"}
    query_tokens = {"usage", "available", "free", "capacity", "space", "show", "display", "available", "disponibile", "libero"}
    return bool(token_set & storage_tokens) and bool(token_set & query_tokens)


def _looks_like_arithmetic_request(intent_text: _IntentText) -> bool:
    token_set = intent_text.token_set
    if not (token_set & {"calculate", "calcolare", "compute", "calcola", "divide", "divided", "moltiplicare", "multiply"}):
        return False
    if len(re.findall(r"\d+", intent_text.text)) < 2:
        return False
    return any(symbol in intent_text.text for symbol in ("÷", "/", "*", "x"))


def _looks_like_base64_transform_request(intent_text: _IntentText) -> bool:
    if "base64" not in intent_text.text:
        return False
    return bool(intent_text.token_set & {"encode", "decode", "codifica", "decodifica"})


def _looks_like_text_path_request(intent_text: _IntentText) -> bool:
    path_tokens = (
        *CODE_FILE_EXTENSIONS,
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".gif",
        "/",
        "\\",
    )
    return any(token in intent_text.text for token in path_tokens)


def _looks_like_local_filesystem_metadata_request(intent_text: _IntentText) -> bool:
    metadata_tokens = {
        "metadata",
        "stat",
        "size",
        "bytes",
        "mtime",
        "modified",
        "modification",
        "newest",
        "latest",
        "oldest",
        "permissions",
        "permission",
        "mode",
        "exists",
        "exist",
        "metadati",
        "dimensione",
        "dimensioni",
        "byte",
        "modificato",
        "modifica",
        "recente",
        "recenti",
        "nuovo",
        "permessi",
        "permesso",
        "esiste",
    }
    workspace_tokens = {"workspace", "workdir", "directory", "folder", "cartella", "progetto", "project"}
    if not (intent_text.token_set & metadata_tokens):
        return False
    if _looks_like_binary_triage_request(intent_text):
        return False
    return _looks_like_text_path_request(intent_text) or bool(intent_text.token_set & workspace_tokens)


def _looks_like_file_edit_request(intent_text: _IntentText) -> bool:
    if _explicitly_forbids_file_changes(intent_text):
        return False
    if not _matches_any(intent_text, _EDIT_HINTS):
        return False
    if _looks_like_shell_command(intent_text) or _looks_like_arithmetic_request(intent_text):
        return False
    if _looks_like_system_info_request(intent_text):
        return False
    if _looks_like_workspace_discovery_request(intent_text):
        return False
    if _looks_like_text_path_request(intent_text):
        return True
    token_set = intent_text.token_set
    if token_set & {"mkdir", "rmdir", "rm"}:
        return True
    if _matches_any(intent_text, _EDIT_OBJECT_HINTS):
        return True
    if re.search(r"\b(named|called)\s+[a-z0-9_.-]+\b", intent_text.text):
        return True
    return False


def _looks_like_binary_analysis_request(intent_text: _IntentText) -> bool:
    return bool(re.search(r"\bbinar\w*\b", intent_text.text))


def _mentions_binary_format_token(intent_text: _IntentText) -> bool:
    return any(re.search(rf"\b{re.escape(token)}\b", intent_text.text) for token in _BINARY_FORMAT_TOKENS)


def _looks_like_binary_triage_request(intent_text: _IntentText) -> bool:
    if not (_matches_any(intent_text, _BINARY_HINTS) or _mentions_binary_format_token(intent_text)):
        return False
    triage_tokens = {
        "analyze",
        "analysis",
        "analizza",
        "triage",
        "inspect",
        "inspection",
        "static",
        "hash",
        "hashes",
        "container",
        "strings",
        "decompile",
        "identify",
        "type",
    }
    return bool(intent_text.token_set & triage_tokens)


def _looks_like_general_knowledge_request(intent_text: _IntentText) -> bool:
    if _looks_like_system_info_request(intent_text):
        return False
    if _matches_any(intent_text, _GENERAL_KNOWLEDGE_HINTS):
        return True
    return bool(re.search(r"\bwhat is\b", intent_text.text)) or bool(re.search(r"\bcome funziona\b", intent_text.text))


def _looks_like_code_generation_request(intent_text: _IntentText) -> bool:
    if _looks_like_text_path_request(intent_text):
        return False
    if _matches_any(intent_text, _EDIT_OBJECT_HINTS) and not _explicitly_forbids_file_changes(intent_text):
        return False
    if _matches_any(intent_text, _GENERAL_KNOWLEDGE_HINTS):
        return False
    token_set = intent_text.token_set
    if not token_set & frozenset(_CODE_GENERATION_HINTS):
        return False
    if _matches_any(intent_text, ("file", "files", "folder", "directory", "cartella", "cartelle", "save", "salva")) and not _explicitly_forbids_file_changes(intent_text):
        return False
    return bool(token_set & {"write", "create", "generate", "genera", "scrivi", "produce", "make"})


def _explicitly_forbids_file_changes(intent_text: _IntentText) -> bool:
    text = intent_text.text
    file_terms = ("file", "files", "filesystem", "workspace", "document", "documents")
    if not any(term in text for term in file_terms):
        return False
    forbid_patterns = (
        r"\bdo not\s+(?:create|write|modify|edit|save)\b",
        r"\bdon't\s+(?:create|write|modify|edit|save)\b",
        r"\bwithout\s+(?:creating|writing|modifying|editing|saving)\b",
        r"\bno\s+(?:file|files|filesystem)\s+(?:changes?|writes?|edits?|modifications?)\b",
    )
    return any(re.search(pattern, text) for pattern in forbid_patterns)


def _looks_like_creative_generation_request(intent_text: _IntentText) -> bool:
    if _looks_like_text_path_request(intent_text):
        return False
    if _matches_any(intent_text, _EDIT_OBJECT_HINTS):
        return False
    if _matches_any(intent_text, ("save", "salva", "file", "files", "folder", "directory", "cartella", "cartelle")):
        return False
    token_set = intent_text.token_set
    if not (token_set & frozenset(_CREATIVE_GENERATION_HINTS)):
        return False
    if "make up" in intent_text.text:
        return True
    return bool(token_set & {"write", "create", "generate", "compose", "invent", "genera", "scrivi", "inventa", "crea"})


def _looks_like_current_factual_lookup(intent_text: _IntentText) -> bool:
    if _matches_any(intent_text, _FACTUAL_HINTS):
        return True
    if _is_time_lookup(intent_text.token_set):
        return True
    if _is_online_lookup(intent_text.token_set):
        return True
    return False


def _looks_like_explicit_web_fetch_request(intent_text: _IntentText) -> bool:
    return _is_explicit_web_fetch_request(intent_text.token_set)


def _looks_like_codebase_inspection_request(intent_text: _IntentText) -> bool:
    if _matches_any(intent_text, _CODEBASE_HINTS):
        return True
    token_set = intent_text.token_set
    code_tokens = {"code", "codice", "repo", "project", "progetto", "architecture", "architettura"}
    inspection_tokens = {
        "stability", "stabilita", "stabilità", "maintainability", "maintenance", "manutenzione",
        "attention", "attenzione", "important", "importanti", "files", "file", "read", "leggere",
        "inspect", "inspect", "summary", "riassumi", "summarize", "review", "risk", "risks",
        "rischi", "bug", "bugs", "findings", "finding", "issue", "issues", "weakness", "weaknesses",
        "problema", "problemi", "debolezze", "vulnerability", "vulnerabilities", "vuln", "vulns",
        "security", "insecure", "exploit", "exploitable", "cve", "vulnerabilità", "vulnerabilita",
        "sicurezza", "falla", "falle",
    }
    return bool(token_set & code_tokens) and bool(token_set & inspection_tokens)


def _looks_like_workspace_security_search_request(intent_text: _IntentText) -> bool:
    token_set = intent_text.token_set
    if not any(_is_search_verb(token) for token in token_set):
        return False
    if token_set & {"online", "web", "internet"}:
        return False
    workspace_tokens = {"workspace", "workdir", "directory", "folder", "cartella", "repo", "repository", "project", "progetto"}
    issue_tokens = {
        "security",
        "issue",
        "issues",
        "vulnerability",
        "vulnerabilities",
        "vuln",
        "vulns",
        "secret",
        "secrets",
        "password",
        "token",
        "credential",
        "credentials",
        "insecure",
        "bug",
        "bugs",
        "rischio",
        "rischi",
        "sicurezza",
        "vulnerabilita",
        "vulnerabilità",
        "segreto",
        "segreti",
        "credenziali",
        "problema",
        "problemi",
    }
    return bool(token_set & workspace_tokens) and bool(token_set & issue_tokens)


def _looks_like_explicit_code_file_review_request(intent_text: _IntentText) -> bool:
    token_set = intent_text.token_set
    request_text = intent_text.text
    review_tokens = {
        "analyze",
        "analizza",
        "check",
        "problem",
        "problems",
        *CODE_REVIEW_REQUEST_HINTS,
    }
    if not _looks_like_text_path_request(intent_text):
        return False
    if not ((token_set & review_tokens) or any(hint in request_text for hint in CODE_REVIEW_REQUEST_HINTS)):
        return False
    return has_code_file_extension(request_text) or has_code_language_hint(request_text) or "this file" in request_text or "questo file" in request_text


def _looks_like_workspace_discovery_request(intent_text: _IntentText) -> bool:
    token_set = intent_text.token_set
    workspace_tokens = {"directory", "folder", "folders", "cartella", "cartelle", "workdir", "workspace"}
    listing_tokens = {"contain", "contains", "content", "contents", "inside", "files", "show", "mostra", "contiene", "ci", "sono", "elenca", "which", "quali"}
    if not token_set & workspace_tokens:
        return False
    if any(token in intent_text.text for token in (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", "/", "\\")):
        return False
    return bool(token_set & listing_tokens)


def _looks_like_pure_chitchat(intent_text: _IntentText) -> bool:
    text = intent_text.text
    if not _matches_any(intent_text, _CHITCHAT_HINTS):
        return False
    if _matches_any(intent_text, _EDIT_HINTS):
        return False
    if _matches_any(intent_text, _FACTUAL_HINTS):
        return False
    if _matches_any(intent_text, _COMMAND_HINTS):
        return False
    if _looks_like_text_path_request(intent_text):
        return False
    if _looks_like_shell_command(intent_text):
        return False
    if "http://" in text or "https://" in text:
        return False
    cleaned = re.sub(r"[^\w\s]", " ", text)
    tokens = [token for token in cleaned.split() if token]
    if len(tokens) > 6:
        return False
    return True


def _looks_like_assistant_persona_request(intent_text: _IntentText) -> bool:
    text = intent_text.text
    patterns = (
        r"\bwho are you\b",
        r"\bwhat are you\b",
        r"\bhow old are you\b",
        r"\bwhat is your age\b",
        r"\bwhat model are you\b",
        r"\bwhich model are you\b",
        r"\bhow were you trained\b",
        r"\bchi sei\b",
        r"\bcosa sei\b",
        r"\bquanti anni hai\b",
        r"\bche modello sei\b",
        r"\bquale modello sei\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)

def _is_time_lookup(token_set: set[str]) -> bool:
    time_tokens = {"time", "hour", "hours", "ora", "ore", "orario"}
    query_tokens = {"what", "current", "now", "adesso", "quanto", "che"}
    return bool(token_set & time_tokens) and bool(token_set & query_tokens)


def _is_online_lookup(token_set: set[str]) -> bool:
    web_tokens = {"online", "web", "internet"}
    info_tokens = {"info", "informazioni", "about", "su", "who", "chi", "news", "notizie", "documentation", "documentazione", "docs"}
    return (any(_is_search_verb(token) for token in token_set) or bool(token_set & web_tokens)) and bool(token_set & info_tokens)


def _is_search_verb(token: str) -> bool:
    return token.startswith("cerc") or token in {"search", "lookup", "look", "trova"}


def _has_explicit_web_url(text: str) -> bool:
    return "http://" in text.text or "https://" in text.text


def _is_explicit_web_fetch_request(token_set: set[str]) -> bool:
    fetch_tokens = {
        "fetch",
        "open",
        "check",
        "visit",
        "read",
        "summarize",
        "show",
        "tell",
        "say",
        "says",
        "written",
        "summary",
        "controlla",
        "apri",
        "leggi",
        "leggimi",
        "riassumi",
        "mostra",
        "riporta",
        "dimmi",
        "dammi",
        "dice",
        "scritto",
    }
    web_tokens = {"url", "link", "sito", "site", "web", "pagina", "page", "qui", "here", "qua"}
    return bool(token_set & fetch_tokens) or bool(token_set & web_tokens)


_INTENT_RULES = _intent_rules()
