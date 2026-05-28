from __future__ import annotations

import re


WORD_TOKEN_RE = re.compile(r"[a-z0-9àèéìòù]+")

ENGLISH_MARKERS = {
    "the", "this", "project", "codebase", "which", "parts", "without", "unnecessary",
    "stability", "maintainability", "who", "created", "you", "describe", "yourself",
    "explain", "simple", "terms", "message", "messages", "step", "steps", "would",
    "before", "running", "decide", "whether", "search", "documentation", "report",
    "tool", "tools", "calling", "error", "errors", "critical", "log", "review",
    "security", "vulnerability", "vulnerabilities", "issue", "issues", "file",
    "files", "workspace", "inspect", "analyze",
}

ITALIAN_MARKERS = {
    "questo", "progetto", "codice", "quali", "punti", "senza", "inutili", "stabilita", "stabilità",
    "manutenzione", "spiegami", "semplice", "passo", "passi", "prima", "eseguire", "strumento",
    "strumenti", "errore", "errori", "critici", "log", "cerca", "documentazione", "riassumi",
}


def word_tokens(text: str) -> list[str]:
    return WORD_TOKEN_RE.findall(text.lower())


def prefers_english_output(text: str) -> bool:
    token_set = set(word_tokens(text))
    return len(token_set & ENGLISH_MARKERS) > len(token_set & ITALIAN_MARKERS)
