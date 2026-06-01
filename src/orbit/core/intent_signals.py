from __future__ import annotations

import re


LEARNING_MARKERS = (
    "ask me",
    "quiz me",
    "test me",
    "give me a question",
    "fammi una domanda",
    "fammi domande",
    "interrogami",
    "mettimi alla prova",
)

SECURITY_DISCOURSE_MARKERS = (
    "i tried ",
    "i used ",
    "i think ",
    "in my opinion",
    "the future of",
    "nowday",
    "nowadays",
    "all the ",
    "ho provato ",
    "secondo me",
    "il futuro",
)

DISCUSSION_MARKERS = (
    "what do you think",
    "cosa ne pensi",
    "che ne pensi",
    "secondo te",
    "explain",
    "spiega",
    "why",
    "perche",
    "perchè",
    "how does",
    "how do",
    "come funziona",
    "tell me about",
    "parlami di",
)

OPERATIONAL_ACTION_PREFIX_RE = re.compile(
    r"^\s*(?:\d+[\.)]|[-*])\s*(?:analyze|analyse|analizza|inspect|ispeziona|perform|run|extract|estrai)\b"
)

NUMBERED_OR_BULLETED_RE = re.compile(r"^\s*(?:\d+[\.)]|[-*])\s+")


def contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def looks_like_discursive_security_text(text: str) -> bool:
    if contains_phrase(text, SECURITY_DISCOURSE_MARKERS):
        return True
    return bool(NUMBERED_OR_BULLETED_RE.search(text)) and not bool(OPERATIONAL_ACTION_PREFIX_RE.search(text))
