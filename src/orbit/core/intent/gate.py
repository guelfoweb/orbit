from __future__ import annotations

from dataclasses import dataclass
import re

from .router import (
    INTENT_CLASS_AMBIGUOUS,
    INTENT_CLASS_BINARY_ANALYSIS,
    INTENT_CLASS_PDF_ANALYSIS,
    INTENT_CLASS_WEB_LOOKUP,
)
from ..tools.router import ToolRoute


INTENT_GATE_SYSTEM_PROMPT = (
    "You are an intent gate for a local CLI. "
    "Answer only YES or NO. "
    "YES means the user is asking Orbit to use an available local tool: inspect or modify local workspace files, "
    "run a bounded shell command, search/fetch web content, or perform local binary/static analysis. "
    "NO means conversation, explanation, opinions, learning, a quiz, unsupported external actions such as sending email, "
    "or a vague edit/action without a concrete target."
)


@dataclass(frozen=True)
class IntentGateDecision:
    confirm: bool
    reason: str


def should_confirm_tool_route(user_input: str, route: ToolRoute | None) -> bool:
    return intent_gate_decision(user_input, route).confirm


def intent_gate_decision(user_input: str, route: ToolRoute | None) -> IntentGateDecision:
    if route is None or not route.categories:
        return IntentGateDecision(confirm=False, reason="no tool route")
    if route.intent_class == INTENT_CLASS_BINARY_ANALYSIS:
        if is_clear_local_binary_operation(user_input):
            return IntentGateDecision(confirm=False, reason="clear local binary operation")
        return IntentGateDecision(confirm=True, reason="ambiguous binary/static route")
    if route.intent_class == INTENT_CLASS_PDF_ANALYSIS:
        return IntentGateDecision(confirm=False, reason="clear pdf analysis")
    if route.intent_class == INTENT_CLASS_WEB_LOOKUP:
        if is_clear_web_lookup_operation(user_input):
            return IntentGateDecision(confirm=False, reason="clear web lookup")
        return IntentGateDecision(confirm=True, reason="weak web lookup")
    if route.intent_class == INTENT_CLASS_AMBIGUOUS:
        return IntentGateDecision(confirm=True, reason="ambiguous route")
    return IntentGateDecision(confirm=False, reason="clear routed intent")


def intent_gate_messages(*, user_input: str, route: ToolRoute) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": INTENT_GATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Candidate route: {route.intent}\n"
                f"User prompt:\n{user_input}\n\n"
                "Should Orbit proceed with local tool-based analysis for this prompt? Answer only YES or NO."
            ),
        },
    ]


def parse_intent_gate_reply(content: object) -> bool | None:
    normalized = str(content or "").strip().upper()
    if normalized.startswith("NO"):
        return False
    if normalized.startswith("YES"):
        return True
    return None


def is_clear_local_binary_operation(user_input: str) -> bool:
    lowered = user_input.lower()
    if _contains_local_artifact_path(lowered):
        return True
    explicit_phrases = (
        "perform static analysis",
        "run static analysis",
        "static triage",
        "collect initial evidence",
        "file type and hashes",
        "all malware samples",
        "malware directory",
        "malware folder",
        "analisi statica",
        "evidenze iniziali",
    )
    if any(phrase in lowered for phrase in explicit_phrases):
        return True
    action_pattern = r"\b(analyze|analyse|analizza|inspect|ispeziona|extract|estrai|decompile|triage|try|prova)\b"
    target_pattern = r"\b(file|sample|samples|campione|campioni|apk|pdf|zip|binary|binario|directory|folder|cartella|workdir)\b"
    return bool(re.search(action_pattern, lowered) and re.search(target_pattern, lowered))


def _contains_local_artifact_path(lowered: str) -> bool:
    return bool(re.search(r"[^\s]+(?:\.pdf|\.apk|\.zip|\.jar|\.aar|\.ipa|\.dex|\.so|\.bin|\.exe|\.dll)\b", lowered))


def is_clear_web_lookup_operation(user_input: str) -> bool:
    lowered = user_input.lower()
    if "http://" in lowered or "https://" in lowered:
        return True
    explicit_web_phrases = (
        "search online",
        "search the web",
        "search web",
        "look up online",
        "lookup online",
        "on the web",
        "cerca online",
        "cercami online",
        "cerca sul web",
        "cerca in rete",
        "cercami in rete",
    )
    if any(phrase in lowered for phrase in explicit_web_phrases):
        return True
    entity_lookup_phrases = (
        "who is",
        "chi è",
        "chi e",
        "informazioni su",
        "information about",
        "info su",
    )
    if any(phrase in lowered for phrase in entity_lookup_phrases):
        return True
    current_or_external_terms = {
        "latest",
        "recent",
        "current",
        "today",
        "now",
        "news",
        "weather",
        "forecast",
        "online",
        "internet",
        "web",
        "docs",
        "documentation",
        "ultimo",
        "ultime",
        "recente",
        "recenti",
        "attuale",
        "oggi",
        "ora",
        "notizie",
        "meteo",
        "previsioni",
        "documentazione",
    }
    return bool(set(re.findall(r"[a-z0-9_àèéìòù]+", lowered)) & current_or_external_terms)
