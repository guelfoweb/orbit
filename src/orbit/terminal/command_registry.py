from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    name: str
    category: str
    usage: str
    description: str
    handler: str


@dataclass(frozen=True)
class CommandInvocation:
    spec: CommandSpec
    arguments: str


COMMANDS = (
    CommandSpec("/read", "Files & Web", "/read <source> [prompt]", "Read a local document or URL safely.", "read"),
    CommandSpec(
        "/search",
        "Files & Web",
        "/search <query> [source] [prompt]",
        "Search the web, a local document, or one URL.",
        "search",
    ),
    CommandSpec("/ls", "Files & Web", "/ls [path]", "List a directory inside the workdir.", "list"),
    CommandSpec("/models", "Models", "/models", "Show discovered and supported models.", "models"),
    CommandSpec("/clear", "Session", "/clear", "Clear the interactive terminal display.", "clear"),
    CommandSpec("/compact", "Session", "/compact [tools]", "Compact memory or old tool results.", "compact"),
    CommandSpec("/continue", "Session", "/continue", "Continue an answer that reached max_tokens.", "continue"),
    CommandSpec("/reset", "Session", "/reset", "Clear the conversation and saved session.", "reset"),
    CommandSpec("/sessions", "Session", "/sessions clear", "Delete saved sessions for this workdir.", "sessions"),
    CommandSpec("/exit", "Session", "/exit", "Exit interactive mode.", "exit"),
    CommandSpec("/health", "Runtime", "/health", "Check backend health.", "health"),
    CommandSpec(
        "/max-tokens",
        "Runtime",
        "/max-tokens [n]",
        "Show or set the output token limit.",
        "max_tokens",
    ),
    CommandSpec("/think", "Runtime", "/think [off|on]", "Show or set thinking visibility.", "think"),
    CommandSpec("/status", "Runtime", "/status [ctx]", "Show runtime status or context usage.", "status"),
    CommandSpec("/props", "Runtime", "/props", "Show backend properties.", "props"),
    CommandSpec(
        "/tools",
        "Runtime",
        "/tools [off|on|status|refresh]",
        "Show tool access or local capabilities.",
        "tools",
    ),
    CommandSpec(
        "/analysis",
        "Session",
        "/analysis <path>",
        "Analyse one local artifact in an isolated workspace.",
        "analysis",
    ),
    CommandSpec(
        "/report",
        "Session",
        "/report [question]",
        "Report on the evidence already collected, running no analysis action.",
        "report",
    ),
    CommandSpec("/chat", "Session", "/chat", "Return to normal chat mode.", "chat"),
    CommandSpec("/help", "Help", "/help", "Show command help.", "help"),
)


_BY_NAME = {command.name: command for command in COMMANDS}
if len(_BY_NAME) != len(COMMANDS):
    raise RuntimeError("duplicate slash command name")


def resolve_command(value: str) -> CommandInvocation | None:
    stripped = value.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped.split(maxsplit=1)
    name = parts[0]
    arguments = parts[1] if len(parts) == 2 else ""
    spec = _BY_NAME.get(name)
    if spec is None:
        return None
    return CommandInvocation(spec=spec, arguments=arguments.strip())


def commands_matching(prefix: str) -> tuple[CommandSpec, ...]:
    normalized = prefix.strip()
    if normalized and not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return tuple(command for command in COMMANDS if command.name.startswith(normalized))
