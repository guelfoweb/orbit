from __future__ import annotations


DEFAULT_THINKING = False
THINK_USAGE = "off|on"


def normalize_think_spec(value: object, *, key: str = "think") -> bool:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError(f"invalid config key {key}: expected string or boolean")
    raw = value.strip().lower()
    if raw == "on":
        return True
    if raw == "off":
        return False
    raise ValueError(f"invalid config key {key}: unsupported value {raw!r}; expected one of off, on")


def think_text(current: bool | None = None) -> str:
    if current is None:
        return "\n".join(
            [
                "Use:",
                "  /think off = suppress reasoning and return only the final answer",
                "  /think on  = show reasoning before the final answer",
            ]
        )
    state = "on" if current else "off"
    return "\n".join(
        [
            f"think: {state}",
            "",
            "Use:",
            "  /think off = suppress reasoning and return only the final answer",
            "  /think on  = show reasoning before the final answer",
        ]
    )
