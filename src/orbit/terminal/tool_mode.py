from __future__ import annotations


ToolSpec = str

DEFAULT_ON_TOOL_NAMES = ("exec_shell_full_command", "fetch_url")

SPECIAL_TOOL_SPECS = ("off", "on")
USAGE = "off|on|status|refresh"


def normalize_tool_spec(value: object, *, key: str = "tools") -> ToolSpec:
    if not isinstance(value, str):
        raise ValueError(f"invalid config key {key}: expected string")
    raw = value.strip().lower()
    if not raw:
        raise ValueError(f"invalid config key {key}: expected non-empty tool spec")
    if raw in SPECIAL_TOOL_SPECS:
        return raw
    allowed = ", ".join(SPECIAL_TOOL_SPECS)
    raise ValueError(f"invalid config key {key}: unsupported value {raw!r}; expected one of {allowed}")


def tools_are_enabled(spec: ToolSpec) -> bool:
    return spec != "off"


def allowed_tool_names_for_spec(spec: ToolSpec) -> tuple[str, ...] | None:
    if spec == "off":
        return ()
    if spec == "on":
        return tuple(dict.fromkeys(DEFAULT_ON_TOOL_NAMES))
    return ()
