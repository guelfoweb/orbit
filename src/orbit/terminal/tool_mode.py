from __future__ import annotations


ToolSpec = str

FILES_TOOL_NAMES = ("list_files", "read_file", "stat_path", "file_glob_search", "grep_search")
EDIT_TOOL_NAMES = ("read_file", "write_file", "edit_file", "apply_diff", "make_directory", "delete_path")
WEB_TOOL_NAMES = ("search_web", "fetch_url")
SHELL_TOOL_NAMES = ("exec_shell_command", "get_datetime")

TOOL_GROUPS: dict[str, tuple[str, ...]] = {
    "files": FILES_TOOL_NAMES,
    "edit": EDIT_TOOL_NAMES,
    "web": WEB_TOOL_NAMES,
    "shell": SHELL_TOOL_NAMES,
}

SPECIAL_TOOL_SPECS = ("off", "on")
USAGE = "off|on|files|edit|web|shell|group[,group...]"


def normalize_tool_spec(value: object, *, key: str = "tools") -> ToolSpec:
    if not isinstance(value, str):
        raise ValueError(f"invalid config key {key}: expected string")
    raw = value.strip().lower()
    if not raw:
        raise ValueError(f"invalid config key {key}: expected non-empty tool spec")
    if raw in SPECIAL_TOOL_SPECS:
        return raw
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not parts:
        raise ValueError(f"invalid config key {key}: expected non-empty tool spec")
    invalid = [part for part in parts if part not in TOOL_GROUPS]
    if invalid:
        allowed = ", ".join((*SPECIAL_TOOL_SPECS, *TOOL_GROUPS.keys()))
        raise ValueError(f"invalid config key {key}: unsupported value {invalid[0]!r}; expected one of {allowed}")
    return ",".join(dict.fromkeys(parts))


def tools_are_enabled(spec: ToolSpec) -> bool:
    return spec != "off"


def allowed_tool_names_for_spec(spec: ToolSpec) -> tuple[str, ...] | None:
    if spec == "off":
        return ()
    if spec == "on":
        return None
    names: list[str] = []
    for part in spec.split(","):
        names.extend(TOOL_GROUPS[part])
    return tuple(dict.fromkeys(names))
