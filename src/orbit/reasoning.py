"""Validated, request-scoped named reasoning effort for native rendering."""
from contextlib import contextmanager
from contextvars import ContextVar

REASONING_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_effort = ContextVar("orbit_reasoning_effort", default=None)


def parse_reasoning_effort(value):
    if value is None:
        return None
    if not isinstance(value, str) or value not in REASONING_EFFORT_LEVELS:
        raise ValueError("reasoning_effort must be one of: " + ", ".join(REASONING_EFFORT_LEVELS))
    return value


def current_reasoning_effort():
    return _effort.get()


@contextmanager
def reasoning_effort_scope(value):
    token = _effort.set(parse_reasoning_effort(value))
    try:
        yield
    finally:
        _effort.reset(token)
