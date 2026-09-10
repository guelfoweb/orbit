"""Network policy for autonomous static analysis.

Autonomous static analysis of a local artifact must never reach the network.
A discovered URL, domain or IP is *evidence* -- something to extract,
canonicalise, store and report -- never permission to contact it. The autonomy
loop already runs its model-authored programs in a bubblewrap sandbox with
`--unshare-all` (no network namespace) and exposes only `read_file`, so a
program cannot open a socket. This module adds the second, mandatory layer the
safety contract asks for: an EXECUTION-TIME denial at the one place a network
request is actually made, so that even a stale, malformed or future-miswired
tool request refuses the outbound request instead of trusting the sandbox or
the tool grammar as the only boundary.

The signal is a process/async-safe `ContextVar` set for the lifetime of an
autonomous analysis run. Outside that context -- ordinary CHAT tool use -- the
policy is inert and network tools behave exactly as before. Nothing here reads
model text or forms an opinion; it is a binary contract flag the runtime owns.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

# True only while an autonomous static-analysis run is on the stack. Default
# False so every non-analysis caller (CHAT, direct tool use, tests) is
# unaffected. A ContextVar rather than a module global so concurrent sessions
# and async tasks each see their own value.
_ANALYSIS_NETWORK_DENIED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "orbit_analysis_network_denied", default=False
)

# The deterministic reason recorded when a network tool is refused under the
# analysis contract. Stable text so a report and a test can both recognise it,
# and phrased as policy -- not a backend/network failure -- because no socket
# was opened and nothing failed.
ANALYSIS_NETWORK_DENIED_REASON = (
    "network retrieval is denied during autonomous static analysis: a discovered "
    "URL is evidence, not permission to contact it; the remote resource was not "
    "retrieved"
)


def network_retrieval_denied() -> bool:
    """Whether the current context forbids outbound network retrieval."""
    return _ANALYSIS_NETWORK_DENIED.get()


@contextmanager
def analysis_network_denied() -> Iterator[None]:
    """Deny outbound network retrieval for the duration of the block.

    Wrap an autonomous analysis run in this. It is re-entrant and restores the
    prior value on exit, so nesting and interleaving with non-analysis work
    both behave.

    Maintainer note: this relies on the analysis run being single-threaded --
    a `ContextVar` set here is NOT seen by a worker thread or executor started
    inside the block (a fresh thread starts from the default, False). Today the
    autonomy loop, sandbox and backend run synchronously on one thread, so the
    flag covers the whole run. If analysis work is ever moved onto a thread or
    executor, propagate the context (`contextvars.copy_context().run(...)`) or
    the denial will silently lapse on that thread.
    """
    token = _ANALYSIS_NETWORK_DENIED.set(True)
    try:
        yield
    finally:
        _ANALYSIS_NETWORK_DENIED.reset(token)
