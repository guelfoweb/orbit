# Tool protocol safety

A parsed tool call is a proposal, not permission to execute. The shared native
and runtime paths require all of the following:

- A final native parse consumes the entire response with a complete envelope.
  Lenient, partial ASTs are used only for streaming previews. Complete JSON
  arguments alone do not prove that the surrounding protocol is complete.
- Reasoning remains outside tool authority. The native bridge carries the
  protocol's reasoning delimiters into AST validation. Unexpected reasoning
  markers in visible content, or a tool used to terminate an unclosed reasoning
  block, fail closed. Delimiters inside argument values remain literal data.
- Dispatch requires a terminal `stop` or `tool_calls` completion, then the
  existing canonical schema, permission and safety checks. Length, cancellation,
  timeout, incomplete and unknown completion states cannot dispatch calls,
  including calls handed off from routing or returned by an existing retry.
  Disabling canonical validation or formal healing does not disable this check.

The parser uses the existing protocol grammar. In particular, Qwen3-Coder's
existing alternative opener permits a leading `<function=...>` without the
outer opening `<tool_call>`; it still requires the protocol's function and tool
closing markers. This is not a general XML repair rule. Ambiguous multiple calls
remain subject to the existing single-call canonical gate.

Reserved reasoning markers in visible prose or code are conservatively rejected
as ambiguous, including tools-off output. This is a protocol boundary check,
not a classifier of natural-language intent. Literal markers in structured
argument values remain valid.

Prompt rendering, generation grammar, sampling, budgets and retry counts are
unchanged. The existing FINISH-only parameter-order recovery still requires a
complete original envelope and exact argument equality after native reparsing.
Failed or cancelled attempts retain their diagnostic completion state. The
existing tools-off finalization of committed shell results can still produce a
fresh answer after a cancelled response with no calls. It cannot dispatch that
response or bypass required artifact verification. No safety rejection is
converted into a successful execution.

This hardening does not enable MiMo or qualify model reasoning. MiMo's template
is retained as an offline regression fixture only. Template/protocol compatibility
is separate from model-profile enablement and native model quality.

Run the deterministic gates with:

```sh
TMPDIR=/tmp PYTHONPATH=src python3 -m unittest \
  tests.test_native_tool_protocol_strict \
  tests.test_tool_completion_authority \
  tests.test_finish_parameter_order \
  tests.test_finish_constrained_decoding -q
```

The native parser tests compile a small probe against the vendored library and
use scripted responses; no model weights, inference or real tool execution are
needed. Rebuild native artifacts with `python3 scripts/build_native.py` after
updating the vendored parser or bridge. The pinned upstream revision is unchanged;
Orbit's source/patch and bridge identities bind the rebuilt content.
