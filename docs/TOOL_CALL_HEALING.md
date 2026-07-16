# Deterministic Tool-Call Healing

Orbit applies a strict canonical contract to active tool calls. Deterministic
healing is enabled by default and runs only after existing normalization fails
to produce a tool call. Set `ORBIT_TOOL_CALL_HEALING=0` for an immediate kill
switch. An invalid value also disables healing safely.

The whitelist is intentionally closed:

- remove one known tool-mode envelope;
- remove trailing commas;
- decode `arguments` only when it is a complete JSON object string;
- unwrap one registered wrapper.

Healing never closes missing delimiters or strings, maps aliases, accepts
top-level arguments, changes tools, creates or removes arguments, converts
types, applies defaults or clamps, or retries the model. One complete candidate,
strong tool-mode structure, a non-`length` finish, exact tool identity, value
invariance, idempotence, and passing schema, permission, policy, and operational
limits are mandatory. Every other result fails closed.

The runtime order is:

```text
model output
-> active normalization
-> optional formal repair
-> canonical preflight
-> runtime guardrails
-> executor
```

The repair returns the canonical decision that authorized it. The loop and
executor reuse that decision; repaired calls are not validated twice and the
healing module cannot execute tools directly.

Runtime/backend properties expose bounded metadata only:

- `tool_call_healing_enabled`;
- `tool_call_healing_source` (`default` or `stable`);
- `tool_call_healing_config_error`;
- `tool_call_healing_blocked_reason`;
- `tool_call_healing_repair_count`;
- `tool_call_healing_rejection_count`;
- `tool_call_healing_last_rules`.

No prompt, model output, tool arguments, path, URL, evidence, or credential is
included. The native server reports its process-local view; the runtime client
overlays its own counters when reading backend properties.

Real Gemma measurements have not yet shown a recurring natural malformed call
covered by the whitelist. Default enablement therefore makes no performance or
success-rate claim. Do not add rules or a nudge retry without new production
evidence and a separate review.
