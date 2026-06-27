# KV Backend Envelope Diagnostics

## Purpose

`ORBIT_KV_DIAG=1` can now record metadata about the backend-visible request envelope for each model call. This diagnostic is intended to explain why a logically stable prompt prefix can have low reported KV/cache reuse in some tools-on paths.

This is diagnostics only. It does not change prompt content, routing, tool selection, backend behavior, cache behavior, or final answers.

## What Is Recorded

Each `kv_diag_model_call` event includes a `request_envelope` object with metadata such as:

- backend class name
- endpoint/path when it can be inferred from already-cached backend state
- `stream`
- `cache_prompt`
- `continue_current`
- hashed runtime session key, when available
- message count
- role sequence only, for example `["system", "user"]`
- whether a tools parameter was present
- tool count
- prompt-layout common-token estimate, when available

The diagnostics also keep the existing model-call fields:

- `request_id`
- `model_call_id`
- `phase`
- `tools_mode`
- `prompt_tokens`
- `cached_tokens`
- `evaluated_tokens`
- prompt-layout hashes and common-prefix metadata

## Privacy And Safety

The envelope diagnostic must not record:

- raw prompts
- user message content
- tool output
- file content
- web content
- full local paths
- personal data

`role_sequence` records only roles, never message text. Session identity is represented only as an existing hash from the diagnostic request context.

## How To Run

Example:

```bash
ORBIT_KV_DIAG=1 ORBIT_KV_DIAG_FILE=/tmp/orbit_kv_envelope.jsonl orbit --workdir workdir
```

Then run repeated tools-on scenarios in the same session, for example:

```text
/tools on
hi
hi
list files in the workdir
list files in the workdir
```

Inspect only JSON metadata:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for line in Path("/tmp/orbit_kv_envelope.jsonl").read_text().splitlines():
    event = json.loads(line)
    if event.get("event") != "kv_diag_model_call":
        continue
    envelope = event.get("request_envelope", {})
    print(
        event.get("request_id"),
        event.get("phase"),
        event.get("cached_tokens"),
        event.get("prompt_tokens"),
        envelope.get("endpoint"),
        envelope.get("stream"),
        envelope.get("tools_parameter_present"),
        envelope.get("tool_count"),
        envelope.get("message_count"),
        envelope.get("role_sequence"),
    )
PY
```

## Interpreting Cache Hits And Misses

If cache-hit and cache-miss calls have the same:

- endpoint
- stream mode
- `cache_prompt`
- tools parameter presence
- tool count
- role sequence shape
- stable prompt-layout common prefix

but still report very different `cached_tokens`, the likely cause is backend cache/session behavior rather than prompt-layout instability.

If low-cache calls differ in endpoint, stream mode, tools parameter presence, role sequence, or message count, those differences should be investigated before any KV optimization.

## Candidate Patch This Can Inform

This diagnostic can justify a later, separate patch only if the metadata shows a clear backend-visible difference between cache-hit and cache-miss requests. Possible later work could include explicit backend cache identity or prompt-layout changes, but neither should be implemented without benchmark evidence.

