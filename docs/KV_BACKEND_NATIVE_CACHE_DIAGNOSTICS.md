# KV Backend Native Cache Diagnostics

## Purpose

The runtime and backend-envelope diagnostics can show that two tools-on calls have the same endpoint, stream mode, and `cache_prompt=true`, while native `cached_tokens` still differs sharply. Native cache diagnostics add one lower-level event from the native llama client to explain whether the backend token cache actually sees a reusable token prefix.

This is diagnostics only. It does not change runtime behavior, prompt content, tool selection, backend cache behavior, or final answer generation.

## Event

When `ORBIT_KV_DIAG=1` is set, the native standard completion path emits:

```json
{"event":"kv_diag_native_cache","endpoint":"/chat/stream","cache_prompt":true,"prompt_tokens":706,"previous_prompt_tokens":0,"longest_common_prefix_tokens":0,"cached_tokens":0,"cache_miss_reason":"no_previous_prompt"}
```

The event contains only metadata:

- backend request id generated inside the native server
- endpoint, stream flag, and received `cache_prompt`
- hashed session/cache key
- slot id
- message count
- role sequence only
- tools parameter presence and tool count
- backend-visible prompt token count
- previous cached prompt token count
- tokenized prompt hash
- tokenized common-prefix hash and length
- longest common prefix token count
- first mismatch token position, when the current and previous prompts diverge
- reused/cached/evaluated token counts
- cache miss reason when deducible

## Privacy And Safety

The diagnostic does not log:

- raw prompt text
- raw token ids
- user message content
- tool output
- file content
- web content
- full local paths
- personal data

Token hashes are one-way diagnostic hashes over token sequences. They are intended only for comparing whether two backend-visible prompts are identical or share a prefix.

## How To Run

Start Orbit normally, then run the client with diagnostics:

```bash
ORBIT_KV_DIAG=1 ORBIT_KV_DIAG_FILE=/tmp/orbit_native_kv.jsonl orbit --workdir workdir
```

Suggested smoke scenarios:

```text
/tools on
hi
hi
what is 2+2?
what is 2+2?
list files in the workdir
list files in the workdir
read README.md and explain it
read README.md and explain it
fetch https://example.com and summarize it briefly
```

Inspect only metadata:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for line in Path("/tmp/orbit_native_kv.jsonl").read_text().splitlines():
    event = json.loads(line)
    if event.get("event") != "kv_diag_native_cache":
        continue
    print(
        event.get("endpoint"),
        event.get("prompt_tokens"),
        event.get("previous_prompt_tokens"),
        event.get("longest_common_prefix_tokens"),
        event.get("cached_tokens"),
        event.get("cache_miss_reason"),
        event.get("role_sequence"),
    )
PY
```

## Interpreting Results

If `longest_common_prefix_tokens` is high and `cached_tokens` is also high, native KV reuse is working for that request.

If `longest_common_prefix_tokens` is high but `cached_tokens` is low, the next suspect is a bug in cache reuse accounting or memory invalidation after the LCP calculation.

If both `longest_common_prefix_tokens` and `cached_tokens` are low, the backend-visible prompt is not prefix-stable at token level. In that case the candidate patch is prompt-layout work, not KV slot work.

`first_mismatch_token` identifies the token position where the current backend-visible prompt first diverges from the previous prompt in the same native session. A low value such as `4` means the backend cannot reuse the large logical runtime/tool prefix because the tokenized request changes before that prefix becomes common.

If `previous_prompt_tokens` is zero, the native client has no previous prompt cache for that slot/session. The event should report `no_previous_prompt`.

If `cache_prompt=false`, the event should report `cache_disabled`; this records the received envelope but does not change current backend behavior.

## Initial Local Finding

Local smoke runs showed that cache hits and misses used the same native slot (`default`), endpoint (`/chat/stream`), and `cache_prompt=true`.

The important difference was tokenized LCP:

- cache-hit cases had high `longest_common_prefix_tokens`, and `cached_tokens` matched it closely
- cache-miss cases had low `longest_common_prefix_tokens`, often `0` or `4`, and `cached_tokens` matched that low value

This suggests the native cache is behaving consistently with the tokenized prompt it receives. The low reuse is more likely caused by backend-visible prompt shape differences between phases or requests, not by a missing cache slot identity.

## Patch Decision This Enables

Use this diagnostic before any KV optimization:

- If hit and miss requests use different slot/session metadata, the candidate patch is stable cache identity or slot reuse.
- If slot/session metadata is stable but tokenized LCP is low, the candidate patch is prompt-layout stabilization.
- If slot/session metadata is stable and tokenized LCP is high but `cached_tokens` is low, the candidate patch is native cache accounting or memory reuse validation.

No performance patch should be implemented until this event identifies which of those cases applies.
