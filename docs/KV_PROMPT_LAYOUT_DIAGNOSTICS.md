# KV Prompt Layout Diagnostics

Status: diagnostics only.

This document describes the prompt layout diagnostics added after the post-route
KV baseline. The goal is to understand why a logically stable tools-on route
prefix can still show low backend cache reuse.

## Problem

The post-route baseline showed:

- `route_no_decision_length_retry = 0`
- valid web and fetch requests stay on `route -> final_from_tool`
- the tools-on route prompt is still large, around 700 tokens
- the logical stable prefix hash is stable
- backend cache reuse for route calls can still drop to only a few cached tokens

This suggests that the logical stable prefix might not map to a reusable prefix
in the serialized/tokenized prompt, or that dynamic conversation content changes
before the backend's reusable prefix boundary.

## What The Diagnostics Measure

When `ORBIT_KV_DIAG=1` is enabled, each `kv_diag_model_call` event includes
prompt layout metadata:

- `prompt_layout_hash`
- `prompt_layout_order`
- `prompt_layout`
- `prompt_layout_common_prefix`

The purpose is to determine whether Orbit's logically stable prefix is also the
real serialized/tokenized prefix that the backend can reuse. These diagnostics do
not imply any KV optimization by themselves.

Each prompt layout block contains only metadata:

- block index
- component name, such as `runtime_policy`, `user_message`, `tool_schema_parameter`
- source, such as `messages` or `tools_parameter`
- role, if the block comes from a message
- content/schema hash
- character length estimate
- token estimate
- estimated character range
- estimated token range

For repeated calls in the same diagnostic scenario, Orbit also reports the
metadata-only common prefix between the previous and current layout:

- whether a previous comparable call exists
- number of common blocks
- estimated common prefix characters
- estimated common prefix tokens
- hash of the common block prefix
- current first divergent component
- previous first divergent component

If a layout divergence is detected, Orbit emits:

```json
{"event":"kv_diag_prompt_layout_mismatch","common_blocks":1,"first_divergence_component":"assistant_history","previous_first_divergence_component":"user_message"}
```

## Privacy And Safety

The diagnostics do not log:

- raw prompt text
- user messages
- assistant content
- tool output
- file contents
- fetched web content
- raw tool schemas

The logs include hashes, lengths, estimated positions, component labels, and
tool counts. They are intended to answer layout questions without exposing
content.

## How To Run

Example:

```bash
ORBIT_KV_DIAG=1 \
ORBIT_KV_DIAG_FILE=/tmp/orbit_kv_layout.jsonl \
PYTHONPATH=src \
python3 -m orbit.terminal.cli --workdir workdir --tools on --no-render-markdown "hi"
```

For repeat scenarios, run multiple turns in one session or run controlled
benchmark scripts with the same `ORBIT_KV_DIAG_FILE`.

## How To Interpret Results

Start with `kv_diag_model_call`:

1. Check `stable_prefix_hash`.
2. Check `prompt_layout_order`.
3. Check whether stable components appear before dynamic components.
4. Compare `prompt_layout_common_prefix.common_tokens_estimate` across repeated calls.
5. Inspect `kv_diag_prompt_layout_mismatch` events for the first divergent component.

Useful patterns:

- If `runtime_policy` is first and stable but the backend still reports low cache,
  the issue may be backend cache behavior or tokenizer/session semantics.
- If `user_message`, `assistant_history`, or `tool_result` appears before a stable
  tool/policy block, prompt layout may prevent longest-prefix reuse.
- If `user_message` appears before `tool_schema_parameter`, treat that as a signal
  to verify the serialized layout and backend cache semantics. It is not itself a
  runtime fix or permission to change routing behavior.
- If `conversation_prefix` changes while `stable_prefix_hash` stays stable,
  Orbit may need better token-position diagnostics before any optimization.

## Next Decision

Do not implement KV optimization from these diagnostics alone.

The next decision should be based on whether the stable route/tool policy is
actually at the beginning of the serialized prompt seen by the backend. If it is
not, the next candidate is a no-semantics-change prompt layout experiment in a
separate branch. If it is already first, the likely blocker is backend cache
behavior rather than Orbit prompt assembly.
