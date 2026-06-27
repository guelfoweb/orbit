# Route Outcome Observability

Status: diagnostics only. This adds no runtime, prompt, routing, tool-selection, backend, final-policy, or KV/cache behavior changes.

## Purpose

The tools-on route pass can either:

- finish a simple no-tool answer in one pass
- parse a tool or chat decision
- produce invalid/truncated output and require a retry/final pass

`ORBIT_KV_DIAG=1` now records a non-content event for this outcome so repeated benchmarks can measure pass count without logging prompt text or model output.

## Event

Event name:

```json
{"event":"kv_diag_route_outcome"}
```

Fields:

- `request_id`
- `model_call_id`
- `pass_index`
- `phase`
- `tools_mode`
- `finish_reason`
- `decision_type`
- `output_chars`
- `output_tokens`
- `retry_reason`
- `outcome`

No raw prompt, raw route output, raw tool output, or user-visible content is logged.

## Classifications

| Outcome | Meaning |
| --- | --- |
| `route_direct_final_stop` | Route produced no parsed decision, stopped normally, and was accepted as the visible final answer. |
| `route_no_decision_length_retry` | Route produced no parsed decision and hit `finish_reason=length`; Orbit rejects the truncated route output and runs `chat_final_retry`. |
| `route_parsed_tool` | Route produced a parsed non-chat route, such as filesystem/web/media/tool access. |
| `route_parsed_chat` | Route explicitly selected chat/final-answer handling. |
| `route_invalid_output` | Route produced invalid control/empty output requiring a repair/final pass. |
| `route_other_retry` | Route required another model-guided routing attempt, such as explicit web search or tool-related length retry. |

## Example

```json
{
  "event": "kv_diag_route_outcome",
  "request_id": "req_000001",
  "model_call_id": "mc_000001",
  "pass_index": 1,
  "phase": "route",
  "tools_mode": "on",
  "finish_reason": "length",
  "decision_type": null,
  "output_chars": 512,
  "output_tokens": 128,
  "retry_reason": "length_without_decision",
  "outcome": "route_no_decision_length_retry"
}
```

## Validation

Unit coverage verifies:

- direct route finalization emits `route_direct_final_stop` without changing the final answer
- length-limited route output emits `route_no_decision_length_retry` without changing retry behavior
- parsed/invalid/other classifications can be emitted as metadata-only events
- event payloads include request/model-call correlation
- event payloads do not include raw prompt or raw output content

## Next Step

Use this metric in route pass-count benchmarks with a normal output budget.

Recommended analysis before any behavior change:

- count `route_direct_final_stop` versus `route_no_decision_length_retry`
- compare output token counts for route-direct final answers
- identify prompts where route writes long prose instead of producing `CHAT`
- only then decide whether a route prompt/contract change is worth benchmarking

Do not mix this observability with KV reuse work.
