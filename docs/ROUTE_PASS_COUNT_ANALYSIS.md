# Route Pass Count Analysis

Status: analysis only. No runtime, prompt, routing, tool-selection, backend, MTP, final-policy, or KV/cache behavior changes are proposed here.

## Analyzed Build

- Commit: `a0a2f16708c2b55f3194a060c2d1daa5e2b9615b`
- Source: `origin/main` after KV Phase 2 analysis merge
- Backend: native `orbit server`
- Mode: no-MTP default
- Diagnostic command: `PYTHONPATH=src python3 scripts/bench_kv_diag.py --max-tokens 120 --timeout 180`
- Direct route probes: local `with_command_system_prompt(...)` calls against the same backend
- Diagnostic artifacts: local JSONL/Markdown under `benchmarks/`, not committed

The earlier Phase 2 benchmark used `--max-tokens 32`. This report uses a larger output budget to avoid over-attributing retries to artificially tiny final-answer budgets. The route pass is still capped by the internal route budget, currently bounded by `ROUTE_MAX_TOKENS`.

## Route Flow

For `/tools on`, `ChatRuntime.ask_auto()` first asks the model to route the request:

1. Build command-routing messages with `with_command_system_prompt(...)`.
2. Call the backend with phase `route`.
3. Parse a route decision from tool calls or text via `parse_command_decision_from_tool_calls(...)` / `parse_command_decision(...)`.
4. If a tool route is parsed, enter the tool loop.
5. If `ToolRoute.CHAT` is parsed, run `chat_final`.
6. If no decision is parsed, Orbit may accept the route output as the visible final answer, but only when it is not empty, not control-channel markup, and not truncated.

The observed `route -> chat_final_retry` path is in the no-decision branch.

## Exact Retry Conditions

`chat_final_retry` is triggered in the no-decision branch when:

```text
decision is None
first.finish_reason == "length"
```

That means:

- the route output did not parse as a tool decision
- the route output did not parse as `CHAT`
- the route pass hit its token limit
- Orbit rejects the truncated route output as final and asks for a normal final answer pass

Other retry/finalization paths exist but are different:

| Trigger | Phase/result | Classification |
| --- | --- | --- |
| route output contains control-channel markup and is not length-limited | `chat_final` via transport environment | correctness repair |
| route output has `finish_reason == "length"` and no parsed decision | `chat_final_retry` | truncation repair |
| route output is empty with `stop` | `chat_final` | empty response repair |
| route output is length-limited and prompt looks tool/file/web-related | `route_retry` may happen first | model-guided command repair |
| parsed `ToolRoute.CHAT` | `chat_final` | explicit model route decision |
| parsed filesystem/web/media tool route | tool loop | explicit model route decision |

The problematic no-tool-needed case is the second row: no parsed decision plus route truncation.

## Benchmark Scenarios

| Scenario | Requests | Model calls | Phases | Prompt tokens / call | Cached tokens / call | Result |
| --- | ---: | ---: | --- | --- | --- | --- |
| `tools_off_repeat` | 2 | 2 | `chat_final`, `chat_final` | `34`, `54` | `6`, `30` | complete |
| `tools_on_repeat_no_tool_needed` | 2 | 2 | `route`, `route` | `345`, `365` | `6`, `341` | complete |
| `tools_on_same_session_repeat` | partial | 3 | `route`, `chat_final_retry`, `route` | `351`, `351`, `488` | `336`, `350`, `347` | runner timeout after useful diagnostics |
| `tools_on_after_reset` | 2 | 2 | `route`, `route` | `345`, `345` | `336`, `344` | complete |
| `tools_on_off_switch` | 3 | 3 | `chat_final`, `route`, `chat_final` | `34`, `365`, `74` | `6`, `6`, `6` | complete |
| `list_directory_repeat` | 2 | 3 | `route`, `final_from_tool`, `route` | `350`, `457`, `520` | `6`, `346`, `453` | complete |
| `system_info_repeat` | 2 | 3 | `route`, `final_from_tool`, `route` | `351`, `547`, `640` | `335`, `347`, `543` | complete |

## Phase Cost With Normal Budget

| Phase | Calls | Avg prompt tokens | Avg evaluated tokens | Cached ratio | Avg completion tokens | Finish reasons |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `chat_final` | 4 | 49.0 | 37.0 | 24.5% | 17.2 | `stop`: 4 |
| `route` | 11 | 405.9 | 128.4 | 68.4% | 40.5 | `stop`: 9, `length`: 2 |
| `chat_final_retry` | 1 | 351.0 | 1.0 | 99.7% | 120.0 | `length`: 1 |
| `final_from_tool` | 2 | 502.0 | 155.5 | 69.0% | 61.5 | `stop`: 2 |

Interpretation:

- `route` is still the dominant tools-on prefix cost.
- Warm route calls often reuse most prompt tokens.
- `chat_final_retry` has almost no evaluated prompt cost when cache is warm, but it still adds another generation pass.
- `final_from_tool` cost is expected because tool evidence is included.

## Direct Route Probe

Direct route probes against `with_command_system_prompt(...)` explain the retry cause:

| Prompt | Route finish | Completion tokens | Parsed decision | Route preview |
| --- | --- | ---: | --- | --- |
| `hi` | `stop` | 9 | none | `Hello! How can I help you today?` |
| `what is 2+2?` | `stop` | 1 | none | `4` |
| `hi, tell me something about yourself` | `length` | 128 | none | starts a long self-description |

For short no-tool prompts, the route pass effectively acts as the final answer and stops. Orbit accepts this as a one-pass tools-on answer.

For `hi, tell me something about yourself`, the route pass starts writing a full assistant self-description under the routing prompt, hits the internal route token budget, and returns `finish_reason=length`. Since the output is truncated and no route decision is parsed, Orbit correctly rejects it and runs `chat_final_retry`.

## Classification

| Cause | Evidence | Classification |
| --- | --- | --- |
| Retry necessary for correctness | route output is truncated (`finish_reason=length`) | yes |
| Output route not valid | no parsed route decision or tool call | yes |
| Output truncated/limited | route completion reaches route budget | yes |
| Parser/schema mismatch | no evidence; output is prose, not malformed command JSON | no |
| Retry avoidable | only if route is prevented from generating long prose or returns explicit `CHAT` | possible future prompt/contract work |
| KV/cache issue | stable prefix reused well in retry (`cached 350/351`) | not primary |

The retry is not caused by backend cache failure. It is caused by route-as-final behavior colliding with a short internal route budget.

## Impact

The problematic request adds:

- one route pass that generates up to the route cap
- one final retry pass that generates the user-visible answer
- additional wall time dominated by generation, not prompt evaluation

In the normal-budget benchmark, the first repeat request had:

- `route`: `351` prompt tokens, `336` cached, `15` evaluated, `120` completion tokens, `length`
- `chat_final_retry`: `351` prompt tokens, `350` cached, `1` evaluated, `120` completion tokens, `length`

The prompt cache is effective here. The wasted cost is mostly the first route generation, not prefill.

## Recommendation

Do not start with KV/prefix reuse for this issue.

The smallest safe next step is to make route outcome diagnostics more explicit before changing behavior:

- record a hash-only/non-content `route_outcome` category in diagnostics
- distinguish `route_direct_final_stop`, `route_no_decision_length_retry`, `route_parsed_tool`, `route_parsed_chat`, `route_empty_retry`, and `route_control_markup_retry`
- keep diagnostics off by default
- do not log route content
- do not change routing behavior

After that, evaluate a prompt/contract change separately:

- make the route pass return an explicit `CHAT` decision for longer direct-chat answers instead of writing the answer in the route pass
- or require direct route answers to be concise enough for the route budget

Both are behavior/prompt changes and need separate benchmarks. They should not be mixed with KV work.

Current conclusion:

- The route retry is necessary under current semantics because the route answer is truncated.
- The cost center is multi-pass generation, not missing KV reuse.
- The next safe change should target route outcome observability first, then route contract design if benchmark evidence supports it.
