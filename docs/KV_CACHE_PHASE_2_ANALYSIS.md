# KV Cache Phase 2 Analysis

Status: analysis only. No runtime, prompt, backend, routing, tool-selection, or cache behavior changes are proposed here.

## Analyzed Build

- Commit: `ce99df82cf5a4c2292f75e92f4ae59d021760c40`
- Source: `origin/main` after KV Phase 1B collector merge
- Backend: native `orbit server`
- Mode: no-MTP default
- Diagnostic command: `PYTHONPATH=src python3 scripts/bench_kv_diag.py --max-tokens 32 --timeout 90`
- Slow network scenario: not included
- Diagnostic artifacts: local JSONL/Markdown under `benchmarks/`, not committed

The benchmark used a short output budget intentionally. This makes multi-pass and retry behavior visible, but length-related retries should be interpreted as diagnostic signals, not release-quality response behavior.

## Scenario Summary

| Scenario | Requests | Model calls | Phases | Prompt tokens / call | Cached tokens / call | Footer correlation | Result |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `tools_off_repeat` | 2 | 2 | `chat_final`, `chat_final` | `34`, `54` | `6`, `30` | yes | complete |
| `tools_on_repeat_no_tool_needed` | 2 | 2 | `route`, `route` | `345`, `365` | `6`, `341` | yes | complete |
| `tools_on_same_session_repeat` | 2 | 4 | `route`, `chat_final_retry`, `route`, `chat_final_retry` | `351`, `351`, `400`, `400` | `336`, `350`, `347`, `399` | yes | complete, length-limited |
| `tools_on_after_reset` | 2 | 2 | `route`, `route` | `345`, `345` | `336`, `344` | yes | complete |
| `tools_on_off_switch` | 3 | 3 | `chat_final`, `route`, `chat_final` | `34`, `365`, `74` | `6`, `6`, `6` | yes | complete |
| `list_directory_repeat` | 1 | 2 | `route`, `final_from_tool` | `350`, `457` | `6`, `346` | yes | runner timeout after useful diagnostics |
| `system_info_repeat` | 1 | 1 | `route` | `351` | `335` | no footer before timeout | runner timeout |

## Phase Cost

| Phase | Calls | Avg prompt tokens | Avg cached tokens | Avg evaluated tokens | Cached ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chat_final` | 4 | 49.0 | 12.0 | 37.0 | 24.5% |
| `route` | 9 | 357.4 | 228.6 | 128.9 | 63.9% |
| `chat_final_retry` | 2 | 375.5 | 374.5 | 1.0 | 99.7% |
| `final_from_tool` | 1 | 457.0 | 346.0 | 111.0 | 75.7% |

Interpretation:

- `route` is the dominant prompt-size phase for tools-on turns.
- Warm `route` calls can reuse most tokens, but cold or invalidated `route` calls still evaluate roughly the whole tools-on prefix.
- `chat_final_retry` reuses almost all prompt tokens, but it still adds an extra model pass and generation latency.
- `final_from_tool` is more expensive than route after tool evidence because it includes tool result context.

## Tools-On Overhead

`tools_off_repeat` uses one `chat_final` call per request:

- first request: `34` prompt tokens, `28` evaluated
- second request: `54` prompt tokens, `24` evaluated

`tools_on_repeat_no_tool_needed` uses one `route` call per request:

- first request: `345` prompt tokens, `339` evaluated
- second request: `365` prompt tokens, `24` evaluated

The cold tools-on overhead is clear: roughly `+311` prompt tokens compared with the first tools-off request. After the first tools-on call, backend cache reuse is high for the repeated route prefix, and evaluated tokens drop to the same order as tools-off.

This means tools-on cost has two distinct components:

- cold prefix cost from the large tool/runtime prompt
- pass-count cost when route does not directly produce the visible answer

## Stable Prefix Findings

Observed stable prefix hashes:

| Phase/tools mode | Stable prefix hash |
| --- | --- |
| `chat_final` / tools off | `86be2d636b3c89b1` |
| `route` / tools on | `8d72bd08f2f92266` |
| `chat_final_retry` / tools on | `8d72bd08f2f92266` |
| `final_from_tool` / tools on | `8d72bd08f2f92266` |

The stable prefix hash stayed stable within each mode. Prefix mismatch events only reported:

- `conversation_prefix`
- `full_prompt`

That is expected in multi-turn sessions because conversation history grows. There is no evidence in this run that tool schema, capability summary, or runtime policy drift is causing poor reuse.

## Metric Coherence

The Phase 1B collector made footer correlation visible:

- Completed scenarios show `footer_correlation_present=yes`.
- Per-call `cached_tokens` and `evaluated_tokens` are internally coherent: `evaluated_tokens = prompt_tokens - cached_tokens`.
- `reused_tokens` is still equivalent to `cached_tokens`, because the backend exposes only cached token count.
- Runner timeouts can still occur after useful diagnostic events are emitted; timeout means the scenario exceeded the script budget, not that correlation failed.

The main remaining ambiguity is semantic rather than structural: `cached_tokens` is a backend metric and may not fully describe reusable KV prefix policy. For Phase 2 decisions, it is sufficient to distinguish prompt instability from backend reuse behavior.

## Key Findings

1. `tools_on_same_session_repeat` performs two model calls per user request:
   `route` followed by `chat_final_retry`.

2. The repeated `chat_final_retry` calls have very high cache reuse, but they still cost an extra generation pass and create latency independent of prefix evaluation.

3. `tools_on_repeat_no_tool_needed` can be a single `route` call per request. The first call pays the large tools-on prefix; the second call reuses it well.

4. `/reset` does not obviously destroy route-prefix reuse in this run. `tools_on_after_reset` shows high cache on both route calls.

5. Switching tools on/off invalidates reuse as expected. Treat tool mode as a separate cache/prefix domain.

6. Dedicated tools reduce tool-output noise, but `final_from_tool` still has real cost because it must include evidence context and generate the final answer.

7. The stable prefix is reusable in principle. The problem observed here is not prompt instability.

## Cost Classification

| Possible cause | Evidence | Classification |
| --- | --- | --- |
| Multi-pass overhead | `tools_on_same_session_repeat` has `route` + `chat_final_retry` per request | significant |
| Large tools prefix | tools-on route prompt ~`345` tokens vs tools-off chat ~`34` tokens | significant cold-start cost |
| Missing KV reuse | warm repeated route calls cache well; stable prefix hash is stable | not primary in this run |
| Metric ambiguity | `reused_tokens` maps to `cached_tokens`; backend semantics remain coarse | residual |
| Backend cache behavior | cache improves after repeated prefixes but invalidates across mode switches | expected, needs backend-specific care |

## Recommendation

Do not start with a KV cache optimization patch.

The immediate next analysis should focus on the tools-on no-tool-needed route path and why it sometimes falls into `chat_final_retry`. Reducing avoidable model passes is likely safer and higher impact than adding explicit prefix/KV reuse now, because:

- the stable prefix is already stable
- repeated route calls already show high cache reuse
- extra passes add generation latency even when prompt tokens are cached
- route/final retry behavior is easier to reason about before touching cache semantics

Recommended next phase:

1. Analyze route outcomes that lead to `chat_final_retry`.
2. Separate cases where route is correctly acting as final answer from cases where route output is unusable or length-limited.
3. Measure with a normal output budget, not only `--max-tokens 32`, to avoid over-attributing length-limited diagnostics.
4. Keep KV reuse work paused until route pass count and finalization behavior are characterized.

Proceed to KV/prefix optimization only after the multi-pass route path is understood and benchmarked.

