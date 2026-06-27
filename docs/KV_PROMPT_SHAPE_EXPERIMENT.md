# KV Prompt Shape Experiment

## Scope

Commit under test: local branch `kv-backend-envelope-diagnostics` on top of `origin/main` `8a7cdb8`, plus the local experiment patch.

This experiment is feature-flagged by:

```bash
ORBIT_KV_PROMPT_SHAPE_EXPERIMENT=1
```

The flag is off by default. With the flag off, prompt construction is unchanged.

The experiment does not change routing, tool selection, final policy, backend decoding, cache logic, or visible runtime behavior by default.

## Hypothesis

Diagnostics showed that low KV reuse in tools-on no-tool paths is caused by low backend-visible tokenized longest common prefix (LCP), not by slot identity, endpoint, stream mode, or `cache_prompt`.

The common miss pattern is:

```text
route prompt -> chat_final prompt -> next route prompt
```

The route prompt starts with the route/system/tools contract. The chat final prompt starts with the short chat system prompt. The next route call then shares only a very small token prefix with the previous chat final prompt.

## Medium prompts used

- `Explain in two short paragraphs how a local AI runtime can reduce latency after the first turn.`
- `Explain the tradeoff between correctness and latency in a local agent runtime.`

These prompts are:

- no-tool
- no-web
- no-file
- no-URL
- no-system-info
- neutral and non-personal

Both were verified with `ORBIT_KV_DIAG=1` to use `route -> chat_final` on the first turn.

## Experiment

When `ORBIT_KV_PROMPT_SHAPE_EXPERIMENT=1`, tools-on `chat_final` uses a route-prefixed final-answer system prompt:

```text
<route contract prefix>

Final answer phase: ... answer normally ... do not return JSON ... do not call tools ...
```

This keeps the backend-visible prefix stable between:

```text
route -> chat_final -> route
```

The first attempted version also applied the same shape to `final_from_tool`. That was rejected locally because it enlarged tool-final prompts and produced a timeout on the file-read repeat scenario. The narrower version only targeted tools-on `chat_final` no-tool paths.

## Benchmark

Server: native `orbit server`, no-MTP default, warm process.

Max output tokens: 96.

Diagnostics: `ORBIT_KV_DIAG=1`.

Scenario names intentionally avoid raw user prompt content.

Each scenario was run as three same-session turns per flag. The table reports aggregate request-level cached/evaluated token sums from `kv_diag_request_summary`.

### OFF Baseline

| Scenario | Phase sequence | Request cached/evaluated summary | Result |
| --- | --- | --- | --- |
| short chat | `route -> chat_final -> route -> route` | `4/736`, `4/722`, `722/24` | cache miss on route after chat final |
| trivial chat | `route -> route -> route` | `696/16`, `708/22`, `726/22` | already efficient |
| medium direct chat | `route -> route -> route` | `696/15`, `707/26`, `729/26` | route direct final, not the target path |
| listing | `route -> final_from_tool -> route -> route_retry -> final_from_tool -> route` | `1403/132`, `828/1419`, `704/299` | listing behavior preserved |
| file read | `route -> tool_call -> tool_call -> final_from_tool -> route -> final_from_tool -> route` | `1175/897`, `827/923`, `823/172` | content evidence preserved |
| web search | `route -> final_from_tool` repeated 3 times | `1406/455`, `1705/923`, `1420/2484` | valid 2-call web path |
| fetch URL | `route -> final_from_tool` repeated 3 times | `1410/153`, `1719/172`, `2047/172` | valid 2-call fetch path |

### ON Restricted

| Scenario | Phase sequence | Request cached/evaluated summary | Result |
| --- | --- | --- | --- |
| short chat | `route -> chat_final -> route -> route` | `1387/63`, `691/35`, `722/24` | improved target path |
| trivial chat | `route -> route -> route` | `696/16`, `708/22`, `726/22` | unchanged |
| medium direct chat | `route -> route -> route` | `696/15`, `707/26`, `729/26` | unchanged direct route |
| listing | `route -> final_from_tool -> route -> route_retry -> final_from_tool -> route` | `1403/132`, `828/1419`, `704/299` | unchanged |
| file read | `route -> tool_call -> tool_call -> final_from_tool -> route -> final_from_tool -> route` | `1175/897`, `827/923`, `823/172` | unchanged, content evidence preserved |
| web search | `route -> final_from_tool` repeated 3 times | `1406/455`, `1705/857`, `1420/2322` | unchanged 2-call web path |
| fetch URL | `route -> final_from_tool` repeated 3 times | `1410/153`, `1719/172`, `2047/172` | unchanged 2-call fetch path |

### Medium no-tool prompts that actually use `chat_final`

| Scenario | Flag | Request cached/evaluated summary | Extra passes | Result |
| --- | --- | --- | --- | --- |
| medium latency | `OFF` | `8/766`, `818/812`, `1717/97` | 2 x `route_no_decision_length_retry` | unstable |
| medium latency | `ON` | `2143/138`, `1451/71`, `1609/97` | 1 x `chat_final_completion_repair`, 2 x `route_no_decision_length_retry` | lower cost, but more phases |
| medium prefix | `OFF` | `701/63`, `808/802`, `1692/92` | 2 x `route_no_decision_length_retry` | unstable |
| medium prefix | `ON` | `2138/128`, `1441/61`, `1584/92` | 1 x `chat_final_completion_repair`, 2 x `route_no_decision_length_retry` | lower cost, but more phases |

## Findings

- `cached_tokens` continues to match backend tokenized LCP.
- The restricted experiment increases cached tokens in tools-on no-tool chat paths where route and chat-final phases alternate.
- The improvement is visible on the short chat target path: request one went from `4/736` cached/evaluated to `1387/63`.
- Trivial and direct-route chat paths were already efficient and remain unchanged.
- Valid web and fetch paths remain `route -> final_from_tool`.
- File-read evidence behavior remains model-guided and still obtains content evidence.
- The broad `final_from_tool` version should not be promoted: it increased prompt size and caused a timeout in local file-read repeat testing.
- The restricted `chat_final` version still introduces an extra `chat_final_completion_repair` pass on both medium no-tool prompts.
- The restricted version does not reduce `route_no_decision_length_retry` on the repeated medium prompts; it only makes the repeated passes cheaper once they happen.

## Risks

- The route-prefixed chat final prompt is semantically more complex than the default short chat prompt.
- Although it explicitly says not to return JSON or call tools, it still includes the route contract earlier in the same system message.
- The extra `chat_final_completion_repair` pass is a visible control-flow regression even when the final answer remains correct.

## Recommendation

Reject.

The smaller scoped experiment is not safe to promote in its current form. It improves cache reuse and wall time on the short chat target, but it also adds a completion-repair pass on medium no-tool prompts and does not remove repeated `route_no_decision_length_retry` on those scenarios.

The code patch should be removed. The report is still useful as evidence that:

- prompt-shape alignment can materially increase backend-visible LCP;
- broad prompt-shape changes around final phases are risky;
- future work should target a smaller semantic delta or a backend-side reuse mechanism instead of reusing the route contract inside `chat_final`.

Removal criteria met:

- extra phase introduced on medium no-tool prompts;
- no reduction of `route_no_decision_length_retry` on repeated medium prompts;
- benefit is strong but too narrow and too fragile for promotion.
