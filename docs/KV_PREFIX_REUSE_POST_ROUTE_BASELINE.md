# KV Prefix Reuse Post-Route Baseline

Commit analyzed: `f6b5afb` (`origin/main`, PR #51 merged)

Server mode: local `orbit server`, default no-MTP path.

Diagnostics:

- `ORBIT_KV_DIAG=1`
- `ORBIT_KV_DIAG_FILE=/tmp/orbit_kv_post_route_53mbp_e3/*.jsonl`
- Runtime behavior, prompts, backend, KV/cache behavior, and tool selection were not changed.
- Benchmark artifacts were kept in `/tmp` and are not part of this repository.

## Summary

PR #51 fixed the route-length retry issue for the tested path. Across this baseline,
`route_no_decision_length_retry` stayed at `0`.

The main remaining cost is not route retry. It is:

- the large tools-on route prefix, around `706-715` prompt tokens for simple requests
- inconsistent backend cache reuse for the same stable route prefix across one-shot requests
- extra model passes for model-guided evidence acquisition, especially file-read requests
- conversation prefix changes in same-session repeat turns, which prevent full-prompt reuse even when the stable route prefix is unchanged

## Scenario Table

| Scenario | Wall | Calls | Phases | Route outcome | Prompt tokens by call | Cached tokens by call | Evaluated tokens by call | Completion tokens by call | Finish | Prefix mismatch | Tool |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| `tools_off_chat_short` | 4.0s | 1 | `chat_final` | n/a | `34` | `24` | `10` | `9` | `stop` | 0 | none |
| `tools_on_chat_hi` | 67.2s | 2 | `route -> chat_final` | `route_parsed_chat` | `706, 34` | `4, 4` | `702, 30` | `5, 9` | `stop, stop` | 0 | none |
| `tools_on_trivial_math` | 60.0s | 1 | `route` | `route_direct_final_stop` | `712` | `4` | `708` | `1` | `stop` | 0 | none |
| `tools_on_chat_dante` | 88.6s | 2 | `route -> chat_final` | `route_parsed_chat` | `712, 40` | `696, 4` | `16, 36` | `5, 256` | `stop, length` | 0 | none |
| `tools_on_web_search` | 121.8s | 2 | `route -> final_from_tool` | `route_parsed_tool` | `713, 1168` | `4, 709` | `709, 459` | `14, 38` | `stop, stop` | 0 | `orbit-web-search "Mario Nobile"` |
| `tools_on_fetch_url` | 31.2s | 2 | `route -> final_from_tool` | `route_parsed_tool` | `715, 842` | `696, 711` | `19, 131` | `9, 34` | `stop, stop` | 0 | `fetch_url` |
| `tools_on_file_read` | 90.5s | 4 | `route -> tool_call -> tool_call -> final_from_tool` | `route_parsed_tool` | `712, 558, 589, 213` | `696, 0, 479, 0` | `16, 558, 110, 213` | `8, 20, 15, 20` | `stop, stop, stop, stop` | 1 | `cat README.md` |
| `tools_on_listing` | 84.3s | 2 | `route -> final_from_tool` | `route_parsed_tool` | `711, 818` | `4, 707` | `707, 111` | `9, 47` | `stop, stop` | 0 | `list_directory` |
| `same_session_repeat_tools_on` | 68.3s | 3 | `route -> chat_final -> route` | `route_parsed_chat`, `route_direct_final_stop` | `706, 34, 726` | `696, 4, 4` | `10, 30, 722` | `5, 9, 9` | `stop, stop, stop` | 2 | none |

## Tools-On Overhead

The tools-off chat baseline used `34` prompt tokens and completed in one `chat_final`
call.

The simplest tools-on chat request used a route prompt of `706` tokens before the
normal `chat_final` call. That is the dominant overhead for no-tool-needed
tools-on requests.

The route contract now prevents long prose in route and avoids
`route_no_decision_length_retry`, but every tools-on request still pays the route
classification prefix unless it can correctly finish directly from the route pass.

## Cache Ratio By Phase

Observed cache ratios:

- `chat_final` tools-off short: `24/34` cached, about `71%`
- cold or poorly reused `route`: often `4/706-715` cached, about `1%`
- warm `route`: `696/706-715` cached, about `97-99%`
- `final_from_tool` after compact tools: `707-711` cached for `818-842` prompt tokens, about `84-86%`
- `final_from_tool` after web search: `709/1168` cached, about `61%`
- file-read internal `tool_call` pass 2: `0/558` cached
- file-read internal `tool_call` pass 3: `479/589` cached, about `81%`
- file-read `final_from_tool`: `0/213` cached

This shows that prefix reuse can work for the route prefix, but it is not reliable
across all one-shot requests or across all phases.

## Cold Vs Warm

The stable route prefix hash stayed constant for tools-on route calls:

```text
17217847d0dc9f1f
```

Despite the stable hash, cold/warm behavior varied:

- `tools_on_chat_hi`, `tools_on_trivial_math`, `tools_on_web_search`, and `tools_on_listing` evaluated nearly the entire route prompt.
- `tools_on_chat_dante`, `tools_on_fetch_url`, `tools_on_file_read`, and the first route in `same_session_repeat_tools_on` reused most of the route prefix.

The likely cause is backend/session cache state rather than prompt instability in
the stable prefix. The diagnostics did not report `stable_prefix_hash` changes for
the route prefix.

## Same-Session Repeat

The same-session repeat produced two user requests:

1. request 1: `route -> chat_final`
2. request 2: `route`

The first route call reused the stable route prefix:

```text
prompt_tokens=706 cached_tokens=696 evaluated_tokens=10
```

The second route call did not:

```text
prompt_tokens=726 cached_tokens=4 evaluated_tokens=722
```

Diagnostics reported prefix mismatch events for:

- `conversation_prefix`
- `full_prompt`

The stable route prefix hash did not change. The conversation prefix changed, as
expected after the first turn. This suggests the backend is not reusing the stable
route prefix independently once the conversation prefix changes before or around
the reusable segment.

## What Is Already Cached

Reliable or useful reuse was observed in:

- warm route calls with identical stable route prefix
- `final_from_tool` for `fetch_url` and `list_directory`
- the second internal file-read `tool_call` pass
- the tools-off short chat prompt

The dedicated tools remain useful:

- `fetch_url` stayed compact and reused most of the route/tool prefix.
- `list_directory` stayed compact and reused most of the `final_from_tool` prefix.
- web search stayed at the correct `route -> final_from_tool` path, though provider/tool output size still affects final prompt size.

## What Is Not Reused

The main gaps are:

- one-shot tools-on route prompts sometimes evaluate the full `706-715` token prefix
- same-session second route call can lose reuse despite an unchanged stable route prefix
- file-read `tool_call` and final phases use different stable prefixes, limiting reuse across phases
- conversation history changes cause `conversation_prefix_hash` and `full_prompt_hash` changes, which may prevent backend longest-prefix reuse

No evidence in this baseline points to `route_no_decision_length_retry` as the
remaining performance issue.

## Recommendation For Phase 3B

Do not implement KV optimization yet.

The next safe phase should be analysis/instrumentation focused on prefix layout and
backend cache semantics:

1. Confirm whether the backend cache only reuses a strict longest prefix from token
   position zero.
2. Measure whether the stable route prefix appears before or after conversation
   content in the exact serialized prompt sent to the backend.
3. Add diagnostics for token position ranges of stable components without logging
   raw prompt content.
4. Compare a no-behavior-change prompt assembly experiment in a throwaway branch:
   stable route/tool policy first, dynamic conversation after it.
5. Keep any Phase 3B work diagnostics-only until it proves that prompt layout, not
   backend cache behavior, is the blocker.

If Phase 3B shows that stable policy/tool definitions are not at the reusable
prefix start, then the smallest future optimization candidate is a prompt-layout
refactor that preserves content and model-guided behavior while placing stable
route/tool instructions in the true prefix. That must be benchmarked before merge.
