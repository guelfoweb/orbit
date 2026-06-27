# Route Contract Phase 3 Plan

Status: analysis plus local Phase 3 contract patch validation.

The local Phase 3 patch changes only the model-facing route contract and parser support for the explicit model-selected `{"route":"CHAT"}` decision. It does not add runtime auto-routing, prompt-specific semantic shortcuts, forced tools, runtime-authored final answers, backend changes, MTP changes, KV/cache changes, or final-policy changes.

## Analyzed Build

- Diagnostic build: `37a6f01f06337a3c5b382ba9c34d9da5b348b4ec`
- Base commit: `a38e5eb085cccbbe92fba3399a48ce56117ff625`
- Backend: native `orbit server`
- Mode: no-MTP default
- Diagnostics: `ORBIT_KV_DIAG=1` with `kv_diag_route_outcome`
- Command style: one-shot `orbit --tools on --no-render-markdown --max-tokens 160 --workdir workdir <prompt>`
- Diagnostic artifacts: local files under `/tmp`, not committed

The one-shot runs use separate processes, so `request_id` values reset per run. Rows below are mapped by execution order, not by `request_id`.

## Scenarios Tested

| Scenario | Prompt class | Route outcome | Route finish | Route output tokens | Model calls | Total evaluated tokens | Approx wall time |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `hi` | short chat | `route_direct_final_stop` | `stop` | 9 | 1 | 9 | 4.2s |
| `what is 2+2?` | short factual | `route_direct_final_stop` | `stop` | 1 | 1 | 16 | 1.7s |
| `hi, tell me something about yourself` | medium open chat | `route_no_decision_length_retry` | `length` | 128 | 2 | 17 | 105.2s |
| `Who was Dante Alighieri?` | general knowledge | `route_no_decision_length_retry` | `length` | 128 | 2 | 17 | 101.3s |
| local AI runtimes paragraph | longer explanation | `route_direct_final_stop` | `stop` | 88 | 1 | 25 | 33.9s |

## Retry Cost

For the two `route_no_decision_length_retry` cases:

| Scenario | Phase sequence | Prompt tokens / call | Cached tokens / call | Evaluated tokens / call | Completion tokens / call | Finish reasons |
| --- | --- | --- | --- | --- | --- | --- |
| self-description | `route`, `chat_final_retry` | `351`, `351` | `335`, `350` | `16`, `1` | `128`, `160` | `length`, `length` |
| Dante | `route`, `chat_final_retry` | `351`, `351` | `335`, `350` | `16`, `1` | `128`, `160` | `length`, `length` |

The cache is effective during the retry. The waste is not repeated prefill; it is the first route generation reaching the internal route budget and then requiring a second generation pass.

## Probable Cause

The current route prompt starts with:

```text
Answer normally unless shell is needed.
```

It also says:

```text
Return valid one-line JSON only.
```

In practice, the model treats direct no-tool requests as permission to answer in normal prose during the route pass. That is intentional enough to support one-pass tools-on chat for short answers, but it creates a failure mode:

1. The user asks a no-tool question.
2. The route pass writes a direct prose answer.
3. If the answer is short, it stops and Orbit accepts it as `route_direct_final_stop`.
4. If the answer is longer than the internal route budget, it ends with `finish_reason=length`.
5. Because no tool/chat decision was parsed and the prose is truncated, Orbit correctly runs `chat_final_retry`.

This is not a parser/schema mismatch. The route output is plain prose, not malformed tool JSON.

This is not primarily a KV reuse issue. The retry call is almost fully cached.

## Options Considered

| Option | Description | Pros | Risks |
| --- | --- | --- | --- |
| Keep current behavior | Accept route-as-final when it stops; retry only on truncation | No behavior change | Long route prose can waste a generation pass |
| Increase route budget | Let more prose complete in route pass | Fewer truncation retries for medium answers | More route generation cost; hides routing contract ambiguity |
| Force runtime auto-routing to chat for no-tool prompts | Runtime decides no tool is needed and skips route/prose | Potential latency win | Violates model-guided routing; deterministic task shortcut |
| Add semantic prompt heuristics | Detect prompts likely to produce long answers | May reduce retries | Hardcoded semantic fix; brittle |
| Require explicit `CHAT` route for direct answers | Route emits compact model-selected chat decision; final answer happens in `chat_final` | Model-guided and compact route | Makes short tools-on chat two-pass unless carefully scoped |
| Allow concise direct route answer, otherwise explicit `CHAT` | Route may answer only if short; for longer answers model returns compact chat decision | Preserves one-pass short chat and avoids long route prose | Requires prompt/contract change and benchmark validation |

## Recommended Phase 3 Change

The smallest safe candidate is a prompt/contract change, not runtime auto-routing:

```text
If no shell/tool is needed, either answer directly in one short sentence, or return a compact CHAT decision for a normal final answer pass. Do not write long prose in the route pass.
```

This keeps the model responsible for deciding whether the request needs tools or chat. It does not map user prompts deterministically to a tool or answer. It only tightens the technical contract of the route pass so the route phase stays compact.

The local implementation uses the explicit JSON decision `{"route":"CHAT"}`. The runtime only parses that model-emitted decision; it does not infer CHAT from the user prompt.

## Local Phase 3 Patch Smoke

Command style:

```text
PYTHONPATH=src ORBIT_KV_DIAG=1 ORBIT_KV_DIAG_FILE=/tmp/orbit_route_contract_phase3_matrix_final.jsonl python3 -m orbit.terminal.cli --tools on --no-render-markdown --max-tokens 512 --workdir . <prompt>
```

The diagnostic JSONL and terminal captures were kept under `/tmp` and are not repository artifacts.

The first contract draft over-preferred `{"route":"CHAT"}` and caused `read README.md and explain it` to route as chat. The final local contract now declares tool/evidence tasks first, then the no-tool direct-or-CHAT choice.

Three runs per scenario:

| Scenario | Prevalent outcome | Outcome count | Phase sequence | Route completion tokens | Model calls | Wall time |
| --- | --- | --- | --- | --- | --- | --- |
| `hi` | `route_parsed_chat` | 3/3 | `route`, `chat_final` | `5, 5, 5` | `2, 2, 2` | `9.1s, 8.7s, 8.6s` |
| `what is 2+2?` | `route_direct_final_stop` | 3/3 | `route` | `1, 1, 1` | `1, 1, 1` | `43.9s, 38.2s, 39.0s` |
| `hi, tell me something about yourself` | `route_parsed_chat` | 3/3 | `route`, `chat_final` | `5, 5, 5` | `2, 2, 2` | `104.9s, 106.8s, 111.9s` |
| `Who was Dante Alighieri?` | `route_parsed_chat` | 3/3 | `route`, `chat_final` | `5, 5, 5` | `2, 2, 2` | `211.1s, 215.3s, 243.6s` |
| local AI runtimes paragraph | `route_parsed_chat` | 3/3 | `route`, `chat_final` | `5, 5, 5` | `2, 2, 2` | `78.6s, 92.6s, 101.0s` |
| `read README.md and explain it` | `route_parsed_tool` | 3/3 | `route`, `final_from_tool`, `final_from_tool_retry` | `9, 9, 9` | `3, 3, 3` | `65.2s, 71.8s, 71.3s` |
| `list files in the workdir` | `route_parsed_tool` | 3/3 | `route`, `final_from_tool` | `9, 9, 9` | `2, 2, 2` | `42.2s, 41.8s, 44.3s` |

Observed route outcome count after the final local patch:

- `route_direct_final_stop`: 3
- `route_parsed_chat`: 12
- `route_parsed_tool`: 6
- `route_no_decision_length_retry`: 0

Direct route response classification:

- `what is 2+2?`: 3/3 direct route answers were brief/complete.
- No direct route answer was classified as long/risky.

Notes:

- `hi` still chooses compact `{"route":"CHAT"}` in this model/build, but `what is 2+2?` preserves one-pass direct final. The merge criterion does not require both trivial scenarios to remain direct.
- `Who was Dante Alighieri?` hit the normal final-answer `max_tokens` budget in the sampled runs. The route pass itself stayed compact and did not retry from `length`.

## Why Not KV First

The measured retry calls have high cache reuse:

- route prompt evaluation after warm cache is small
- `chat_final_retry` evaluated only about one token in the sampled runs
- wall time is dominated by generated tokens in the wasted route pass and the retry answer

Optimizing KV/prefix reuse would not remove the extra route generation. Reducing avoidable route prose is the more direct target.

## Validation Criteria For A Future Phase 3 Patch

A future patch must be rejected if it introduces runtime deterministic routing or semantic task shortcuts.

Required validation:

- `route_direct_final_stop` still works for very short direct answers.
- `route_no_decision_length_retry` decreases for medium no-tool prompts.
- Tool prompts still produce `route_parsed_tool`.
- Explicit URL/file/system/listing requests still route through model-selected tools.
- No prompt-specific hardcoding.
- No runtime-authored final answers.
- No backend/native/MTP/KV changes.
- `python3 -m unittest discover -s tests -q` passes.
- Live smoke with `/tools on`:
  - `hi`
  - `what is 2+2?`
  - `hi, tell me something about yourself`
  - `Who was Dante Alighieri?`
  - `read README.md and explain it`
  - `list files in the workdir`
  - explicit URL fetch

Recommended metrics:

- route outcome counts
- model calls per user request
- completion tokens generated in route
- wall time per scenario
- correctness of final visible answer

The next patch should be reversible and limited to the route contract plus tests. It should not include KV work.
