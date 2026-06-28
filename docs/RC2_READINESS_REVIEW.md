# RC2 Readiness Review

Commit analyzed: `7e810a5` (`origin/main`, PR #57 merged).

This document prepares for a possible RC2. It does not publish a release, create
a tag, or change version metadata.

## Feature State

`ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1` is available as an opt-in experiment.

- Default: OFF
- Scope: native backend, tools-on route pass only
- Excluded: `chat_final`, `final_from_tool`, `tool_call`
- No deterministic routing
- No file/web/fetch/listing special-casing
- No prompt semantic change
- Fallback: baseline behavior on mismatch, restore failure, invalid checkpoint,
  or unsupported backend path

Checkpoint memory observed locally: about `238 MB`.

## Tests

Required release-hardening tests:

```bash
PYTHONPATH=src python3 -m unittest tests.test_native_bindings tests.test_prefix_anchor tests.test_prefix_anchor_probe tests.test_kv_diag tests.test_native_chat_template -q
PYTHONPATH=src python3 -m unittest tests.test_payloads tests.test_native_server_protocol tests.test_native_server_think tests.test_llama_server_backend -q
python3 -m unittest discover -s tests -q
python3 -m compileall -q src tests scripts
git diff --check
```

Status: PASS in this branch.

## Smoke OFF/ON

Native smoke should be run with the same model and workdir used for PR #57:

- OFF: `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT` unset
- ON: `ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT=1`

Scenarios:

1. `hi`
2. `hi` repeat
3. `what is 2+2?`
4. `Explain in two short paragraphs how a local AI runtime can reduce latency after the first turn.`
5. `Explain the tradeoff between correctness and latency in a local agent runtime.`
6. `list files in the workdir`
7. `read README.md and explain it`
8. valid web search
9. valid `fetch_url`

Status: PASS in this branch.

| Scenario | Flag | Phases | Route cached/evaluated | Request cached/evaluated | Wall ms | Outcome |
| --- | --- | --- | --- | --- | ---: | --- |
| `hi` | OFF | `route -> chat_final` | `0/706` | `4/736` | 54193 | baseline, no anchor |
| `hi` repeat | OFF | `route -> chat_final` | `4/702` | `8/732` | 62879 | baseline, no anchor |
| `what is 2+2?` | OFF | `route` | `4/708` | `4/708` | 63996 | direct route |
| medium latency prompt | OFF | `route -> chat_final` | `696/27` | `700/74` | 53721 | no route retry |
| medium tradeoff prompt | OFF | `route -> chat_final` | `4/714` | `8/756` | 178828 | cache miss, no route retry |
| listing | OFF | `route -> final_from_tool` | `4/707` | `711/886` | 141956 | `ListDir` preserved |
| README read | OFF | `route -> final_from_tool` | `696/16` | `1404/424` | 92601 | `Read:` content evidence |
| web search | OFF | `route -> final_from_tool` | `696/18` | `1406/455` | 116654 | `orbit-web-search` |
| fetch URL | OFF | `route -> final_from_tool` | `696/18` | `1406/149` | 51892 | `Fetch:` |
| `hi` | ON | `route -> chat_final` | `0/706` | `4/736` | 94347 | capture miss |
| `hi` repeat | ON | `route -> chat_final` | `693/13` | `697/43` | 13694 | restore hit |
| `what is 2+2?` | ON | `route` | `693/19` | `693/19` | 3613 | direct route, restore hit |
| medium latency prompt | ON | `route -> chat_final` | `693/30` | `697/77` | 45297 | restore hit, no route retry |
| medium tradeoff prompt | ON | `route -> chat_final` | `693/25` | `697/67` | 47442 | restore hit |
| listing | ON | `route -> final_from_tool` | `693/18` | `1386/211` | 62310 | `ListDir` preserved |
| README read | ON | `route -> final_from_tool` | `693/19` | `1386/442` | 113071 | `Read:` content evidence |
| web search | ON | `route -> final_from_tool` | `693/21` | `1386/455` | 123467 | `orbit-web-search` |
| fetch URL | ON | `route -> final_from_tool` | `693/21` | `1386/169` | 51595 | `Fetch:` |

ON anchor events:

- first ON route: capture miss, `checkpoint_size_bytes=238454176`
- subsequent ON routes: restore hit, `restore_used=true`
- route prefix: `693` cached tokens on restore
- `route_no_decision_length_retry`: `0` in all smoke scenarios

The two medium final answers ended with `finish_reason=length` because the smoke
used `--max-tokens 64`; this was not a route retry or repair path.

## Expected Acceptance Criteria

- OFF remains baseline.
- ON produces restore hits on repeated tools-on route calls.
- ON does not increase `model_calls`.
- ON does not increase repair/retry.
- `route_no_decision_length_retry` remains zero or does not regress.
- file-read preserves content evidence and does not use directory listing as a
  substitute.
- listing preserves `list_directory`.
- web/fetch remain `route -> final_from_tool` when provider/network succeeds.
- checkpoint memory cost remains documented.
- prefix-anchor remains default OFF.

## Risks

- First capture miss remains expensive.
- Checkpoint memory cost is material.
- Web smoke depends on provider/network behavior.
- The experiment is not ready to be default.

## Recommendation

Recommendation: ready for RC2 after normal release packaging checks.

Do not publish RC2 until:

- no unexpected worktree changes remain
- release notes explicitly state that prefix-anchor is opt-in and default OFF
- the release command/tagging step is explicitly authorized
