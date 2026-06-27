# Web Search A/B Regression Check

## Scope

This report compares the tools-on web path between:

- A: `origin/main` at `a38e5eb`
- B: local `local-main-integration-test` at `8a21619`

The goal is to determine whether the observed manual slowdown on web search is caused by:

- an extra model pass
- web provider/network latency
- a longer final answer
- route contract changes affecting web routing

No runtime behavior, prompt behavior, backend behavior, KV/cache behavior, or tags were changed during this check.

## Deterministic Harness Results

The deterministic harness used fake backend outputs and mocked web/fetch tool execution. It verifies controller behavior without network or model variability.

| Scenario | A phases | A calls | B phases | B calls | Tool | Result |
| --- | --- | ---: | --- | ---: | --- | --- |
| chat normal tools-off | `route` | 1 | `route` | 1 | none | same |
| read file valid | `route -> final_from_tool` | 2 | `route -> final_from_tool` | 2 | `exec_shell_full_command` | same |
| web parenthesized valid | `route -> final_from_tool` | 2 | `route -> final_from_tool` | 2 | `exec_shell_full_command` via `orbit-web-search` | same |
| web normalized valid | `route -> final_from_tool` | 2 | `route -> final_from_tool` | 2 | `exec_shell_full_command` via `orbit-web-search` | same |
| fetch_url valid | `route -> final_from_tool` | 2 | `route -> final_from_tool` | 2 | `fetch_url` | same |
| explicit web fallback | `route -> route_retry -> final_from_tool` | 3 | `route -> route_retry -> final_from_tool` | 3 | `exec_shell_full_command` via `orbit-web-search` | same |

Conclusion from harness:

- The runtime still supports the normalized path.
- Valid web route output uses `initial_tool_calls` and skips the intermediate `tool_call` model pass.
- `web_search_results` is reinjected as a normal tool result after internal `orbit-web-search` execution.
- The fallback `explicit_web_search` path legitimately uses `route_retry`.

## Live A/B Results

Live runs used the local server at `http://127.0.0.1:12120` with tools enabled.

### Generic Web Search

Prompt:

```text
search online for information about Mario Nobile
```

| Metric | A: origin/main | B: local integration |
| --- | ---: | ---: |
| wall time | 82.793s | 148.096s |
| model calls | 2 | 3 |
| phases | `route -> final_from_tool` | `route -> route_retry -> final_from_tool` |
| route completion tokens | 18 | 14 |
| final completion tokens | 49 | 47 |
| tool | `exec_shell_full_command` | `exec_shell_full_command` |
| tool wall time | 1.093s | 1.044s |
| tool result size | 2435 chars | 2365 chars |
| route_retry present | no | yes |

Additional B diagnostic with `ORBIT_KV_DIAG=1`:

| B pass | Phase | Outcome | Decision | Prompt tokens | Cached | Evaluated | Completion | Wall |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `route` | `route_other_retry` | null | 474 | 457 | 17 | 17 | 6.546s |
| 2 | `route_retry` | `route_parsed_tool` | `FILESYSTEM` | 188 | 4 | 184 | 17 | 19.117s |
| 3 | `final_from_tool` | n/a | n/a | 929 | 4 | 925 | 48 | 90.248s |

The route retry reason was `explicit_web_search`.

### Explicit Fetch URL

Prompt:

```text
fetch https://example.com and summarize it in one sentence
```

| Metric | A: origin/main | B: local integration |
| --- | ---: | ---: |
| wall time | 21.778s | 4.034s |
| model calls | 2 | 1 |
| phases | `route -> final_from_tool` | `route` |
| tool | `fetch_url` | none |
| tool wall time | 0.165s | n/a |
| B route outcome | n/a | `route_direct_final_stop` |

Additional B diagnostic:

| B pass | Phase | Outcome | Decision | Prompt tokens | Cached | Evaluated | Completion | Wall |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `route` | `route_direct_final_stop` | null | 478 | 457 | 21 | 7 | 3.762s |

This is not a performance regression, but it is a correctness concern for explicit URL fetch requests: the route contract allowed a direct short answer where a tool decision was expected.

## Diagnosis

The slowdown in the web search scenario is caused by an extra model call on B:

```text
route -> route_retry -> final_from_tool
```

instead of the expected valid route path:

```text
route -> final_from_tool
```

It is not caused by the web provider or network:

- A web tool wall time: 1.093s
- B web tool wall time: 1.044s

It is not caused by a substantially longer final answer:

- A final completion tokens: 49
- B final completion tokens: 47 to 48

The route contract patch appears to alter model routing for web/tool tasks. In the observed B web run, the initial route emitted a short non-decision answer, which triggered the explicit web-search guard and `route_retry`. In the observed B fetch_url run, the route emitted a short direct final answer and no tool was used.

## Comparison With Normalized Tool Paths

When the route output is valid, all normalized tool paths behave as intended:

```text
route -> tool execution -> final_from_tool
```

This applies to:

- parenthesized `orbit-web-search`
- normalized `call:orbit-web-search{...}`
- `fetch_url`
- file read via shell content evidence

The regression is therefore not in `web_search_results` reinjection or `initial_tool_calls`. It is in the model's first route decision under the current route contract.

## Recommendation

Do not merge the current route contract behavior as-is.

Smallest safe next step:

- refine the route contract so direct short answers are allowed only for no-tool tasks
- make explicit that web/search/fetch/URL requests must return a tool decision, not a direct short answer
- keep this model-guided: no deterministic runtime auto-routing, no hardcoded prompt mapping, no runtime-authored final answers

Validation criteria for the refinement:

- generic web search returns `route_parsed_tool` directly when the route emits a valid tool decision
- explicit fetch URL returns a `fetch_url` decision, not `route_direct_final_stop`
- valid WEB remains `route -> final_from_tool` with 2 model calls
- fallback explicit web search remains allowed as `route -> route_retry -> final_from_tool`
- no changes to backend, KV/cache, MTP, final policy, or tool selection policy

## After Refinement

The route contract was refined without adding runtime auto-routing:

- direct short answers are explicitly limited to no-tool tasks that need no external evidence
- web/search/latest/current/online and URL fetch/read/open/explain/summarize/analyze requests are explicitly tool tasks
- file read/explain/summarize/analyze requests are explicitly content-evidence tasks, not directory-listing tasks
- directory listing remains reserved for user requests that ask to list files or inspect structure

No backend, KV/cache, MTP, final policy, tool selection runtime, or deterministic routing changes were introduced.

### Smoke Outcomes

| Scenario | Phases | Calls | Route outcome | Tool | Retry | Result |
| --- | --- | ---: | --- | --- | --- | --- |
| web search generic | `route -> final_from_tool` | 2 | `route_parsed_tool` | `exec_shell_full_command` / `orbit-web-search` | no | PASS |
| fetch URL | `route -> final_from_tool` | 2 | `route_parsed_tool` | `fetch_url` | no | PASS |
| `what is 2+2?` | `route -> chat_final` | 2 | `route_parsed_chat` | none | no | PASS correctness, slower than ideal |
| `hi` | `route -> chat_final` | 2 | `route_parsed_chat` | none | no | PASS correctness, slower than ideal |
| `Who was Dante Alighieri?` | `route -> chat_final` | 2 | `route_parsed_chat` | none | no | PASS |
| `read README.md and explain it` | `route -> final_from_tool` | 2 | `route_parsed_tool` | `exec_shell_full_command` / `cat README.md` | no | PASS |
| `summarize README.md` | `route -> final_from_tool` | 2 | `route_parsed_tool` | `exec_shell_full_command` / `cat README.md` | no | PASS |
| `list files in the workdir` | `route -> final_from_tool` | 2 | `route_parsed_tool` | `list_directory` | no | PASS |

Observed `route_no_decision_length_retry`: `0`.

The refinement restores the expected web/fetch and file-content route shape while preserving model-guided routing. The model still chooses the tool decision; the runtime does not map prompts to tools deterministically.

Residual caveat:

- trivial no-tool prompts such as `what is 2+2?` and `hi` currently choose compact `{"route":"CHAT"}` and therefore use two model calls
- this is a performance caveat, not a correctness/evidence blocker
- do not fix it by weakening file/web evidence routing or adding runtime deterministic shortcuts
