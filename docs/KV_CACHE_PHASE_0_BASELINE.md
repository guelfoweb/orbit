# KV Cache Phase 0 Baseline

Status: baseline report. No runtime, backend, prompt, or KV cache behavior was changed for this phase.

## Environment

- Date: 2026-06-27
- Commit tested: `4db269c8b2792d4fca17766b80aabf062dacbd9a`
- Branch: `kv-cache-reuse-phase-0-baseline`
- Client command: `PYTHONPATH=src python3 -m orbit.terminal.cli`
- Server: existing local `orbit server`
- Base URL: `http://127.0.0.1:12120`
- Model: `gemma4:12b-it-native`
- Context: 8192
- Backend runtime: `orbit-native`
- Backend mode: `no-mtp`
- MTP enabled: `no`
- Tools available: `exec_shell_full_command`, `fetch_url`, `list_directory`, `system_info`
- Rendering during capture: `--no-render-markdown`, to keep logs parseable

Notes:

- Runs used a warm server.
- Most one-shot runs used `--max-tokens 160`; `fetch_url_vatican` used `--max-tokens 96` but still reached a 256-token final pass through the runtime path.
- Multi-turn rows report the last visible footer for the scenario, not every internal pass.
- TTFT is not directly reported by the current CLI footer, so this baseline records wall time and backend footer metrics instead.

## `/status` Summary

Observed runtime status before the baseline:

```text
backend: orbit-native
backend_mode: no-mtp
session_id: default
threads: 6
threads_batch: 6
ctx_size: 8192
batch_size: 256
ubatch_size: 128
parallel_slots: 1
mtp_available: yes
mtp_enabled: no
model: gemma4:12b-it-native
```

## Scenario Results

| Scenario | Tools | Prompt | Tool Calls | Tokens | Cached | Cache | Prefill/s | Gen/s | Stop | Wall |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `chat_short_tools_off` | off | `hi, tell me something about yourself` | none | 40->160 | 39 | 98% | 3.2 | 3.3 | length | 48s |
| `chat_multiturn_tools_off` | off | 3 short turns | none | 248->25 | 108 | 44% | 13.5 | 3.6 | stop | 17s |
| `chat_tools_on_no_tool_needed` | on | `hi, tell me something about yourself` | none | 351->160 | 350 | 100% | 3.1 | 2.9 | length | 2m 4s |
| `file_read_readme` | on | `read README.md...` with `--workdir workdir` | `ListDir`, `ListDir recursive` | 450->23 | 0 | 0% | 12.7 | 3.2 | stop | 3m 31s |
| `file_read_readme_repo_root` | on | `read README.md...` with repo root workdir | `Read` | 757->35 | 351 | 46% | 12.3 | 3.1 | stop | 50s |
| `fetch_url_vatican` | on | explicit Vatican URL fetch | `Fetch` | 1724->256 | 388 | 23% | 11.3 | 2.8 | length | 3m 49s |
| `list_directory_flat` | on | `list files in the workdir` | `ListDir` | 457->47 | 346 | 76% | 12.5 | 3.2 | stop | 28s |
| `list_directory_recursive` | on | `list files recursively in the workdir` | `ListDir recursive` | 581->104 | 347 | 60% | 12.5 | 3.1 | stop | 56s |
| `system_info` | on | `tell me the specs of this computer` | `SystemInfo` | 547->149 | 347 | 63% | 12.3 | 3.5 | stop | 1m 8s |
| `long_prompt_tools_off` | off | controlled repeated long prompt | none | 843->41 | 6 | 1% | 13.2 | 3.4 | stop | 1m 15s |
| `same_session_repeat` | on | same specs prompt twice in one REPL session | `SystemInfo` twice | 909->134 | 4 | 0% | 13.2 | 3.3 | stop | 3m 28s |
| `reset_reuse` | on | `hi`, `/reset`, `hi` | none | 345->9 | 344 | 100% | 3.5 | 3.6 | stop | 2s |
| `tools_on_off_switch` | off/on/off | `hi`, `/tools on`, `hi`, `/tools off`, `hi` | none | 74->22 | 6 | 8% | 14.3 | 3.7 | stop | 10s |

## Caveats

- `file_read_readme` with `--workdir workdir` is a setup issue: the clean fixture workdir did not contain `README.md`. The model correctly used directory listing and reported that the file was absent. The actual file-read baseline is `file_read_readme_repo_root`.
- Some rows stopped because of the configured output cap. They are still useful for prompt/cache/prefill measurement, but not for final-answer quality comparison.
- Multi-turn rows only show the final footer. Per-turn cache analysis needs a more precise collector.
- The CLI footer exposes cached tokens and prefill/generation throughput, but not first-token latency as a separate metric.

## Initial Observations

1. Tools-on no-tool-needed prompts have a measurable static prompt overhead.

   `chat_short_tools_off` used 40 input tokens, while `chat_tools_on_no_tool_needed` used 351 input tokens for the same prompt. This is the clearest direct cost of tool schema, runtime policy, and capability summary exposure.

2. Dedicated compact tools keep common tool paths bounded.

   `list_directory_flat`, `list_directory_recursive`, and `system_info` stayed under 600 input tokens in the final measured pass. These are good examples of reducing noisy shell reinjection without deterministic auto-routing.

3. Direct URL fetch remains a high-cost path.

   `fetch_url_vatican` used 1724 input tokens and took 3m49s. This is expected because the page content is real evidence and must be reinjected, but it is a strong candidate for measuring prefix reuse and payload shaping separately.

4. Repeated tools-on session reuse was not reliable in this run.

   `same_session_repeat` ended with 909 input tokens and only 4 cached tokens. This is the most important finding for the KV phase because a repeated same-session prompt should be one of the easiest places to observe prefix reuse if the prompt prefix remains stable.

5. Reset behavior can still show high cache reuse for tiny prompts.

   `reset_reuse` showed 344 cached tokens out of 345 in the final footer. This needs closer instrumentation before drawing conclusions because the prompt was small and the row only captures the final turn.

## Stable Prompt Prefix Candidates

Observed candidates for reuse:

- base system prompt
- tools-on runtime policy
- tool definitions for the stable tool set
- capability summary when `/tools refresh` has not run
- session prefix before recent turns

The biggest currently visible prefix delta is tools off vs tools on.

## Frequently Changing Sections

Observed dynamic sections:

- user message
- tool calls
- tool results
- file/URL/directory/system evidence
- conversation history in multi-turn sessions
- mode transitions such as `/tools on`, `/tools off`, and `/reset`

These should be treated as invalidation boundaries until measured more precisely.

## Phase 1 Recommendation

Proceed to phase 1 diagnostics only, not an optimization patch.

Recommended next step:

- add or use a non-invasive benchmark collector that captures per-turn and per-pass footer metrics
- keep prompts, tool outputs, system prompt, and runtime behavior unchanged
- measure prefix stability for tools-on no-tool-needed and same-session repeat
- identify why `same_session_repeat` produced near-zero cached-token reuse in this run
- separately measure direct URL payload cost vs static prefix cost

Do not proceed to a KV reuse implementation until:

- per-pass metrics are available
- the stable prefix is identified exactly
- invalidation keys are defined for `/reset`, tools mode, thinking mode, capability refresh, memory refresh, and tool evidence
- cross-session contamination risks are ruled out

## Decision

Proceed to phase 1: yes, for diagnostics and benchmark instrumentation only.

Proceed to KV cache reuse patch: no.
