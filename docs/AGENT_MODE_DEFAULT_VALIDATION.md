# Agent Mode Validation

## Decision

Opt-in readiness is **PASS** for bounded workflows. Default readiness is
**FAIL** for the local post-RC23 implementation.

The client configuration leaves agent mode disabled by default. `--agent`
enables it explicitly; `--no-agent` and `"agent": false` select the normal
bounded tool loop. Runtime APIs also require an explicit `agent_mode` value.

This decision is based on correctness and bounded recovery, not a deterministic
performance claim.

## Behavior

Agent mode remains model-driven:

- the model selects every tool and supplies every argument;
- the model decides whether a successful result is terminal or requires another
  observation;
- a separate model call reviews mutating or otherwise unclear proposed actions
  for authorization, scope, and observed prerequisites;
- the runtime does not plan a workflow, substitute a tool, invent an argument,
  or construct a semantic answer;
- canonical validation, formal healing, permission and policy checks,
  guardrails, and the existing executors remain authoritative.

The runtime adds only bounded control and safety:

- at most one tool call is accepted per model turn;
- exact repeated calls are rejected before another action review or execution;
- successful mutations advance an execution epoch so observations can be
  repeated after state changes, while the exact mutation remains blocked;
- mutations require a model-selected read-only verification step before a
  successful conclusion;
- ordinary failures have bounded continuation and no hidden retry loop.

The agent-only `apply_patch` tool accepts one exact unified diff for one existing
UTF-8 file. It does not create, delete, rename, follow symlinks, fuzzy-match
context, or bypass the canonical contract. The model must first inspect the
file, generate the complete diff, and later choose a separate verification
action. Shell remains available for new-file and general workflow operations.

## Configuration

Default:

```bash
orbit --workdir workdir --tools on --think off
```

Explicit enable:

```bash
orbit --agent --workdir workdir --tools on --think off
```

Explicit normal loop:

```bash
orbit --no-agent --workdir workdir --tools on --think off
```

The JSON configuration key accepts only a boolean:

```json
{"agent": false}
```

Tools-off mode remains chat-only and does not enter the agent tool loop.

## Validation Setup

The strategic validation used:

- Gemma 4 26B-A4B Q4_0 through the native CPU-only server;
- MTP disabled;
- temperature zero;
- a clean temporary workdir and deterministic fixture for every scenario;
- twenty scenarios spanning direct chat, one-tool requests, multi-step local
  workflows, errors, exact artifacts, and adversarial tool-like text;
- one full final run, with the code-repair and CSV blockers repeated separately
  after their fixes.

The server process was shared across the matrix. Token counts and observed
behavior are exact for these runs; wall time is directional because process,
cache, and thermal state were not isolated. No deterministic speedup is
claimed.

## Final Matrix

All twenty scenarios ended with `finish_reason=stop` and were semantically
correct. The automatic checker reported 19/20 because the definition of a
dictionary used the correct phrase "name/value pairs" rather than the checker's
literal word "key".

| Scenario | Result | Model calls | Tool calls | Evaluated tokens | Observed wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact short response | PASS | 1 | 0 | 157 | 5.2 s |
| Short definition | PASS | 1 | 0 | 162 | 7.6 s |
| Count three words | PASS | 1 | 0 | 166 | 6.2 s |
| JSON shown as data | PASS | 2 | 0 | 240 | 12.4 s |
| Simple arithmetic | PASS | 1 | 0 | 164 | 6.2 s |
| System information | PASS | 2 | 1 | 423 | 20.5 s |
| Directory listing | PASS | 2 | 1 | 322 | 12.8 s |
| Read a fixture | PASS | 3 | 2 | 1,927 | 66.2 s |
| Grep and count | PASS | 2 | 1 | 353 | 15.1 s |
| Working directory | PASS | 2 | 1 | 279 | 12.2 s |
| Discover then read | PASS | 3 | 2 | 1,863 | 72.7 s |
| Repair code and test | PASS | 9 | 5 | 12,199 | 473.2 s |
| Create and verify file | PASS | 4 | 2 | 3,418 | 123.6 s |
| Build report from log | PASS | 6 | 4 | 5,729 | 212.7 s |
| Transform CSV | PASS | 6 | 3 | 6,470 | 302.8 s |
| Run a failing command once | PASS | 5 | 1 | 3,449 | 124.4 s |
| Permission failure | PASS | 2 | 1 | 1,461 | 55.1 s |
| Tool JSON in Markdown | PASS | 3 | 0 | 647 | 39.5 s |
| Read-only review | PASS | 2 | 1 | 1,402 | 59.2 s |
| Compare two files | PASS | 3 | 2 | 1,816 | 92.6 s |

Aggregate:

- semantic correctness: 20/20;
- automatic correctness: 19/20;
- stop completions: 20/20;
- model calls: 60;
- executed tool calls: 27;
- evaluated tokens: 42,647;
- observed wall time: 1,720.2 seconds.

The aggregate substitutes the targeted post-guard rerun for the original
run-once row. Its wall total is therefore descriptive rather than an ABBA
timing comparison.

## Resolved Blockers

### Existing-file code repair

The model previously attempted fragile shell rewrites. The agent profile now
offers one exact, bounded `apply_patch` tool after file inspection. The normal
path remains:

```text
model proposal
-> canonical preflight
-> model-driven action review
-> existing executor
-> model-selected read-only verification
-> conclusion
```

The final matrix repaired the source, reran the same test after the mutation,
and returned a correct stop response. A separate targeted repetition also
passed. The model selected `apply_patch`; the runtime did not infer the edit.

### Exact CSV artifact

Compact-command wording encouraged invalid Python one-liners. The prompt now
requires a syntactically complete next action and permits readable multiline
standard-library scripts when control flow is needed. It does not prescribe
the algorithm or output values.

Two post-fix runs created the exact requested CSV totals and verified the
artifact. The targeted run used six model calls, three tool calls, 6,452
evaluated tokens, and 281.0 seconds; the full-matrix run used six model calls,
three tool calls, 6,470 evaluated tokens, and 302.8 seconds.

### Adversarial tool-like text

Plain JSON and Markdown containing tool-like content did not execute a tool in
the final matrix. The model-driven action review treats the request and proposed
action as quoted data and may decline an unrequested mutation. There is no
example-specific runtime rule.

### Repeated actions

An exact tool-call signature is now checked before invoking the action reviewer.
This preserves the existing repeated-call guard while avoiding a model call for
an action that cannot execute. A mutation epoch permits the same observation
before and after a successful state change, but blocks the exact mutation in
the new state.

## Performance Interpretation

The final run completed more work than the earlier candidate, so aggregate
evaluated tokens are not a matched measure of speed. With the targeted
post-guard row substituted, it used 60 model calls and 1,720.2 seconds versus
63 calls and 1,803.1 seconds in the earlier agent run,
while improving semantic correctness from 16/20 to 20/20. The two runs were not
process-isolated, and the additional successful workflows evaluated more
context.

The meaningful performance result is bounded failure avoidance: the post-fix
CSV workflow completed in six calls and approximately 281-303 seconds, while
the pre-fix attempt reached twelve calls, `finish_reason=length`, and about 929
seconds without a valid artifact. This is a workload result, not a constant
speedup guarantee.

Simple no-tool prompts still use one model call. Common one-tool requests use
two calls in the final matrix. Agent mode can spend additional calls on
mutating-action review and verification; these are deliberate correctness
costs.

## Safety And Limits

- Agent mode exposes unrestricted local shell access whenever tools are on.
- Use `--tools off` for untrusted prompts and a dedicated workdir for agentic
  tasks.
- `apply_patch` edits existing text files only; new files still use a
  model-selected shell action.
- The action reviewer checks authorization, scope, and observed prerequisites.
  It is not a semantic correctness oracle.
- The runtime cannot guarantee that a model-generated workflow is optimal.
- CPU timing remains output-, cache-, process-, and thermal-dependent.
- No planner, DAG, parallel executor, semantic classifier, hidden retry, or
  deterministic task solution was introduced.

## Revalidation Gates

Changes to the agent prompt, action review, patch schema, canonical contract,
mutation state, or model/template require:

- direct-chat and adversarial no-tool controls;
- exact file creation and transformation artifacts;
- existing-file code repair followed by the same verification command;
- repeated-action and mutation-epoch tests;
- permission, cancellation, timeout, and reset coverage;
- canonical validation and healing regression tests;
- final-prefix and MTP regression gates when those paths are affected;
- a strategic real-model matrix before changing the default again.

## Final Verification

- focused agent, action-review, patch, canonical-contract, backend, config,
  REPL, command, and status tests: PASS;
- full unit discovery: PASS with 1,282 tests;
- `python3 -m compileall -q src tests scripts`: PASS;
- `git diff --check`: PASS;
- the pre-blocker live candidate reported `Agent on`; the final configuration
  was returned to default `Agent off`;
- live unambiguous system-information request: correct two-line response,
  `finish_reason=stop`, final prefill `cached=64`;
- live `--no-agent`: `Agent off`, normal tool set without `apply_patch`, one
  successful `pwd` execution, and `finish_reason=stop`;
- cancellation, timeout, reset, canonical/healing, final-prefix, and native MTP
  regression coverage: PASS in the full suite.

The active validation server intentionally had MTP disabled. No live MTP timing
or speed claim is made for this agent-mode change.

## Default-Promotion Blocker

The strategic matrix did not cover a broad read-only source audit with several
independent deliverables. A later public-style audit exposed a different
failure mode:

- one four-deliverable audit executed eight tools, consumed 375.7 seconds, and
  returned incomplete and incorrect configuration-default claims;
- a narrower two-resolver audit still executed five tools, consumed 400.6
  seconds, ended with `finish_reason=length`, and failed to confirm one default
  that was present in direct source evidence;
- a model-review experiment for broad recursive discovery added six review
  calls and was cancelled after 943.3 seconds; it did not prevent redundant
  scans and was removed.

Prompt-only guidance to prefer targeted source searches did not change the
observed command sequence and was removed. No schema, evidence-selection,
round-limit, or final-budget workaround was introduced.

One formal defect was fixed independently: if the model places the exact name
of a registered structured tool in the first shell-command position, runtime
rejects that malformed shell action before execution and requests one bounded
model revision. Runtime does not convert it into a tool call or choose a
replacement.

Additional default-readiness probes isolated the remaining failure:

- exposing the registered schemas directly fixed the malformed
  `list_directory`-as-shell form, but the end-to-end audit still repeated broad
  discovery, missed requested configuration defaults, and returned an
  incomplete conclusion;
- native thinking did not increase prefill in the isolated probe, but it spent
  substantial decode budget restating the task and repeatedly reached
  `finish_reason=length` before emitting the next tool call;
- a bounded model-driven revision after an exact repeated action avoided that
  duplicate but expanded the conversation and still failed to cover all
  deliverables;
- using one compact tool-decision prompt for every round did not fix the search
  strategy. The longest probe used twelve model calls, nine executed tools,
  and 664.7 seconds without a clean final answer.

These variants were removed. They did not meet the combined correctness,
latency, and prefill gates, and retaining them would make successful bounded
workflows slower without solving the broad-audit failure. The normal non-agent
path is unchanged.

A later recovery run corrected an incomplete diagnostic fixture and repeated
the four-subject configuration audit against a clean temporary copy of the
complete `src/`, `tests/`, `scripts/`, and `docs/` trees plus the relevant root
files. The model used five calls and three tools over 168.2 seconds. It listed
the tree and then used one recursive search whose production output was
bounded. The final answer ended with `finish_reason=length`, left `agent`,
`think`, and every precedence rule unresolved, and incorrectly treated the
planning-shadow allowlist as the normal tools default. The three tool outputs
were valid; the failure was strategic evidence acquisition and attribution,
not tool syntax, canonical validation, or missing source files.

### Structured source-query experiment

An agent-only, read-only structured source-query prototype tested whether a
compact grouped search result could replace broad shell output. The model
selected every regular expression and path. Runtime only validated bounds and
paths, searched UTF-8 repository text without a shell, preserved one group per
query, and returned exact bounded `file:line` snippets with explicit
continuation metadata. The experiment was disabled by default and did not
affect non-agent prompts.

The hypothesis failed with the current model and template:

- the schema-only run did not select the new tool and used ten model calls, six
  tools, 7,831 evaluated tokens, and 390.608 seconds before returning an
  unresolved stop response;
- after a compact availability hint, the model selected the tool but collapsed
  all subjects into one query instead of the required query objects. Canonical
  validation rejected the call; the run used five calls, two tools, 6,530
  evaluated tokens, and 258.338 seconds;
- a flatter schema made the call structurally valid, but the model still used
  one combined expression over the repository root. The bounded expansion
  rejected that broad request; the model returned to recursive listings and
  used seven calls, four tools, 6,405 evaluated tokens, and 295.259 seconds;
- the final calibration provided only a generic shape example, explicitly
  requested one query per independent subject, and excluded non-source
  artifacts from directory expansion. The model still emitted one combined
  expression. Its four bounded hits all came from the first alphabetic source
  file, after which it resumed recursive directory listings. The run used six
  calls, three tools, 5,635 evaluated tokens, 288 output tokens, and 252.570
  seconds. The post-tool route reached `finish_reason=length`; the final stop
  response left every requested default, precedence rule, and authoritative
  source unresolved.

The measured root cause is therefore not only the size of shell output. Gemma
4 26B-A4B Q4_0 did not preserve the independent search subjects when choosing
the structured query and did not use continuation to recover from bounded
results. The runtime cannot split the expression, choose source paths, or rank
results without taking over semantic decisions.

The first calibrated smoke failed the correctness gate, so the required three
candidate repetitions and the remaining source-audit matrix were not run.
Repeating a known failing workload would not justify promotion. The prototype,
flag, schema, evidence branch, and tests were removed completely. Reopen this
specific approach only for a model or template that naturally emits separate
model-chosen query groups, uses bounded continuation, and passes the
configuration audit smoke before any larger matrix.

The compact model-owned workflow-state hypothesis was also rejected before
production integration:

- a separate bounded initializer could enumerate the four requested subjects,
  but adding that immutable checklist to continuation calls increased the
  model/tool count and did not associate later evidence with each subject;
- the production Gemma tool template terminates immediately after a closed
  tool envelope, so a sideband workflow object cannot accompany a normal tool
  call in the same result;
- a registered wrapper carrying workflow state and an inner model-selected
  action did not satisfy its strict schema: the richer form omitted required
  fields and invented `search_repository`, while the minimal form replaced the
  requested subjects with `list_files` and invented that tool name;
- broader final evidence and focused-audit prompt variants remained incomplete
  while increasing calls, context, or both.

No workflow-state module, wrapper tool, prompt variant, additional retry, or
evidence-selection path was retained. These generation probes are not
production features and do not justify default promotion.

### Isolated model-authored decomposition experiment

Before the candidate run, direct source inspection established the independent
expected result:

- `agent` defaults to `false` in `src/orbit/terminal/config.py`. The JSON
  boolean is used when the CLI is silent, while `--agent` and `--no-agent`
  override it. CLI construction forwards the resolved value to the runtime.
- `tools` defaults to `on`. A CLI `--tools` value overrides the resolved
  configuration; otherwise `ORBIT_TOOLS` overrides JSON, `tool_mode` is the
  preferred JSON key, and `tools` is the compatibility JSON key. An invalid
  environment value fails safely to `off`.
- `think` defaults to `false` through
  `src/orbit/terminal/think_mode.py`. CLI `--think` overrides the JSON
  boolean or `on`/`off` value, and the interactive `/think` command updates
  the subsequent REPL, backend, and runtime state.
- `max_tokens` defaults to `512` in `AppConfig`. CLI `--max-tokens` overrides
  JSON within the configured range, and interactive `/max-tokens` updates
  subsequent REPL turns. Per-phase completion caps remain separate runtime
  policy and do not change this user-facing default.

An off-by-default prototype then allowed the existing route inference to emit
two to four model-authored deliverables. Runtime performed structural bounded
validation only. A valid proposal would have run each deliverable serially in
an isolated agent context with the original request, separate evidence and
action state, and the unchanged canonical, healing, permission, guardrail, and
executor paths. One final model call would have received the bounded child
results. Runtime did not derive, rewrite, merge, or complete deliverables.

The first clean configuration-audit smoke failed its gate:

- the initial route used all 64 existing output tokens and ended with
  `finish_reason=length`;
- it did not produce a valid decomposition;
- the unchanged fallback loop executed broad `ls`, `find`, and recursive
  `grep` discovery;
- the complete run used nine model calls, five tool executions, 5,103
  evaluated tokens, 424 output tokens, and 281.257 seconds;
- the terminal response stopped normally but reported `agent`, `tools`,
  `think`, and `max_tokens` all unresolved.

This was worse than the recovered baseline of five model calls, three tools,
3,416 evaluated tokens, 330 output tokens, and 168.2 seconds, which was already
incorrect and ended at length. Because the candidate failed the first
correctness gate, the remaining two configuration repetitions, the other
source audits, and the twenty-scenario matrix were not run. Increasing the
route budget was prohibited and would not establish a bounded default.

The flag, parser, isolated-child runtime path, benchmark harness, and focused
tests were removed. No decomposition behavior remains. Reopen this mechanism
only for a model or template that naturally emits a complete faithful
decomposition within the existing route budget and passes the first
configuration-audit smoke without broad recursive fallback.

### Two-stage decomposition follow-up

A follow-up tested a materially different hypothesis instead of asking Gemma
to fit the full decomposition into the 64-token route. The existing route was
allowed to select only exact `{"route":"DECOMPOSE"}`. That explicit decision
would start one separate model call with a fixed 256-token budget. Only the
second call could emit two to six bounded deliverables. Structurally valid
deliverables would run serially in isolated agent contexts, followed by one
model-authored synthesis. The experiment was off by default and did not change
the existing route, tool, or final budgets.

The first clean smoke passed the syntactic stages:

- route output was exact `{"route":"DECOMPOSE"}` with
  `finish_reason=stop`, using seven output tokens;
- decomposition output was valid JSON with four unique deliverables and
  `finish_reason=stop`, using 135 output tokens;
- no decomposition retry or fallback into the parent audit loop occurred.

The model-authored decomposition was strategically wrong for the requested
audit. It selected these four axes:

1. CLI parsing for all four settings;
2. JSON configuration for all four settings;
3. runtime construction for all four settings;
4. tests for all four settings.

That merged `agent`, `tools`, `think`, and `max_tokens` inside every child
rather than making each requested subject independently complete across CLI,
JSON, runtime, and tests. Runtime did not rewrite or replace the model's
deliverables.

The isolated execution did not recover:

- every child began with a broad recursive directory listing;
- the children used broad `find` or recursive `grep` calls rather than direct
  reads of authoritative sources;
- each child exhausted its four-round bound without returning a usable
  deliverable result;
- the final model call stopped normally but reported every setting unresolved;
- no source claim was supported and correctness remained 0/4.

Measured comparison:

| Metric | Recovered baseline | Two-stage candidate |
| --- | ---: | ---: |
| Model calls | 5 | 23 |
| Executed tools | 3 | 14 |
| Evaluated tokens | 3,416 | 21,081 |
| Output tokens | 330 | 905 |
| Wall time | 168.2 s | 881.824 s |
| Final finish reason | `length` | `stop` |
| Correct requested subjects | 0/4 | 0/4 |

The dedicated phase therefore solved only decomposition truncation. It did not
solve faithful decomposition or precise source acquisition, and isolation
multiplied the same broad-discovery strategy. Increasing child rounds would
increase an already uncontrolled cost without correcting the model-authored
axes, so it was not tested. The first smoke failed the correctness and
performance gates; the remaining repetitions, other source audits, and
twenty-scenario matrix were not run.

The flag, dedicated budget, parser, child-context integration, harness, and
focused tests were removed. Agent mode remains opt-in and no decomposition
execution behavior remains. Reopen decomposition only for a new model or
template with independent evidence that it preserves requested semantic
subjects, chooses precise source acquisition, and passes the first clean audit
before any larger matrix.

### Structural Python-inspection experiment

The final source-acquisition probe tested one agent-only, read-only structural
Python tool behind an off-by-default flag. The model could choose up to eight
exact AST queries across bounded roots. Supported query kinds were identifier,
string literal, function definition, class definition, assignment target, and
function-call name. Runtime preserved one result group per query and returned
deduplicated exact source ranges, enclosing definitions, bounded snippets,
total counts, truncation, and continuation cursors. It performed no shell
execution, semantic ranking, fuzzy matching, or source selection.

A direct capability check established that the implementation could retrieve
the relevant configuration definitions and option strings from authoritative
source locations. This check established tool capability only; it was not
credited as model audit correctness because the model did not choose those
queries.

In the first clean real-model audit, Gemma never selected the structural tool:

- model-selected structural queries: none;
- structural-tool calls: zero;
- executed tools: one `list_directory` and eight shell calls;
- shell discovery: two broad `find` calls and six recursive `grep` variants;
- several searches repeated the same bounded output without reaching the
  authoritative definitions;
- the final response used `finish_reason=stop` but explicitly left `agent`,
  `tools`, `think`, and `max_tokens` unresolved.

Measured comparison:

| Metric | Recovered baseline | Structural AST candidate |
| --- | ---: | ---: |
| Model calls | 5 | 13 |
| Executed tools | 3 | 9 |
| Structural-tool calls | 0 | 0 |
| Evaluated tokens | 3,416 | 19,165 |
| Output tokens | 330 | 571 |
| Wall time | 168.2 s | 908.218 s |
| Final finish reason | `length` | `stop` |
| Correct requested subjects | 0/4 | 0/4 |

The first smoke failed the mandatory selection, correctness, repeated-scan,
and bounded-cost gates. The remaining repetitions, other source audits, and
twenty-scenario matrix were therefore not run. The result isolates the
remaining problem: exact structural evidence was available, but the model's
source-acquisition strategy did not choose it and continued broad repository
scans. Adding a schema alone did not resolve that model behavior.

The flag, structural tool, schema integration, executor branch, harness, and
tests were removed. Agent mode remains opt-in and no Python-inspection runtime
path remains. Reopen only for a model or template with independent evidence
that it naturally selects precise structural queries and passes the first
configuration-audit smoke without broad recursive fallback.

### Structured profile without unrestricted shell

A final follow-up isolated tool competition as one variable. Behind an
off-by-default flag, agent mode exposed exactly these model-selectable tools:

1. `inspect_python`;
2. `read_file`;
3. bounded non-recursive `list_directory`.

The profile registered no unrestricted shell tool or equivalent process
runner. The existing route still determined whether tools were needed, after
which the normal tool loop presented only those three schemas. Runtime did not
convert a route command into a structured call or select a replacement tool.
Any shell proposal in the actual tool-mode phase would have been rejected by
the existing canonical permission decision. No such proposal occurred in the
measured run.

The direct implementation check again found the expected AST assignments and
option strings, proving that exact source evidence was available if the model
selected it. The first clean real-model audit selected this complete action
sequence instead:

1. `list_directory({"path":"."})`;
2. no further tool;
3. final response.

The listing correctly showed `src/`, `tests/`, `scripts/`, `docs/`, and the
root files. Gemma nevertheless concluded that repository contents were not
available. It selected no `inspect_python` query and no `read_file` path.
Consequently, evidence and correctness were:

- `agent`: unresolved, no source evidence;
- `tools`: unresolved, no source evidence;
- `think`: unresolved, no source evidence;
- `max_tokens`: unresolved, no source evidence.

Measured comparison:

| Metric | Recovered baseline | Restricted structured profile |
| --- | ---: | ---: |
| Model calls | 5 | 3 |
| Executed tools | 3 | 1 |
| AST queries | 0 | 0 |
| File reads | 0 | 0 |
| Evaluated tokens | 3,416 | 2,064 |
| Output tokens | 330 | 92 |
| Wall time | 168.2 s | 91.798 s |
| Final finish reason | `length` | `stop` |
| Correct requested subjects | 0/4 | 0/4 |

The restricted surface removed recursive shell scans and reduced cost, but it
did not improve correctness or induce source acquisition. Tool competition
was therefore not sufficient to explain the failure. The first smoke failed
the mandatory 4/4 correctness and natural `inspect_python` selection gates, so
no repetitions or broader matrices were run.

#### Terminal prompt capture and bounded round correction

A subsequent benchmark-only wrapper captured the exact model-facing messages,
tool schemas, phase, budget, and token metrics for every call. The terminal
call after the root listing had these properties:

- phase: `final_from_tool`;
- completion budget: 96 tokens;
- prompt/cached/evaluated tokens: 911/0/911;
- the original request was present byte-for-byte and named `agent`, `tools`,
  `think`, and `max_tokens`;
- the complete bounded root listing was present through its evidence ID,
  compatibility excerpt, and evidence card;
- no tool definition was transmitted;
- no instruction said that another action could be selected.

The preceding route and tool calls used 211 and 990 evaluated tokens
respectively. The complete captured run used three model calls, one executed
tool, 2,112 evaluated tokens, 99 output tokens, and 91.181 seconds. This
demonstrated a real orchestration defect: the tool loop inferred its round
limit from unrestricted-shell availability, so the shell-free profile reached
the one-round non-agent bound and entered a final-only prompt view.

A minimal probe changed only that coupling. The structured profile received
the same existing eight-round agent bound without changing route, tool, final,
or global completion budgets. After each observation, the model-facing input
then contained the complete request, accumulated observations, all three
structured tool schemas, and the existing explicit instruction to select the
next required action.

The first clean corrected-prompt smoke selected this sequence:

1. `list_directory({"path":"."})`;
2. `list_directory({"path":"src"})`;
3. `list_directory({"path":"src/orbit"})`;
4. `list_directory({"path":"src/orbit/runtime"})`;
5. the exact fourth action again, rejected before execution by the existing
   repeated-call guard;
6. final response.

Gemma selected neither `inspect_python` nor `read_file`. It acquired no source
content and reported `agent`, `tools`, `think`, and `max_tokens` unresolved.
Every model call stopped normally, including the final, but correctness
remained 0/4.

| Metric | Prompt-capture run | Corrected-round smoke |
| --- | ---: | ---: |
| Model calls | 3 | 7 |
| Executed tools | 1 | 4 |
| AST queries | 0 | 0 |
| File reads | 0 | 0 |
| Evaluated tokens | 2,112 | 4,385 |
| Output tokens | 99 | 227 |
| Wall time | 91.181 s | 203.952 s |
| Final finish reason | `stop` | `stop` |
| Correct requested subjects | 0/4 | 0/4 |

The complete prompt view corrected the premature forced final, but it did not
correct the model's source-acquisition strategy. The candidate therefore
failed its first mandatory smoke. The three repetitions, other source audits,
and strategic matrix were not run.

#### Completion verifier and restricted recovery

One final off-by-default probe tested whether a separate model-driven verifier
could reject the incomplete terminal answer and unlock exactly one recovery
epoch. The verifier received only the original request, proposed final answer,
and bounded evidence that had actually been observed. Its valid outputs were
exact `accept` or bounded `continue` with model-authored missing items. A valid
`continue` would have exposed only `inspect_python` and `read_file`; runtime
would still not choose a tool, path, query, claim, or answer.

The first clean smoke selected the following initial actions:

1. one recursive bounded `list_directory`;
2. three increasingly broad `find` commands;
3. one exact repeated shell proposal, rejected by the existing repeat guard;
4. a final answer explicitly stating that file listings were insufficient to
   inspect configuration defaults, precedence rules, or authoritative source
   lines.

The verifier nevertheless returned exact `{"decision":"accept"}`. It did not
identify any missing item, so the recovery epoch was never entered and neither
restricted recovery tool was exposed or selected. All four requested subjects
remained unresolved.

| Metric | Recovered baseline | Verifier candidate |
| --- | ---: | ---: |
| Model calls | 5 | 8 |
| Executed tools | 3 | 4 |
| Verifier calls | 0 | 1 |
| Recovery epochs | 0 | 0 |
| Evaluated tokens | 3,415 | 7,739 |
| Output tokens | 330 | 244 |
| Wall time | 167.103 s | 326.954 s |
| Final finish reason | `length` | `stop` |
| Correct requested subjects | 0/4 | 0/4 |

The one-token difference from the previously recorded 3,416-token baseline is
normal prompt accounting variation in the clean rerun; its behavior and
failure mode were unchanged. The verifier itself evaluated 2,449 tokens and
used five output tokens.

This result rejects the candidate for the current model and template. A
same-model completeness review cannot be treated as an independent correctness
oracle when it accepts an answer that explicitly declares the requested audit
incomplete. The mandatory sequence therefore failed before recovery, and the
remaining repetitions, source audits, and strategic matrix were not run.

The experimental resolver, tool schemas, AST and file-read modules, route
bridge, temporary round-limit correction, prompt-capture harness, and tests
were removed. The later verifier resolver, parser, recovery state, restricted
tool integration, benchmark harness, and focused tests were also removed. No
verifier module, recovery flag, or restricted recovery path remains. Agent
mode remains opt-in; normal agent and non-agent prompts, tool sets, prefill,
and lifecycle are unchanged.

Reconsider default promotion only after production-like multi-deliverable
source audits complete correctly and stop within the existing round and final
budgets, without broad-scan reviewer calls or deterministic semantic routing.
A replacement model or template must pass at least twenty strategic scenarios,
including repeated broad audits, with complete requested facts, no skipped
tools, no repeated-action loop, clean stop responses, no prefill regression,
and no material matched-workload wall-time regression.
