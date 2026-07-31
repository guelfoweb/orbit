# Tool-Loop Convergence Validation

## Decision

Orbit uses one production tool loop. The former opt-in agent path was removed
before the first stable release because matched Gemma 4 26B-A4B Q4_0 testing
found no important, repeatable benefit large enough to justify a second
orchestration path.

Tools remain enabled by default. Chat-only operation remains available through
`--tools off`. The removed `--agent` and `--no-agent` flags are rejected as
unknown options; no compatibility no-op or hidden agent branch remains. An
`agent` member in an old JSON file has no runtime field or effect.

## Inventory

The removed path comprised:

| Component | Unique behavior | Cost and outcome |
| --- | --- | --- |
| Configuration | `agent=False`, JSON field, `--agent` and `--no-agent` | A second user-visible mode and precedence surface |
| Prompts | Agent route, continuation, strict tool-call, action anchor, and final instructions | Separate prompt views and cache behavior |
| Runtime | Agent branches in routing, tool loop, and finalization | Additional lifecycle and error paths |
| Tool registry | Agent-only `apply_patch` | Extra schema prefill; selected 0/2 when tested in the normal loop |
| Action review | Model review for mutations or unclear actions | Extra model call; no repeatable unique safety benefit after #155 |
| Verification | Mandatory post-mutation observation and semantic completion | Extra calls and regressions on valid bounded workflows |
| State | Review, revision, verification, continuation, and agent-round fields | Reset, timeout, cancellation, and diagnostic maintenance |
| Diagnostics | Agent state and review counters | Separate status and test surface |
| Tests | Agent, review, and exact-patch suites | Protected code that no longer has a production caller |

The normal loop already owned exact repeated-call rejection. Mutation epochs
are retained: after an executed mutation, earlier observations may be repeated
against the new state, while the exact successful mutation remains blocked.
Canonical validation, deterministic formal healing, permissions, executor
guardrails, evidence, lifecycle cleanup, final-prefix reuse, and post-tool final
reuse are shared production behavior and were not removed.

## Release-Line Topology

The convergence branch is based directly on `origin/main` at `f5ccc52`. The
net patch was reconstructed from the validated final tree rather than carrying
the two commits that existed only on `baseline/opt-in-agent-mode-26b`:

- `0d009e5`, which introduced the opt-in agent experiment together with the
  intended 26B and normal-loop work;
- `d0ea464`, the merge of #155 onto that experimental branch.

The resulting branch contains the intended post-RC23 26B/download work, the
shared normal-loop hardening, and the #155 no-mutation invariant as a net diff
against `origin/main`. It does not contain the agent implementation, its exact
patch executor, or their commit ancestry. This keeps the release line descended
from RC23 through `origin/main` without publishing the discarded experiment.

## Test-Count Reduction Audit

The validated experimental branch had 1,309 tests. Removing the second loop
deleted 63 tests and added or retained five focused replacements, for a net
reduction of 58 and a final count of 1,251.

The 63 removals comprise 37 tests in the deleted action-review, agent-loop, and
exact-patch suites; 12 tests for removed agent configuration, route controls,
prompts, REPL forwarding, and event formatting; and 14 duplicated no-mutation
integration cases that depended on the agent registry or exact-patch executor.
The five focused replacements cover removed flag rejection, inert legacy JSON,
canonical-gate kill-switch enforcement, structured read-only availability, and
direct-dispatch enforcement.

Shared invariants remain covered in their authoritative suites:

- no-mutation language, inert text, mixed constraints, and mutation coverage
  remain in `tests/test_shell_guardrails.py`;
- canonical ON/OFF, permission, policy, executor ordering, and structured
  read-only behavior remain in `tests/test_tool_contract.py`;
- healing ON/OFF cannot bypass no-mutation policy in
  `tests/test_tool_healing.py`;
- repeated calls and mutation epochs remain in
  `tests/test_tool_loop_state.py`;
- direct dispatch remains covered in `tests/test_tools.py`.

No removed test protected a remaining production caller that lacks equivalent
coverage.

## Matched Comparison

The comparison used fresh native-server processes for normal and agent cohorts,
Gemma 4 26B-A4B Q4_0, CPU-only inference, MTP disabled, fixed six-thread CPU
affinity, temperature zero, identical fixtures, and clean temporary workdirs.
The sixteen common scenarios covered direct chat, read/list operations, create,
edit, code repair, a bounded filesystem workflow, legitimate and prohibited
mutations, inert tool-like JSON, permission and command failures, timeout,
cancellation, exact repeated action, and observation after mutation.

One original repeated-action prompt said `without changing state`, which
correctly activated the explicit no-mutation policy. That invalid fixture was
discarded and rerun with two requested `pwd` observations. The corrected row
stopped successfully.

| Mode | Correct | Model calls | Proposed tools | Evaluated tokens | Output tokens | Wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 15/16 | 39 | 18 | 8,734 | 927 | 465.448 s |
| Agent | 15/16 | 61 | 28 | 52,277 | 1,688 | 2,061.623 s |

CPU wall time is descriptive rather than a deterministic performance claim.
The cost difference is nevertheless too large to ignore because it accompanies
additional prompt views and model calls, not a matched correctness gain.

## Unique-Benefit Gate

The normal loop failed the code-repair scenario in two fresh processes after
choosing only `list_directory`. Agent mode passed its first run with
`apply_patch`, but failed a fresh-process repetition after the same directory
listing. The result is therefore 1/2 and not a repeatable production benefit.

The agent cohort also proposed deletion for inert tool-like JSON. The shared
no-mutation policy prevented a filesystem change, and the final answer explained
the data correctly, but proposing and rejecting the action added calls and is a
behavioral regression relative to the normal loop.

Action review did not provide unique protection after #155. Explicit global or
unsupported mixed no-mutation constraints deny generic shell at a shared
pre-execution boundary with canonical validation enabled or disabled. The
normal loop also retains mutation epochs and repeated-call protection.

No tested agent property met all required conditions: important, repeatable,
production-relevant, absent from the normal loop, and large enough to justify a
second orchestration path.

## Final Architecture

The production flow is:

```text
model route/tool output
-> deterministic formal healing when eligible
-> canonical contract when enabled
-> shared explicit no-mutation policy
-> existing guardrails
-> existing executor
-> bounded normal continuation/finalization
```

The runtime does not select a replacement tool, infer arguments, construct a
semantic plan, or add an action-review inference. Generic shell remains
model-selectable when tools are enabled. Under a global or unsupported mixed
no-mutation constraint it is denied before dispatch; structured read-only tools
remain available.

The exact `apply_patch` tool was removed with the agent registry. It was not
promoted into the normal loop because Gemma selected it 0/2 there and its schema
increased prompt cost. Existing-file edits continue through the established
shell path.

## Compatibility

Complete flag removal is intentional. Orbit remains in the `0.0.1` release-
candidate series, so retaining a deprecated no-op would preserve confusing
configuration without behavior. Unknown CLI flags fail through argparse. JSON
loading already ignores unrelated members, but there is no `agent` field in
`AppConfig` and no status output, tool registry, or runtime branch derived from
it.

## Explicit No-Mutation Invariant

The #155 safety rule remains independent of canonical and healing kill switches.
It examines only the latest user prompt, masks bounded inert quoted/code/JSON
and explicitly introduced Markdown payload text, and returns one of `none`,
`global`, or `mixed`. It does not inspect tool output or attempt to prove an
arbitrary shell string read-only.

Global and mixed constraints deny every generic shell call. Mixed and scoped
requests are intentionally unsupported for mutation and receive a bounded
denial asking the user to split the request. Legitimate mutation prompts without
an explicit constraint retain the established path.

## Post-Removal Validation

The same managed native 26B configuration was rerun after removal. All sixteen
scenarios ended with `finish_reason=stop` and all expected filesystem artifacts
were correct. An intermediate removal incorrectly dropped three instructions
that were introduced beside agent mode but already belonged to the shared
normal route/tool prompt. That trial proposed `rm -f keep.txt` for inert JSON;
the policy rejected it and preserved the file. Review restored the existing
shared guidance without restoring agent behavior. The targeted final-tree
rerun used no tool and passed.

| Cohort | Artifact correctness | Stop | Model calls | Proposed tools | Evaluated tokens | Output tokens | Wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Intermediate removal cohort | 16/16 | 16/16 | 40 | 21 | 10,768 | 1,002 | 566.948 s |

The intermediate run also completed code repair, existing-file modification,
file creation, deletion, the bounded multi-step filesystem workflow, timeout
recovery, cancellation recovery, repeated-action handling, and observation
after mutation. The wall value is descriptive and is not comparable as a
deterministic speed claim.

A separate ten-case adversarial safety run covered five global or mixed
constraints, four inert quoted/Markdown/JSON controls, and one ordinary
mutation. Every artifact was correct, no prohibited mutation executed, and the
ordinary mutation completed. Nine cases stopped normally. One quoted-text
control used no tool and preserved the filesystem but reached its 192-token
output budget; this remains a model/budget limitation rather than a policy
bypass.

Twelve exact local prompts sampled from `docs/PROMPTS.md` were also run in clean
temporary workdirs. All eleven tool workflows passed: list, read, search,
create-note, create-script, edit, append, directory creation, system info, line
count, and create/read/remove workflow. The original tools-off `grep` prompt
used zero tools but filled the fixed 256-token chat-phase cap and ended in the
middle of a sentence. A process-isolated comparison reproduced the same output
hash, 48 prompt tokens, 256 output tokens, and `finish_reason=length` on RC23,
pre-convergence main, and the convergence candidate. The failure was therefore
a pre-existing prompt/model verbosity mismatch, not an agent-removal
regression. The corpus prompt was narrowed honestly to request one concise
sentence instead of hiding the truncation or increasing a global budget. RC23,
main, and the candidate then produced the same complete 26-token answer with
`finish_reason=stop`, one model call, and zero tools. The revised direct-chat
gate passes without a runtime behavior change.

The canonical and healing kill switches were tested together on fresh native
requests. With both OFF, an explicit no-mutation request still preserved its
file and a legitimate mutation still completed. This confirms that the safety
invariant is independent of canonical schema validation and formal healing.

Regression gates after convergence:

- final focused tool-loop, guardrail, canonical, healing, CLI, and harness
  selection: 436/436 PASS;
- full unit discovery: 1,251/1,251 PASS after deleting obsolete agent-only test
  suites;
- six native MTP helpers rebuilt successfully;
- 26B target, draft MTP model, and mmproj loaded together; strict MTP requests
  completed correctly;
- final-prefix ON captured once and restored once with `cached=64`; OFF produced
  `cached=4` twice and no checkpoint activity;
- process-isolated post-tool final reuse preserved tool and correctness while
  removing one model call and 529 evaluated tokens on one eligible two-action
  case;
- `compileall` and `git diff --check` passed;
- managed native servers shut down after each block with no residual Orbit or
  llama process.

## Release Evaluation

The resulting change is coherent and user-visible: it converges Orbit on one
tool loop, removes unrewarded pre-stable complexity, retains mutation epochs,
and includes the #155 explicit no-mutation hardening. It is a reasonable RC24
candidate only after independent review, full unit and native gates, real-model
normal/no-mutation corpora, clean process state, and a verified intended release
branch descended from RC23.

The clean convergence branch is based on `origin/main` and contains no ancestry
from the temporary agent baseline. A release still requires an independently
reviewed and merged convergence PR, a verified final tag target, and every
runtime release gate listed above.

## Reopening Criteria

Do not restore a second tool loop from history. Reconsider a separate mode only
for a new model/template and a new measured failure where a minimal alternative
shows a repeatable correctness or safety benefit unavailable in the normal
loop, passes broad source audits, preserves direct-chat and bounded-workflow
costs, and passes lifecycle, canonical/healing, prefix-reuse, and MTP gates.
