# Read-Only Mutation Guardrail Analysis

## Scenario

During PR #67 review, a read-only file inspection turn was followed by a fresh web-search turn. The first turn read the requested fixture file, but the model then emitted an additional mutating shell edit command before answering.

Expected behavior:

- Read-only file requests may read file content.
- Listing evidence must not become file-content evidence.
- Mutating commands must not run unless the latest user turn explicitly asks for a modification.
- A following web request must use the web path and must not reuse stale local evidence.

## Matrix

| Condition | Branch/config | Result | Classification |
| --- | --- | --- | --- |
| A | `main`, `ORBIT_KV_PREFIX_PREWARM` unset, `ORBIT_KV_PREFIX_ANCHOR` unset | Reproduced: read command ran, then a mutating shell edit command ran. Fixture diff was saved and restored. | FAIL on main |
| B | PR #67 branch, prewarm unset/off | Reproduced the same read-then-mutate pattern. Fixture diff was saved and restored. | FAIL without startup prewarm |
| C | PR #67 branch, startup prewarm | The original PR #67 review smoke observed the same failure under startup prewarm. After A and B reproduced the same bug without prewarm, C is not the root cause. | Not prewarm-specific |

Saved diffs:

- `/home/guelfoweb/LAB/orbit-backups/readonly-mutation-investigation/A_main_default.edit-target.diff`
- `/home/guelfoweb/LAB/orbit-backups/readonly-mutation-investigation/B_pr67_off.edit-target.diff`

## Diagnosis

This is a pre-existing read-only safety gap in the shell tool loop.

The runtime already classified mutating shell commands for post-mutation verification, but it did not reject a mutating shell command before execution when the latest user turn was read-only. As a result, a model-emitted edit command could run even though the user only requested inspection.

A follow-up smoke exposed a second detail: the intent detector treated the `edit` substring inside a path-like filename as a mutation verb. The fix now removes path-like tokens and URLs before checking for mutation intent, so filenames do not change read-only classification.

This is not caused by:

- PR #67 startup prewarm.
- Route KV prefix-anchor.
- `ORBIT_KV_PREFIX_ANCHOR=auto`.
- Tool alias normalization from PR #59.

## Fix

The fix adds a pre-execution guardrail:

- If the latest user turn is read-only and does not explicitly request modification, mutating shell commands are rejected before execution.
- The guardrail uses existing generic shell mutation detection; it does not hardcode the fixture path or a specific utility.
- `exec_shell_full_command` remains available for non-mutating reads, web search, fetch-style shell commands, and explicit mutation requests.
- Explicit edit requests still allow mutating shell commands and retain existing mutation verification behavior.

The retry remains model-guided. The runtime does not choose a replacement command; it only rejects the unsafe side effect and asks the model to answer from existing evidence or provide a non-mutating evidence command.

The shell write-operator detector was also tightened so quoted angle brackets inside command arguments are not mistaken for shell redirection.

## Validation Plan

Required tests:

- Read-only request rejects mutating shell commands before execution.
- Read-only request rejects Python file writes before execution.
- Explicit edit request still permits mutating shell commands.
- Read file path remains valid.
- Listing remains valid.
- Web/fetch regressions remain valid.
- PR #59 stale-evidence scenario remains valid.
- PR #62 listing-to-read evidence scenario remains valid.
- KV/prewarm is not changed.

Required smoke:

- Read-only local file turn followed by fresh web turn.
- Explicit edit request still mutates only when requested.
- Read/list/web/fetch baseline regressions.

## Release State

No tag or GitHub Release changes are required. `v0.0.1-rc2` remains a historical prerelease tag and release.
