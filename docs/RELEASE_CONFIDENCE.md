# Release confidence suite

This suite checks Orbit as a lightweight coding agent before release. It uses isolated fixtures copied into `/tmp` and validates final behavior, not implementation details such as whether the model used `sed`, `python`, `perl`, or another shell strategy.

Run it with a healthy local `llama-server`:

```bash
python3 scripts/release-confidence.py --keep-failed
```

The suite writes a machine-readable report to `/tmp/orbit-release-confidence.json`.

## Current release result

Latest release-confidence run: **12/15 PASS**.

The remaining 3 failures are classified as known limitations, not release blockers:

- `html_multiline_title`: Gemma may choose fragile `sed` commands for multiline HTML. The checker is valid, but the failure is an expected local patch-generation limitation.
- `css_regex_sensitive`: Gemma may choose brittle regex/shell commands when CSS selectors or values contain regex-sensitive characters. This is a command-generation limitation.
- `shell_script_hardening`: Gemma may keep attempting long heredocs/full rewrites for shell scripts instead of a minimal local patch. Guardrails reduce risk but do not deterministically solve the task.

They are not blockers for this release because:

- The runtime does not falsely mark these cases as behaviorally correct.
- The failures are bounded to robust local patch generation, not core routing, shell execution, repair, verification, session handling, or read-only review.
- Fixing them reliably would require stronger deterministic patch tooling or a redesigned edit workflow, which is outside the current lightweight model-driven release scope.

Release criterion for this phase:

- Unit tests pass.
- Core benchmark/smoke suite does not regress.
- Release confidence stays at or above 12/15.
- Known limitations are documented and not hidden behind fragile checkers.
- No additional deterministic task fast paths are introduced.

## Patch workflow experiments

Two experimental branches were evaluated and intentionally not merged:

- `patch-workflow-experiment`: tested a hidden `orbit-patch` pseudo-command inside `exec_shell_full_command`. It recovered 0/3 target failures; the model continued to prefer `sed` or heredoc strategies.
- `patch-tool-experiment`: tested a model-facing `apply_patch` tool. It recovered 0/3 target failures in the final critical run and introduced temporary regressions during release-confidence testing.

These results indicate that an optional patch tool is not sufficient for the current residual failures. The future direction, if these cases become release blockers, is a dedicated `deterministic-edit-workflow` forced by runtime state, not another optional patch command.

## Principles

- Check final files, behavior, tests, or expected output.
- Do not assert fragile command choices.
- Do not mutate the repository `workdir/`.
- Preserve model-driven behavior: the runtime may guard contracts and retries, but the model chooses commands.
- Classify failures before changing code.

## Test matrix

| ID | Fixture | Prompt intent | Behavioral checker | Risk covered | Release status |
| --- | --- | --- | --- | --- | --- |
| `html_multiline_title` | `index.html` with multiline `<title>` | Change title | Extract final title value | HTML multiline patch without fragile `sed` | known limitation |
| `css_regex_sensitive` | CSS selector and URL with `{}`, `/`, `.`, spaces | Update color | Balanced braces, preserved selector/URL, updated color | Regex-sensitive CSS edit | known limitation |
| `python_tiny_function` | Broken tiny Python function | Fix function | Import and assert behavior | Small local Python patch | blocker release |
| `shell_script_hardening` | Unquoted shell script | Harden script | Run with spaced paths and check strict mode | Patch completeness for shell scripts | known limitation |
| `json_config_update` | Nested JSON config | Update values | Parse JSON and check values | Valid config mutation | blocker release |
| `yaml_config_update` | Simple YAML config | Update values | Check YAML shape and values | Valid config mutation without extra deps | blocker release |
| `fix_failed_test` | Failing unittest fixture | Run, inspect, fix | `python -m unittest` passes | Content evidence recovery | blocker release |
| `rename_symbol` | Python function and local use | Rename symbol | Import new name and check behavior | Rename consistency | blocker release |
| `file_with_spaces` | File path with spaces | Update file | Read exact final file | Path quoting | blocker release |
| `recoverable_command_error` | Slash-heavy route string | Replace path | Final file contains new value | Shell Repair Loop | blocker release |
| `noop_mutation` | INI-style setting with spaces | Enable setting | Final setting changed | Mutation Verification | blocker release |
| `metadata_trap` | Test describes bug | Inspect content and fix | `python -m unittest` passes | Metadata-only discovery trap | blocker release |
| `long_command_pressure` | Large file with tiny broken function | Minimal edit | Import and assert behavior | Minimal Patch Guard | blocker release |
| `read_only_review` | Vulnerable Python snippet | Review only | File hash unchanged and issue identified | Read-only analysis boundary | blocker release |
| `ambiguous_suggest_then_fix` | Broken normalization function | Suggest, then fix | First prompt does not mutate; follow-up behavior passes | Ambiguous mutation boundary | blocker release |

## Robust local patch generation

Recent benchmarks showed that insufficient discovery is not the only failure mode. The harder residual problem is robust local patch generation: the model may choose brittle quoting, complex regex, long heredocs, or full rewrites when a small local patch is safer.

The suite covers that area explicitly:

- `html_multiline_title` verifies multiline HTML modification without relying on fragile line-local assumptions.
- `css_regex_sensitive` stresses characters that often break regex or shell quoting.
- `python_tiny_function` requires a small Python edit without a long heredoc.
- `shell_script_hardening` checks that the shell patch is complete, not merely changed.
- `recoverable_command_error` exercises Shell Repair Loop behavior on slash-heavy replacements.
- `noop_mutation` verifies that silent no-op mutations do not get reported as success.
- `long_command_pressure` checks that a large file with a tiny change does not force a full rewrite.

## Failure classification

Classify each failure into one dominant category before changing code:

- `discovery insufficiente`
- `command generation fragile`
- `quoting/sed fragile`
- `heredoc lungo`
- `patch incompleta`
- `checker fragile`
- `known limitation`

Fix only release blockers and real regressions. Do not overfit Orbit to artificial fixtures, fragile checkers, or cases that require architectural redesign.

## Guardrails added during this phase

The release-confidence fixes are intentionally limited to model-guidance guardrails:

- Mutation Verification is stricter: verification must show direct evidence of the requested value or state, not only metadata, paths, tags, fields, or key names.
- Minimal Patch Guard can trigger earlier for broad rewrites of existing files and asks the model for a minimal local patch.
- Mutative request detection now covers common operational verbs such as `set`, `enable`, `disable`, and `configure`.
- The release confidence suite uses isolated `/tmp` fixtures and behavioral checkers.

## Residual risks

- Local patch generation is still model-dependent and may fail on multiline HTML, regex-sensitive CSS, YAML-like edge cases, or shell scripts requiring multiple semantic edits.
- Some passing cases can be slow on CPU-only machines because guardrails add model turns when the model chooses fragile commands.
- The suite is a confidence gate, not a deterministic proof: release decisions should consider repeated runs when changing guardrails or prompt-adjacent behavior.
