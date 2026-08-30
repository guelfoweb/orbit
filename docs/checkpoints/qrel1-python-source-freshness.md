# QREL-1: source-fresh Python qualification

Qualification-tooling reliability. No Orbit runtime behaviour changes.

## Baseline

`a4e13b7dba8920ecd45d59424edfa6a47fa71a45`

## The REF-6 incident, and the corrected diagnosis

A test in the REF-6 candidate failed with `capture_count 2 != 1` -- the exact
signature of a mutant that makes the restore site count a capture. The source
was correct, and `inspect.getsource` on the *loaded* module also showed
`restored=True`, so it looked like a transient artifact of the mutation run.

That first diagnosis was wrong. The failure reproduced in isolation.
Disassembling the loaded function showed `KW_NAMES ('captured',)` at **both**
call sites: the interpreter was running stale bytecode.

**REF-6 shipped correct source.** The defect was in the qualification
environment, not the candidate.

### Why the cache was trusted

CPython validates a cached `.pyc` against the source's `(mtime, size)` pair.
The edit replaced `captured` with `restored` -- both eight characters, so the
file length did not move -- and landed inside the same filesystem-timestamp
second. Neither half of the pair changed, so the stale `.pyc` was served.

Mutation testing produces exactly this shape of edit on purpose: a keyword,
operator or constant swapped for one of equal length, repeatedly and fast. The
hazard is not exotic; it is the normal case for this workflow.

### Why source inspection could not clear it

`inspect.getsource` reads the *file*. It reported the new source while the old
code object executed, and actively concealed the divergence. Only disassembly
of the loaded code object distinguished them.

**Source inspection is not proof of what executes.** That is the lesson the
tests here encode: every one asserts on executed behaviour, never on source
shape or environment variables.

## Deterministic reproducer

Built before any fix, with no reliance on timing luck:

1. write version A, import it so a `.pyc` exists
2. record the source mtime and size
3. overwrite with version B of **identical length**
4. restore the recorded mtime with `os.utime`
5. run a plain interpreter

Result: source reads `restored`, execution yields `captured`. The condition is
reproduced exactly, on demand.

## Execution-path audit

The repository runs **unittest, not pytest**. There is no CI, no `conftest.py`,
no pytest configuration, and **no automated mutation runner** -- REF-1..REF-6
campaigns were run by hand. That is precisely why the hazard was unguarded: the
risky workflow is ad-hoc shell, not a script.

Four sites launch `[sys.executable, "-m", "unittest"]`:

| site | workdir | in scope |
|---|---|---|
| `scripts/orbit_qualify_lifecycle.py:182` | **`cwd=ROOT`** | **yes** |
| `src/orbit/qualification/runner.py:704` | fresh `TemporaryDirectory` | no |
| `src/orbit/dev/release_confidence.py:194,275` | fresh `TemporaryDirectory` | no |
| `tests/*` | own fixtures | no |

Only the first runs tests against the Orbit worktree, so it is the only place a
stale worktree `.pyc` can be executed. The other two run generated fixtures in
temporary directories and are production code reachable from the CLI; per the
mission's ownership rule they were deliberately left alone.

No masked exit-code pipeline exists in tracked code. The `build_native.py |
tail` incident recorded in `native-build-reliability.md` was shell usage at the
invocation layer, not a defect in a file.

## Mechanism

`scripts/qualify_fresh.py` -- `run_fresh()` launches a **fresh interpreter**
with `PYTHONPYCACHEPREFIX` pointing at a **freshly created per-run**
`mkdtemp()` root, removed in `finally`. Two properties are load-bearing and
neither suffices alone:

* **Fresh process.** A cache root cannot help a module already in
  `sys.modules`; re-importing in a live interpreter keeps the old code object.
* **Fresh root per run.** An isolated but *reused* root fails exactly like the
  worktree's -- measured against the same unchanged `(mtime, size)` pair, so
  mutation N executes mutation N-1's bytecode. This was measured, not assumed.

`scripts/` is not imported by `src/orbit/`, so this is engineering tooling with
no path into the runtime.

### Rejected alternatives

| mechanism | why rejected |
|---|---|
| `python -B` | suppresses only *writing*; a stale `.pyc` is still read and executed -- verified |
| `PYTHONDONTWRITEBYTECODE=1` | same defect, verified: executed `captured` while source said `restored` |
| `find . -name __pycache__ -delete` | races, mutates unrelated developer state, and fails open -- a skipped cleanup silently restores the hazard |
| hash-based `.pyc` | does not help an `UNCHECKED_HASH` pyc, and requires every producer to opt in |
| `importlib.reload` | cannot fix an already-executed module graph; a subprocess is simpler and stronger |

## Proof it works on real Orbit code

The REF-6 campaign was rerun with same-length mutants and a pinned mtime -- the
exact condition that fooled it:

* plain runner: the `restored`→`captured` mutant **SURVIVED** (invisible)
* `qualify_fresh`: the same mutant was **caught**

All three REF-6 mutants were caught under identical size and pinned mtime, and
`final_prefix_store.py` was restored byte-identically (`36100f86…`).

## Adversarial results

Attacks that the mechanism defeats: caller-supplied `PYTHONPYCACHEPREFIX`,
`UNCHECKED_HASH` (PEP 552) pycs, stale `.pyc` in a `sys.path` directory outside
cwd, zipimport, and a `sitecustomize` hijack -- the prefix is read at
interpreter startup, before user code runs.

**Known boundary:** a `.pyc` whose `.py` no longer exists is imported
*sourceless* and no cache setting can help, because there is no source to
correspond to. Outside this invariant, and it does not arise here: the worktree
holds zero orphaned or legacy `.pyc`.

`-E`/`-I` passed in `args` would neutralize the prefix. Not reachable from the
call site; guarding it would be over-engineering.

## Qualification

* **13 mutants caught.** Including: isolated prefix removed, shared/reused root,
  fixed non-per-run root, write-suppression substituted for isolation, env not
  passed, env key typo, `PYTHONPATH` default dropped, forced `PYTHONPATH`
  overwrite, exit code swallowed, result forced to success, cleanup removed,
  cleanup only on success, and the call site reverted to a bare
  `subprocess.run`.
* **M6** (cleanup moved to before the child) preserves *freshness* -- CPython
  recreates the root -- but is **not** behaviourally equivalent: it leaks one
  temp directory per run, measured at 4 over 4 runs, and three cleanup tests
  kill it. Calling it simply "equivalent" overstated the analysis; it is a
  caught mutant, killed by a property other than the one first examined.
* focused: 31 freshness tests, plus 78 across qualification, build-exit-code and
  CLI-dispatch suites
* `compileall` clean, `git diff --check` clean
* full suite: 3849 collected, **3841 passed, 8 skipped, 0 failed, real exit 0**
  -- green under both the fresh runner and the plain one (baseline 3818)
* cleanup verified on success, failure, timeout, spawn failure and
  `KeyboardInterrupt`; zero leaked roots over 20 sequential runs. `SIGKILL`
  strands one root, which is inert -- never reused, so it cannot serve stale
  bytecode.
* concurrency: 6 parallel runs all executed current source; `mkdtemp` is atomic,
  so no locking was added

## Review

One cycle, independent and adversarial: **BLOCKER 0 / MAJOR 1 / MINOR 3**.

* **MAJOR-1 — fixed.** Reordering the environment merge so a caller-supplied
  `PYTHONPYCACHEPREFIX` wins survived all 30 tests: the ordering was correct but
  entirely unpinned. Since the real call site passes `env={**os.environ, ...}`,
  an exported prefix would have silently killed the guarantee -- the same class
  of unnoticed regression QREL-1 exists to prevent. A test now pins it, and
  catches the mutant.
* **MINOR-2 — recorded**, in the M6 paragraph above.
* **MINOR-3** (sourceless `.pyc`) and **MINOR-4** (`-E`/`-I`) recorded as
  boundaries; the reviewer independently confirmed zero orphaned `.pyc`.

## Invariant

For every qualification execution launched through `run_fresh`, the executed
Python code corresponds to the current source bytes -- even when size and mtime
are unchanged, a stale worktree cache exists, and source is mutated repeatedly.

## SHA256SUMS

```
4b509e18356e3e06d68fd506895f2168f12e5615bbbe8568246c3d8c0773814b  scripts/qualify_fresh.py
b45347943bedc83f135ec9d5c2ac8709aff167db700e86674bed4e90dca7625f  tests/test_qualify_fresh.py
0291061f195d2163b1c1655cd148c0231addb28b0231f6c7912caca5f3682cd7  scripts/orbit_qualify_lifecycle.py
```
