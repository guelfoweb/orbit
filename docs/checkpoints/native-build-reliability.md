# Native build reliability: packaged shim rebuilds fail closed

## Starting baseline

`a9fa2c31674ffd9da81ff13cab6a0cc2b2a87f28`

## Root cause

The reported symptom was "`build_native.py` prints a failure and still exits 0".
**That does not reproduce.** Every failure path in `build_cli.main` returns
non-zero, and `scripts/build_native.py` propagates it via
`raise SystemExit(main(...))`. The observed exit 0 came from shell usage:

```
build_native.py … | tail -3     # reports tail's status, not the build's
${PIPESTATUS[0]}                # → 1, the real status
```

`scripts/build_native.py` is therefore unchanged by this work.

The **real** defect is different. Six shim builders short-circuit to a packaged
artifact without consulting the source:

* `build_persistent_mtp_shim` — on an exported-**symbol** match
* `mtp_probe`, `mtp_dry_run`, `mtp_accept_probe`, `mtp_decode_probe`,
  `mtp_completion` — on mere file **existence**

Neither reaches `compile_cpp_helper`, so its freshness check is bypassed. That
is correct for a *runtime* that only needs a usable shim, and wrong for an
*explicit build*, which must produce artifacts from the current source.

## Historical reproducer

With `#error` appended to `vendor/shim/orbit_persistent_mtp.cpp`:

| condition | before | after |
|---|---|---|
| pre-existing `.so` present | **exit 0**, "completed in 1s", compiler never invoked, stale binary shipped | **exit 1**, compiler diagnostic, stale artifact removed |
| `.so` absent | exit 1 | exit 1 |

## The fix

1. `force: bool = False` on all six builders; when true the short-circuit is
   skipped and `force` is threaded into `compile_cpp_helper`.
   `build_cli._build_packaged_shims` passes `force=True` to all six. **No
   runtime caller passes it**, so the runtime fast path is unchanged.
2. `compile_cpp_helper` under `force` unlinks the previous artifact *before*
   invoking the compiler, then raises on a non-zero return code, on a missing
   output, and on a directory or zero-byte output. Post-build existence
   therefore proves *this* invocation created the artifact.
3. `build_cli.main` verifies the six shim artifacts after the build, using
   `persistent_mtp_shim_filename()` for the platform-dependent name.
4. `except OSError` prints the usual `error: …` line and returns 1, re-raising
   under `--verbose` so a genuine bug keeps its traceback.

Accepted trade: a forced build that fails now leaves no artifact, where the
stale one previously survived. That is the point — keeping it is the defect —
but a failed explicit build leaves the tree without a loadable shim until the
source is fixed. Only the explicit build path is affected.

## Review history

Four independent review cycles. Every cycle found the same *class* of weakness
in the qualification, never in the production fix:

| cycle | verdict | what it caught |
|---|---|---|
| 1 | 0 / 2 / 3 | AST test counting `force=True` literals — defeated by `try/except: pass`; build succeeded with zero shim artifacts |
| 2 | 0 / 2 / 3 | the behavioural replacement mocked away all real-compiler coverage (N2a survived); existence-only checks admitted a stale artifact; `.so` hardcoded → macOS; `--help` test proved nothing |
| 3 | 0 / 1 / 2 | deleting **all six** `force=True` passed ~735 tests — the replacement asserted *propagation*, not *forcing* |
| final | 0 / 1 / 2 | the macOS guard was a source-shape test; calling the helper into an unused variable defeated it |

The recurring lesson: **a test that asserts source shape is not a test of
behaviour.** Each attempt was defeated a different way — by an error-swallowing
wrapper, by mocks removing the real path, by redundant guards masking each
other, and by satisfying the letter of a substring check. What works is
asserting the specific observable property at the boundary where it is decided.

## Final mutation evidence — 12/12 caught

| mutant | result |
|---|---|
| all six `force=True` removed (`sed`) | CAUGHT (6 failures) |
| `force` removed from the persistent shim only | CAUGHT |
| short-circuit ignores `force` | CAUGHT |
| `compile_cpp_helper` returncode check disabled (N2a) | CAUGHT |
| `_run` returncode check disabled (N2b) | CAUGHT |
| unlink-before-forced-build removed (S1) | CAUGHT |
| `output.exists()` check removed | CAUGHT |
| empty/directory check removed | CAUGHT |
| macOS filename hardcoded to `.so` | CAUGHT |
| shim-artifact check disabled | CAUGHT |
| `OSError` handler returns success | CAUGHT |
| wrapper discards `main()`'s return value | CAUGHT |

## Qualification

* focused: **315/315 PASS**
* full suite: 3733 collected, **3725 passed, 8 skipped, 0 failed, real exit 0**
* real build: exit 0; artifact mtime advances, proving this invocation created it
* shim exports: **64** (list in `native-build-exports.txt`)
* provenance V2: OK, 63 patched paths
* `client.py`: **zero diff**; five route/anchor guards intact; strict invariant
  intact; `use_mtp_experimental` remains `False`

## Self-MTP status (unchanged by this work)

FUNCTIONAL: PASS · PERFORMANCE: FAIL (−46.8 %) · AUTO-MTP DEFAULT: NO ·
PERFORMANCE LINE: CLOSED. See `docs/selfmtp-closure.md`.
