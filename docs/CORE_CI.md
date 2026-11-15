# Core CI scope

`python3 scripts/check_core.py` runs an explicit, source-fresh subset of the
unittest suite. The same command runs locally and in GitHub Actions on Linux
with Python 3.11 and 3.12. No package installation, native build, model weights,
external corpus or account secrets are needed.

Run it from a Git checkout with a HEAD commit: the retained replay CLI records
the source revision. A source archive alone needs a Git HEAD before this gate.

The selected modules cover route parsing, canonical tool validation, formal
healing, terminal-completion authority, server request validation, model
contracts, research admission, cache-profile membership, Qualification Harness
fixtures and recorded tool-intent replay. Their exact list lives in
`scripts/check_core.py`. New core tests must remain usable in an artifact-free
checkout. An empty suite, any failed test or any skip fails this gate.

The runner delegates to `scripts/qualify_fresh.py`, obtaining a new interpreter
and private bytecode cache for each invocation, and returns the real child
exit code. It never derives success from log text or a shell pipeline.

`.github/workflows/core.yml` also compiles Python sources and checks changed
lines for whitespace errors. It runs on pull requests, pushes to `main` and
explicit workflow dispatch. Actions are pinned to complete commit hashes;
checkout credentials are not persisted and token permissions are read-only.

## Boundaries

This is a core regression gate, not the complete non-live suite. The native ABI,
library-family/runpath, compiled-shim and real PEG gates still require a local
native build. Frozen-corpus tests require the retained hash-pinned artifacts.
Live model qualification, semantic review and performance measurements remain
separate. The workflow neither supplies corpus substitutes nor sets
`ORBIT_ALLOW_MISSING_CORPUS` to waive the deterministic corpus gate.

Full contribution and release gates remain in `AGENTS.md`; CI success does not
replace them. In a provisioned checkout:

```bash
TMPDIR=/tmp PYTHONPATH=src python3 -m unittest discover -s tests -q
```

## Cache-profile declarations

`native_llama/cache_profile_policy.py` centralizes only repeated membership
decisions for the existing qualified profile IDs. It declares fixed-prefix
startup paths, rolling route/analysis eligibility, post-final shadow membership
and the existing preservation rules. Unknown or research IDs get no behavior.

State, cache identities, tokenizer boundaries, native decode-call sequences,
capture/restore, invalidation and scheduling remain with their current owners.
Verified-profile, thinking, MTP, configuration and exact-prefix checks remain
at their existing call sites. Model-specific algorithms and configuration
switches remain explicit where their behavior differs.

The Qwen3-Coder startup still calls the analysis-prewarm function, which reports
the existing model-ineligible skip; Flash Next still omits that call. These
presentation and ordering details are part of the preserved behavior. No new
profile, optimization, prewarm or performance qualification is introduced.

This declaration cleanup does not reopen the R2–R6 extraction technical stops.
