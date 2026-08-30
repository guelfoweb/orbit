# REF-1: committed-identity extraction

First step of the incremental, behaviour-preserving runtime refactor.

## Baseline

`8db0158a92f83a06170cb220c05ce56baea2b7fe`

## What moved

Committed-identity ownership left `client.py` for a new `CommittedIdentity`
(`src/orbit/native_llama/committed_identity.py`). Four client methods became
pure delegates:

| client method | owner method |
|---|---|
| `_invalidate_committed_sequence` | `invalidate` |
| `_resident_prefix_len_for_mtp` | `resident_prefix_len_for_mtp` |
| `_publish_mtp_committed_identity` | `publish_from_mtp` |
| `_commit_sequence` | `commit` |

The immutable invariant is unchanged and now lives in one place:

    prompt_tokens[:len(committed)] == committed

## One owner, no dual state

`NativeSessionState` holds **no token copy**. `committed_sequence_tokens` is a
property over the owner, created lazily so a session is never without one. The
first draft kept a `_committed_sequence_tokens` field as pre-bind storage; it
was removed once probing showed it could never hold a live second copy, because
every write goes through the setter, which materialises the owner first. Dead
state in a refactor about ownership is the wrong thing to leave behind.

Sessions that are not a `NativeSessionState` — several call sites build a client
via `object.__new__` with a stand-in carrying only the token attribute — get an
`AttributeBackedIdentity`, which applies the same policy while reading and
writing through that attribute. Before the extraction any such object was a
valid session; requiring a concrete type would have been a behaviour change.

## Dependencies

* new module imports only `kv_diag`; it never imports the client, so no cycle
* the owner takes three narrow callables (tokenize, coder opt-out, ids for
  diagnostics) rather than the client object
* no factory, provider, registry, protocol hierarchy or DI
* `client.py` 4919 → 4900 lines; `committed_identity.py` 196; `session_state.py`
  58 → 120

## Behaviour

No intentional change, and none measured. The one substantive question was that
`_resident_prefix_len_for_mtp` previously read its `committed` ARGUMENT and the
delegate reads the owner. Independently verified: production snapshots at
`client.py:2315` and uses at `:2371`, and every call in that window
(`_RequestTiming.start`, `_thinking_enabled`, `_prepare_mtp_prompt`, …) provably
cannot touch `_session`, so the two always agree.

## Review history

Two cycles. Both found real defects, none in the *idea* of the extraction and
all in its edges:

| cycle | verdict | what it caught |
|---|---|---|
| 1 | 0 / 2 / 3 | delegates required a `NativeSessionState`-shaped session, breaking 10 duck-typed tests; two suites survived only by monkeypatching; a vacuous assertion; dead `snapshot()`; `commit` untested |
| 2 | 0 / 1 / 2 | the adapter fix was **itself untested** — three mutants reverting it all passed; a `MagicMock` session silently no-opped every operation; dataclass `__eq__`/`replace`/`asdict` hazards |

Two defects were found by me rather than by review, and both are worth recording
because they are the same mistake in different clothes:

* a `reusable_prefix_len` written for a future caller that never existed. A
  mutation "survived" on it — not because it was equivalent, but because the
  code was unreachable. Removed.
* the adapter stubbed the tokenizer, so a duck-typed session's resident claim
  returned 0 where baseline returned 3 — silently refusing all reuse. Fixing
  that exposed a third layer: `CommittedIdentity` read `self._tokens` directly,
  bypassing the subclass property entirely, so the first fix had no effect. The
  parent now reads via `self.tokens` and writes via `self.adopt`.

The lesson from cycle 2 is the sharpest: **a fix found by eye and not by a test
is unguarded**. The duck-typed test covered `commit` and `invalidate` — the two
operations that did *not* regress — while derivation, the one that did, had no
coverage at all.

## Qualification

* mutants: **12 caught** (strict equality, invalidate no-op, publication removed,
  pair-failure publishes, proper-prefix removed, coder opt-out removed, bind
  carry-over dropped, commit drops generated, parent bypasses subclass, owner not
  cached, setter bypasses owner, delegate rejects duck-typed sessions)
* two survivors, both analysed and equivalent in production: `if not resident:`
  → `if False:` (`list(()) == []` is identical to invalidate), and removing the
  `isinstance` guard (no test now constructs a `MagicMock` session)
* focused: 270/270; identity surface incl. previously-missed modules: PASS
* full suite: 3739 collected, **3731 passed, 8 skipped, 0 failed, real exit 0**
  (baseline 3733; +6 new tests, zero regressions)
* hot path: no new hashing, no extra tokenization (one call per turn, as
  before), no native calls, no context-length-proportional copies, no per-token
  work. The delegates are per-turn.

## Deferred

`_committed_identity_owner` is a dataclass field, so it participates in the
generated `__eq__`/`__repr__` and is shared by `dataclasses.replace`/`copy.copy`.
Nothing in `src/` or `tests/` does any of that today, so the fix
(`compare=False, repr=False`, plus `__deepcopy__`) is left to a follow-up rather
than smuggled into a behaviour-preserving extraction. Recorded as a TODO at the
field.
