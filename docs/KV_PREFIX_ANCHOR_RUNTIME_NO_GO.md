# KV Prefix Anchor Runtime No-Go

Commit analyzed: `6eb3acc`

## Correction To The Previous Rejection

The earlier blocker was too strict.

A route-wide prefix anchor is **not** deterministic routing by itself.

If Orbit restores only a stable tools-on route prefix and then still appends the
same dynamic suffix, the model still makes the same route decision in the same
model-guided way.

So this line is conceptually valid:

- tools-on route pass
- stable route prefix restored from real native KV state
- dynamic suffix decoded normally
- model still decides CHAT, file, listing, web, fetch, or other tool paths

That part is **not** the blocker anymore.

## Stable Route Prefix Boundary

For the current route path, the useful stable prefix is the first system message:

- route system prompt
- route contract
- stable surrounding template framing

The dynamic suffix is the remaining conversation:

- prior history
- current user turn
- any other non-stable messages

So the route boundary is conceptually available.

## Real Blocker

Verdict is still `no-go`, but for a different reason:

### 1. Route currently shares the normal chat lineage

In the standard native path, the route pass is sent through the same native chat
path that also serves other non-tool completions.

It does **not** have a dedicated cache lane or dedicated native sequence in the
standard path.

### 2. The active lineage is still single-context and mutable

The standard path relies on one live `llama_context` plus one active prompt
lineage:

- one `ctx_tgt`
- one `cached_prompt_tokens`
- one active memory state

Injecting route-prefix checkpoint restore into that same active lineage would
mutate the context that later phases also rely on.

### 3. Safe restore semantics are still unproven in this shared path

What is still missing is not the abstract route boundary.

What is missing is a proven safe invariant for this exact standard path:

1. restore route-prefix checkpoint
2. append dynamic route suffix
3. complete route normally
4. continue subsequent phases with no change in:
   - logits
   - route outcome
   - retry behavior
   - continuation readiness
   - later cache behavior

That invariant is not established yet for the single shared standard lineage.

### 4. The current cache mode does not isolate route from later phases

The route pass currently goes through the standard native chat path rather than
a dedicated route-specific native mode.

So a route-prefix restore experiment would need either:

- a route-specific isolated native lane, or
- a proven restore contract showing that shared-lineage restore is behaviorally
  identical across repeated route calls and later phases

Neither exists yet in a small measured form.

## What This Means

The conceptual objection is removed:

- route-wide prefix anchor is allowed in principle
- no no-tool preclassification is required
- no prompt semantic rewrite is required

But the implementation is still blocked because the restore would happen inside
the same shared native lineage used by later standard-chat behavior.

Without a stronger native isolation or restore guarantee, this is not yet safe
to wire into production runtime flow, even behind a flag.

## Smallest Next Patch That Would Change The Decision

The next acceptable patch must solve one of these two technical issues first:

1. **Shared-lineage proof path**
   - show, with a bounded experiment, that restoring a route-prefix checkpoint
     into the current standard lineage is behaviorally identical to baseline
   - prove no regressions in:
     - model call count
     - repair/retry
     - route outcome
     - file/web/fetch evidence behavior
     - later phase cache behavior

2. **Route-isolated native lane**
   - add a dedicated native route lineage or safe sequence boundary
   - keep the restored route prefix out of later unrelated standard-chat state

## Current Recommendation

Do not wire route-prefix restore into the runtime yet.

Keep:

- the native bindings
- the lifecycle scaffolding
- the documentation

Do not add a production runtime hookup until the shared-lineage restore safety
question is answered directly.
