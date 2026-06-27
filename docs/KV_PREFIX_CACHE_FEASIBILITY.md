## KV Prefix Cache Feasibility

Commit analyzed: `8a7cdb8`

### Goal

Find a stable way to improve tools-on prefill by reusing KV for the stable system/tools/capabilities prefix without changing the semantic prompt seen by the model.

### Current native path

The native standard path currently keeps one live prompt-cache state for the default session:

- `NativeSessionState.cached_prompt_tokens` stores only one tokenized prompt history.
- `_prepare_memory_for_prompt()` computes a single longest common token prefix against that one cached prompt.
- `llama_memory_clear()` or `llama_memory_seq_rm()` mutates the active memory in place.
- `_ensure_prompt_cache_mode()` resets the session state when the cache mode changes.
- `validate_session_id()` accepts only the default session.

This means the backend can reuse KV only for the single prompt lineage that is currently resident in the active llama context.

### Why the obvious options are not safe today

#### Option A: backend prefix prefill anchor

Not currently feasible in the standard path.

Reason:

- The standard native client does not expose a generic "prefill stable prefix once, then append different suffixes later" anchor API.
- The runtime can compute logical stable components, but the backend only reuses KV from the currently resident token sequence.
- Implementing an anchor without native checkpoint/restore would require replaying the anchor tokens on each switch, which is not real KV reuse.

#### Option B: stable prefix cache by prefix hash

Not currently feasible in the standard path.

Reason:

- A prefix hash alone is not enough.
- The client has no generic snapshot/restore of prompt KV for standard chat.
- There is no stored backend state object keyed by prefix hash.
- Adding a runtime-side mapping from prefix hash to token list would be a fake cache unless the backend can restore the corresponding KV state.

#### Option C: separate cache modes or lanes for route/chat_final

Not safe with the current standard implementation.

Reason:

- The client has one active llama context and one `cached_prompt_tokens` lineage.
- Separate logical lanes would still need real KV state restore when switching lanes.
- The current code can reset the active state, but it cannot restore a previous standard-chat KV snapshot for another lane.
- Rebuilding a lane by replaying its full prompt would increase work instead of reducing it.

### Existing support that does not solve this

There is persistent session machinery for experimental MTP, including checkpoint/restore counters, but that support is specific to the MTP path and is not wired as a generic standard-chat KV snapshot facility.

So the required primitive exists only in a specialized path, not in the normal tools-on chat path we need to accelerate.

### Technical no-go

With the current standard native backend, a safe prefix-cache patch is not available without adding at least one of these new backend capabilities:

1. Generic standard-chat KV checkpoint/save and restore.
2. Multiple real cache lanes or session states backed by distinct restorable KV memory.
3. A native prefix-anchor API that can materialize a cached stable prefix once and reuse it across later compatible prompts.

Without one of those, any "prefix cache" patch in the runtime would either:

- replay tokens instead of reusing KV,
- change semantic prompt shape,
- or introduce hidden backend behavior not justified by the existing architecture.

### Recommendation

Verdict: `no-go`

Do not implement a runtime-level prefix cache patch on top of the current standard path.

The next safe lever is backend-native:

- add a generic KV checkpoint/restore primitive for the standard chat path,
- keep it feature-flagged,
- and benchmark it first on tools-on no-tool chat before promoting it.

### Smallest credible next patch

If work continues, the smallest credible patch is:

- native backend support for saving and restoring standard-chat KV state keyed by a runtime-provided cache namespace or stable prefix hash,
- with diagnostics proving that restore reuses tokenized prefix KV instead of replaying tokens.

That patch should stay off by default and must preserve:

- prompt semantics,
- model call counts,
- routing behavior,
- file/web evidence policy,
- and visible output.

### Benchmark required before accepting any future patch

Any future implementation should compare OFF vs ON for:

- repeated tools-on short chat,
- repeated tools-on medium no-tool chat that goes through `chat_final`,
- listing,
- file read,
- valid web search,
- valid `fetch_url`.

Acceptance should require:

- higher `cached_tokens` or lower `evaluated_tokens` on tools-on no-tool paths,
- no increase in model calls or retries,
- no regression in file/web/listing correctness,
- and no semantic prompt changes.
