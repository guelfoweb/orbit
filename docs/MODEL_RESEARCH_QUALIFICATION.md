# Model support and research qualification

Orbit distinguishes three questions:

1. Does this artifact match a reviewed technical model profile?
2. Which operations does that profile support?
3. What did a particular qualification run establish about those operations?

`NativeModelProfile.verified` retains its existing production compatibility
meaning. `supported_capabilities` declares the technical text operations
independently; shipped profiles preserve their existing supported operations.
The legacy `/props` `capabilities` mapping remains available and still requires
a verified profile. Its `capability_scope` explicitly describes technical
support, not measured quality. No semantic qualification is inferred from it.

Qualification reports are separate records bound to the artifact SHA-256,
the co-located native-library/bridge hashes, the source hashes and configuration,
and the retained evidence bytes. Results are per capability: load, CHAT,
tool intent and ANALYSIS semantics. A result for one never grants another.
These reports are observational; production admission never imports or reads
the qualification package or its output.

## Reasoning contract

Profiles can declare `none`, `optional` or `always` reasoning, and whether the
backend separates reasoning from visible content. `/props` distinguishes
`thinking_requested` from effective `thinking_enabled`. A model whose template
always opens reasoning cannot be described as thinking-off just because the
caller requested it.

New servers publish `separated_reasoning` explicitly. The client consumes that
boolean for verified profiles without assuming a model family. For older
servers without the field, the existing verified `qwen-think` behavior remains
the compatibility fallback. An explicit false or malformed capability never
falls back to an inferred true. This does not change templates, decoding,
budgets, retries or current profile eligibility.

## Decision budgets for required reasoning

ROUTE and tool-call phases retain their existing small base budgets for
optional/off reasoning and unknown backends. A native server explicitly
declaring `reasoning_behavior=always` and `separated_reasoning=true` adds a fixed
1024-token internal reserve to the ordinary decision budget. This increases
the **total generation ceiling**, including reasoning and the visible decision;
it does not independently limit reasoning or force its closure.

For the frozen observer's caller limit of 256, ROUTE has a 64-token base plus
1024 reserve (1088 total); an ordinary tool call has 96 plus 1024 (1120 total).
The base still respects its phase/caller limit. This reserve belongs only to
internal decisions; dedicated CHAT/final budget policies remain unchanged.
The combined internal ceiling also covers visible decision text or prose
returned or reused from that phase; it does not independently cap that prose.
An existing length retry retains at least the same decision ceiling without
adding another retry. Mutation/recovery base policies remain intact.

The declaration informs budgeting even for an unverified research profile;
it grants no tool authority and does not promote the profile. Exact context
admission sees the expanded total before inference. Truncation still fails
closed, and a CHAT fallback after a truncated decision cannot pass the gate.
No prompt, grammar, parser, oracle, or repair policy changes with this budget.

## Explicit research admission

The source-checkout command below admits an exact, unqualified, single-file
GGUF to an independently managed CPU-only server. It uses the existing native
loader, revision-bound Jinja bridge, parser, HTTP backend and runtime observer.
There is no production `orbit server` flag, environment setting or registry
entry that enables experimental models.

Create an identity manifest from authoritative artifact information and
inspected GGUF metadata, for example:

```json
{
  "version": 1,
  "artifact_sha256": "d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d",
  "artifact_size": 14526258464,
  "template_sha256": "ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef",
  "model_name": "LFM2.5-24B-A2B",
  "architecture": "lfm2moe",
  "metadata": {
    "general.name": "LFM2.5-24B-A2B",
    "general.architecture": "lfm2moe",
    "general.file_type": "15",
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "lfm2"
  },
  "reasoning_behavior": "always",
  "separated_reasoning": true
}
```

This example pins an experimental artifact; it is not a qualification result
or an enablement. Declared reasoning behavior is a test hypothesis to check
against the retained rendering and responses, not proof of model adherence.

```bash
TMPDIR=/tmp python3 scripts/orbit_qualify_model.py \
  --identity /path/to/identity.json \
  --model /path/to/model.gguf \
  --output /path/to/new-evidence-directory
```

The output directory must not exist. Use an isolated source checkout: source
and native libraries must not change during a run. The command downloads
nothing, never changes the model store configuration, and never registers a
model. It checks complete file size, SHA-256, exact declared metadata and
template before native weight loading; it rejects changed file identity before
admission. This is startup attestation, not continuous integrity protection
against another process modifying a model during inference.

The research profile remains `verified=false`, all production capability flags
remain false, and `/props` identifies its artifact, template and process ID.
MTP, GPU and low-memory mode cannot be enabled through research admission.
History is passed through without a model-specific serializer. Native template
or parser incompatibility must surface as failure; the command adds no grammar,
semantic repair, prompt tuning or alternative parser.

The default controlled profile is ctx 8192, threads 6/6, batch 256, ubatch 128,
thinking requested off, repack off and startup prewarm off. These are explicit
probe settings, not a qualified or optimal profile for an unknown model. The
five numeric options can be overridden explicitly. The parent has a four-hour
wall ceiling, configurable with `--max-wall-seconds`; startup is limited to
300 seconds. It stops only its own child server on completion or failure.

## Sequence, evidence and acceptance

1. **Load:** a ready, idle native server must match the exact experimental
   artifact, template and newly spawned PID. Its `/props` is retained.
2. **CHAT:** retain five greedy tools-off observations covering Italian,
   English, JSON and a two-turn recall. Normal stopped responses with visible
   output produce `MANUAL_CHECK`; incomplete, empty or tool-producing results
   fail. The command does not judge factual accuracy, language, JSON purpose,
   memory correctness or absence of reasoning leakage. Review the exact output.
3. **Tool intent:** run the byte-pinned 24-case corpus through the unchanged
   `observe_case` and `summarize` functions. Each case gets a fresh runtime and
   a reset of this dedicated server. Executor substitution and canonical
   validation are unchanged. Actual model-selected tool executions are zero.

Canonical validation must also be enabled in the invoking process. An explicit
`ORBIT_TOOL_CALL_CANONICAL_GATE=0` is rejected before starting the server.

The existing [production-profile gate](TOOL_INTENT_QUALIFICATION.md) remains
unchanged: its CLI still requires a verified profile. The research command is
an explicitly different admission path to the same observer. A research PASS
is bounded evidence for the pinned configuration; it does not certify a future
production integration, argument semantics, generated code, later workflow
steps or ANALYSIS narrative.

`qualification.json` records separate hash-bound outcomes. `load.json`,
`chat.json`, `tool_intent.json` and the detailed `tool-intent.json` preserve
stage evidence; progress files retain completed observations if a later stage
fails. `server.log` preserves loader diagnostics. ANALYSIS semantics remains
`NOT_RUN`. A failed stage cannot silently become a pass or promote the model.

Exit 1 means a measured stage failed. Exit 2 means the technical observations
finished but manual CHAT review remains. This command never exits 0 to imply
general qualification. An explicitly reviewed production profile, normal
integration tests and any applicable semantic baseline remain separate work.

## Post-tool finals with required reasoning

The same strict native `always` + separated-reasoning declaration now reserves
1024 internal tokens alongside each existing CHAT post-tool final allowance.
The small visible-answer policies remain 96/160/192/256 tokens as applicable;
for example a directory-listing final receives a total ceiling of 1120 instead
of 96. Each existing length retry, completion repair and compact repair computes
its own ordinary base and adds the reserve once. There are no new attempts.
The total is a combined generation ceiling, not separately enforced reasoning
and visible-text quotas; either can consume it. Exact context admission receives the entire ceiling before inference and
refuses a request that cannot fit. Optional/unknown declarations retain their
previous limits. Pure CHAT, full-document and ANALYSIS finalization budgets are
unchanged. A bounded reserve is not a guarantee against every possible length
failure and does not certify the answer's semantics.

Native separated-reasoning pure CHAT now uses the streaming transport when a
visible callback is provided; it no longer forces a non-stream request just
because thinking is enabled. Separated reasoning is not stored as visible
assistant content. Existing evidence/citation withholding remains authoritative:
a session with an evidence store can still deliver the checked final in one
block. This change does not remove that guard or promise universal streaming.

Cache eligibility is independent of these budgets. In retained Dell observations
for Inkling, three identical 507-token inputs reused zero tokens: the native
memory refused partial sequence removal and Orbit cleared it for a safe cold
prefill. A tested 523-token continuation also reused zero. This is measured
behavior of those sequences, not proof that every possible continuation misses.
No cache profile, snapshot policy, prompt identity, native library or MTP setting
is changed by this budget fix. Diagnostic evidence is retained under
`workdir/diag/inkling_final_budget_cache_1/`.

## Named reasoning effort

The verified Inkling Q4 profile accepts named effort on ordinary native CHAT
and exact token-count requests. The pinned model template maps these inputs:

| Effort | Template value |
|---|---:|
| `none` | 0 |
| `minimal` | 0.1 |
| `low` | 0.2 |
| `medium` | 0.7 |
| `high` | 0.9 |
| `xhigh` | 0.99 |
| `max` | 0.99 |

The HTTP API accepts `reasoning_effort` at the top level or in
`chat_template_kwargs`; conflicting declarations and unknown/non-string values
are rejected. Only these named values are exposed, not arbitrary numeric input.
Omitting the field preserves the legacy thinking-derived template behavior.
Explicit effort overrides that template choice, while the thinking flag remains
separate. Unsupported/unverified profiles and older bridges reject explicit
requests rather than silently ignoring the selected effort. Artifact-content
requests do not accept it; required-tool decoding accepts only None effort.
Continuation retains already-rendered state and cannot change explicit effort.

A validated ContextVar scope surrounds both native generation and exact token
counting under the model lock, and resets on success or exceptions. This keeps
simultaneous HTTP requests isolated and makes context admission count the same
render the model receives. The optional mainline-only bridge export sets the
template's declared `reasoning_effort` keyword. Existing exports, context ABI,
legacy native family, prompts, tool grammars/parsers and retry counts remain
unchanged. Rebuild `--family mainline-support` to expose the new symbol; capability
publication requires both the verified profile and the actual upgraded bridge.
The generic template capability probe need not enumerate every model-specific
alias; the pinned template's own map remains authoritative.

The local English GUI replaces the checkbox with all seven options on this
profile, starts at None, and sends the exact named choice through the runtime
backend, including token counting. Other profiles retain legacy Off/On choices.
Settings are disabled while saving, stale status polls cannot overwrite an
acknowledged choice, and a rejected change restores the last confirmed setting.
None requests minimum reasoning; it cannot guarantee no internal reasoning.
Budgets are unchanged, including the fixed required-reasoning reserve. A higher
level is not a hard token quota or a new tool/ANALYSIS semantic qualification.
The aligned checkpoint remains eligible only for None/legacy thinking off;
higher named efforts invalidate that checkpoint pending separate qualification.

Evidence: `workdir/diag/reasoning_effort_1/`. Native rendering verified all seven
values; separate old/new processes retained identical None/High prompt bytes,
full-vocabulary prefill logits and 24-token generated sequences. Seven real
GUI/native tools-off smokes completed with stop and the requested value attested
at the native generation render. On one fixed math prompt, two cold observations
per level in None/Max/Max/None order both produced 91 completion tokens: mean GUI
elapsed 19.625 seconds for None and 19.77 for Max. No speed difference is
established by that small, host-specific sample. It is not a broad reasoning or
quality benchmark. A second cold comparison used the short OK request from the
GUI smokes, again in None/Max/Max/None order: None produced 41 completion tokens
and averaged 14.66 seconds; Max produced 62 and averaged 20.325 seconds. The
per-token decode rates were similar. Both cases are retained, including the
case with no observed speed improvement. Native/model identity and benchmark
metadata are retained.

## Inkling aligned prefill checkpoint

A later speed follow-up adds one bounded native sequence checkpoint for the
verified `orbit-inkling2-mini-native-v1` profile. Eligibility is deliberately
narrow: CPU/text, ctx8192, threads6/6, batch256, ubatch128, progress_step64,
repack off, low-memory off, thinking off, native CHAT prompt mode (without a
native tools payload), None/omitted effort, and all MTP/probe modes off. Other
profiles retain their existing cache policies. `ORBIT_ALIGNED_PREFIX_REUSE=0`
is the opt-out; invalid values also disable it.

Capture occurs only from cold prefill at token 1024, on the unchanged 64-token
decode boundary. No generated token enters the checkpoint. Restore requires a
longer prompt and exact first-1024 token equality, plus the model/profile,
template, metadata, native binary, full configuration, context pointer, session
and reset identity. At least one suffix token is evaluated to refresh logits.
Existing prefix/rolling strategies take precedence. Partial resident append
cannot capture a differently segmented state; short intervening finals preserve
the checkpoint. This is inference-state reuse, not answer caching or evidence
selection.

Only one snapshot is held, capped at 256 MiB before allocation; the measured Q4
state is 205,187,404 bytes. Capture and restore temporarily require another copy.
Reset, close, cancellation (including idle cancel), completion errors and
ineligible mode/configuration release it. An unavailable identity uses ordinary
prefill. Partial or failed native restore invalidates the snapshot and clears
physical memory before normal fallback; committed tokens are published only
after successful restore. No native library, prompt, parser, budget or MTP
algorithm changes are involved.

Native equivalence and timing evidence, source regression and fault mutations
are retained in `workdir/diag/inkling_cache_speed_1/`. Timing depends on prompt
length, shared-prefix length, output length and host load. First requests and
short prompts retain their cold cost; model reasoning and decode speed are
unchanged. This does not extend model or ANALYSIS semantic qualification.

## GUI conversation turn ownership

`ChatRuntime` accepts an optional keyword-only `user_turn_namespace` containing
1–64 ASCII letters, digits, underscores or hyphens. The default remains `None`
and preserves existing `turn_1`, `turn_2`, ... IDs. An opting-in consumer gets
`turn_<namespace>_<counter>` IDs. The local GUI creates a fresh UUID namespace
for each new/reconnected runtime, including a newly opened or resumed session.
This prevents its restarted turn counter from making a previous conversation's
records eligible for current-turn citation.

The evidence store and retained records remain scoped as before. This is turn
ownership protection, not deletion of prior evidence, a general evidence-store
isolation boundary, a semantic classifier or a relaxation of re-attestation.
The observed pre-fix GUI `system_info` completion with a previous directory
citation is retained as a rejected attribution observation in the diagnostics.
It is not a semantic qualification for the model.

## Deterministic checks

```bash
TMPDIR=/tmp PYTHONPATH=src python3 -m unittest \
  tests.test_model_contract tests.test_research_admission tests.test_model_probe \
  tests.test_qualification_tool_intent -q
```
