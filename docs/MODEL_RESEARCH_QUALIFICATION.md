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

ROUTE and tool-call phases retain their existing small caps for optional/off
reasoning and unknown backends. A native server explicitly declaring
`reasoning_behavior=always` and `separated_reasoning=true` lets these phases use
up to 1024 **total** completion tokens, including reasoning. An explicit caller
limit remains the ceiling: the frozen 24-intent observer requests 256, so its
reasoning-required decisions can use at most 256 total tokens. This is not a
separate reasoning allowance, a forced reasoning stop, or a retry. Existing
larger mutation budgets are preserved. User-visible budgets are unchanged.

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

## Deterministic checks

```bash
TMPDIR=/tmp PYTHONPATH=src python3 -m unittest \
  tests.test_model_contract tests.test_research_admission tests.test_model_probe \
  tests.test_qualification_tool_intent -q
```
