# Qwen 3.6 Native Compatibility

## Supported Profile

Orbit has a verified native compatibility profile for:

- model family: Qwen 3.6;
- model identity: `Qwen3.6-35B-A3B`;
- GGUF architecture: `qwen35moe`;
- quantization identity: `general.file_type=15` (`Q4_K_M`);
- tokenizer metadata: `gpt2` with `qwen35` pre-tokenization;
- validated artifact: `Qwen3.6-35B-A3B-Q4_K_M.gguf`;
- llama.cpp vendor: Orbit's current revision-bound native build.

Detection does not use the filename. The native backend verifies the GGUF
architecture, model identity, tokenizer metadata, and exact embedded template
hash before selecting the profile. An unrecognized Qwen variant or template
fails before inference instead of running the Gemma protocol silently.

The reviewed embedded template is byte-identical to the official Qwen 3.6
template used for validation. Its SHA-256 is
`e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`.

## Rendering And Parsing

Qwen rendering and output parsing use a revision-bound native Orbit bridge
compiled against the active llama.cpp headers and libraries. Python passes
JSON messages, JSON tool definitions, primitive flags, and an opaque model
handle. It does not reproduce the upstream chat-parser structures.

The bridge:

- applies the embedded template through llama.cpp's Jinja implementation;
- passes `enable_thinking` explicitly;
- separates visible content from reasoning content;
- parses the Qwen 3.6 XML tool protocol into OpenAI-compatible tool calls;
- preserves exact JSON tool arguments for Orbit's canonical validation;
- is co-located with, and cryptographically bound to, the active native
  libraries, headers, compiler identity, source provenance, and bridge source.

Orbit keeps the existing runtime tool loop. Canonical validation, formal
healing, permissions, no-mutation policy, repeated-call protection, and tool
executors are unchanged.

## Thinking

Thinking is disabled by default. Orbit also forces it off for strict structured
phases, including route selection, tool selection, document-search planning,
document-search verification, and formal repair.

When the user explicitly enables thinking, the native parser returns reasoning
separately from visible assistant content. Reasoning is never sent to route or
tool-call parsers and is not retained in normal conversation history. A small
thinking budget can end before Qwen produces visible content; this is reported
honestly as `finish_reason=length`.

The verified Qwen profile uses one native completion for explicit chat
thinking instead of Orbit's legacy two-pass reasoning prompt. The backend
returns reasoning and visible content as separate fields, while history keeps
only the visible answer. In the validation host, a greeting exhausted a
256-token budget in reasoning but completed with `finish_reason=stop` at a
512-token user-selected budget. Orbit does not raise that budget automatically.

## Tools And History

The profile supports the Qwen 3.6 tool envelope:

```text
<tool_call><function=NAME><parameter=ARG>VALUE</parameter></function></tool_call>
```

Tool declarations, assistant tool calls, tool results, and post-tool
continuations are rendered by the official template. A budget-truncated or
malformed envelope cannot become an executable call merely because the parser
found a partial structure. Orbit accepts at most one executable call per model
turn under the existing canonical contract.

The official template permits a system message only at the beginning. Later
Orbit evidence cards keep their chronological position and exact content but
are serialized as ordinary input turns for this profile. Tool messages and
assistant tool-call history retain their native roles.

## Startup

Build the native libraries and compatibility bridge, then start the server with
the explicit GGUF path:

```bash
python3 scripts/build_native.py
orbit server \
  --model models/ggml-org--Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --think off
```

Inspect bounded compatibility diagnostics:

```bash
curl -s http://127.0.0.1:12120/props | python3 -m json.tool
```

Relevant fields include the detected family, compatibility profile, template
source and hash, reasoning protocol, tool protocol, effective thinking state,
and verification result. Prompts and generated reasoning are not included.

## Route Prefix Reuse

The exact verified Qwen 3.6 Q4_K_M profile reuses a process-local checkpoint
for the first 768 tokens of tools-on route prompts. The boundary is part of
the invariant official-template prefix and stops before user or conversation
content. No padding or prompt text was added to create the boundary.

The first eligible route performs its normal prefill and captures the complete
hybrid sequence state. Later routes restore both attention KV and recurrent
state, then evaluate only the dynamic suffix. The checkpoint is model,
quantization, context, template, tokenizer, route-contract, backend-build, and
process bound. It is never shared with Gemma, MTP, thinking-enabled phases, or
another Qwen profile.

Reuse is enabled by default only after exact cold, segmented, and restored
logits equivalence was established for the verified profile. Disable it
immediately with:

```bash
ORBIT_QWEN_ROUTE_PREFIX_REUSE=0 orbit server \
  --model models/ggml-org--Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf
```

`ORBIT_QWEN_ROUTE_PREFIX_REUSE=1` enables it explicitly. Invalid values disable
reuse safely. `/props` reports bounded capture, restore, fallback,
invalidation, identity, token-count, and checkpoint-size diagnostics without
prompt content.

The validated checkpoint is 81,608,684 bytes. It is invalidated on cancel,
timeout, reset, restart, model or context replacement, profile/template/schema
identity change, restore failure, and completion error. A restore error clears
partial state and performs one cold prefill without recursive retry.

The verified boundary is 768 production tokens. The full invariant prefix is
810 tokens for the measured route fixtures; 768 is the longest useful
production-batch-aligned boundary below it. The token immediately after the
checkpoint is token ID 1802. The boundary token hash is
`095416e9dcf73f8a9db7b39b1d3f289da66a705c32418c194964938d58876b9b`.

Cold full prefill, explicit segmentation at token 768, and capture/restore
produced byte-identical logits in the native validation probe: maximum absolute
logit difference `0.0`, identical logits hash, next token, ordered candidates,
generated route, and finish reason. A process-isolated 11-case route corpus
also produced identical output hashes and decisions. OFF evaluated 9,094
prompt tokens in 287.8 seconds; ON evaluated 1,414 in 57.9 seconds, including
the first capture. Ten restores avoided exactly 7,680 evaluated tokens. Median
prefill after capture was 1.76 seconds versus 24.38 seconds OFF on the measured
CPU host. These timings are descriptive and not a deterministic performance
guarantee.

## Current Limits

- Only the exact verified 35B-A3B identity and reviewed template are enabled.
- Qwen MTP is disabled. The available Qwen draft GGUF has not been qualified
  against Orbit's target/draft lifecycle and must not be treated as supported.
- Gemma-specific route and final-prefix checkpoints remain disabled for Qwen;
  Qwen route reuse has its own identity, storage, eligibility, and lifecycle.
- The Qwen hybrid/recurrent memory layout can reject partial KV removal. Orbit
  uses complete sequence-state capture/restore for the verified route prefix
  and falls back to a cold prefill for other incompatible cache transitions.
- Qwen thinking can consume a substantial decode budget before visible output.
- CPU wall time depends on model, prompt, context, threads, and host state; no
  deterministic speed claim is made.

Gemma 4 remains a separate compatibility profile. Qwen support does not change
the Gemma renderer, tokenizer, tool protocol, MTP path, or prefix-reuse gates.
