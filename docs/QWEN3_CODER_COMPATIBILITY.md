# Qwen3-Coder Native Compatibility

## Verified Profile

Orbit supports exactly this native CPU profile:

- profile: `orbit-qwen3-coder-native-v1`;
- model: `Qwen3-Coder-30B-A3B-Instruct`;
- GGUF architecture: `qwen3moe`;
- quantization: `general.file_type=15` (`Q4_K_M`), quantization version 2;
- tokenizer: `gpt2` with `qwen2` pre-tokenization, EOS 151645, padding 151654,
  and no added BOS;
- context metadata: 262144;
- experts: 128, with 8 active per token;
- embedded template SHA-256:
  `87710339d25b4e789c1d723f93c91ee861a86d305bb3d20a845536f251d6ea8a`.

Detection uses GGUF metadata and the exact embedded-template hash. A filename
does not authorize the profile. A changed architecture, model identity,
quantization, tokenizer, expert layout, context identity, or template fails
before inference.

## Chat And Tools

The profile uses the official embedded ChatML template through Orbit's
revision-bound llama.cpp chat bridge. The same bridge parses the model's native
XML tool protocol and preserves typed JSON arguments for the canonical runtime
contract. Chat, route selection, tool execution, tool-result history,
post-tool finalization, existing-file modification, failure recovery, and
inert tool-like data are supported.

The template has no qualified thinking mode. `thinking=true` fails closed.
Reasoning markup is not accepted as visible or structured output.

## Artifact Content

Orbit's shared `write_artifact` contract is generative: the model selects one
non-empty UTF-8 file body. It is not a promise to reproduce arbitrary supplied
bytes exactly.

The generic content-only chat prompt caused Qwen3-Coder to wrap generative file
bodies in a Markdown presentation fence. The verified backend protocol keeps
the official ChatML rendering but pre-opens one JSON string and applies a
profile-local grammar to its generated value. The model emits the complete file
body as JSON string characters or escapes, closes the string, and ends its
assistant turn. The backend decodes that value reversibly with strict UTF-8;
quotes, backslashes, Markdown fences, protocol-like text, Unicode, and trailing
newlines remain file content rather than transport delimiters.

Missing closures, malformed escapes or framing, invalid UTF-8, output-budget exhaustion,
cancellation, and unexpected message state fail closed. Orbit does not trim,
normalize Unicode or newlines, retry, regex-clean, complete, or semantically
repair content. Publication, path safety, atomic commit, lifecycle cleanup, and
ephemeral read-only `verify_artifact` remain in the shared runtime.

## Startup And Diagnostics

```bash
orbit server \
  --model models/unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --ctx 8192 --threads 6 --threads-batch 6 --batch 256 --ubatch 128 \
  --think off
```

`/props` reports the profile ID, template hash, renderer, tool/history
protocols, artifact protocol, verified quantization, and bounded capability
flags. It does not expose prompt content.

## Validation And Limits

The corrected production profile passed a four-case fail-fast smoke and an
eight-workflow process-isolated corpus. The corpus covered chat, exact route,
tool+final, artifact+verification, existing-file modification, failed-command
recovery, inert payloads, autonomous path/format choice, Markdown and Python
artifacts. Every visible result stopped normally and no model process remained.
Focused protocol fixtures cover HTML, JavaScript, Python, JSON, Markdown
fences, Unicode, YAML/TOML, shell text, XML/tool-like text, framing failure,
invalid Unicode scalars, cancellation, output exhaustion, and sampler cleanup.

Independent review rejected the initially qualified triple-backtick transport:
a file ending in the same fence made a missing transport closure ambiguous,
and the generic replacement-mode token decoder could silently substitute
invalid sampled UTF-8. The JSON-string grammar, strict decoder, and profile-
local sampler correct both defects. A real-model edge probe preserved a file's
terminal Markdown fence. Requests for authoritative byte-exact Unicode remain
outside the production contract and are not represented as passing generative
artifact tests.

Final evidence distinguishes overwrite authorization from the actual
publication action. In the corrected real-model smoke, a new target was
reported as created rather than overwritten, while path, UTF-8 integrity, byte
count and SHA-256 verification remained present.

At context 8192, six CPU threads, batch 256, ubatch 128, temperature zero,
thinking off, and MTP off, qualification measured approximately 18.80 prefill
tokens/s, 11.35 decode tokens/s, and 31.14 GiB peak RSS. Timings are
descriptive.

Unsupported capabilities:

- Qwen3-Coder MTP;
- multimodal input;
- Qwen 3.6 route-prefix checkpoint reuse;
- arbitrary exact-copy artifact guarantees;
- empty artifacts;
- binary artifacts or hidden multi-file planning;
- other Qwen3-Coder variants, templates, or quantizations.
