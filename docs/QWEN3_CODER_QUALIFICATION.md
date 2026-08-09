# Qwen3-Coder 30B-A3B qualification and production validation

## Status and scope

This document records the qualification and production integration of
`Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`. The exact verified profile is
`orbit-qwen3-coder-native-v1`; it remains distinct from
`orbit-qwen36-native-v1`.

The completed GGUF was inspected with `vocab_only=true` on 2026-08-08. The
observed identity is:

- file size: `18,556,689,568` bytes;
- file SHA-256:
  `fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad`;
- architecture: `qwen3moe`;
- model name: `Qwen3-Coder-30B-A3B-Instruct`;
- file type: `15` (`Q4_K_M`);
- tokenizer: `gpt2` with `qwen2` pre-tokenizer;
- declared context: `262144` tokens;
- experts: `128`, with `8` selected per token;
- embedded-template SHA-256:
  `87710339d25b4e789c1d723f93c91ee861a86d305bb3d20a845536f251d6ea8a`.

The template uses its own Qwen XML function-call protocol, does not reference
`enable_thinking`, and rendered identical chat fixtures when thinking was
requested on and off. This identity is therefore distinct from the verified
Qwen 3.6 profile. Chat, routing and tool-history checks passed. A later audit
showed that the initial exact-copy artifact oracle was stronger than Orbit's
production generative-artifact contract. Qualification now passes with the
profile-specific reversible content transport documented below. That exact
profile and protocol are now implemented in production.

The production-profile runner is `scripts/qualify_qwen3_coder.py`. Its `ready`
operation does not open the GGUF. `inspect` requires the final regular file,
rejects an active Orbit download and uses the current native API with
`vocab_only=true`. `smoke` requires the exact inspection manifest and loads the
normal production profile without an override. Qwen 3.6 route checkpoint
reuse, Gemma final-prefix reuse, thinking and MTP are disabled in that smoke.

## Sequence

Run from the repository root after the downloader has completed:

```bash
MODEL=models/unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
OUT=/tmp/orbit-qwen3-coder-qualification
mkdir -p "$OUT"

python3 scripts/qualify_qwen3_coder.py --model "$MODEL" ready
python3 scripts/qualify_qwen3_coder.py --model "$MODEL" inspect \
  --output "$OUT/metadata.json" \
  --template-output "$OUT/chat-template.jinja"
python3 scripts/qualify_qwen3_coder.py --model "$MODEL" smoke \
  --manifest "$OUT/metadata.json" \
  --output "$OUT/first-smoke.json"
```

Do not continue after a non-zero exit. In particular, do not create a profile
when native `vocab_only` loading, bridge rendering or any first-smoke case
fails.

## Metadata and template gate

The inspection records:

- exact GGUF byte size and SHA-256;
- architecture, model name, file type and quantization metadata;
- tokenizer model, pre-tokenizer, BOS, EOS, padding and add-token metadata;
- context, embedding, block, attention, expert and rope metadata exposed by
  the GGUF;
- sorted metadata-key-set hash and selected identity-metadata hash;
- exact embedded template in a private output file plus SHA-256 and bounded
  marker counts;
- thinking-off, thinking-on, tool-declaration and tool-history render hashes;
- revision-bound Orbit chat-bridge identity.

Review the complete private template before inference. Determine its official
upstream identity and verify the reasoning delimiters, tool declaration shape,
tool-call envelope, tool-result roles, assistant-history rules, stop markers
and `enable_thinking` behavior. Filename similarity is not evidence.

## First smoke gate

The fail-fast smoke runs exactly:

1. direct chat with exact visible output;
2. one route that must resolve to `exec_shell_full_command` with exactly
   `{"command":"pwd"}`;
3. execution of that command plus a clean final answer;
4. model-selected `write_artifact`, content-only generation, model-selected
   `verify_artifact` and exact artifact validation.

Stop on template/render failure, reasoning or control markup leakage,
malformed route/tool/arguments, non-stop structured output, incompatible tool
history/result serialization, or structural artifact failure. The candidate
override used during the original investigation was retired after identity and
regression review; the current runner exercises the exact production profile.

## Matched comparison corpus

Only after the first smoke passes, run Qwen3-Coder and the current verified
Qwen 3.6 profile in separate clean processes with CPU-only context 8192, six
threads, batch 256, ubatch 128, temperature zero, thinking off and MTP off.
Qwen3-Coder must not use the Qwen 3.6 route checkpoint. Use one excluded warmup
and three measured repetitions where practical.

The prepared corpus is:

| Case | Fixture and required result |
| --- | --- |
| Simple chat | Exact short visible response, no tool or control markup |
| Long prefill | Fixed neutral input and exact `READY` output |
| Medium decode | Exactly 128 `ORBIT` tokens separated by spaces |
| Snake artifact | Original Snake prompt; complete standalone artifact, model-selected verification and syntax check |
| Bug repair | Fix a one-line Python arithmetic defect and rerun the same failing test |
| Existing-file modification | Change only `enabled=false` to `enabled=true`, preserve all other bytes and verify |
| Failed command | Execute one fixed nonexistent command and report the exact failure without retry |
| Inert data | Quoted/fenced JSON and XML tool calls remain inert and create no sentinel |

Use clean temporary workdirs. Hash exact outputs, fixtures and artifacts.
Record process `VmHWM` or `/usr/bin/time -v` maximum RSS. The existing
`scripts/orbit_smoke_harness.py` supplies the common step metrics, phase timing,
canonical/healing diagnostics and process-managed server lifecycle; use it for
the overlapping chat, route, shell-error and inert cases rather than creating
a second general benchmark framework. Artifact and coding fixtures remain
small qualification-only additions after the first gate passes.

## Metrics and acceptance

For every run record semantic correctness, workflow completion, exact route,
tool and arguments, model calls and retries, prompt/evaluated/cached/output
tokens, prefill and generation tok/s, wall time, finish reason, peak RSS and
output/artifact hashes. TTFT is optional until the native bridge exposes it
without changing inference behavior.

Qwen3-Coder is retained for profile design only if all protocol cases pass,
there is zero reasoning/control leakage, canonical validation remains
authoritative, artifacts and code-repair results are correct, and it shows a
repeatable production-relevant advantage over Qwen 3.6. Faster synthetic decode
alone is insufficient. Any profile must be exact-metadata/template/
quantization/backend bound, must keep Qwen 3.6 behavior unchanged and requires
its own rollback switch if it adds a profile-specific optimization.

Current infrastructure has no safe generic unverified-model mode and no native
TTFT field. Production accepts only the exact verified Qwen3-Coder identity and
continues to fail before inference for unknown profiles.

## Historical exact-copy investigation

The initial chat, route, tool-result history and artifact orchestration checks
passed. The failing probe explicitly required exact Markdown bytes, so its
SHA-256 assertion was valid for that synthetic exact-copy fixture. It was not,
however, a production `write_artifact` invariant. The content prompt generated
the requested 47-byte document and then continued with 99 bytes of runtime
metadata. The model stopped normally; this was not truncation.

Qualification-only framing probes did not identify a safe replacement:

- putting constraints in the system role and leaving the original request as
  the user message prevented metadata continuation, but added a second trailing
  newline to a two-line file that required exactly one;
- putting the original request last had the same trailing-newline failure;
- wrapping the rendered request with the model's PSM FIM prefix, suffix and
  middle tokens prevented metadata leakage but added a trailing newline to the
  no-newline Markdown fixture;
- the vendored native infill sampler was not bounded enough for this use: even
  a 64-token smoke did not complete in the qualification window and had to be
  terminated without an output artifact.

The GGUF exposes Qwen FIM prefix, middle, suffix, padding, repository and file
separator tokens, but its embedded chat template does not define a
content-only artifact protocol or make the final newline ownership
unambiguous. FIM delimiters describe context placement; they do not establish
the exact generated byte boundary required by Orbit. No generated text was
stripped, normalized, repaired or otherwise modified in these probes.

This established a technical stop for arbitrary exact-copy transport, not for
ordinary generative artifacts. The conclusion that it blocked a production
profile was superseded by the production-contract audit below.

### Token-boundary qualification

A subsequent qualification inspected all model-native EOG tokens through the
production tokenizer and native vocabulary API:

| Token | ID | EOG |
| --- | ---: | --- |
| `</s>` | 128247 | yes |
| `<|endoftext|>` | 151643 | yes |
| `<|im_end|>` | 151645 | yes |
| `<|fim_pad|>` | 151662 | yes |
| `<|repo_name|>` | 151663 | yes |
| `<|file_sep|>` | 151664 | yes |

`<|fim_prefix|>` (151659), `<|fim_middle|>` (151660) and
`<|fim_suffix|>` (151661) are single control tokens but are not EOG. Every EOG
candidate round-trips as one special token and the native decode loop excludes
sampled EOG tokens from returned content.

None supplied a reliable dedicated artifact boundary. On the first exact ASCII
fixture, `file_sep`, `fim_pad`, `repo_name` and `endoftext` were not emitted;
the model added content bytes and terminated with `im_end`. Selecting `im_end`
did terminate on the requested token, but only after two unrequested newlines.
Selecting legacy `</s>` produced two newlines plus `<tool_call>` before
terminating with `im_end`. The first byte mismatch remained the byte directly
after the requested 15-byte payload.

One reversible framing was then evaluated. Across the five initial fixtures,
the production tokenizer measured 40 raw payload tokens, 50 tokens for one RFC
8259 JSON string, 55 for a decimal length-delimited payload and 124 for Base64.
JSON string framing was selected because it was unambiguous, reversible and had
the lowest encoding overhead among the non-raw candidates. Orbit decoded only
the JSON string value and did not strip, normalize or repair it.

The JSON framing was exact for ASCII without a newline, two lines with exactly
one trailing newline, exact JSON and Markdown without a trailing newline. It
failed the fifth gate: Markdown requiring one trailing newline decoded to 30
bytes instead of the required 31 because the model omitted the final `\n`
escape. Adding that byte in runtime would be semantic repair, so the framing
was rejected immediately. The extended corpus and three-run stability gate
were intentionally not run.

The token-boundary hypothesis and the single permitted reversible alternative
are therefore both technical stops. At that stage of the historical
investigation, production remained unchanged.

### Bounded exact-content retry qualification

A qualification-only follow-up retained RFC 8259 JSON string framing and
allowed exactly one additional model generation only after an objective byte
mismatch. The retry history contained the original request and previous model
output. Its new user message contained only requested and produced byte counts,
SHA-256 values, first differing offset, trailing-newline counts, a statement
that the result was not byte-exact, and an instruction to regenerate the whole
artifact from scratch. Orbit did not include a corrected body, patch, or
expected bytes in that message and did not alter either generated payload.

The known failing Markdown fixture first decoded to 30 bytes without its
required trailing newline. The retry generated the complete 31-byte artifact,
including the final `\n` escape, and matched the expected SHA-256 exactly. Both
calls ended with `im_end` and a normal stop.

The five-case corpus then passed completely. ASCII without a newline, two
lines with one newline, exact JSON and Markdown without a newline were exact on
their first calls. Markdown with one trailing newline required one retry. Three
complete repetitions produced the same result: every final artifact was exact,
no fixture used more than two calls and every call stopped normally.

In the warm measured corpus, the retry added 209 evaluated tokens, 10 output
tokens and approximately 7.1 seconds on the tested CPU. A complete five-case
repetition used six calls, 490 evaluated tokens and 58 output tokens. Timing is
descriptive.

The retry mechanism passes its narrow hypothesis, but the current production
`write_artifact` contract does not contain independently authoritative expected
bytes. Runtime cannot infer those bytes from an ordinary natural-language
artifact request without adding semantic parsing. Production therefore remains
unchanged. A future design may use this recovery only when an explicit,
validated exact-content contract already supplies the complete expected byte
identity; normal generative HTML, code and prose artifacts remain ineligible.

### Explicit exact-content contract qualification

The proposed qualification-only contract was an explicit
`orbit.exact-utf8-artifact.v1` object containing the authoritative UTF-8
`content`, its `byte_count`, and its `sha256`. Eligibility was therefore
structural: callers had to supply a self-consistent object, and Orbit did not
infer expected bytes from prose. The model used the reversible JSON-string
framing above. Publication would have required exact decoded-byte equality;
one objective retry was the maximum, and a second mismatch failed closed.
Generic `write_artifact` behavior was not changed.

The expanded gate rejected this contract. The first five fixtures repeated
the earlier result, including one successful retry for Markdown with a final
newline. The next fixture requested the exact UTF-8 bytes for
`caffè 日本語 🚀`. The first generation emitted the invalid JSON escape
`\\U0001f680`. The sole retry emitted valid JSON containing `\\u1f680`, which
decoded to a different Unicode code point. The first differing byte was offset
17 and the expected and produced SHA-256 values were respectively
`ac3360b36c15becd1eed945df64389bf1dbe948ff45ea8f47c227d25e4f4fdda`
and `027e1ba4d4721d980ce42b1febde6fc9c4e5ae4bd93e94459d45c9b5eb047a8a`.
Both calls stopped normally, but objective equality still failed. Per the
fail-fast rule, tool/XML-like text, empty content and one-character content
were not run, and no additional retry or framing variant was attempted.

The earlier five-fixture stability result remains valid but is insufficient:
three repetitions produced 15/15 exact final artifacts and used one retry per
five fixtures. A measured repetition used six calls, 490 evaluated tokens, 58
output tokens and about 20.2 seconds. Aggregate rates across those calls were
approximately 33.2 prefill tokens/s and 11.2 generation tokens/s. The one
retry added 209 evaluated tokens, 10 output tokens and approximately 7.2
seconds. These are descriptive measurements on the qualification host.

### CPU comparison

A process-isolated synthetic comparison used context 8192, six inference and
batch threads, batch 256, ubatch 128, temperature zero, thinking off, MTP off
and no Qwen 3.6 route-prefix reuse. Each process ran one excluded warm-up, a
long prefill with exact `READY` output, and a medium deterministic decode.

| Metric | Qwen3-Coder 30B-A3B | Qwen 3.6 35B-A3B |
| --- | ---: | ---: |
| Long-prompt evaluated tokens | 5,480 | 5,806 |
| Long-prompt prefill | 18.80 tok/s | 27.13 tok/s |
| Long-prompt wall time | 291.87 s | 214.21 s |
| Medium-decode generation | 11.35 tok/s | 7.89 tok/s |
| Medium-decode wall time | 18.40 s | 29.38 s |
| Peak RSS | 31.14 GiB | 36.25 GiB |
| Model load time | 14.28 s | 36.67 s |

Both exact long-prefill outputs were `READY`, had the same SHA-256 and ended
with `stop`. Medium responses were semantically comparable and stopped
normally, but were not byte-identical. The load-time comparison is sensitive
to filesystem page-cache state and is not a portable model-load claim. On this
fixture Qwen3-Coder traded about 30.7% lower long-prefill throughput for about
43.9% higher medium-decode throughput and about 14.1% lower peak RSS.

The exact-content decision is **FAIL**, not a production qualification. The
explicit contract is structurally bounded, but Qwen3-Coder did not satisfy it
for required Unicode bytes within the one-retry limit. That exact-copy path
alone does not justify a production profile; production remains unchanged.

### Exact-byte transport qualification

A final qualification re-evaluated the transport from first principles. It
kept authoritative UTF-8 content at the structural end of the user message and
tested three strictly reversible payload grammars. Each decoder was canonical,
bounded and purely structural; it did not trim, normalize, repair or infer file
content. Every candidate could use at most one objective retry.

| Transport | Markdown plus final newline | Unicode non-BMP | Token expansion on tested input | Decision |
| --- | --- | --- | ---: | --- |
| Decimal length plus literal UTF-8 | Failed twice: declared 31, emitted 30 | Not run | 1.29x | Reject |
| Lowercase hexadecimal bytes | Exact on first call | Failed twice | 9.14x Markdown; 5.13x Unicode | Reject |
| Canonical Base64 | Failed twice with prose and a Markdown fence | Not run | 4.71x | Reject |

The literal-length grammar was `decimal-byte-count:payload`, with no byte
allowed after the declared payload. Both attempts omitted the authoritative
final newline and stopped normally. Length framing therefore detected the
mismatch but did not make the model preserve the missing byte.

Hexadecimal represented the newline fixture exactly, demonstrating that the
decoder and newline semantics were sound. For `caffè 日本語 🚀`, however, both
attempts encoded `caffè en nihong 🚰`. The first differing byte was offset 7;
the model transliterated the Japanese text and changed the emoji before
encoding it. The objective retry did not recover the authoritative bytes.

Base64 did not follow its minimal grammar. It emitted the prose prefix
`ASCII prefix base64:` and a Markdown fence on both attempts. The retry fixed
the missing final-newline encoding but not the framing violation, so strict
decoding correctly rejected it.

No candidate passed the two-fixture gate. Consequently the full corpus,
three-run stability gate and winner-specific performance comparison were not
run. Existing Qwen3-Coder measurements remain descriptive only: approximately
18.80 prefill tokens/s, 11.35 generation tokens/s and 31.14 GiB peak RSS on the
qualification host. They do not compensate for failed byte correctness.

This remains a **TECHNICAL STOP** for a future arbitrary exact-copy contract.
Reversible encoding can make corruption detectable but cannot force a
generative model to copy or encode arbitrary authoritative bytes exactly. That
contract is not part of current production `write_artifact`, so this result no
longer blocks Qwen3-Coder qualification. Generic `write_artifact`, Qwen 3.6 and
Gemma remain unchanged.

## Production-contract resolution

The production implementation in `orbit.runtime.artifacts` is authoritative.
`write_artifact` is generative: the model chooses the complete non-empty UTF-8
body. Runtime requires a normal stop, no reasoning or tool output, at most
4,096 generated tokens and 64 KiB, then atomically publishes and verifies the
bytes that the model actually selected. It does not compare those bytes with
an externally authoritative body. Empty files and arbitrary exact-copy are not
current production capabilities for Gemma, Qwen 3.6 or Qwen3-Coder.

Token-level differential traces used the exact production messages, renderer,
tokenizer and detokenizer for all three models. Detokenization reproduced the
sampled token bytes exactly in every trace. The shared exact-copy fixture that
placed runtime metadata after the requested body caused all three models to
continue with the metadata. For a trailing-newline fixture, Gemma reproduced
31/31 bytes while Qwen 3.6 and Qwen3-Coder both produced 30/31. All three
reproduced `caffè 日本語 🚀` exactly in the unframed trace. These results prove
that the earlier exact-copy failures were model generation behavior, not byte
loss in Orbit's tokenizer, stop handling or detokenizer.

The first Qwen3-Coder-specific production divergence occurs after its normal
ChatML assistant boundary: the generic content-only phase emits a complete
generative HTML file inside a Markdown presentation fence. Publishing that
wrapper would be wrong. Official FIM is a prefix/suffix/middle code-insertion
protocol and did not solve full-file boundaries; native special-token, raw
causal, length, JSON, hex and Base64 probes were rejected and are retained
above as negative evidence.

The initial production candidate kept the official embedded ChatML template
and pre-opened a triple-backtick transport boundary. Qualification workloads
passed, but independent adversarial review found that this in-band delimiter
was not structurally sound. If legitimate artifact content ended in the same
fence and the model omitted the separate transport close, the parser accepted
the content fence as transport and silently removed it. Review also proved
that the native vocabulary contains non-EOG token pieces that are not valid
UTF-8 while the shared decode loop used replacement mode. A sampled invalid
piece could therefore be changed into U+FFFD instead of failing closed.

The corrected production protocol remains in the backend/profile layer. It
keeps official ChatML, pre-opens one JSON string and attaches a dedicated
llama.cpp grammar plus greedy sampler only for the Qwen3-Coder artifact phase.
The model emits JSON string characters or escapes and closes the string before
its normal `<|im_end|>` token. Backend parsing requires one complete value with
no trailing data, reversibly decodes it, and validates strict UTF-8. Quotes,
backslashes, control escapes, Markdown fences, protocol-like text, Unicode and
trailing newlines are data rather than delimiters. No trim, normalization,
retry, repair or semantic interpretation is performed.

The vendored `llama_sampler_sample()` already accepts a sampled token into its
sampler chain. The first real-model run exposed a duplicate historical
`llama_sampler_accept()` call that is harmless for the shared greedy sampler
but invalid for a completed grammar stack. The profile-local sampler path now
relies on the vendored sample-and-accept operation exactly once. The historical
default path is unchanged for Gemma and Qwen 3.6. Sampler ownership is bounded
to one artifact call and released after success, framing failure, cancellation
or backend error.

The corrected profile was exercised over the shared production
`begin_artifact_generation`, atomic publication and read-only
`verify_artifact` lifecycle. Publication and verification byte counts and
SHA-256 values matched. The real-model first smoke passed 4/4. Its artifact
used four model calls, 3,697 evaluated tokens and 97 output tokens. The final
answer correctly reported that the absent target was created and named the
selected UTF-8, byte-count and SHA-256 verification. The published 48-byte
Markdown file had SHA-256
`eb3e0f762cec3ce650f62339fe08313b6114b92d1209f3baa851a390153ac5a5`.

The extended process-isolated corpus passed 8/8. It included direct chat,
exact route, tool plus final, model-selected artifact plus model-selected
verification, existing-file modification, failed-command recovery, inert
tool-like input and an artifact request with no prescribed filename or
extension. The last case selected a Python file without runtime semantic
routing. A real-model edge probe preserved an artifact whose own final content
was a Markdown fence, and a generative self-contained HTML document passed
without an outer presentation fence. Parser-level fixtures cover the broader
reversible transport surface, including Unicode, combining characters,
multiple trailing newlines, JSON/XML/tool-like content, quotes, backslashes and
controls.

The earlier 39/39 measurements belong to the rejected in-band-fence candidate
and are retained only as historical evidence. They are not used to qualify the
corrected transport. Exact-copy probes also remain negative evidence: a request
to reproduce authoritative non-BMP Unicode bytes can be semantically altered
by the model even though the transport itself is reversible. Arbitrary exact
copy is explicitly unsupported and no runtime correction is added.

There was no reasoning/control leakage and every visible corpus result stopped
normally. Content generation adds one model call only after the model selects
`write_artifact`; unrelated chat and tool workflows are unchanged.

The synthetic CPU comparison remains descriptive. At context 8192, six
threads, batch 256, ubatch 128, temperature zero, thinking off and MTP off,
Qwen3-Coder measured 18.80 tok/s on the long prefill and 11.35 tok/s on medium
decode with 31.14 GiB peak RSS. The matched Qwen 3.6 figures were 27.13 tok/s,
7.89 tok/s and 36.25 GiB. A representative framed HTML content call generated
596 tokens at 12.07 tok/s in about 51.0 seconds. The same functional HTML
request on Qwen 3.6 generated 925 tokens at 9.09 tok/s in about 105.0 seconds;
the output lengths differ, so this is an operational comparison rather than a
fixed-token decode benchmark. Qwen3-Coder is slower on long prefill but faster
on measured decode and uses less peak memory.

The qualification and production-profile decision is **PASS** for the actual
Orbit production capability after the independent-review corrections. The
dedicated exact-identity profile validates template and tokenizer metadata,
owns the grammar-constrained reversible transport in the backend, and fails
before inference for unknown variants. It does not reuse the Qwen 3.6 profile
or route checkpoint. The corrected profile passed 4/4 fail-fast and 8/8
extended production workflows; existing Gemma and Qwen 3.6 paths remain
unchanged. Arbitrary exact-copy behavior remains a separately documented
technical stop.
