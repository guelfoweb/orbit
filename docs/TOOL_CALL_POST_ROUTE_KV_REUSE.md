# Tool-Call and Post-Tool KV Reuse Audit

## Scope

This audit measured native Gemma 4 12B prompt reuse across `tool_call` and
`post_tool_route`. It did not change prompts, model-call count, tool behavior,
or active checkpoint selection.

The measured CPU-only profile used context 8192, six threads, batch 256,
ubatch 128, thinking off, tools on, fixed affinity, and temperature zero. The
sample covered 23 workflows across shell, system information, directory list,
file read, and content search families.

The boundary and restore measurements below are deterministic probes. A later
production-like opportunity sample observed no useful real restore
opportunity. Probe savings therefore establish mechanism feasibility, not
production utility.

## Prompt Layout

The native backend already reused the exact longest common prefix. Adjacent
tool/post-tool prompts diverged before generated tokens because runtime placed
`capability_context` after the current conversation on each loop. The first
tool prompt had roles `system,user,system`; the next prompt had roles
`system,user,assistant,tool,system`. Moving the capability block after the
assistant call and tool result changed the prefix. This was prompt layout, not
checkpoint invalidation or a backend LCP defect.

## Boundary Probe

The production tokenizer measured these candidate boundaries:

| Boundary | Meaning | Cold/segmented max logits difference | Decision |
| ---: | --- | ---: | --- |
| 766 | existing route prewarm | 2.8360815 | reject |
| 824 | stable tool system/schema end | 2.8860884 | reject |
| 838 | before capability context | 2.9907508 | reject |
| 768 | production 64-token batch boundary | 0.0 | numerically viable |

The 768-token boundary used meaningful production content with no padding,
prompt rewrite, or extra prefill. Cold full prefill, segmentation at 768, and
checkpoint restore produced identical token positions, logits hashes, next
tokens, ordered top candidates, tool calls, arguments, outputs, and finish
reasons across the measured system, list, read, grep, and shell families.

| Family | Cold evaluated | Restore evaluated | Cold prefill | Restored suffix |
| --- | ---: | ---: | ---: | ---: |
| system info | 926 | 158 | 69.29 s | 13.00 s |
| list directory | 925 | 157 | 69.36 s | 12.87 s |
| read file | 928 | 160 | 69.54 s | 13.15 s |
| grep search | 934 | 166 | 70.06 s | 13.67 s |
| shell success | 931 | 163 | 69.98 s | 13.38 s |

The exact saving was 768 evaluated tokens for an eligible cold tool call. A
12-output process-isolated comparison retained identical output and tool-call
hashes and measured about 55.5-56.5 seconds less prefill on that machine. This
was workload-specific and is not a deterministic performance guarantee.

The checkpoint measured 264,260,776 bytes. Peak process RSS was roughly 499
MiB above cold because capture and restore temporarily copied the buffer. RSS
remained flat across ten restores, and invalidation removed the retained blob
and released approximately 252 MiB.

## Conclusion

The optimization only helped cold tool calls with no useful normal LCP. MTP
remained authoritative and incompatible with the candidate path. No useful
real restore opportunity was observed in the production-like sample, while
the checkpoint retained about 252 MiB. The measured utility did not justify
retaining an unpromoted runtime feature after RC23.

All tool-prefix runtime integration, configuration, payload fields,
checkpoint state, diagnostics, and tests were discarded. The audit remains as
numerical evidence only; tool-prefix reuse is not active in production.

## Reopening Criteria

Do not restore the implementation unless a new production-like sample shows a
repeatable rate of cold calls with zero useful normal LCP and enough successful
restore opportunities to justify the resident checkpoint. Any new proposal
must also reproduce exact token, logits, tool, argument, output, and finish
reason equivalence; pass process-isolated timing; bound RSS and temporary
copies; preserve MTP authority; and pass cancel, timeout, reset, restart,
identity-mismatch, and restore-failure lifecycle gates.
