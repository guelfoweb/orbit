# Post-Tool Model-Call Audit

## Scope

This observational audit followed default post-tool final prose reuse. It did
not change routing, tool selection, execution, finalization, retries, prompts,
schemas, or backend behavior.

The measured corpus combined 50 process-isolated post-tool reuse scenarios,
five additional tool scenarios, and three route scenarios. In total, 58
correct scenarios produced 121 correlated model calls. Diagnostics contained
metadata only, without prompts, tool arguments, evidence, or complete outputs.

## Measured Call Map

| Category | Calls | Evaluated tokens | Output tokens | Wall seconds | Disposition |
| --- | ---: | ---: | ---: | ---: | --- |
| `tool_call` | 55 | 12,566 | 1,395 | 1,436.7 | necessary |
| `post_tool_route` | 50 | 14,639 | 749 | 1,455.3 | necessary; prose reused |
| `initial_route` | 8 | 359 | 119 | 75.3 | necessary |
| `final_from_tool` synthesis | 5 | 712 | 340 | 166.8 | necessary |
| bounded confirmation | 2 | 284 | 26 | 30.0 | theoretical candidate only |
| `chat_final` | 1 | 270 | 76 | 45.8 | necessary |

All measured calls ended with `finish_reason=stop`. No retry, formatting
repair, duplicate route decision, or uncorrelated backend call was observed.

## Conclusions

Tool selection, post-tool decisions that may select another tool, evidence
interpretation, requested synthesis, and CHAT finalization remained necessary
model work. No additional frequent inference was safely eliminable in the
measured sample.

The two bounded confirmation calls could not be removed safely: runtime had no
prior prose to reuse and could not prove that raw tool output was a complete
user-facing response without taking over a semantic decision.

## Exact Replay

An exact final-input replay counter was considered as a hypothesis for an
observational follow-up. A safe design would require a hash over the complete
rendered input and all model, tokenizer, template, backend, phase,
temperature, token-budget, thinking, and runtime identities. It would also
require session-local bounded state and lifecycle invalidation.

No replay implementation was promoted or retained. Repeated opportunities and
full equivalence were not established, so exact replay remains a technical
stop rather than a production feature.

## Other Technical Stops

- Structured `read_file` and `grep_search` schemas increased cold prefill and
  reduced argument fidelity in the measured candidate.
- Production schema compaction changed tool selection, exact arguments, or
  adversarial behavior in every useful compact variant.
- Additional generic output bounding was not justified by observed downstream
  evidence; existing shell/search bounds and final evidence compaction already
  covered the sample.
- Bounded multi-action planning failed its semantic smoke gate with Gemma 4
  12B and is not active.

No runtime implementation was retained for exact replay, structured file
tools, schema compaction, generic output bounding, or tool-prefix reuse. The
bounded planning analyzer, `ORBIT_TOOL_PLAN_SHADOW`, generation-only harness,
and related tests remain as OFF-by-default observational infrastructure. They
never execute a plan, and the planning technical stop remains in force.

## Reopening Criteria

- Exact replay requires repeated byte-identical final inputs in real sessions,
  full compatibility identity, process-isolated equivalence, bounded storage,
  lifecycle invalidation, and an OFF-by-default shadow phase before any reuse.
- Structured file tools require a new model/template result with no tool or
  argument regression and lower end-to-end evaluated-token and wall cost.
- Schema compaction requires exact tool and payload invariance across complex
  paths and adversarial negatives, not merely fewer prompt tokens.
- Generic output bounding requires observed oversized downstream evidence with
  material token cost and proof that all user-required content is preserved.
- Bounded planning requires a materially different model/template to pass the
  documented JSON, exact-plan, unsupported-plan, and zero-wrong-plan smoke
  gates before repetitions or execution can be reconsidered.
