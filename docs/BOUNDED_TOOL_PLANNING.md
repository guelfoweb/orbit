# Minimal Two-Step Tool Planning Shadow

## Status

The experiment is observational only. It requires both
`ORBIT_TOOL_PLAN_SHADOW=1` and
`scripts/orbit_smoke_harness.py --tool-plan-shadow`. The resolver is OFF by
default; `0` and invalid values stop the benchmark before server startup.
Production routing, canonical validation, healing, tool execution, MTP, final
prefix reuse, and finalization are unchanged. No plan is executed.

## Minimal Contract

The model may return exactly one of two objects:

```json
{
  "type": "tool_plan",
  "steps": [
    {"name": "system_info", "arguments": {}},
    {"name": "list_directory", "arguments": {"path": ".", "max_entries": 20}}
  ]
}
```

```json
{"type": "unsupported_plan"}
```

The plan has exactly two steps. The runtime assigns internal `step_1` and
`step_2` IDs after parsing. The model does not choose IDs, expectations,
dependencies, branching, completion behavior, or execution policy.

Only `system_info` and `list_directory` are eligible. Both steps must have
literal arguments known before execution and must independently pass the
existing canonical contract. Optional arguments retain only defaults already
declared by the canonical schema; the planning layer does not add values.

`unsupported_plan` is required when any action needs another tool, mutation,
network access, an intermediate result, or cannot be completed by exactly two
eligible calls.

## Prompt Views

All views use the same schema, tool definitions, corpus, model configuration,
and output budget.

1. `contract`: minimal schemas and unsupported conditions.
2. `json_only`: `contract` plus one instruction prohibiting prose and Markdown.
3. `exactness`: `json_only` plus one instruction prohibiting substitution of an
   unsupported requested action with an allowed tool.

This isolates one prompt change between adjacent views.

## Corpus

The nine-case corpus contains:

- two `system_info` plus `list_directory` plans in different orders;
- two independent `list_directory` pairs with exact paths and bounds;
- five unsupported tasks covering shell, mutation, output dependency, network,
  and a missing file-read tool.

Correctness is measured against exact tool order and exact argument values.
Canonical validity alone is not semantic correctness. A valid plan returned for
an unsupported task, or a plan with changed arguments, is `wrong_plan`.

## Gemma 4 12B Result

The one-repetition smoke used native Gemma 4 12B Q4_K_M, the production
template, `ctx=8192`, six threads, temperature zero, and 160 output tokens.
All 27 calls were evaluable with healthy cleanup. Tool executions and
finalizations were zero.

| View | JSON | Exact positive plans | Unsupported accuracy | Wrong-plan rate | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `contract` | 44.4% | 0% | 80% | 0% | FAIL |
| `json_only` | 55.6% | 0% | 100% | 0% | FAIL |
| `exactness` | 88.9% | 50% | 100% | 11.1% | FAIL |

`exactness` produced two exact plans, one positive plan with the correct tool
sequence but different arguments, and one positive response with prose
leakage. The other views leaked prose for every positive plan. No invalid JSON
object was observed after a candidate marker; one negative `contract` response
contained no plan marker and was classified as prose/no-plan.

Per-view cost for nine calls:

| View | Prompt tokens | Evaluated tokens | Output tokens | Median wall | Total wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `contract` | 5,504 | 809 | 284 | 5.88 s | 174.18 s |
| `json_only` | 5,621 | 714 | 239 | 5.48 s | 133.97 s |
| `exactness` | 5,729 | 713 | 265 | 4.70 s | 141.45 s |

No model-call reduction is credited. Two exact shadow plans imply only a
structural possibility; they were not executed and the view failed the semantic
gate. The five-repetition phase was not run because no view achieved at least
90% JSON compliance, zero wrong plans, and complete positive/negative accuracy.

## Decision

Runtime parsing and validation remain feasible, but model adherence is not
sufficient even for the minimal two-step contract. The experiment remains
closed to execution.

Do not add tolerant parsing, grammar, retries, semantic repair, tool
substitution, or a larger schema from this result. Reopen only with new model or
template evidence that can achieve zero wrong plans and exact arguments on the
same minimal corpus.
