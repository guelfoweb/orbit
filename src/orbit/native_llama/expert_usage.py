from __future__ import annotations

from typing import Sequence


PHASES = ("prefill", "decode")
MAX_LAYERS = 256
MAX_EXPERTS = 512


def summarize_expert_usage(
    counts: Sequence[int], tokens: Sequence[int], *, layers: int, experts: int, active: int
) -> dict[str, object]:
    if not 0 < layers <= MAX_LAYERS or not 0 < experts <= MAX_EXPERTS:
        raise ValueError("invalid MoE shape")
    if not 0 < active <= experts:
        raise ValueError("invalid active expert count")
    if len(counts) != len(PHASES) * MAX_LAYERS * MAX_EXPERTS or len(tokens) != len(PHASES) * MAX_LAYERS:
        raise ValueError("invalid expert-usage snapshot size")
    phase_rows: list[list[list[int]]] = []
    result: dict[str, object] = {}
    for phase_index, name in enumerate(PHASES):
        base = phase_index * MAX_LAYERS * MAX_EXPERTS
        rows = [
            [int(counts[base + layer * MAX_EXPERTS + expert]) for expert in range(experts)]
            for layer in range(layers)
        ]
        phase_rows.append(rows)
        routed = [int(tokens[phase_index * MAX_LAYERS + layer]) for layer in range(layers)]
        result[name] = _summary(rows, routed, experts, active)
    aggregate = [
        [phase_rows[0][layer][expert] + phase_rows[1][layer][expert] for expert in range(experts)]
        for layer in range(layers)
    ]
    aggregate_tokens = [int(tokens[layer]) + int(tokens[MAX_LAYERS + layer]) for layer in range(layers)]
    return {
        "layers": layers,
        "experts": experts,
        "experts_per_token": active,
        "phases": result,
        "aggregate": _summary(aggregate, aggregate_tokens, experts, active),
    }


def _summary(
    rows: Sequence[Sequence[int]], routed: Sequence[int], experts: int, active: int
) -> dict[str, object]:
    selections = sum(sum(row) for row in rows)
    ids = {expert for row in rows for expert, count in enumerate(row) if count}
    top_n = {}
    for n in (16, 32, 64, 96, 128, 192, 256):
        if n <= experts:
            value = sum(sum(sorted(row, reverse=True)[:n]) for row in rows)
            top_n[str(n)] = value / selections if selections else None
    details = []
    for layer, row in enumerate(rows):
        total = sum(row)
        ordered = sorted(row, reverse=True)
        details.append({
            "layer": layer,
            "routed_tokens": int(routed[layer]),
            "selections": total,
            "experts_observed": sum(bool(value) for value in row),
            "top_1_share": ordered[0] / total if total else None,
            "top_8_share": sum(ordered[:8]) / total if total else None,
            "counts": list(row),
        })
    nonzero_routed = {value for value in routed if value}
    routed_tokens = nonzero_routed.pop() if len(nonzero_routed) == 1 else None
    return {
        "selections": selections,
        "routed_tokens": routed_tokens,
        "routed_token_events": sum(routed),
        "selection_accounting_valid": selections == sum(routed) * active,
        "experts_observed": len(ids),
        "layer_experts_observed": sum(sum(bool(value) for value in row) for row in rows),
        "top_n_coverage": top_n,
        "layers": details,
    }
