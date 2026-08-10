from __future__ import annotations

import json
from pathlib import Path

from .schema import QualificationRun, as_primitive


def result_json(run: QualificationRun) -> str:
    return json.dumps(as_primitive(run), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_result(run: QualificationRun, path: Path | str) -> None:
    Path(path).write_text(result_json(run), encoding="utf-8")


def format_summary(run: QualificationRun) -> str:
    metrics = run.aggregate_metrics
    lines = [f"Profile: {run.provenance.profile_identity}", "", "Common"]
    lines.extend(f"  {item.name:<16} {item.status.value}" for item in run.common)
    lines.extend(("", "Fixtures"))
    lines.extend(f"  {item.name:<16} {item.status.value}" for item in run.fixtures)
    lines.extend(
        (
            "",
            "Metrics",
            f"  input           {_count(metrics.input_tokens)}",
            f"  evaluated       {_count(metrics.evaluated_tokens)}",
            f"  cached          {_count(metrics.cached_tokens)}",
            f"  output          {_count(metrics.output_tokens)}",
            f"  calls           {metrics.calls}",
            f"  prefill         {_rate(metrics.prefill_tokens_per_second)}",
            f"  decode          {_rate(metrics.generation_tokens_per_second)}",
            f"  fixture wall sum {metrics.wall_seconds:.2f} s",
            f"  server VmHWM    {_bytes(metrics.peak_rss_bytes)}",
            "  TTFT            unavailable",
            "",
            "Overall:",
            run.overall_detail,
        )
    )
    return "\n".join(lines)


def _rate(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.2f} tok/s"


def _count(value: int | None) -> str:
    return "unavailable" if value is None else str(value)


def _bytes(value: int | None) -> str:
    return "unavailable" if value is None else f"{value / (1024 ** 3):.2f} GiB"
