from __future__ import annotations


def format_tool_compaction_report(report) -> str:
    lines = ["tool result compaction"]
    if not report.candidates:
        lines.append("candidates: none")
        return "\n".join(lines)
    lines.append(f"candidates: {len(report.candidates)}")
    lines.append("candidate details:")
    for candidate in report.candidates:
        lines.append(
            f"- {candidate.tool}: {candidate.estimated_tokens} tokens, age {candidate.age_messages} message(s)"
        )
    if not report.items:
        lines.append("compacted: none")
        return "\n".join(lines)
    lines.append("results:")
    for item in report.items:
        status = "compacted" if item.changed else f"skipped ({item.reason})"
        lines.append(
            f"- {item.tool}: {status}, {item.before_tokens}->{item.after_tokens} tokens, "
            f"saved {item.saved_tokens}, age {item.age_messages} message(s)"
        )
    lines.append(f"total_saved: {report.saved_tokens}")
    return "\n".join(lines)


def format_memory_compaction_report(refresh) -> str:
    saved = max(0, refresh.estimated_tokens_before - refresh.estimated_tokens_after)
    status = "compacted" if refresh.changed else f"skipped ({refresh.reason})"
    lines = [
        "memory compaction",
        f"status: {status}",
        f"tokens: {refresh.estimated_tokens_before}->{refresh.estimated_tokens_after}",
        f"saved: {saved}",
    ]
    if refresh.threshold_tokens is not None and refresh.context_tokens is not None:
        lines.append(f"threshold: {refresh.threshold_tokens}/{refresh.context_tokens}")
    if refresh.elapsed_seconds:
        lines.append(f"time: {refresh.elapsed_seconds:.1f}s")
    return "\n".join(lines)
