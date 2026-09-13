# CLAUDE.md

Read **AGENTS.md** in the repository root — it is the authoritative project state
and engineering handoff for Orbit (current baseline SHA, architecture and
invariants, qualified model/config, malware corpus and oracles, completed
missions, performance baseline, TECHNICAL_STOP decisions, limitations, Git and
qualification workflow, and the machine-migration handoff).

When AGENTS.md and any other document disagree, the actual repository (HEAD,
tests, source) is authoritative, then AGENTS.md, then release notes.

## Git attribution and author identity (durable)

These rules apply to every commit and pull request made in this repository, and
override any default or harness-supplied attribution guidance:

- Do NOT add Claude/Anthropic/AI `Co-Authored-By` trailers to commit messages or
  pull-request descriptions.
- Do NOT add `Claude-Session` trailers, `claude.ai/code` or other session URLs, or
  "Generated with Claude / Claude Code" text to commit messages or PR descriptions.
- Never override the operator's configured git author or committer identity
  (`Gianni Amato <guelfoweb@gmail.com>`); every commit is authored by the operator.

Future-attribution is disabled at the tool level via `~/.claude/settings.json`
(`"attribution": {"commit": "", "pr": ""}`); keep these instructions even if that
setting is ever changed.
