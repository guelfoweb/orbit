# PROMPTS.md

Curated manual regression prompts for Orbit on `llama-server`.

Run from the repository root:

```bash
HOME_DIR="$(mktemp -d)" HOME="$HOME_DIR" orbit --model gemma4:12b --workdir .
```

Use a clean temporary `HOME` for regression runs to avoid old sessions affecting timing or behavior.

Before long summaries, optionally raise the output budget:

```text
/max-tokens 1024
```

## Core prompts

1. `hi, who are you?`

Expected: normal chat answer, no tool call, streamed output, footer shows `model: gemma4:12b`.

2. `tell me what grep is used for, but do not run any command`

Expected: conceptual explanation, no local tool use.

3. `list files and directories in this workdir`

Expected: model uses `list_files`; answer lists repository entries.

4. `read README.md and summarize the project in two concise sentences`

Expected: model uses `read_file`; answer summarizes Orbit, not raw file dump.

5. `read AGENTS.md and tell me the three strictest runtime rules`

Expected: model uses `read_file`; answer extracts concrete rules from the file.

6. `what is the current max token limit?`

Expected: no tool call; either answers from context if known or suggests `/max-tokens`. Then test `/max-tokens`.

7. `/max-tokens 2048`

Expected: command handled locally; output `max_tokens: 2048`; next `/status` reflects it.

8. `write a short explanation of local LLM inference in 12 bullet points`

Expected: streamed output; no `stop: length` with raised max tokens.

9. `read a large UTF-8 text file if present and explain how Orbit should handle it`

Expected: if no suitable file exists, model should not invent contents; if a large file exists, it should use `read_file` chunk mode.

10. `start answering a long explanation, then interrupt with Ctrl+C`

Expected: stream stops, Orbit prints `interrupted`, rolls back the partial turn, and returns to `>`.

## Memory refresh prompt

Use only when intentionally testing session memory because it can be slow on CPU-only hardware:

```bash
CONTEXT_TOKENS=1600 MAX_TOKENS=64 scripts/bench-memory-refresh.sh
```

Expected: when refresh happens, Orbit prints a line like:

```text
memory: 1081->280 est. tokens | saved 801 (74%) | 227.0s | threshold 1200/1600
```
