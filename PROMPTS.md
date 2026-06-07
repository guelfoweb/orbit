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

Expected: model uses one native filesystem tool only, typically `exec_shell_command` with a safe `ls` command or `file_glob_search`; answer lists repository entries without a second equivalent tool call.

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

## Tool backend prompts

11. `/status`

Expected: handled locally in both REPL and one-shot mode; must not call the model. Output shows `tools_llama_server` and `tools_orbit`.

12. `/tools`

Expected: handled locally in both REPL and one-shot mode; output groups native tools under `llama-server` and non-duplicated local tools under `orbit-only`.

13. `read sample.txt and summarize it in one sentence`

Expected: model uses `read_file`; when `llama-server --tools read_file` is enabled, the tool result event shows `src: llama-server`.

14. `read text/divina_commedia_inferno_canto1.txt and summarize it in Italian in 5 lines. Mention the main scene, characters, and symbolic meaning.`

Expected: model uses local file tools, not memory. With llama-server tools enabled, it should prefer bounded native `read_file`; without native tools, Orbit fallback chunking must stay under context.

15. `search inside local text files for the word Virgilio and summarize the matches`

Expected: model uses `grep_search` when exposed by `llama-server`; tool result event shows `src: llama-server`. It should not read the entire long text file.

16. `summarize this URL in one short paragraph: https://example.com`

Expected: model routes to web tools and uses Orbit `fetch_url`; `llama-server` has no native web tool, so source remains Orbit.

## Native tool guardrail prompts

Before running these prompts, create disposable files inside the active `workdir`:

```bash
printf 'red\nblue\ngreen\n' > server-tool-test.txt
printf 'alpha\nbeta\ngamma\n' > patch-tool-test.txt
```

17. `run wc -l on text/summary.txt and tell me only the line count`

Expected: model routes to filesystem tools and may use `exec_shell_command`; command is read-only and result source should be `llama-server`.

18. `run rm server-tool-test.txt`

Expected: if the model calls `exec_shell_command`, Orbit blocks it before `llama-server` with a clear policy error. The file must remain present.

19. `In server-tool-test.txt replace line 2 with BLUE using a file editing tool, then tell me what changed.`

Expected: model routes to file editing tools and should use native `edit_file`; tool result source should be `llama-server`; file content becomes `red`, `BLUE`, `green`.

20. `Apply a unified diff to patch-tool-test.txt that changes beta to BETA and appends delta, then summarize the patch.`

Expected: model routes to file editing tools, reads the file when line context is needed, and uses bounded local `edit_file`; file content becomes `alpha`, `BETA`, `gamma`, `delta`.

After these prompts, clean up:

```bash
rm -f server-tool-test.txt patch-tool-test.txt
```

## Memory refresh prompt

Use only when intentionally testing session memory because it can be slow on CPU-only hardware:

```bash
CONTEXT_TOKENS=1600 MAX_TOKENS=64 scripts/bench-memory-refresh.sh
```

Expected: when refresh happens, Orbit prints a line like:

```text
memory: 1081->280 est. tokens | saved 801 (74%) | 227.0s | threshold 1200/1600
```
