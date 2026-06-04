# BENCHMARK

## Minimal regression benchmark

Use this after runtime, routing, compaction, guardrail, media, or tool changes:

```bash
python3 -m unittest discover -s tests -q
PYTHONPATH=src python3 -m orbit --workdir workdir --model gemma4:e2b-c8k --debug-timing "list all files and directories in the current workspace"
```

`--debug-timing` prints bounded route, intent-gate, tool, and model timings. It is off by default.

## Method

- Target model: `gemma4:e2b-c8k`.
- Target context: `8192`.
- Prompt source: [PROMPTS.md](PROMPTS.md).
- Run prompts sequentially, one at a time.
- Use one isolated session per prompt to avoid history contamination.
- Keep Ollama warm before measuring; exclude the warm-up prompt from benchmark results.
- Use `--debug-timing` on every prompt.
- Treat `src: local` as a valid fast path when Orbit can answer from bounded local evidence without another model call.
- Skill prompts may depend on a local fixture workdir; do not publish fixture contents in benchmark notes.

## Current model profiles

```bash
ollama create gemma4:e2b-c8k -f Modelfile.gemma4-e2b-c8k
ollama create gemma4:e2b-c4k -f Modelfile.gemma4-e2b-c4k
```

```text
FROM gemma4:e2b

# Main profile used for the benchmark on an Intel NUC 10 class CPU-only machine:
# Intel i7-10710U, 6 physical cores / 12 threads, 64 GB RAM.
PARAMETER temperature 0
PARAMETER num_ctx 8192
PARAMETER num_thread 6
PARAMETER num_batch 96
```

```text
FROM gemma4:e2b

# Conservative profile used on an Intel Xeon E3-1275 v6 CPU-only machine:
# 4 physical cores / 8 threads, about 16 GB RAM.
# Also use this when the c8k profile crashes the Ollama runner with GGML scheduler errors.
PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER num_thread 4
PARAMETER num_batch 64
```

If Ollama fails with `GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)`, rerun with `gemma4:e2b-c4k` before changing Orbit runtime behavior.

Recommended Ollama server settings must be applied to the server process:

```bash
OLLAMA_NUM_PARALLEL=1
OLLAMA_KEEP_ALIVE=-1
OLLAMA_MAX_LOADED_MODELS=1
```

## Latest warm run

Run date: 2026-06-01

Warm-up excluded from results: `hi` completed in `2.6s`, `src: model`, `ctx: 1.4%`.

Summary:

| Group | Count | Exit OK | Fastest | Slowest | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Strategic Suite | 12 | 12 | `0.5s` | `91.8s` | local, web, vision, and review paths covered |
| Strong Prompts | 9 | 9 | `0.5s` | `107.5s` | heavy vision/audio/long-text paths covered |
| Ambiguous Intent Prompts | 10 | 10 | `3.3s` | `43.0s` | tool suppression and intent-gate behavior covered |
| Skill Prompts | 1 | 1 | `1.5s` | `1.5s` | skill path covered with fixture output omitted |

Observations:

- All prompts exited successfully.
- Fast bounded local paths completed in about `0.5s`.
- Web exact-match checks over fetched pages completed in about `1.0s` through local evidence extraction.
- Vision comparison required multiple model calls and ranged from about `85s` to `108s` on CPU-only hardware.
- Audio transcription/summary completed in about `66s` to `70s` with chunked preprocessing.
- Ambiguous operational prompts correctly avoided broad tool execution when the intent gate returned `NO`.
- Code review now returns a more evidence-based local finding instead of generic security advice.
- Long literary summarization uses compacted progressive evidence; it is slower than the previous extractive-only path but produces a more coherent summary with lower context pressure than the initial refactor attempt.

## Debug timing table

| ID | Elapsed | Context | Source | Route | Intent Gate | Tools / Media |
| --- | ---: | ---: | --- | --- | --- | --- |
| P01 | `0.5s` | `9.4%` | `local` | text_document_analysis -> filesystem | - | list_files |
| P02 | `0.5s` | `8.9%` | `local` | codebase_inspection -> filesystem | - | list_files |
| P03 | `0.5s` | `6.4%` | `local` | - | - | - |
| P04 | `0.5s` | `8.0%` | `local` | text_document_analysis -> filesystem | - | read_file |
| P05 | `0.5s` | `6.9%` | `local` | text_document_analysis -> filesystem | - | stat_path |
| P06 | `0.5s` | `16.7%` | `local` | text_document_analysis -> filesystem | - | stat_path |
| P07 | `66.1s` | `21.4%` | `tool+model` | codebase_inspection -> filesystem | - | list_files, read_file |
| P08 | `91.8s` | `27.1%` | `tool+model` | current_factual_lookup -> web | - | fetch_url |
| P09 | `87.4s` | `2.1%` | `model` | - | - | vision |
| P10 | `18.8s` | `8.6%` | `tool+model` | current_factual_lookup -> web | - | search_web |
| P11 | `1.0s` | `34.8%` | `local` | current_factual_lookup -> web | - | fetch_url |
| P12 | `1.0s` | `29.6%` | `local` | current_factual_lookup -> web | - | fetch_url |
| P13 | `9.5s` | `9.4%` | `tool+model` | codebase_inspection -> filesystem | - | list_files, read_file |
| P14 | `84.8s` | `2.1%` | `model` | - | - | vision |
| P15 | `31.5s` | `13.8%` | `tool+model` | text_document_analysis -> filesystem | - | read_file chunks |
| P16 | `0.5s` | `36.3%` | `local` | codebase_inspection -> filesystem | - | list_files, read_file |
| P17 | `0.5s` | `17.0%` | `local` | text_document_analysis -> filesystem | - | stat_path |
| P18 | `29.8s` | `5.4%` | `model` | - | - | vision |
| P19 | `107.5s` | `2.1%` | `model` | - | - | vision |
| P20 | `70.0s` | `2.6%` | `model` | - | - | audio |
| P21 | `65.8s` | `2.5%` | `model` | - | - | audio |
| P22 | `34.8s` | `1.4%` | `model` | chitchat | binary_analysis -> NO | - |
| P23 | `3.3s` | `1.5%` | `model` | chitchat | - | - |
| P24 | `29.9s` | `1.6%` | `model` | chitchat | binary_analysis -> NO | - |
| P25 | `7.4s` | `1.4%` | `model` | chitchat | ambiguous -> NO | - |
| P26 | `32.2s` | `1.5%` | `model` | chitchat | - | - |
| P27 | `27.6s` | `11.1%` | `tool+model` | current_factual_lookup -> web | - | search_web |
| P28 | `43.0s` | `1.4%` | `model` | general_knowledge | - | - |
| P29 | `24.4s` | `1.4%` | `model` | general_knowledge | - | - |
| P30 | `32.0s` | `1.4%` | `model` | general_knowledge | - | - |
| P31 | `7.2s` | `1.5%` | `model` | chitchat | ambiguous -> NO | - |
| P32 | `1.5s` | `17.1%` | `local` | text_document_analysis -> filesystem, shell | - | rg |
