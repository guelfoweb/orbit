# BENCHMARK

## Minimal Regression Benchmark

Use this after runtime, routing, compaction, or guardrail changes:

```bash
python3 scripts/run_prompt_benchmark.py --model gemma4:e2b-fast-t6-c8k
python3 scripts/run_prompt_benchmark.py --model gemma4:e2b-fast-t6-c8k --include-heavy
```

For one-off latency diagnosis without changing normal output:

```bash
PYTHONPATH=src python3 -m orbit --workdir workdir --model gemma4:e2b-fast-t6-c8k --debug-timing "list all files and directories in the current workspace"
```

`--debug-timing` prints bounded route, tool, and model timings to stderr. It is off by default.

## Method

- Use one isolated session per prompt.
- Run prompts sequentially only.
- Treat the first prompt after a cold Ollama restart as a cold-start outlier.
- Use unit tests for correctness coverage and this benchmark for behavior and latency regression checks.
- Keep the target model `gemma4:e2b-fast-t6-c8k` unless explicitly comparing model profiles.

## Current Baseline Target

Target machine: CPU-only NUC-class system with 6 cores / 12 threads and 64 GB RAM.

- Fast bounded prompts should complete in about 1s each.
- Heavy prompts may take tens of seconds, but must complete without tool errors or timeouts.
- Review prompts must not execute `read_file` with a missing path.
- Long text prompts must stay bounded and avoid context exhaustion.
- Vision prompts must complete without Ollama runner crashes.
- Audio prompts may take about one model call per 5 second chunk plus one synthesis call; they must complete without raw long-audio runner crashes.
- Local plus web prompts must not claim local evidence unless `read_file` actually ran.

## Current Model Profile

Recommended local profile:

```bash
ollama create gemma4:e2b-fast-t6-c8k -f Modelfile.gemma4-e2b-fast-t6-c8k
```

Model file parameters:

```text
FROM gemma4:e2b
PARAMETER temperature 0
PARAMETER num_ctx 8192
PARAMETER num_thread 6
PARAMETER num_batch 96
```

Ollama server performance settings should be applied to the server process, not only to the `orbit` client process:

```bash
OLLAMA_NUM_PARALLEL=1
OLLAMA_KEEP_ALIVE=-1
OLLAMA_MAX_LOADED_MODELS=1
```

If Ollama runs through systemd, put these values in an `ollama.service` drop-in and restart the service.

## Prompt Set

Fast prompts:

1. `hi, who are you?`
2. `list all files and directories in the current workspace`
3. `decode this string "Y2lhbw==" from base64`
4. `what is the size and modified time of agent.py?`
5. `tell me how many files exist in the workspace and what the newest file is.`

Heavy prompts:

1. `review agent.py for vulnerabilities and security issues`
2. `analyze the text promessi_sposi.txt and summarize it in 5 lines`
3. `compare two images: images/vision-test-1.png and images/vision-test-2.jpg and tell me the differences`
4. `transcribe audio/voice-sample-16k-mono.wav`
5. `summarize audio/voice-sample.wav in one sentence`

## Latest Observed Run

Recent runs with `gemma4:e2b-fast-t6-c8k` showed:

- Fast bounded prompts: about `0.5s` for local deterministic/tool-backed paths.
- Simple chat: about `2-4s` depending on warm state.
- Code review prompt: about `70-75s`.
- Long text summary prompt: about `23-24s`.
- Vision comparison prompt: about `90-115s` when using per-image inspection plus synthesis on CPU-only hardware.
- Audio summary/transcription over a 26s file: about `70-90s` with 5s chunks.
- Generation speed: typically about `13-15 tk/s` on CPU-only hardware.

Known limitation:

- Code review on large files is still semantically weak compared with frontier models. It is useful for bounded local triage, not deep security assurance.
