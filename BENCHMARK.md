# BENCHMARK.md

Manual benchmark prompts for Orbit on a warm `llama-server`.

This benchmark set is the regression suite for the current runtime behavior.
The software performance tuning line is closed: future changes to routing,
tool selection, final-answer policy, prompt payloads, tool payloads, or cache
behavior should be accepted only with strong measured evidence and no
functional regression.

Start the default server first:

```bash
scripts/gemma4-12b-server.sh start --multimodal
```

Then run:

```bash
scripts/bench-core.sh
```

The benchmark covers:

- chat without tools
- directory listing
- small local file read
- long local text summary
- local text search
- bounded URL fetch

Use the footer to compare:

```text
tks: prompt->completion, cached N | pf .../s | gen .../s | stop: ... | time: ...
```

When `llama-server` built-in tools are enabled, tool events should show the source:

```text
read_file {"path":"sample.txt"}
 └ read_file 54 chars | src: llama-server
```

If a built-in tool is unavailable or fails and Orbit has a compatible fallback, the source should be:

```text
 └ read_file 6000 chars | src: orbit
```
