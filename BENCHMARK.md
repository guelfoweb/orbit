# BENCHMARK.md

Manual benchmark prompts for Orbit on a warm `llama-server`.

Start the default server first:

```bash
scripts/orbit-gemma4-12b.sh --multimodal
```

In another terminal, run:

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
