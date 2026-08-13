# Orbit

Orbit is a small Python-first local runtime for Gemma 4 26B-A4B, verified Qwen 3.6 35B-A3B, and verified Qwen3-Coder 30B-A3B profiles on CPU-only machines. The primary path is the native `orbit server` backend, using vendored llama.cpp/ggml libraries built and loaded by Orbit. It does not require an external `llama-server` process for normal use.

Orbit stays model-driven. The runtime enforces safety, size, timeout, context, and tool-contract boundaries, but the model decides whether to answer directly or use exposed tools.

Linux is the main target environment. macOS may work. Windows is not a target.

## Supported Models

| Model | Prefill | Generation | Tools-on chat | Tool + final | Peak RAM |
|---|---:|---:|---:|---:|---:|
| [Gemma 4 26B-A4B](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) | ~24.2 tok/s | ~7.1 tok/s | ~3.0 s | ~20.9 s | ~29.4 GiB |
| [Qwen 3.6 35B-A3B](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF) | ~31.8 tok/s | ~7.6 tok/s | ~6.1 s | ~23.8 s | ~36.4 GiB |
| [Qwen3-Coder 30B-A3B](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | ~26.0 tok/s | ~10.7 tok/s | ~3.0 s | ~18.4 s | ~31.3 GiB |

NUC10 Intel Core i7-10710U (6 cores / 12 threads) with 64 GB RAM, no GPU
Linux, llama.cpp b9551, ctx=8192, 6 threads, batch/ubatch 256/128, Flash Attention AUTO, thinking off, and MTP off.

Chat and tool latencies are warm steady-state medians after one excluded warm-up. Prefill measures evaluated tokens only and excludes cache benefit. These are not universal performance claims; actual results vary with CPU, memory bandwidth, quantization, backend build, and workload.

`orbit server --low-memory` is an opt-in mode supported only for the verified Qwen3-Coder 30B-A3B profile. Default behavior is unchanged and CPU repacking remains enabled unless this flag is specified. On the documented NUC10, peak RSS was approximately 31.3 GiB by default and 18.3 GiB in low-memory mode, a 41.6% reduction, with weighted prefill approximately 13.9% slower and decode approximately 1% slower. 24 GB RAM is the recommended practical minimum for a complete host; 20 GiB was qualified only as a process-memory limit and is not a general host recommendation.

## Requirements

- Python 3.11 or newer
- Linux recommended
- CMake for building the vendored native libraries

## Install

```bash
git clone https://github.com/guelfoweb/orbit.git
cd orbit
sudo apt install cmake
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Build the vendored native libraries if they are not already present:

```bash
python3 scripts/build_native.py
```

Inspect the host and review the recommended server configuration:

```bash
scripts/suggest-server-profile.sh
```

## Quick Start

Start the native server and select an available verified model:

```bash
orbit server
```

In another terminal:

```bash
orbit
```

For a one-shot request:

```bash
orbit --workdir workdir --think off "hi, how are you?"
```
