# Orbit

Orbit is a small Python-first local runtime for Gemma 4 26B-A4B, verified Qwen 3.6 35B-A3B, and verified Qwen3-Coder 30B-A3B profiles on CPU-only machines. The primary path is the native `orbit server` backend, using vendored llama.cpp/ggml libraries built and loaded by Orbit. It does not require an external `llama-server` process for normal use.

Orbit stays model-driven. The runtime enforces safety, size, timeout, context, and tool-contract boundaries, but the model decides whether to answer directly or use exposed tools.

Linux is the main target environment. macOS may work. Windows is not a target.

## Supported Models

| Model | Prefill | Generation | Tools-on chat | Tool + final | Peak RAM |
|---|---:|---:|---:|---:|---:|
| [Gemma 4 26B-A4B](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) | ~28.5 tok/s | ~7.8 tok/s | ~31.7 s | ~22.1 s | ~29.4 GiB |
| [Qwen 3.6 35B-A3B](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF) | ~30.5 tok/s | ~7.4 tok/s | ~25.1 s | ~30.5 s | ~36.4 GiB |
| [Qwen3-Coder 30B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | ~26.6 tok/s | ~7.8 tok/s | ~4.4 s | ~30.1 s | ~31.3 GiB |

Benchmarks are representative CPU-only measurements and vary by hardware, quantization, cache state, prompt, and workload.

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
