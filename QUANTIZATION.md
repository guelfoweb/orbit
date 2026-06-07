# Quantization

This document describes the local conversion and quantization workflow used for
Gemma 4 models with `llama.cpp`.

The recommended path is:

```text
Hugging Face safetensors -> GGUF BF16 -> quantized GGUF
```

Do not quantize from an already quantized GGUF unless the goal is only a quick
experiment. Re-quantization can noticeably degrade quality.

## Prerequisites

- `llama.cpp` built locally
- Python virtual environment for conversion tools
- enough free disk space for the original model, BF16 GGUF, and quantized GGUF

Expected local paths used in this setup:

```bash
LLAMA_CPP=/home/guelfoweb/LAB/llama.cpp
MODELS=/home/guelfoweb/LAB/models/gemma4-12b
HF_MODEL=$MODELS/hf-it
```

## Download the original model

Create the local model directory:

```bash
mkdir -p "$HF_MODEL"
```

Download the small metadata and tokenizer files with the Hugging Face CLI:

```bash
/home/guelfoweb/LAB/models/.venv-hf/bin/hf download \
  google/gemma-4-12B-it \
  README.md \
  chat_template.jinja \
  config.json \
  generation_config.json \
  processor_config.json \
  tokenizer.json \
  tokenizer_config.json \
  --local-dir "$HF_MODEL"
```

Download the large safetensors file with resumable `curl`:

```bash
curl -L -C - --fail --retry 5 --retry-delay 5 --connect-timeout 30 \
  -o "$HF_MODEL/model.safetensors" \
  https://huggingface.co/google/gemma-4-12B-it/resolve/main/model.safetensors
```

`-C -` allows the download to resume if it is interrupted.

Optional progress check:

```bash
python3 - <<'PY'
from pathlib import Path

p = Path("/home/guelfoweb/LAB/models/gemma4-12b/hf-it/model.safetensors")
expected = 23919549408
size = p.stat().st_size if p.exists() else 0

print(f"{size / 1024**3:.2f} GiB / {expected / 1024**3:.2f} GiB ({size / expected * 100:.1f}%)")
PY
```

## Verify BF16 support

The local `llama.cpp` converter supports BF16 output:

```bash
grep -n '"bf16"' "$LLAMA_CPP/convert_hf_to_gguf.py"
```

Expected evidence:

```text
"bf16": gguf.LlamaFileType.MOSTLY_BF16
```

The Gemma 4 12B IT Hugging Face config declares BF16 weights:

```bash
grep -n '"dtype"' "$HF_MODEL/config.json"
```

Expected evidence:

```text
"dtype": "bfloat16"
```

The local converter also supports the Gemma 4 unified architecture:

```bash
grep -R "Gemma4UnifiedForConditionalGeneration" "$LLAMA_CPP/conversion"
```

## Verify the downloaded model

The original Hugging Face file is:

```bash
$HF_MODEL/model.safetensors
```

Verify the expected size:

```bash
stat -c '%s %n' "$HF_MODEL/model.safetensors"
```

Expected size:

```text
23919549408 bytes
```

## Install conversion requirements

Install the `llama.cpp` Hugging Face conversion dependencies into the conversion
virtual environment:

```bash
/home/guelfoweb/LAB/models/.venv-hf/bin/pip install \
  -r "$LLAMA_CPP/requirements/requirements-convert_hf_to_gguf.txt"
```

This installs PyTorch CPU wheels and may take time.

## Fix tokenizer config if needed

With the current `transformers` release used by the converter, Gemma 4 12B IT
may fail while loading the tokenizer with:

```text
AttributeError: 'list' object has no attribute 'keys'
```

This happens because `tokenizer_config.json` declares `extra_special_tokens` as
a list. Keep a copy of the original file and convert that field to a named map:

```bash
cp "$HF_MODEL/tokenizer_config.json" "$HF_MODEL/tokenizer_config.json.orig"

python3 - <<'PY'
import json
from pathlib import Path

p = Path("/home/guelfoweb/LAB/models/gemma4-12b/hf-it/tokenizer_config.json")
data = json.loads(p.read_text(encoding="utf-8"))

if data.get("extra_special_tokens") == ["<|video|>"]:
    data["extra_special_tokens"] = {"video_token": "<|video|>"}

p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
```

Verify that the tokenizer loads:

```bash
/home/guelfoweb/LAB/models/.venv-hf/bin/python - <<'PY'
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/home/guelfoweb/LAB/models/gemma4-12b/hf-it")
print(type(tok).__name__)
print("vocab", len(tok))
print("video_token", getattr(tok, "video_token", None))
PY
```

Expected output:

```text
GemmaTokenizerFast
vocab 262144
video_token <|video|>
```

## Convert to GGUF BF16

Convert from the original Hugging Face model directory:

```bash
/home/guelfoweb/LAB/models/.venv-hf/bin/python \
  "$LLAMA_CPP/convert_hf_to_gguf.py" \
  "$HF_MODEL" \
  --outfile "$MODELS/gemma4-12B-it-BF16.gguf" \
  --outtype bf16
```

If BF16 conversion fails due to a converter limitation, use F16 as fallback:

```bash
/home/guelfoweb/LAB/models/.venv-hf/bin/python \
  "$LLAMA_CPP/convert_hf_to_gguf.py" \
  "$HF_MODEL" \
  --outfile "$MODELS/gemma4-12B-it-F16.gguf" \
  --outtype f16
```

Prefer BF16 when possible because the source model declares `bfloat16`.

## Quantize

Recommended first baseline:

```bash
"$LLAMA_CPP/build-cpu/bin/llama-quantize" \
  "$MODELS/gemma4-12B-it-BF16.gguf" \
  "$MODELS/gemma4-12B-it-Q4_K_M.gguf" \
  Q4_K_M 6
```

Observed result on the Intel NUC 10 class CPU-only machine:

```text
BF16 GGUF size: 23G
Q4_K_M GGUF size: 6.9G
quantize time: about 301s
quantized BPW: 4.95
```

Optional smaller experiment:

```bash
"$LLAMA_CPP/build-cpu/bin/llama-quantize" \
  "$MODELS/gemma4-12B-it-BF16.gguf" \
  "$MODELS/gemma4-12B-it-Q2_K.gguf" \
  Q2_K 6
```

`llama.cpp` exposes `Q2_K`, not `Q2_K_M`.

## Start llama-server

Example Q4 command:

```bash
llama-server \
  -m "$MODELS/gemma4-12B-it-Q4_K_M.gguf" \
  -c 8192 \
  -t 6 \
  -b 128 \
  -ub 128 \
  -np 1 \
  --reasoning off \
  --cache-ram 8192 \
  --alias gemma4:12b-q4 \
  --host 127.0.0.1 \
  --port 18080
```

If another server is already bound to port `18080`, stop it first or use another
port consistently with Orbit.

## Test with Orbit

From the repository:

```bash
cd /home/guelfoweb/LAB/orbit
. .venv/bin/activate
orbit --model gemma4:12b-q4 --workdir .
```

Suggested smoke prompts:

```text
hi, who are you?
list all files in this workspace
read README.md and summarize it in three bullet points
```
