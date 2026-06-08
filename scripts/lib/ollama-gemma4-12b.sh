#!/usr/bin/env sh
set -eu

OLLAMA_MODEL="gemma4:12b"
OLLAMA_MANIFEST_REL="manifests/registry.ollama.ai/library/gemma4/12b"

find_ollama_models_dir() {
  for dir in \
    "${OLLAMA_MODELS:-}" \
    "$HOME/.ollama/models" \
    "/usr/share/ollama/.ollama/models" \
    "/var/lib/ollama/.ollama/models"
  do
    if [ "$dir" != "" ] && [ -d "$dir" ]; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  return 1
}

ensure_ollama_model() {
  if ! command -v ollama >/dev/null 2>&1; then
    echo "error: ollama not found in PATH. Install Ollama, then run: ollama pull $OLLAMA_MODEL" >&2
    echo "recovery: start the server helper again after Ollama is installed and $OLLAMA_MODEL is available" >&2
    return 1
  fi

  models_dir="$(find_ollama_models_dir 2>/dev/null || true)"
  if [ "$models_dir" != "" ] && [ -r "$models_dir/$OLLAMA_MANIFEST_REL" ]; then
    return 0
  fi

  echo "pulling $OLLAMA_MODEL with ollama..." >&2
  ollama pull "$OLLAMA_MODEL"
}

blob_from_manifest() {
  media_type="$1"
  models_dir="$(find_ollama_models_dir)"
  manifest="$models_dir/$OLLAMA_MANIFEST_REL"
  if [ ! -r "$manifest" ]; then
    echo "error: Ollama manifest not found after pull: $manifest" >&2
    echo "recovery: run 'ollama pull $OLLAMA_MODEL' and retry" >&2
    return 1
  fi

  python3 - "$models_dir" "$manifest" "$media_type" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

models_dir = Path(sys.argv[1])
manifest = Path(sys.argv[2])
media_type = sys.argv[3]
data = json.loads(manifest.read_text(encoding="utf-8"))
for layer in data.get("layers", []):
    if layer.get("mediaType") == media_type:
        digest = str(layer.get("digest", ""))
        if digest.startswith("sha256:"):
            path = models_dir / "blobs" / ("sha256-" + digest.removeprefix("sha256:"))
            if path.is_file():
                print(path)
                raise SystemExit(0)
raise SystemExit(f"error: blob not found in Ollama manifest for {media_type}")
PY
}

model_blob_from_manifest() {
  blob_from_manifest "application/vnd.ollama.image.model"
}

projector_blob_from_manifest() {
  blob_from_manifest "application/vnd.ollama.image.projector"
}
