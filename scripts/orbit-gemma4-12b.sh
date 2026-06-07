#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

# shellcheck source=lib/ollama-gemma4-12b.sh
. "$SCRIPT_DIR/lib/ollama-gemma4-12b.sh"

BASE_URL="${BASE_URL:-http://127.0.0.1:18080}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
MODEL_ALIAS="${MODEL_ALIAS:-gemma4:12b}"
CTX_SIZE="${CTX_SIZE:-8192}"
THREADS="${THREADS:-6}"
BATCH_SIZE="${BATCH_SIZE:-128}"
UBATCH_SIZE="${UBATCH_SIZE:-128}"
CACHE_RAM="${CACHE_RAM:-8192}"
PARALLEL_SLOTS="${PARALLEL_SLOTS:-1}"
LLAMA_SERVER_TOOLS="${LLAMA_SERVER_TOOLS:-read_file,write_file,file_glob_search,grep_search,get_datetime,exec_shell_command,edit_file,apply_diff}"
ORBIT_BIN="${ORBIT_BIN:-orbit}"
MULTIMODAL=0

SERVER_PID=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --multimodal)
      MULTIMODAL=1
      shift
      ;;
    --help-script)
      cat <<'EOF'
usage: scripts/orbit-gemma4-12b.sh [--multimodal] [orbit args...]

Starts llama-server with the local gemma4:12b GGUF downloaded by Ollama,
then opens Orbit.

Options consumed by this script:
  --multimodal  also load the Ollama projector blob for image/audio input
  --help-script show this script help

All other arguments are passed to orbit.
EOF
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

cleanup() {
  if [ "$SERVER_PID" != "" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

stop_on_signal() {
  cleanup
  exit 130
}

health_ok() {
  python3 - "$BASE_URL" <<'PY'
from __future__ import annotations

import json
import sys
from urllib.error import URLError
from urllib.request import urlopen

base_url = sys.argv[1].rstrip("/")
try:
    with urlopen(base_url + "/health", timeout=1.5) as response:
        data = json.loads(response.read().decode("utf-8"))
except (OSError, URLError, TimeoutError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if data.get("status") == "ok" else 1)
PY
}

multimodal_ok() {
  python3 - "$BASE_URL" <<'PY'
from __future__ import annotations

import json
import sys
from urllib.error import URLError
from urllib.request import urlopen

base_url = sys.argv[1].rstrip("/")
try:
    with urlopen(base_url + "/props", timeout=1.5) as response:
        data = json.loads(response.read().decode("utf-8"))
except (OSError, URLError, TimeoutError, json.JSONDecodeError):
    raise SystemExit(1)
modalities = data.get("modalities")
if not isinstance(modalities, dict):
    raise SystemExit(1)
raise SystemExit(0 if modalities.get("vision") or modalities.get("audio") else 1)
PY
}

wait_for_server() {
  i=0
  while [ "$i" -lt 120 ]; do
    if health_ok; then
      return 0
    fi
    i=$((i + 1))
    sleep 1
  done
  echo "error: llama-server did not become ready at $BASE_URL" >&2
  return 1
}

if ! command -v llama-server >/dev/null 2>&1; then
  echo "error: llama-server not found in PATH" >&2
  exit 1
fi

if ! command -v "$ORBIT_BIN" >/dev/null 2>&1; then
  if [ -x "$ROOT_DIR/.venv/bin/orbit" ]; then
    ORBIT_BIN="$ROOT_DIR/.venv/bin/orbit"
  else
    echo "error: orbit not found in PATH. Install it with: pip install -e ." >&2
    exit 1
  fi
fi

if health_ok; then
  if [ "$MULTIMODAL" -eq 1 ] && ! multimodal_ok; then
    echo "error: existing llama-server at $BASE_URL is not multimodal; stop it and rerun with --multimodal" >&2
    exit 1
  fi
  echo "using existing llama-server at $BASE_URL" >&2
else
  ensure_ollama_model
  MODEL_BLOB="$(model_blob_from_manifest)"
  SERVER_ARGS="
-m
$MODEL_BLOB
-c
$CTX_SIZE
-t
$THREADS
-b
$BATCH_SIZE
-ub
$UBATCH_SIZE
-np
$PARALLEL_SLOTS
--reasoning
off
--cache-ram
$CACHE_RAM
--alias
$MODEL_ALIAS
--host
$HOST
--port
$PORT
--tools
$LLAMA_SERVER_TOOLS"
  if [ "$MULTIMODAL" -eq 1 ]; then
    MMPROJ_BLOB="$(projector_blob_from_manifest)"
    SERVER_ARGS="$SERVER_ARGS
--mmproj
$MMPROJ_BLOB"
  fi
  echo "starting llama-server for $MODEL_ALIAS at $BASE_URL" >&2
  # shellcheck disable=SC2086
  llama-server $SERVER_ARGS >/tmp/orbit-llama-server.log 2>&1 &
  SERVER_PID="$!"
  trap cleanup EXIT
  trap stop_on_signal INT TERM
  wait_for_server
fi

"$ORBIT_BIN" --base-url "$BASE_URL" --model "$MODEL_ALIAS" "$@"
