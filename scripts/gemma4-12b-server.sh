#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

BASE_URL="${BASE_URL:-http://127.0.0.1:18080}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
MTP_LLAMA_SERVER_BIN="${MTP_LLAMA_SERVER_BIN:-$HOME/LAB/llama.cpp-gemma4-mtp-qualcomm/build/bin/llama-server}"
MODEL_ALIAS="${MODEL_ALIAS:-gemma4:12b-it}"
MODEL_PATH="${MODEL_PATH:-}"
MMPROJ_PATH="${MMPROJ_PATH:-}"
MTP_DRAFT_PATH="${MTP_DRAFT_PATH:-$HOME/LAB/models/gemma-4-12B-it-MTP-Q8_0.gguf}"
CTX_SIZE="${CTX_SIZE:-8192}"
THREADS="${THREADS:-6}"
BATCH_SIZE="${BATCH_SIZE:-256}"
UBATCH_SIZE="${UBATCH_SIZE:-128}"
CACHE_RAM="${CACHE_RAM:-8192}"
PARALLEL_SLOTS="${PARALLEL_SLOTS:-1}"
LLAMA_SERVER_TOOLS="${LLAMA_SERVER_TOOLS:-read_file,write_file,file_glob_search,grep_search,get_datetime,exec_shell_command,edit_file,apply_diff}"
STATE_DIR="${ORBIT_STATE_DIR:-$HOME/.orbit}"
PID_FILE="${PID_FILE:-$STATE_DIR/gemma4-12b-server.pid}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/gemma4-12b-server.log}"
MULTIMODAL=0
MTP=0

usage() {
  cat <<'EOF'
usage: scripts/gemma4-12b-server.sh start [--multimodal] [--mtp]
       scripts/gemma4-12b-server.sh stop
       scripts/gemma4-12b-server.sh status

Starts/stops llama-server for the tuned Gemma 4 12B instruction-tuned Orbit profile.

Prerequisites:
  llama-server must be available in PATH
  gemma-4-12B-it-Q4_K_M.gguf must be available locally
  optional multimodal support requires mmproj-gemma-4-12B-it-Q8_0.gguf
  optional MTP support requires a compatible llama.cpp build and gemma-4-12B-it-MTP-Q8_0.gguf

start       run llama-server in background and return the terminal
stop        stop the background server started by this script
status      show whether the configured endpoint is healthy

Environment overrides:
  LLAMA_SERVER_BIN MTP_LLAMA_SERVER_BIN MODEL_PATH MMPROJ_PATH MTP_DRAFT_PATH HOST PORT
  BASE_URL CTX_SIZE THREADS BATCH_SIZE UBATCH_SIZE CACHE_RAM PARALLEL_SLOTS
  LLAMA_SERVER_TOOLS ORBIT_STATE_DIR PID_FILE LOG_FILE

Common recovery:
  llama-server not found        install/build llama.cpp and add llama-server to PATH
  model not found               set MODEL_PATH=/path/to/gemma-4-12B-it-Q4_K_M.gguf
  projector not found           set MMPROJ_PATH=/path/to/mmproj-gemma-4-12B-it-Q8_0.gguf
  MTP draft not found           set MTP_DRAFT_PATH=/path/to/gemma-4-12B-it-MTP-Q8_0.gguf
  MTP unsupported               set LLAMA_SERVER_BIN=/path/to/compatible/llama-server
  existing non-multimodal server stop it before start --multimodal
  server without pid file       stop the owning process manually or change PORT/BASE_URL
EOF
}

select_llama_server_bin() {
  if [ "$MTP" -eq 1 ] && [ "$LLAMA_SERVER_BIN" = "llama-server" ] && [ -x "$MTP_LLAMA_SERVER_BIN" ]; then
    LLAMA_SERVER_BIN="$MTP_LLAMA_SERVER_BIN"
  fi
}

first_existing_file() {
  for path in "$@"; do
    if [ -f "$path" ]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

find_huggingface_file() {
  pattern="$1"
  for path in $pattern; do
    if [ -f "$path" ]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

resolve_model_path() {
  if [ "$MODEL_PATH" != "" ]; then
    if [ -f "$MODEL_PATH" ]; then
      printf '%s\n' "$MODEL_PATH"
      return 0
    fi
    echo "error: MODEL_PATH does not exist: $MODEL_PATH" >&2
    return 1
  fi
  first_existing_file \
    "$HOME/LAB/models/gemma4-12b/gemma4-12B-it-Q4_K_M.gguf" \
    "$HOME/LAB/models/gemma-4-12B-it-Q4_K_M.gguf" \
    "$HOME/.cache/huggingface/hub/models--ggml-org--gemma-4-12B-it-GGUF/snapshots/44ee90c4b61e888ac5b318a54ec7a94df61e9cd7/gemma-4-12B-it-Q4_K_M.gguf" \
    || find_huggingface_file "$HOME/.cache/huggingface/hub/models--ggml-org--gemma-4-12B-it-GGUF/snapshots/*/gemma-4-12B-it-Q4_K_M.gguf" \
    || {
      echo "error: gemma-4-12B-it-Q4_K_M.gguf not found" >&2
      echo "recovery: set MODEL_PATH=/path/to/gemma-4-12B-it-Q4_K_M.gguf" >&2
      return 1
    }
}

resolve_mmproj_path() {
  if [ "$MMPROJ_PATH" != "" ]; then
    if [ -f "$MMPROJ_PATH" ]; then
      printf '%s\n' "$MMPROJ_PATH"
      return 0
    fi
    echo "error: MMPROJ_PATH does not exist: $MMPROJ_PATH" >&2
    return 1
  fi
  first_existing_file \
    "$HOME/.cache/huggingface/hub/models--ggml-org--gemma-4-12B-it-GGUF/snapshots/44ee90c4b61e888ac5b318a54ec7a94df61e9cd7/mmproj-gemma-4-12B-it-Q8_0.gguf" \
    "$HOME/LAB/models/gemma4-12b/gemma4-12B-it-mmproj-BF16.gguf" \
    || find_huggingface_file "$HOME/.cache/huggingface/hub/models--ggml-org--gemma-4-12B-it-GGUF/snapshots/*/mmproj-gemma-4-12B-it-Q8_0.gguf" \
    || {
      echo "error: multimodal projector not found" >&2
      echo "recovery: set MMPROJ_PATH=/path/to/mmproj-gemma-4-12B-it-Q8_0.gguf" >&2
      return 1
    }
}

resolve_mtp_draft_path() {
  if [ "$MTP_DRAFT_PATH" != "" ]; then
    if [ -f "$MTP_DRAFT_PATH" ]; then
      printf '%s\n' "$MTP_DRAFT_PATH"
      return 0
    fi
    echo "error: MTP_DRAFT_PATH does not exist: $MTP_DRAFT_PATH" >&2
    return 1
  fi
  first_existing_file \
    "$HOME/LAB/models/gemma-4-12B-it-MTP-Q8_0.gguf" \
    "$HOME/LAB/models/gemma4-12b/gemma-4-12B-it-MTP-Q8_0.gguf" \
    || find_huggingface_file "$HOME/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/snapshots/*/gemma-4-12B-it-MTP-Q8_0.gguf" \
    || {
      echo "error: gemma-4-12B-it-MTP-Q8_0.gguf not found" >&2
      echo "recovery: set MTP_DRAFT_PATH=/path/to/gemma-4-12B-it-MTP-Q8_0.gguf" >&2
      return 1
    }
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
  echo "log: $LOG_FILE" >&2
  return 1
}

pid_running() {
  [ -f "$PID_FILE" ] || return 1
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [ "$pid" != "" ] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_server() {
  select_llama_server_bin
  if ! command -v "$LLAMA_SERVER_BIN" >/dev/null 2>&1; then
    echo "error: llama-server not found: $LLAMA_SERVER_BIN" >&2
    echo "install/build llama.cpp and ensure llama-server is available before starting this profile" >&2
    echo "recovery: set LLAMA_SERVER_BIN=/path/to/llama-server" >&2
    exit 1
  fi
  if health_ok; then
    if [ "$MULTIMODAL" -eq 1 ] && ! multimodal_ok; then
      echo "error: existing llama-server at $BASE_URL is not multimodal" >&2
      echo "recovery: stop the existing server, then run: $0 start --multimodal" >&2
      exit 1
    fi
    echo "llama-server already running at $BASE_URL"
    echo "run Orbit with: orbit --base-url $BASE_URL"
    return 0
  fi

  mkdir -p "$STATE_DIR"
  MODEL_BLOB="$(resolve_model_path)"

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
    MMPROJ_BLOB="$(resolve_mmproj_path)"
    SERVER_ARGS="$SERVER_ARGS
--mmproj
$MMPROJ_BLOB"
  fi

  if [ "$MTP" -eq 1 ]; then
    MTP_DRAFT_BLOB="$(resolve_mtp_draft_path)"
    SERVER_ARGS="$SERVER_ARGS
--spec-type
draft-mtp
--model-draft
$MTP_DRAFT_BLOB"
  fi

  echo "starting llama-server for $MODEL_ALIAS at $BASE_URL"
  echo "log: $LOG_FILE"
  if command -v setsid >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    setsid "$LLAMA_SERVER_BIN" $SERVER_ARGS >"$LOG_FILE" 2>&1 < /dev/null &
  else
    # shellcheck disable=SC2086
    nohup "$LLAMA_SERVER_BIN" $SERVER_ARGS >"$LOG_FILE" 2>&1 < /dev/null &
  fi
  echo "$!" >"$PID_FILE"
  wait_for_server
  echo "ready"
  echo "run Orbit with: orbit --base-url $BASE_URL"
}

stop_server() {
  if pid_running; then
    pid="$(cat "$PID_FILE")"
    echo "stopping llama-server pid $pid"
    kill "$pid" 2>/dev/null || true
    i=0
    while [ "$i" -lt 20 ]; do
      if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "stopped"
        return 0
      fi
      i=$((i + 1))
      sleep 0.5
    done
    echo "forcing llama-server pid $pid"
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "stopped"
    return 0
  fi

  rm -f "$PID_FILE"
  if health_ok; then
    echo "llama-server is running at $BASE_URL, but not from this script pid file"
    echo "stop it manually or start this script after freeing port $PORT"
    return 1
  fi
  echo "llama-server is not running at $BASE_URL"
}

status_server() {
  if health_ok; then
    echo "llama-server: ok at $BASE_URL"
    if pid_running; then
      echo "pid: $(cat "$PID_FILE")"
    else
      echo "pid: unknown"
    fi
    echo "log: $LOG_FILE"
    return 0
  fi
  echo "llama-server: unavailable at $BASE_URL"
  return 1
}

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  usage
  exit 0
fi

COMMAND="$1"
shift

while [ "$#" -gt 0 ]; do
  case "$1" in
    --multimodal)
      MULTIMODAL=1
      shift
      ;;
    --mtp)
      MTP=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$COMMAND" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  status)
    status_server
    ;;
  --help|-h|help)
    usage
    ;;
  *)
    echo "error: unknown command: $COMMAND" >&2
    usage >&2
    exit 2
    ;;
esac
