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
STATE_DIR="${ORBIT_STATE_DIR:-$HOME/.orbit}"
PID_FILE="${PID_FILE:-$STATE_DIR/gemma4-12b-server.pid}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/gemma4-12b-server.log}"
MULTIMODAL=0

usage() {
  cat <<'EOF'
usage: scripts/gemma4-12b-server.sh start [--multimodal]
       scripts/gemma4-12b-server.sh stop
       scripts/gemma4-12b-server.sh status

Starts/stops llama-server for the tuned gemma4:12b Orbit profile.

Prerequisites:
  llama-server must be available in PATH
  ollama must be available in PATH to pull/resolve gemma4:12b
  gemma4:12b must be present locally or pullable with ollama

start       run llama-server in background and return the terminal
stop        stop the background server started by this script
status      show whether the configured endpoint is healthy

Environment overrides:
  HOST PORT BASE_URL CTX_SIZE THREADS BATCH_SIZE UBATCH_SIZE CACHE_RAM
  PARALLEL_SLOTS LLAMA_SERVER_TOOLS ORBIT_STATE_DIR PID_FILE LOG_FILE

Common recovery:
  llama-server not found        install/build llama.cpp and add llama-server to PATH
  ollama not found              install Ollama or run ollama pull gemma4:12b elsewhere
  blob/manifest not found       run: ollama pull gemma4:12b
  existing non-multimodal server stop it before start --multimodal
  server without pid file       stop the owning process manually or change PORT/BASE_URL
EOF
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
  if ! command -v llama-server >/dev/null 2>&1; then
    echo "error: llama-server not found in PATH" >&2
    echo "install/build llama.cpp and ensure llama-server is available before starting this profile" >&2
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

  echo "starting llama-server for $MODEL_ALIAS at $BASE_URL"
  echo "log: $LOG_FILE"
  if command -v setsid >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    setsid llama-server $SERVER_ARGS >"$LOG_FILE" 2>&1 < /dev/null &
  else
    # shellcheck disable=SC2086
    nohup llama-server $SERVER_ARGS >"$LOG_FILE" 2>&1 < /dev/null &
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
