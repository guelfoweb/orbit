#!/usr/bin/env sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <context-size>" >&2
  exit 2
fi

CTX_SIZE="$1"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# shellcheck source=ollama-gemma4-12b.sh
. "$SCRIPT_DIR/ollama-gemma4-12b.sh"

if ! command -v llama-server >/dev/null 2>&1; then
  echo "error: llama-server not found in PATH" >&2
  exit 1
fi

ensure_ollama_model
MODEL_BLOB="$(model_blob_from_manifest)"

set -- \
  -m "$MODEL_BLOB" \
  -c "$CTX_SIZE" \
  -t "${THREADS:-6}" \
  -b "${BATCH_SIZE:-128}" \
  -ub "${UBATCH_SIZE:-128}" \
  -np "${PARALLEL_SLOTS:-1}" \
  --reasoning off \
  --cache-ram "${CACHE_RAM:-8192}" \
  --tools "${LLAMA_SERVER_TOOLS:-read_file,write_file,file_glob_search,grep_search,get_datetime,exec_shell_command,edit_file,apply_diff}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-18080}"

if [ "${CACHE_REUSE:-}" != "" ]; then
  set -- "$@" --cache-reuse "$CACHE_REUSE"
fi

exec llama-server "$@"
