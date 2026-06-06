#!/usr/bin/env sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <context-size>" >&2
  exit 2
fi

CTX_SIZE="$1"
MODEL_BLOB="/usr/share/ollama/.ollama/models/blobs/sha256-1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606"
MMPROJ_BLOB="/usr/share/ollama/.ollama/models/blobs/sha256-675ad6e68101ca9413ec806855c452362f0213f2dfc5800996b086fdb8119842"

if ! command -v llama-server >/dev/null 2>&1; then
  echo "error: llama-server not found in PATH" >&2
  exit 1
fi

if [ ! -r "$MODEL_BLOB" ]; then
  echo "error: Gemma4 12B model blob not found: $MODEL_BLOB" >&2
  exit 1
fi

set -- \
  -m "$MODEL_BLOB" \
  -c "$CTX_SIZE" \
  -t "${THREADS:-6}" \
  -b "${BATCH_SIZE:-128}" \
  -ub "${UBATCH_SIZE:-128}" \
  -np "${PARALLEL_SLOTS:-1}" \
  --reasoning off \
  --cache-ram "${CACHE_RAM:-8192}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-18080}"

if [ -r "$MMPROJ_BLOB" ]; then
  set -- "$@" --mmproj "$MMPROJ_BLOB"
fi

if [ "${CACHE_REUSE:-}" != "" ]; then
  set -- "$@" --cache-reuse "$CACHE_REUSE"
fi

exec llama-server "$@"
