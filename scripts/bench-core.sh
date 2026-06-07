#!/usr/bin/env sh
set -eu

BASE_URL="${BASE_URL:-http://127.0.0.1:18080}"
MODEL="${MODEL:-gemma4:12b}"
WORKDIR="${WORKDIR:-workdir}"
ORBIT_BIN="${ORBIT_BIN:-.venv/bin/orbit}"
TIMEOUT="${TIMEOUT:-600}"
MAX_TOKENS="${MAX_TOKENS:-512}"

if ! command -v "$ORBIT_BIN" >/dev/null 2>&1 && [ ! -x "$ORBIT_BIN" ]; then
  echo "error: orbit binary not found: $ORBIT_BIN" >&2
  exit 1
fi

run_prompt() {
  label="$1"
  prompt="$2"
  home_dir="$(mktemp -d)"
  echo
  echo "## $label"
  echo "$prompt"
  HOME="$home_dir" "$ORBIT_BIN" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --workdir "$WORKDIR" \
    --timeout "$TIMEOUT" \
    --max-tokens "$MAX_TOKENS" \
    "$prompt"
  rm -rf "$home_dir"
}

run_prompt "chat" "hi, who are you? Answer in one short sentence."
run_prompt "list files" "list files and directories in this workdir"
run_prompt "small read" "read sample.txt and summarize it in one sentence"
run_prompt "long read" "read text/divina_commedia_inferno_canto1.txt and summarize it in Italian in 5 lines"
run_prompt "grep search" "search inside local text files for the word Virgilio and summarize the matches"
run_prompt "web url" "summarize this URL in one short paragraph: https://example.com"
