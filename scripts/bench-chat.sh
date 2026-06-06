#!/usr/bin/env sh
set -eu

ORBIT="${ORBIT:-.venv/bin/orbit}"
MODEL="${MODEL:-gemma4:12b}"
MAX_TOKENS="${MAX_TOKENS:-96}"

run_prompt() {
  name="$1"
  prompt="$2"
  printf '\n## %s\n' "$name"
  /usr/bin/time -f 'wall: %e s' "$ORBIT" --model "$MODEL" --max-tokens "$MAX_TOKENS" "$prompt"
}

run_prompt "short identity" "Say who you are in one short sentence."
run_prompt "prompt caching" "What is prompt caching? Answer in two short sentences."
run_prompt "cpu summary" "Summarize the tradeoff of running a 12B model on CPU-only hardware in three concise bullet points."

printf '\n## multi turn memory\n'
printf 'Remember this keyword: lighthouse-cache.\nWhat keyword did I ask you to remember? Answer only the keyword.\n/exit\n' \
  | /usr/bin/time -f 'wall: %e s' "$ORBIT" --model "$MODEL" --max-tokens 32

if [ "${CACHE_BENCH:-0}" = "1" ]; then
  tmp1="${TMPDIR:-/tmp}/orbit-cache-1.txt"
  tmp2="${TMPDIR:-/tmp}/orbit-cache-2.txt"
  python3 - <<'PY' > "$tmp1"
prefix = " ".join(["local cpu prompt cache benchmark"] * 220)
print(prefix + "\nQuestion: summarize the repeated context in one sentence.")
PY
  python3 - <<'PY' > "$tmp2"
prefix = " ".join(["local cpu prompt cache benchmark"] * 220)
print(prefix + "\nQuestion: extract the four repeated words only.")
PY
  printf '\n## long prefix cache first pass\n'
  /usr/bin/time -f 'wall: %e s' "$ORBIT" --model "$MODEL" --max-tokens 32 "$(cat "$tmp1")"
  printf '\n## long prefix cache second pass\n'
  /usr/bin/time -f 'wall: %e s' "$ORBIT" --model "$MODEL" --max-tokens 32 "$(cat "$tmp2")"
fi
