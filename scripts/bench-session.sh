#!/usr/bin/env sh
set -eu

ORBIT="${ORBIT:-.venv/bin/orbit}"
HOME_DIR="${HOME_DIR:-$(mktemp -d)}"
WORKDIR="${WORKDIR:-$(mktemp -d)}"

printf 'home: %s\n' "$HOME_DIR"
printf 'workdir: %s\n' "$WORKDIR"

printf '\n## first process writes session\n'
printf 'Remember this keyword: session-cache-check.\n/exit\n' \
  | /usr/bin/time -f 'wall: %e s' env HOME="$HOME_DIR" "$ORBIT" \
      --workdir "$WORKDIR" \
      --max-tokens 32

printf '\n## second process reads persisted session\n'
printf 'What keyword did I ask you to remember? Answer only the keyword.\n/exit\n' \
  | /usr/bin/time -f 'wall: %e s' env HOME="$HOME_DIR" "$ORBIT" \
      --workdir "$WORKDIR" \
      --max-tokens 32

printf '\n## reset clears persisted session\n'
printf '/reset\n/exit\n' \
  | env HOME="$HOME_DIR" "$ORBIT" \
      --workdir "$WORKDIR" \
      --max-tokens 16

remaining=$(find "$HOME_DIR/.orbit/sessions" -type f 2>/dev/null | wc -l)
printf 'remaining session files: %s\n' "$remaining"
