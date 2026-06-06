#!/usr/bin/env sh
set -eu

ORBIT="${ORBIT:-.venv/bin/orbit}"
MODEL="${MODEL:-gemma4:12b}"
HOME_DIR="${HOME_DIR:-$(mktemp -d)}"
WORKDIR="${WORKDIR:-$(mktemp -d)}"

mkdir -p "$WORKDIR/docs"
touch "$WORKDIR/alpha.txt" "$WORKDIR/beta.md"
printf 'Orbit reads UTF-8 text files correctly.\n' > "$WORKDIR/note.txt"
python3 - <<'PY' > "$WORKDIR/large.txt"
print("START large file marker.")
print("A" * 270000)
print("END large file marker.")
PY
printf '%%PDF-1.7\n' > "$WORKDIR/report.pdf"
cat > "$WORKDIR/web.html" <<'EOF'
<html>
  <head><title>Orbit web smoke</title><script>ignored()</script></head>
  <body><h1>Readable page</h1><p>Orbit fetches explicit URLs and extracts bounded readable text.</p></body>
</html>
EOF

printf 'home: %s\n' "$HOME_DIR"
printf 'workdir: %s\n' "$WORKDIR"

WEB_PORT="$(python3 - <<'PY'
import socket
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
)"
python3 -m http.server "$WEB_PORT" --bind 127.0.0.1 --directory "$WORKDIR" >/dev/null 2>&1 &
WEB_PID="$!"
trap 'kill "$WEB_PID" 2>/dev/null || true' EXIT

printf '\n## operational prompt should use list_files\n'
printf 'list files in this directory\n/exit\n' \
  | env HOME="$HOME_DIR" "$ORBIT" \
      --workdir "$WORKDIR" \
      --model "$MODEL" \
      --max-tokens 96

printf '\n## conceptual prompt should not need tools\n'
printf 'tell me what a filesystem is in one short sentence\n/exit\n' \
  | env HOME="$HOME_DIR" "$ORBIT" \
      --workdir "$WORKDIR" \
      --model "$MODEL" \
      --max-tokens 64

printf '\n## read_file should read UTF-8 text\n'
printf 'read note.txt and summarize it in one short sentence\n/exit\n' \
  | env HOME="$HOME_DIR" "$ORBIT" \
      --workdir "$WORKDIR" \
      --model "$MODEL" \
      --max-tokens 96

printf '\n## stat_path should inspect metadata without shell commands\n'
printf 'what is the size and modified time of note.txt?\n/exit\n' \
  | env HOME="$HOME_DIR" "$ORBIT" \
      --workdir "$WORKDIR" \
      --model "$MODEL" \
      --max-tokens 96

printf '\n## fetch_url should fetch explicit URLs and extract readable text\n'
printf 'summarize this URL in one short sentence: http://127.0.0.1:%s/web.html\n/exit\n' "$WEB_PORT" \
  | env HOME="$HOME_DIR" "$ORBIT" \
      --workdir "$WORKDIR" \
      --model "$MODEL" \
      --max-tokens 96

printf '\n## search_web should return bounded structured results\n'
printf 'search online for Dante Alighieri and return two result titles\n/exit\n' \
  | env HOME="$HOME_DIR" "$ORBIT" \
      --workdir "$WORKDIR" \
      --model "$MODEL" \
      --max-tokens 96

printf '\n## read_file must not read PDF\n'
printf 'use available tools to read report.pdf and tell me what it contains\n/exit\n' \
  | env HOME="$HOME_DIR" "$ORBIT" \
      --workdir "$WORKDIR" \
      --model "$MODEL" \
      --max-tokens 96

printf '\n## read_file chunk mode should handle oversized UTF-8 text\n'
printf 'read large.txt and tell me the first marker you see; use chunks if the file is too large\n/exit\n' \
  | env HOME="$HOME_DIR" "$ORBIT" \
      --workdir "$WORKDIR" \
      --model "$MODEL" \
      --max-tokens 96
