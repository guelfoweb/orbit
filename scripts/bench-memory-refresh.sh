#!/usr/bin/env sh
set -eu

ORBIT="${ORBIT:-.venv/bin/orbit}"
HOME_DIR="${HOME_DIR:-$(mktemp -d)}"
WORKDIR="${WORKDIR:-$(mktemp -d)}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-1600}"
MAX_TOKENS="${MAX_TOKENS:-96}"

printf 'home: %s\n' "$HOME_DIR"
printf 'workdir: %s\n' "$WORKDIR"
printf 'context_tokens override: %s\n' "$CONTEXT_TOKENS"

prompt_file="$(mktemp)"
python3 - <<'PY' > "$prompt_file"
details = []
for i in range(18):
    details.append(
        "Remember operational detail D%02d: file_%02d.txt was inspected, result R%02d was accepted, "
        "and constraint C%02d must remain active. " % (i, i, i, i)
        + ("context padding " * 18)
    )
print("Store these operational details for later. Answer READY only.\\n" + "\\n".join(details[:9]))
print("Store these additional operational details for later. Answer READY only.\\n" + "\\n".join(details[9:]))
print("/status")
print("Based on the operational details I asked you to remember earlier, list three user-provided active constraints by identifier.")
print("/exit")
PY

/usr/bin/time -f 'wall: %e s' env HOME="$HOME_DIR" "$ORBIT" \
  --workdir "$WORKDIR" \
  --context-tokens "$CONTEXT_TOKENS" \
  --max-tokens "$MAX_TOKENS" \
  < "$prompt_file"
