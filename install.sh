#!/bin/sh
set -eu

REPO_URL="${ORBIT_REPO_URL:-https://github.com/guelfoweb/orbit.git}"
INSTALL_DIR="${ORBIT_INSTALL_DIR:-$HOME/.local/share/orbit}"
BIN_DIR="${ORBIT_BIN_DIR:-$HOME/.local/bin}"
CONFIG_DIR="$HOME/.orbit"
CONFIG_FILE="$CONFIG_DIR/config.json"
MODEL_NAME="${ORBIT_MODEL_NAME:-gemma4:e2b-fast-t6-c8k}"

info() {
    printf '%s\n' "$*"
}

error() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || error "required command not found: $1"
}

need_cmd git
need_cmd python3

python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    || error "Python 3.10+ is required"

mkdir -p "$BIN_DIR" "$CONFIG_DIR"

if [ -d "$INSTALL_DIR/.git" ]; then
    info "Updating Orbit in $INSTALL_DIR"
    git -C "$INSTALL_DIR" pull --ff-only
elif [ -e "$INSTALL_DIR" ]; then
    error "$INSTALL_DIR exists but is not a git checkout"
else
    info "Cloning Orbit into $INSTALL_DIR"
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

info "Creating virtual environment"
python3 -m venv "$INSTALL_DIR/.venv"

info "Installing Python package"
"$INSTALL_DIR/.venv/bin/python" -m pip install -e "$INSTALL_DIR"

ln -sfn "$INSTALL_DIR/.venv/bin/orbit" "$BIN_DIR/orbit"

if [ ! -f "$CONFIG_FILE" ]; then
    info "Creating default config at $CONFIG_FILE"
    cat >"$CONFIG_FILE" <<EOF
{
  "model": "$MODEL_NAME",
  "host": "http://127.0.0.1:11434",
  "workdir": ".",
  "timeout": 300,
  "think": "off",
  "debug_timing": false,
  "ui": {
    "markdown": true,
    "collapse_long_input": true,
    "long_input_preview_chars": 50
  },
  "tools": {
    "max_loops": 10
  }
}
EOF
else
    info "Keeping existing config at $CONFIG_FILE"
fi

info "Installed Orbit"
info "Binary: $BIN_DIR/orbit"
info "Config: $CONFIG_FILE"
info "Run: orbit"
info "Optional e2b tuning: $INSTALL_DIR/tune-e2b.sh"

if ! printf '%s' ":$PATH:" | grep -q ":$BIN_DIR:"; then
    info "Note: add $BIN_DIR to PATH if 'orbit' is not found."
fi
