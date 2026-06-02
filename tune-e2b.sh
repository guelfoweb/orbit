#!/bin/sh
set -eu

BASE_MODEL="gemma4:e2b"
TUNED_MODEL="gemma4:e2b-fast-t6-c8k"
INSTALL_DIR="${ORBIT_INSTALL_DIR:-$HOME/.local/share/orbit}"
MODEFILE_NAME="Modelfile.gemma4-e2b-fast-t6-c8k"
MODEFILE_PATH="$INSTALL_DIR/$MODEFILE_NAME"
MODEFILE_URL="https://raw.githubusercontent.com/guelfoweb/orbit/main/$MODEFILE_NAME"

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

if ! command -v ollama >/dev/null 2>&1; then
    error "ollama is not installed. Install Ollama first: https://ollama.com/download"
fi

if ! ollama list | awk '{print $1}' | grep -qx "$BASE_MODEL"; then
    info "Base model not found: $BASE_MODEL"
    info "Pulling $BASE_MODEL"
    ollama pull "$BASE_MODEL"
else
    info "Base model already present: $BASE_MODEL"
fi

if [ ! -f "$MODEFILE_PATH" ]; then
    need_cmd curl
    mkdir -p "$INSTALL_DIR"
    info "Downloading $MODEFILE_NAME"
    curl -fsSL "$MODEFILE_URL" -o "$MODEFILE_PATH"
fi

info "Creating tuned model profile: $TUNED_MODEL"
ollama create "$TUNED_MODEL" -f "$MODEFILE_PATH"

info "Model profile ready: $TUNED_MODEL"
info "Run:"
info "  orbit --model $TUNED_MODEL"
