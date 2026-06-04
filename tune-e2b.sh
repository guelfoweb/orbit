#!/bin/sh
set -eu

BASE_MODEL="gemma4:e2b"
PRIMARY_MODEL="gemma4:e2b-c8k"
SAFE_MODEL="gemma4:e2b-c4k"
INSTALL_DIR="${ORBIT_INSTALL_DIR:-$HOME/.local/share/orbit}"
C8K_MODEFILE_NAME="Modelfile.gemma4-e2b-c8k"
C4K_MODEFILE_NAME="Modelfile.gemma4-e2b-c4k"

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

ensure_modelfile() {
    name="$1"
    path="$INSTALL_DIR/$name"
    url="https://raw.githubusercontent.com/guelfoweb/orbit/main/$name"
    if [ -f "$path" ]; then
        return
    fi
    need_cmd curl
    mkdir -p "$INSTALL_DIR"
    info "Downloading $name"
    curl -fsSL "$url" -o "$path"
}

ensure_modelfile "$C8K_MODEFILE_NAME"
ensure_modelfile "$C4K_MODEFILE_NAME"

info "Creating tuned model profile: $PRIMARY_MODEL"
ollama create "$PRIMARY_MODEL" -f "$INSTALL_DIR/$C8K_MODEFILE_NAME"

info "Creating conservative model profile: $SAFE_MODEL"
ollama create "$SAFE_MODEL" -f "$INSTALL_DIR/$C4K_MODEFILE_NAME"

info "Model profiles ready:"
info "  $PRIMARY_MODEL"
info "  $SAFE_MODEL"
info "Run:"
info "  orbit --model $PRIMARY_MODEL"
info "If Ollama crashes with a GGML scheduler assert, try:"
info "  orbit --model $SAFE_MODEL"
