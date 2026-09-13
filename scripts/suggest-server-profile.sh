#!/usr/bin/env sh
# Print the conservative heuristic server profile for this host.
#
# This is a PRESENTATION WRAPPER. It used to re-derive thread, batch, ubatch and
# cache_ram numbers in shell, which made it a second implementation of a policy
# that also lives in Python -- and the two drifted: the script printed
# `export THREADS=...` for an `orbit server` that read no environment at all, so
# its advice had no effect. The policy now lives in exactly one place,
# `orbit.native_server.server_profile`, and this prints what that says.
#
# Note what this does NOT do. It is the heuristic only: it never measures and
# never reads the profile cache, because a suggestion meant for review should
# not depend on a benchmark the reader did not watch run. For the real resolved
# profile, including any measurement Orbit has cached for this machine, use:
#
#   orbit server --show-profile
#
# and to discard a stored measurement and take it again:
#
#   orbit server --recalibrate
#
# ORBIT_CACHE_RAM is advisory. Orbit's native server has no cache-RAM knob to
# pass it to -- it is computed conservatively here for an operator running an
# external llama-server, and `orbit server` reads the variable only so that
# `--show-profile` can report what an operator set.
#
# `orbit server` already resolves all of this by itself. These exports exist for
# overriding it explicitly; anything you export here wins over what Orbit would
# have chosen, and anything you pass on the command line wins over both.
set -eu

usage() {
    cat <<'USAGE'
usage: suggest-server-profile.sh [--shell] [--ctx N]

  --shell   print an env block suitable for `eval`
  --ctx N   context size to size the cache reserve against (default 8192)
USAGE
}

for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
    esac
done

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

# The repository's own interpreter when there is one, otherwise whatever python3
# is on PATH -- the module imports nothing outside the standard library, so an
# uninstalled checkout still works.
if [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "suggest-server-profile: no python3 found" >&2
    exit 1
fi

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    exec "$PYTHON" -m orbit.native_server.server_profile "$@"
