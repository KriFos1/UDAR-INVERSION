#!/bin/bash
# Runs or continues the chain of a config on this machine. Usage: ./run.sh [config.toml] [udarinv run options]
# Needs UDARINV_FWD2D (the 2D solver build, see solver/README.md). Stop it between points with: touch <output>/PAUSE
set -euo pipefail
cd "$(dirname "$0")"
CONFIG=${1:-config.toml}
shift || true
: "${UDARINV_FWD2D:?Set UDARINV_FWD2D to the fwd2D program of a solver build}"
exec udarinv run "$CONFIG" "$@"
