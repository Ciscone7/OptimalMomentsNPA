#!/bin/bash
#
# SpinsSDP Experiments Runner
#
# Usage:
#   ./run_experiments.sh           # Full mode
#   ./run_experiments.sh --test    # Test mode (reduced params)
#   ./run_experiments.sh --dry-run # Preview commands
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Activate venv if exists
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

# Run the Python runner with all arguments
python3 -m spins_sdp.run.runner "$@"
