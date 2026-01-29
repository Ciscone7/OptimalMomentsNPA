#!/usr/bin/env bash
# Setup script for SpinsSDP.
#
# What it does:
#   1) Creates a virtual environment in .venv/
#   2) Installs dependencies
#   3) Installs the project (editable)
#
# Usage (from repo root):
#   bash spins_sdp/run/setup_env.sh
#   bash spins_sdp/run/setup_env.sh --dev
#

set -euo pipefail

INSTALL_DEV=0
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
	echo "Usage: bash spins_sdp/run/setup_env.sh [--dev]"
	echo "  --dev   Also install requirements-dev.txt (notebooks/plotting/dev tools)"
	exit 0
fi
if [[ "${1:-}" == "--dev" ]]; then
	INSTALL_DEV=1
	shift
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=========================================="
echo "SpinsSDP Setup"
echo "Repo: ${REPO_ROOT}"
echo "Venv: ${VENV_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "=========================================="
echo

cd "$REPO_ROOT"

echo "[1/4] Checking Python..."
"$PYTHON_BIN" --version
echo

echo "[2/4] Creating virtual environment (.venv)..."
if [[ -d "$VENV_DIR" ]]; then
	echo "  .venv already exists; reusing it"
else
	if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
		echo
		echo "ERROR: Failed to create venv. On Debian/Ubuntu you may need:" >&2
		echo "  sudo apt-get install -y python3-venv" >&2
		echo "Or use conda/mamba on the cluster." >&2
		exit 1
	fi
fi

VENV_PY="${VENV_DIR}/bin/python"

echo "  Using: $VENV_PY"
"$VENV_PY" --version
echo

echo "[3/4] Installing dependencies..."
"$VENV_PY" -m pip install -U pip setuptools wheel

# requirements.txt is the most reliable source of deps in this repo.
if [[ -f "requirements.txt" ]]; then
	"$VENV_PY" -m pip install -r requirements.txt
else
	echo "  NOTE: requirements.txt not found; installing from pyproject.toml only"
fi

if [[ "$INSTALL_DEV" == "1" ]]; then
	echo
	echo "  Installing dev dependencies (requirements-dev.txt)..."
	if [[ -f "requirements-dev.txt" ]]; then
		"$VENV_PY" -m pip install -r requirements-dev.txt
	else
		echo "  NOTE: requirements-dev.txt not found; skipping"
	fi
fi

echo
echo "[4/4] Installing SpinsSDP (editable)..."
"$VENV_PY" -m pip install -e .

echo
echo "Verifying imports (core):"
"$VENV_PY" -c "import numpy; import spins_sdp; print('  OK: numpy + spins_sdp')"
"$VENV_PY" -c "import cvxpy; print('  OK: cvxpy')" || true
"$VENV_PY" -c "import mosek; print('  OK: mosek')" || true
"$VENV_PY" -c "import qutip; print('  OK: qutip')" || true

echo
echo "=========================================="
echo "Setup complete"
echo "=========================================="
echo
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python -m spins_sdp.run.estimate_runtime --test --dry-run"
echo "  ./spins_sdp/run/run_experiments.sh --test --dry-run"
