#!/bin/bash
#
# Setup script for deployment
#
# This script:
# 1. Creates a Python virtual environment
# 2. Installs the project and all dependencies
# 3. Verifies the installation
#
# After running, activate the environment with:
#   source venv/bin/activate
#

set -e

echo "=========================================="
echo "SpinsSDP Setup"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✓ Virtual environment created and activated"
echo ""

# Upgrade pip, setuptools, wheel
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel
echo "✓ Package tools upgraded"
echo ""

# Install project dependencies
echo "Installing SpinsSDP and dependencies..."
pip install -e .
echo "✓ SpinsSDP installed"
echo ""

# Install optional features (all variants)
echo "Installing optional features (Bayesian, RBM, HDF5)..."
pip install -e ".[all]"
echo "✓ Optional features installed"
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "import spins_sdp; print(f'  ✓ spins_sdp imported successfully')"
python3 -c "import mosek; print(f'  ✓ mosek available')"
python3 -c "import qutip; print(f'  ✓ qutip available')"
python3 -c "import cvxpy; print(f'  ✓ cvxpy available')"
echo ""

echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Test locally: ./spins_sdp/run/run_experiments.sh --test"
echo ""
