#!/bin/bash

################################################################################
# 1. Heisenberg chain: Exact + DMRG + Relaxation bounds
# 2. Heisenberg J2 sweep: Fixed N=40, varying J2 with DMRG + Relaxation
# 3. Optimization sweep: Random sampling vs optimization for various k
#
# All models use periodic boundary conditions (PBC) as in the paper.
################################################################################

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SpinsSDP Paper Experiments Suite${NC}"
echo -e "${BLUE}========================================${NC}\n"

################################################################################
# SECTION 1: SIMPLE HEISENBERG CHAIN (J1 only)
################################################################################

echo -e "${GREEN}[SECTION 1] HEISENBERG CHAIN (Simple Heisenberg)${NC}"
echo "Models: Exact (N≤15), DMRG (N≤100), Relaxation (N≤60,80,100)"
echo "Basis: Paper basis (heisenberg_simple)"
echo ""

# --- Subsection 1a: Exact ground energies (N up to 15) ---
echo -e "${YELLOW}1a. Computing exact ground energies (N=2..15)${NC}"
python -m spins_sdp.scripts.exact_ground_energy \
    --model heisenberg \
    --N-min 2 \
    --N-max 15 \
    --boundary periodic \
    --solver MOSEK \
    --resume \
    --verbose

echo -e "${YELLOW}   ✓ Exact ground energies computed${NC}\n"

# --- Subsection 1b: DMRG upper bounds (N up to 100) ---
echo -e "${YELLOW}1b. Computing DMRG upper bounds (N=4..100, chi_max=256)${NC}"
python -m spins_sdp.scripts.dmrg \
    --model heisenberg \
    --N-min 4 \
    --N-max 100 \
    --chi-max 256 \
    --mixer \
    --boundary periodic \
    --resume \
    --verbose

echo -e "${YELLOW}   ✓ DMRG upper bounds computed${NC}\n"

# --- Subsection 1c: SDP relaxations using paper's basis ---
echo -e "${YELLOW}1c. Computing SDP relaxations (paper basis, N=4,6,8,...,60,80,100)${NC}"

# Use heisenberg_simple basis (paper basis) for N up to 60
python -m spins_sdp.scripts.npa_energy_lb \
    --model heisenberg \
    --basis heisenberg_simple \
    --N-min 4 \
    --N-max 60 \
    --boundary periodic \
    --solver MOSEK \
    --resume \
    --verbose

# Also compute for N=80, N=100
for N in 80 100; do
    echo "Computing relaxation for N=$N..."
    python -m spins_sdp.scripts.npa_energy_lb \
        --model heisenberg \
        --basis heisenberg_simple \
        --Ns $N \
        --boundary periodic \
        --solver MOSEK \
        --resume \
        --verbose
done

echo -e "${YELLOW}   ✓ SDP relaxations computed (paper basis)${NC}\n"

################################################################################
# SECTION 2: HEISENBERG J2 SWEEP
################################################################################

echo -e "${GREEN}[SECTION 2] HEISENBERG J2 SWEEP${NC}"
echo "Fixed: N=40, boundary=periodic"
echo "Sweep: J2 from 0 to 2 (increment 0.1)"
echo "Methods: DMRG (chi_max=512) + Relaxation (adaptive basis)"
echo ""

echo -e "${YELLOW}2a. Computing DMRG upper bounds for J2 sweep (N=40, J2=0..2)${NC}"

for J2_int in {0..20}; do
    J2=$(echo "scale=1; $J2_int / 10" | bc)
    echo "  J2=$J2..."
    python -m spins_sdp.scripts.dmrg \
        --model heisenberg_j2 \
        --Ns 40 \
        --J2 $J2 \
        --chi-max 512 \
        --mixer \
        --boundary periodic \
        --resume \
        --verbose
done

echo -e "${YELLOW}   ✓ DMRG J2 sweep completed${NC}\n"

echo -e "${YELLOW}2b. Computing SDP relaxations for J2 sweep (N=40, J2=0..2)${NC}"

for J2_int in {0..20}; do
    J2=$(echo "scale=1; $J2_int / 10" | bc)
    
    # Choose basis based on J2 value
    if (( $(echo "$J2 <= 1.0" | bc -l) )); then
        BASIS="heisenberg_j2_weak"
    else
        BASIS="heisenberg_j2_strong"
    fi
    
    echo "  J2=$J2 (basis=$BASIS)..."
    python -m spins_sdp.scripts.npa_energy_lb \
        --model heisenberg_j2 \
        --basis $BASIS \
        --Ns 40 \
        --J2 $J2 \
        --boundary periodic \
        --solver MOSEK \
        --resume \
        --verbose
done

echo -e "${YELLOW}   ✓ SDP J2 sweep completed${NC}\n"

################################################################################
# SECTION 3: OPTIMIZATION SWEEPS
################################################################################

echo -e "${GREEN}[SECTION 3] OPTIMIZATION SWEEPS${NC}"
echo "Setup: Start=NPA1, End=NPA4, boundary=periodic"
echo "Methods: Random sampling baseline + Simulated annealing optimization"
echo ""

# Define system sizes for optimization (practical subset)
OPTIMIZATION_Ns=(4 6 8 10)
NUM_SEEDS=10
K_MAX=20

for N in "${OPTIMIZATION_Ns[@]}"; do
    echo -e "${YELLOW}3.$N. Optimization sweep for N=$N${NC}"
    
    # --- 3a: Random sampling baseline ---
    echo "  a) Random sampling baseline (k=0..$K_MAX, seeds=$NUM_SEEDS)..."
    python -m spins_sdp.scripts.optimization_sweep \
        --model heisenberg \
        --N $N \
        --boundary periodic \
        --start-level 1 \
        --end-level 4 \
        --method random \
        --k-max $K_MAX \
        --num-seeds $NUM_SEEDS \
        --resume \
        --verbose
    
    echo "     ✓ Random baseline done"
    
    # --- 3b: Simulated annealing ---
    echo "  b) Simulated annealing (k=0..$K_MAX, seeds=$NUM_SEEDS)..."
    python -m spins_sdp.scripts.optimization_sweep \
        --model heisenberg \
        --N $N \
        --boundary periodic \
        --start-level 1 \
        --end-level 4 \
        --method sa \
        --sa-steps 100 \
        --k-max $K_MAX \
        --num-seeds $NUM_SEEDS \
        --resume \
        --verbose
    
    echo "     ✓ Simulated annealing done"
    
    # --- 3c: Parallel tempering ---
    echo "  c) Parallel tempering (k=0..$K_MAX, seeds=$NUM_SEEDS)..."
    python -m spins_sdp.scripts.optimization_sweep \
        --model heisenberg \
        --N $N \
        --boundary periodic \
        --start-level 1 \
        --end-level 4 \
        --method pt \
        --pt-chains 4 \
        --pt-epochs 5 \
        --pt-steps-per-epoch 50 \
        --k-max $K_MAX \
        --num-seeds $NUM_SEEDS \
        --resume \
        --verbose
    
    echo "     ✓ Parallel tempering done"
    
    # --- 3d: Bayesian optimization ---
    echo "  d) Bayesian optimization (k=0..$K_MAX, seeds=$NUM_SEEDS)..."
    python -m spins_sdp.scripts.optimization_sweep \
        --model heisenberg \
        --N $N \
        --boundary periodic \
        --start-level 1 \
        --end-level 4 \
        --method bo \
        --bo-n-init 10 \
        --bo-n-iter 30 \
        --bo-candidates-per-iter 50 \
        --k-max $K_MAX \
        --num-seeds $NUM_SEEDS \
        --resume \
        --verbose
    
    echo "     ✓ Bayesian optimization done"
    echo ""

done

################################################################################
# COMPLETION
################################################################################

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ All experiments completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results saved to: spins_sdp/results/"
echo ""
