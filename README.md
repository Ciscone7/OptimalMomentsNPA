# SpinsSDP

**Certification of many-body quantum properties using semidefinite programming and the NPA hierarchy.**

SpinsSDP is a Python framework for computing certified bounds on ground-state energies and observables of quantum spin chains, as well as for certifying Bell nonlocality. It implements the Navascués–Pironio–Acín (NPA) moment-matrix hierarchy, assembles and solves the resulting semidefinite programs (SDPs) via MOSEK, and provides a suite of combinatorial optimisers for intelligently selecting which monomials to include in the relaxation basis — allowing tight bounds with smaller SDPs.

> **Authors:** Francesco Flora (ICFO) and Losel Matos (École Polytechnique)

---

## Table of Contents

- [Overview](#overview)
  - [What Does This Framework Do?](#what-does-this-framework-do)
  - [Key Ideas](#key-ideas)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Core Installation](#core-installation)
  - [Optional Extras](#optional-extras)
- [Quickstart](#quickstart)
- [Core Modules](#core-modules)
  - [Spin-Chain Module (`spins/`)](#spin-chain-module-spins)
  - [Bell Module (`bell/`)](#bell-module-bell)
  - [Optimisation Backends (`optimize/`)](#optimisation-backends-optimize)
  - [Artifact Manager](#artifact-manager)
  - [Plotting](#plotting)
- [Experiment Scripts](#experiment-scripts)
  - [Available Scripts](#available-scripts)
  - [Configuration System](#configuration-system)
  - [Pre-Made Experiment Configs](#pre-made-experiment-configs)
- [Demo Notebooks](#demo-notebooks)
- [Results Storage](#results-storage)

---

## Overview

### What Does This Framework Do?

Given a quantum spin-chain Hamiltonian (Ising, Heisenberg, Heisenberg-J₂, …), SpinsSDP can:

1. **Lower-bound the ground-state energy** via NPA moment-matrix relaxations solved as SDPs.
2. **Upper-bound the ground-state energy** via DMRG (Density Matrix Renormalisation Group).
3. **Bound any ground-state observable** (magnetisation, correlations, structure factors, …) by adding energy-window constraints to the SDP.
4. **Optimise the monomial basis** used in the NPA relaxation — selecting the *k* best monomials from a large candidate pool to achieve the tightest bound at a given SDP size.
5. **Certify Bell nonlocality** — compute NPA bounds on Bell operators (e.g., CHSH) for arbitrary bipartite projective-measurement scenarios, with the same basis-optimisation machinery.

### Key Ideas

- **NPA hierarchy.** Instead of optimising over all quantum states directly (intractable), we relax the problem to a semidefinite program over a *moment matrix* whose entries correspond to expectation values of Pauli monomials. The resulting bound is rigorous: it can only be looser than the true quantum value. Higher NPA levels include more monomials and yield tighter bounds, but the SDP grows rapidly.

- **Basis selection as combinatorial optimisation.** Going to a high NPA level (e.g. NPA-4) may produce an SDP that is too large to solve in practice. Instead, we start from a small basis (e.g. NPA-1) and *select* the most informative monomials from a higher level — framing this as a black-box combinatorial optimisation problem over fixed-Hamming-weight binary vectors. Several metaheuristic solvers are provided: Simulated Annealing, Parallel Tempering, Bayesian Optimisation (Random Forest), and an RBM-based REINFORCE sampler.

- **Symmetry reduction.** Physical symmetries of the Hamiltonian (translation, reflection, axis permutation, sign symmetries, rotation) are exploited to block-diagonalise the moment matrix and identify equivalent monomials, dramatically shrinking the SDP.
---

## Project Structure

```
SpinsSDP/
│
├── spins/                           # Spin-chain module
│   ├── models.py                    #   Hamiltonians & observables (Ising, Heisenberg, J₂)
│   ├── pauli_logic.py               #   Pauli algebra engine (bitmask representation)
│   ├── basis_builder.py             #   NPA & custom basis generators
│   ├── symmetry.py                  #   Symmetry manager (translation, reflection, …)
│   ├── spins_sdp.py                 #   SDP assembly & solving (CVXPY + MOSEK)
│   ├── spins_optimize.py            #   Basis-selection optimisation for spins
│   └── variational.py               #   DMRG upper bounds (TeNPy)
│
├── bell/                            # Bell nonlocality module
│   ├── bell_logic.py                #   Bell projector algebra & scenarios
│   ├── bell_sdp.py                  #   SDP assembly for Bell relaxations
│   └── bell_optimize.py             #   Basis-selection optimisation for Bell
│
├── optimize/                        # Metaheuristic optimisation backends
│   ├── montecarlo.py                #   Simulated Annealing & Parallel Tempering
│   ├── bayesian.py                  #   Bayesian Optimisation (Random Forest + UCB)
│   └── rbm.py                       #   RBM + REINFORCE (JAX/Equinox)
│
├── scripts/                         # Reproducible experiment scripts
│   ├── config_utils.py              #   YAML config loader with CLI override
│   ├── spins/                       #   Spin-chain experiments (see below)
│   └── bell/                        #   Bell experiments
│
├── experiments/                     # Pre-made YAML config files
├── demo_notebooks/                  # Interactive Jupyter tutorials
├── results/                         # Saved experiment results
├── plots/                           # Analysis notebooks
│
├── artifact_manager.py              # Experiment persistence (meta.json + data.npz)
├── plots.py                         # Plotting utilities for result visualisation
├── pyproject.toml                   # Package metadata & dependencies
├── requirements.txt                 # Pinned runtime dependencies
└── requirements-dev.txt             # Development dependencies
```

---

## Installation

### Core Installation

Requires **Python ≥ 3.10** and a valid [MOSEK](https://www.mosek.com/) license (free for academics).

```bash
# Clone and enter the project
git clone <repo-url>
cd SpinsSDP

# Create a virtual environment and install in editable mode
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the core dependencies: `numpy`, `scipy`, `cvxpy`, `mosek`, `qutip`, `physics-tenpy`, `tqdm`, `pyyaml`.

### Optional Extras

Additional features can be installed as extras:

```bash
# Plotting and analysis (matplotlib, pandas)
pip install -e ".[plotting]"

# Bayesian optimisation (scikit-learn)
pip install -e ".[bayesian]"

# RBM-based optimiser (JAX, Equinox, Optax)
pip install -e ".[rbm]"

# HDF5 storage support
pip install -e ".[hdf5]"

# Everything at once
pip install -e ".[all]"

# Full development environment (linting, testing, notebooks, all optional features)
pip install -r requirements-dev.txt
```

---

## Quickstart

**Compute the exact ground energy and NPA lower bound for the Heisenberg chain:**

```python
from spins.models import heisenberg_hamiltonian_dict, heisenberg_hamiltonian_exact
from spins.basis_builder import generate_npa_basis
from spins.symmetry import SymmetryManager
from spins.spins_sdp import solve_pauli_relaxation

N = 8

# Exact ground energy via QuTiP diagonalisation
H_exact = heisenberg_hamiltonian_exact(N, boundary="periodic")
E0 = H_exact.groundstate()[0]
print(f"Exact ground energy: {E0:.6f}")

# NPA-2 lower bound via SDP
H_dict = heisenberg_hamiltonian_dict(N, boundary="periodic")
basis = generate_npa_basis(N, k=2)
sym = SymmetryManager.default_for_heisenberg(N)
E_lb = solve_pauli_relaxation(basis, H_dict, sym)
print(f"NPA-2 lower bound:   {E_lb:.6f}")
```

**Bound an observable in the ground state:**

```python
from spins.models import staggered_magnetization_z_dict
from spins.spins_sdp import bound_observable

obs = staggered_magnetization_z_dict(N)
result = bound_observable(basis, H_dict, obs, energy_lb=E_lb, energy_ub=E0, symmetry_manager=sym)
print(f"Staggered Mz ∈ [{result.lb:.6f}, {result.ub:.6f}]")
```

**Run an experiment script from a YAML config:**

```bash
# Exact diagonalisation for the Ising model
python scripts/spins/exact_ground_energy.py --config experiments/spins_exact.yaml

# NPA lower bound for the Heisenberg model
python scripts/spins/npa_energy_lb.py --config experiments/spins_energy_lb.yaml

# DMRG upper bound
python scripts/spins/dmrg.py --config experiments/spins_dmrg.yaml

# Basis-selection optimisation sweep
python scripts/spins/optimization_sweep.py --config experiments/spins_optimization.yaml
```

---

## Core Modules

### Spin-Chain Module (`spins/`)

#### `pauli_logic.py` — Pauli Algebra Engine

The foundation of the framework. Pauli strings on N qubits are encoded as `PauliWord` objects using a compact **bitmask representation**: two integers `(x_mask, z_mask)` where bit *k* determines the local operator on site *k*:

| `x_mask[k]` | `z_mask[k]` | Operator |
|:---:|:---:|:---:|
| 0 | 0 | I |
| 1 | 0 | X |
| 0 | 1 | Z |
| 1 | 1 | Y |

Multiplication of two Pauli words computes the product word and phase (a power of *i*) entirely via bitwise operations — making it extremely fast. The module also handles:

- The **Ỹ = iY real-basis transformation**, which makes the moment matrix real-symmetric without losing tightness.
- **Moment matrix compilation**: given a basis of Pauli words, constructs the symbolic representation M\[i,j\] = (a + ib) · y\[label\] linking matrix entries to moment variables.

#### `models.py` — Hamiltonians & Observables

Provides Hamiltonians in two representations:
- **Exact form** (`Qobj`): for QuTiP diagonalisation.
- **Pauli dictionary** (`Dict[PauliWord, complex]`): for SDP construction.

Supported models:
| Model | Hamiltonian |
|-------|-------------|
| **Transverse-field Ising** | $H = -J\sum_i Z_iZ_{i+1} - h\sum_i X_i - k\sum_i Z_i$ |
| **Heisenberg (XXX)** | $H = \tfrac{1}{4}\sum_i (\sigma_i^x\sigma_{i+1}^x + \sigma_i^y\sigma_{i+1}^y + \sigma_i^z\sigma_{i+1}^z)$ |
| **Heisenberg-J₂** | Heisenberg + $J_2$ next-nearest-neighbour coupling |

Built-in observables: uniform/staggered magnetisation, nearest-neighbour correlations, two-point correlators, single-site Pauli operators.

#### `basis_builder.py` — NPA Basis Generation

Generates the set of Pauli monomials used in the moment-matrix relaxation:

- **`generate_npa_basis(N, k)`**: Standard NPA hierarchy up to level *k* via BFS from the identity using single-site generators.
- **`generate_heisenberg_paper_basis(N, r)`**: Tailored basis for 1D periodic Heisenberg chains with monomials up to degree 4 (identity, single-site, pairs up to distance *r*, contiguous triples, contiguous quadruples).
- **J₂ variants**: `heisenberg_j2_basis_weak` (J₂ ≤ 1) and `heisenberg_j2_basis_strong` (J₂ > 1) with modified triple structures.
- **Signature splitting**: `split_basis_by_signature` partitions words by their (s_xy, s_yz) parity for block-diagonalisation under rotation symmetry.

#### `symmetry.py` — Symmetry Manager

The `SymmetryManager` is a frozen dataclass with boolean flags controlling which physical symmetries are applied:

| Symmetry | Effect |
|----------|--------|
| **Rotation** | Block-diagonalises the moment matrix by (s_xy, s_yz) signature into up to 4 independent blocks |
| **Sign** | Zeros out moments with odd parity in N_X, N_Y, or N_Z |
| **Translation** | Identifies cyclic translation orbits (for periodic chains) |
| **Mirror** | Identifies spatial reflection orbits |
| **Axis permutation** | Identifies orbits under S₃ permutation of {X, Y, Z} |
| **Real basis** | Uses Ỹ = iY to make the moment matrix real-symmetric |

The `canonicalize(word)` method maps any Pauli word to its unique canonical representative under all active symmetries, and returns `None` if the word is killed by sign symmetry. Factory methods `default_for_heisenberg(N)` (all symmetries on) and `none(N)` (no symmetries) are provided for convenience.

#### `spins_sdp.py` — SDP Assembly & Solving

Converts compiled moment-matrix representations into CVXPY semidefinite programs solved by MOSEK:

- **`solve_pauli_relaxation(basis, operator, symmetry_manager)`**: End-to-end pipeline — builds block representations, assembles the block-diagonal SDP, and solves.
- **`build_block_diagonal_sdp(reps, objective, ...)`**: Constructs the SDP with per-signature-block PSD constraints. Real-symmetric blocks use a direct PSD constraint; complex-Hermitian blocks use the 2×2 real embedding.
- **`bound_observable(basis, H, O, energy_lb, energy_ub, ...)`**: Solves two SDPs (minimise and maximise ⟨O⟩) subject to the energy constraint E_lb ≤ ⟨H⟩ ≤ E_ub, returning certified bounds on the observable.

#### `spins_optimize.py` — Basis-Selection Optimisation

The core optimisation loop for spin chains. Given a **starting set** of monomials (e.g., NPA-1) and an **adding set** of candidate monomials (e.g., NPA-2 \ NPA-1):

1. Choose *k* monomials from the adding set.
2. Build the augmented basis and solve the SDP.
3. Return the objective value (energy lower bound or observable gap).

This black-box objective is then minimised by one of the metaheuristic solvers. Key entry points:

- **`optimize_ground_energy(...)`**: Single (k, seed) energy optimisation.
- **`optimize_observable(...)`**: Single (k, seed) observable-gap optimisation.
- **`sweep_k_values(...)`** / **`sweep_observable_k_values(...)`**: Sweep over multiple *k* values and seeds, with optional warm-start feedback chaining (the best mask from k is used to initialise k+1), atomic checkpointing, and resume support.

#### `variational.py` — DMRG Upper Bounds

Wraps [TeNPy](https://tenpy.readthedocs.io/) to compute variational upper bounds on the ground-state energy using DMRG. Supports all three Hamiltonian models with periodic boundary conditions and configurable bond dimension ramp-up schedules.

---

### Bell Module (`bell/`)

Mirrors the spin module but for **bipartite Bell scenarios** with projective measurements.

#### `bell_logic.py` — Bell Projector Algebra

- **`BellScenario`**: Defines a scenario with m_A Alice settings, m_B Bob settings, d_A Alice outcomes, d_B Bob outcomes. Factory methods: `symmetric(m, d)` and `asymmetric(...)`.
- **`BellWord`**: A reduced monomial in the projector algebra, represented as two tuples of (setting, outcome) pairs. Supports idempotence (A² = A), orthogonality (A_{x|a}·A_{x|a'} = 0 for a ≠ a'), and Alice-Bob commutation (\[A, B\] = 0).
- **`compile_moment_matrix_rep(basis)`**: Compiles the moment matrix for a Bell basis, canonicalising via min(w, w†).
- Built-in operators: `chsh_operator`, `probability_word`.

#### `bell_sdp.py` — Bell SDP Assembly

Builds and solves CVXPY SDPs for Bell NPA relaxations. The constraints are normalisation (y\[I\] = 1) and positive semidefiniteness (M ⪰ 0). Completeness is baked into the generator design (last outcomes are expanded via the completeness relation).

#### `bell_optimize.py` — Bell Basis Optimisation

Same optimisation machinery as the spin case, applied to Bell scenarios. Sweeps over *k* values and seeds, with feedback chaining and resume support.

---

### Optimisation Backends (`optimize/`)

All optimisers work on the same abstraction: minimise a black-box function over binary vectors of fixed Hamming weight *k* drawn from {0, 1}^L.

| Method | Module | Description |
|--------|--------|-------------|
| **Simulated Annealing (SA)** | `montecarlo.py` | Single-chain SA with exponential cooling. Proposes moves by swapping a selected and an unselected index. |
| **Parallel Tempering (PT)** | `montecarlo.py` | Replica-exchange Monte Carlo with geometrically-spaced temperatures. Runs multiple SA chains in parallel (one per CPU core) and periodically attempts temperature swaps between adjacent replicas for better mixing. |
| **Bayesian Optimisation (BO)** | `bayesian.py` | Iterative surrogate-model approach using a Random Forest regressor. Selects candidates via Upper Confidence Bound (UCB) acquisition. Requires `scikit-learn`. |
| **RBM + REINFORCE** | `rbm.py` | Restricted Boltzmann Machine trained with the REINFORCE policy-gradient algorithm. Uses Gumbel-top-k sampling to enforce the Hamming-weight constraint. Requires `jax`, `equinox`, `optax`. |
| **Random** | (baseline) | Uniform random selection of *k* indices — used as a baseline for comparison. |

All methods support warm-starting from a previous best solution.

---

### Artifact Manager

The `ArtifactManager` (in `artifact_manager.py`) provides clean, organised persistence for experiment results using plain files:

```
<results_root>/<artifact_name>/v<version>/<run_name>/
    meta.json      ← config, provenance, timestamps, platform info
    data.npz       ← result arrays (NumPy compressed)
```

Key features:
- **Config hashing** for integrity and deduplication.
- **Atomic writes** via temp-file → rename to prevent corruption.
- **Run discovery** by name, config-hash prefix, or recursive meta-field queries.
- **Resumable experiments**: scripts check for existing data and skip already-computed points.


## Experiment Scripts

### Available Scripts

All scripts live under `scripts/` and can be run either with pure CLI flags or with a YAML config file (`--config`).

#### Spin-Chain Scripts (`scripts/spins/`)

| Script | Purpose |
|--------|---------|
| `exact_ground_energy.py` | Exact diagonalisation of spin-chain Hamiltonians (QuTiP) |
| `npa_energy_lb.py` | NPA moment-relaxation **lower bound** on the ground-state energy |
| `dmrg.py` | DMRG variational **upper bound** on the ground-state energy (TeNPy) |
| `optimization_sweep.py` | Basis-selection optimisation sweep for **energy** lower bounds |
| `observable_sweep.py` | Basis-selection optimisation to minimise the **observable gap** (ub − lb) |
| `fraction_scaling.py` | Fixed-fraction scaling: select k = ⌊p × L⌋ monomials across system sizes |
| `fraction_scaling_observable.py` | Same fixed-fraction scaling, but for observable gaps |

#### Bell Scripts (`scripts/bell/`)

| Script | Purpose |
|--------|---------|
| `optimization_sweep.py` | Basis-selection optimisation sweep for Bell operators (e.g. CHSH) |

> **Note:** All scripts support `--resume` for automatic checkpointing and resumption, `--force` to overwrite existing results, and `--verbose` for detailed logging.

### Configuration System

Every experiment script supports two invocation styles:

**1. With a YAML config file** (recommended for reproducibility):
```bash
python scripts/spins/npa_energy_lb.py --config experiments/spins_energy_lb.yaml
```

**2. Purely from the CLI:**
```bash
python scripts/spins/exact_ground_energy.py \
    --model ising --Ns 4 6 8 --h 0.7 --k 0.3 --boundary periodic
```

**3. Config file + CLI overrides** (CLI wins):
```bash
python scripts/spins/npa_energy_lb.py \
    --config experiments/spins_energy_lb.yaml --npa-level 3
```

#### Priority Rules

```
CLI flags  >  YAML config  >  argparse defaults
```

When a parameter falls back to its built-in default, the script prints a warning:

```
Loading config: experiments/spins_exact.yaml
Using default values (not in config or CLI):
  J2 = 0.0
  N_max = None
```

#### Mutually Exclusive Parameters

Some parameters are mutually exclusive (e.g., `--Ns 4 6 8` vs `--N-min 4 --N-max 8`). The config system handles conflicts automatically — CLI flags take priority and conflicting YAML keys are removed.

#### Writing Your Own Config Files

Config files are flat YAML mappings (no nesting). Keys correspond to CLI argument names with underscores:

```yaml
# my_experiment.yaml
model: heisenberg
Ns: [4, 6, 8, 10, 12]
boundary: periodic
npa_level: 2
use_all_symmetries: true
mosek_tol: 1.0e-9
repeats: 3
```

**Tip:** Run any script with `--help` to see all available parameters.

### Pre-Made Experiment Configs

The `experiments/` directory contains ready-to-use configs:

| Config File | Description |
|-------------|-------------|
| `spins_exact.yaml` | Exact diagonalisation — Ising model, N ∈ {4, 6, 8} |
| `spins_energy_lb.yaml` | NPA-2 lower bound — Heisenberg, N ∈ {4, 6, 8} |
| `spins_dmrg.yaml` | DMRG upper bound — Heisenberg, N ∈ {4, 6, 8}, χ_max = 64 |
| `spins_optimization.yaml` | SA optimisation sweep — Heisenberg N = 4, NPA 1→2 |
| `spins_observable.yaml` | SA observable-gap sweep — Heisenberg N = 6, staggered Mz |
| `bell_optimization.yaml` | SA Bell sweep — CHSH (2,2,2,2 scenario), NPA 1→2 |
| `observable_cn2_paper_basis.yaml` | Large-scale PT sweep — C_{N/2} on N = 10, paper basis |
| `observable_cn2_npa4.yaml` | Large-scale PT sweep — C_{N/2} on N = 10, NPA-4 basis |

---

## Demo Notebooks

Interactive Jupyter tutorials are provided in `demo_notebooks/`:

| Notebook | Description |
|----------|-------------|
| **`spins_tutorial.ipynb`** | Complete walkthrough of the spin-chain pipeline: Hamiltonians, Pauli algebra, NPA bases, SDP relaxations, symmetry reduction, basis optimisation, and persistence. |
| **`bell_tutorial.ipynb`** | Complete walkthrough of the Bell module: scenarios, projector algebra, NPA bases, CHSH bound (recovers Tsirelson's bound 2√2), optimisation sweeps, and persistence. |
| **`general_observables.ipynb`** | Bounding arbitrary ground-state observables via energy-constrained SDPs: magnetisation, correlations, structure factor, scaling with system size. |
| **`observable_optimization.ipynb`** | Visualising large-scale overnight experiment results for the half-chain X-correlation C_{N/2}, comparing paper-basis vs NPA-4 strategies. |
| **`artifact_manager.ipynb`** | Tutorial on the persistence layer: saving/loading results, config hashing, run discovery. |

Each tutorial notebook ships with pre-computed demo results (in `*_demo_results/` directories) so they can be explored without running expensive computations.

---

## Results Storage

Experiment outputs follow the `ArtifactManager` directory convention:

```
results/
├── bell_optimization_sweep/        # Bell optimisation results
├── energy_lower_bound/             # NPA energy lower bounds
├── observable_bound/               # Observable bound sweeps
├── reference/                      # Reference values
├── spin_fraction_scaling/          # Fraction-scaling experiments (energy)
└── spin_fraction_scaling_observable/  # Fraction-scaling experiments (observables)
```

Each artifact directory contains versioned runs:
```
results/<artifact>/v1/<run_name>/
    meta.json       # Full config, timestamps, platform info, config hash
    data.npz        # NumPy arrays with all computed results
```

Results can be loaded programmatically:

```python
from artifact_manager import ArtifactManager

am = ArtifactManager("results")
run = am.open_run("spin_moment_energy_lb", "my_heisenberg_run")
data = run.load_table()
print(data["N"], data["E_lb"])
```
