# SpinsSDP

Repo for certification of many-body properties of spin systems using the NPA hierarchy.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running Experiment Scripts](#running-experiment-scripts)
  - [Available Scripts](#available-scripts)
  - [YAML Config Files](#yaml-config-files)
  - [CLI Override](#cli-override)
  - [Default Value Warnings](#default-value-warnings)
  - [Mutually Exclusive Parameters](#mutually-exclusive-parameters)
  - [Writing Your Own Config Files](#writing-your-own-config-files)

---

## Project Structure

```
SpinsSDP/
├── bell/                        # Bell inequality module
│   ├── bell_logic.py
│   ├── bell_optimize.py
│   └── bell_sdp.py
├── spins/                       # Spin-chain module
│   ├── basis_builder.py
│   ├── models.py
│   ├── pauli_logic.py
│   ├── spins_optimize.py
│   ├── spins_sdp.py
│   ├── symmetry.py
│   └── variational.py
├── optimize/                    # Optimisation backends (SA, PT, BO, RBM)
│   ├── bayesian.py
│   ├── montecarlo.py
│   └── rbm.py
├── scripts/
│   ├── config_utils.py          # Shared YAML config loader (used by all scripts)
│   ├── spins/                   # Spin-chain experiment scripts
│   │   ├── _common.py           # Shared CLI utilities for spin scripts
│   │   ├── exact_ground_energy.py
│   │   ├── npa_energy_lb.py
│   │   ├── dmrg.py
│   │   └── optimization_sweep.py
│   └── bell/                    # Bell experiment scripts
│       └── optimization_sweep.py
├── experiments/                 # Pre-made YAML config files
│   ├── spins_exact.yaml
│   ├── spins_energy_lb.yaml
│   ├── spins_dmrg.yaml
│   ├── spins_optimization.yaml
│   └── bell_optimization.yaml
└── demo_notebooks/              # Jupyter tutorials
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Running Experiment Scripts

Every experiment script can be invoked in two ways:

1. **Purely from the CLI** — passing every parameter as a flag (the traditional way).
2. **With a YAML config file** — passing `--config path/to/experiment.yaml` and optionally overriding individual parameters from the CLI.

### Available Scripts

| Script | Purpose |
|--------|---------|
| `scripts/spins/exact_ground_energy.py` | Exact diagonalisation of spin-chain Hamiltonians |
| `scripts/spins/npa_energy_lb.py` | NPA moment-relaxation lower bound via SDP |
| `scripts/spins/dmrg.py` | DMRG upper bound (via TeNPy) |
| `scripts/spins/optimization_sweep.py` | Spin basis optimisation sweep (SA/PT/BO/RBM) |
| `scripts/bell/optimization_sweep.py` | Bell NPA basis optimisation sweep |

> **Note:** Make sure you have installed the project in editable mode (`pip install -e .`) so that all packages are importable.

### YAML Config Files

Instead of typing dozens of CLI flags, you can write all your experiment parameters in a single YAML file and pass it with `--config`:

```bash
# Run exact diagonalisation using a config file
python scripts/spins/exact_ground_energy.py \
    --config experiments/spins_exact.yaml

# Run Bell optimisation using a config file
python scripts/bell/optimization_sweep.py \
    --config experiments/bell_optimization.yaml
```

The YAML file is a flat key-value mapping where keys correspond to CLI argument names (using underscores). For example, the CLI flag `--npa-level 2` corresponds to `npa_level: 2` in YAML.


### CLI Override

The configuration system follows a strict **three-level priority**:

```
CLI flags  >  YAML config  >  argparse defaults
```

Any CLI flag you add alongside `--config` will override the corresponding YAML value.


### Default Value Warnings

When a parameter is not specified in either the CLI or the config file, the script falls back to its built-in default and **prints a warning** so you always know exactly what values are being used:

```
Loading config: experiments/spins_exact.yaml
Using default values (not in config or CLI):
  J2 = 0.0
  N_max = None
  N_min = None
```

### Mutually Exclusive Parameters

Some parameters belong to mutually exclusive groups (e.g., you specify system sizes either with `--Ns 4 6 8` or with `--N-min 4 --N-max 8`, but not both). The config system handles this correctly:

- If the YAML file sets `Ns: [4, 6, 8]` but you pass `--N-min 4 --N-max 10` on the CLI, the CLI wins and the YAML's `Ns` is automatically removed to avoid a conflict.

### Writing Your Own Config Files

Config files are flat YAML mappings. No nesting is allowed - every key must map to a scalar or a list:

```yaml
# my_experiment.yaml
model: heisenberg
Ns: [4, 6, 8, 10, 12]
boundary: periodic
J: 1.0
h: 0.5
repeats: 3
```

**Tip:** Check each script's `--help` to see all available parameters and their dest names.


## Quick-Start Examples

```bash
# 1. Run an experiment from a config file
python scripts/spins/exact_ground_energy.py \
    --config experiments/spins_exact.yaml

# 2. Override a single parameter
python scripts/spins/exact_ground_energy.py \
    --config experiments/spins_exact.yaml --boundary open

# 3. Run purely from CLI (no config file)
python scripts/spins/exact_ground_energy.py \
    --model ising --Ns 4 6 8 --h 0.7 --k 0.3

# 4. See all available parameters for a script
python scripts/spins/exact_ground_energy.py --help
```

The `experiments/` folder contains ready-to-use config files for quick testing.
