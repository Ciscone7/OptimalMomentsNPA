from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Tuple
import argparse

from spins_sdp import models
from spins_sdp.basis_builder import generate_npa_basis


def time_best_avg(fn, repeats: int) -> Tuple[Any, float, float]:
    """Return (result, best_time, avg_time) over `repeats` runs using perf_counter."""
    times: List[float] = []
    out: Any = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, float(min(times)), float(sum(times) / len(times))


def parse_ns_from_args(args: argparse.Namespace) -> List[int]:
    """Parse either --Ns or --N-min/--N-max into a sorted unique int list."""
    if getattr(args, "Ns", None):
        Ns = list(args.Ns)
    else:
        if getattr(args, "N_min", None) is None or getattr(args, "N_max", None) is None:
            raise SystemExit("Provide either --Ns or both --N-min and --N-max")
        if args.N_min > args.N_max:
            raise SystemExit("--N-min must be <= --N-max")
        Ns = list(range(int(args.N_min), int(args.N_max) + 1))

    Ns = [int(n) for n in Ns]
    if any(n <= 0 for n in Ns):
        raise SystemExit("All N must be >= 1")
    return sorted(set(Ns))


def model_params_from_args(args: argparse.Namespace) -> Dict[str, float]:
    """Extract model-specific params from parsed args (ising/heisenberg/heisenberg_j2)."""
    if args.model == "ising":
        return {"J": float(args.J), "h": float(args.h), "k": float(args.k)}
    if args.model == "heisenberg":
        return {}
    if args.model == "heisenberg_j2":
        return {"J2": float(args.J2)}
    raise SystemExit(f"Unknown model: {args.model}")


def hamiltonian_exact_fn(model_name: str):
    """Return the exact (QuTiP) Hamiltonian builder for a model."""
    if model_name == "ising":
        return models.ising_hamiltonian_exact
    if model_name == "heisenberg":
        return models.heisenberg_hamiltonian_exact
    if model_name == "heisenberg_j2":
        return models.heisenberg_j2_hamiltonian_exact
    raise ValueError(f"Unknown model: {model_name}")


def hamiltonian_dict_fn(model_name: str):
    """Return the Pauli-operator dict builder for a model."""
    if model_name == "ising":
        return models.ising_hamiltonian_dict
    if model_name == "heisenberg":
        return models.heisenberg_hamiltonian_dict
    if model_name == "heisenberg_j2":
        return models.heisenberg_j2_hamiltonian_dict
    raise ValueError(f"Unknown model: {model_name}")


def build_npa_basis_sets(
    N: int,
    start_level: int,
    end_level: int,
):
    """Return (starting_set, adding_set, final_set) for NPA levels."""
    full_basis = generate_npa_basis(N=N, k=end_level)
    
    # Extract starting set: concatenate levels 0 through start_level (inclusive)
    starting_set = []
    for level_idx in range(min(start_level + 1, len(full_basis.levels))): # The min is in case start_level == max level
        starting_set.extend(full_basis.levels[level_idx])
    
    # Extract adding set: concatenate levels start_level+1 through end_level (inclusive)
    adding_set = []
    for level_idx in range(start_level + 1, min(end_level + 1, len(full_basis.levels))):
        adding_set.extend(full_basis.levels[level_idx])
    
    # Final set is just all words
    final_set = full_basis.words
    
    return starting_set, adding_set, final_set
