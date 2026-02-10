"""Run optimization sweeps for moment selection and save results.

This script finds optimal subsets of monomials to add to a starting basis,
using simulated annealing, parallel tempering, Bayesian optimization, or RBM-based
REINFORCE optimization.

The random sampling method (`--method random`) provides a baseline for comparison:
it randomly selects k monomials without any optimization.

Results are saved in a resumable way:
  - Results are grouped by a hash of the full configuration.
  - If a `data.npz` already exists, only missing (k, seed) pairs are computed.

On-disk layout (default):
    spins_sdp/results/spin_optimization_sweep/v1/<config_hash>/
    ├── meta.json
    └── data.npz

`data.npz` contains arrays:
  - run_idx (int)             # sequential run index
  - k (int)                   # number of monomials selected (Hamming weight)
  - seed (int)                # random seed for this run
  - best_value (float)        # best SDP lower bound found
  - elapsed_s (float)         # wall-clock time (seconds)
    - n_obj_evals (int)         # number of objective evaluations (proxy for cost)
  - mask_bits (uint8)         # bit-packed selection mask, shape (n_runs, ceil(L/8))

Examples:
    python -m spins_sdp.scripts.optimization_sweep --model heisenberg --N 4 --boundary periodic --start-level 1 --end-level 2 --method sa --ks 0 1 2 3 --seeds 42 43 44
    python -m spins_sdp.scripts.optimization_sweep --model ising --N 6 --J 1.0 --h 0.5 --k 0.3 --start-level 1 --end-level 2 --method sa --k-max 10 --num-seeds 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    import pandas as pd

from spins_sdp import models
from spins_sdp.sdp import solve_pauli_relaxation
from spins_sdp.scripts._artifact_io import (
    config_hash,
    atomic_save_npz,
    upsert_meta_json,
    utc_now_iso,
)
from spins_sdp.scripts._common import (
    add_symmetry_args,
    build_basis_sets,
    build_npa_basis_sets,
    hamiltonian_dict_fn,
    model_params_from_args,
    symmetry_config_from_args,
)
from spins_sdp.symmetry import SymmetryManager

from src.optimalsdp.montecarlo import parallel_tempering, simulated_annealing



SCHEMA_VERSION = 1
ARTIFACT_NAME = "spin_optimization_sweep"


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pandas is required for results_to_dataframe(). Install with `pip install pandas` "
            "or `pip install -r requirements-dev.txt`."
        ) from e
    return pd


# -----------------------------------------------------------------------------
# Data schema for data.npz
# -----------------------------------------------------------------------------
# Note: mask_bits is handled separately (2D array)
_SCALAR_FIELDS = {
    "k": np.dtype("int32"),
    "seed": np.dtype("int32"),
    "best_value": np.dtype("float64"),
    "elapsed_s": np.dtype("float64"),
    "n_obj_evals": np.dtype("int32"),
}


def _load_existing_runs(data_path: Path) -> Tuple[Dict[Tuple[int, int], int], Dict[str, List]]:
    """Load existing runs from data.npz.
    
    Returns:
        - completed: dict mapping (k, seed) -> run_idx
        - arrays: dict of existing arrays (for appending)
    """
    if not data_path.exists():
        return {}, {}
    
    data = np.load(data_path, allow_pickle=False)
    
    # Build completed set
    ks = data["k"]
    seeds = data["seed"]
    run_idxs = data["run_idx"]
    
    completed = {}
    for i in range(len(ks)):
        completed[(int(ks[i]), int(seeds[i]))] = int(run_idxs[i])
    
    # Extract all arrays
    arrays = {key: list(data[key]) for key in data.files}
    
    return completed, arrays


def _save_runs(
    data_path: Path,
    run_idx: List[int],
    k: List[int],
    seed: List[int],
    best_value: List[float],
    elapsed_s: List[float],
    n_obj_evals: List[int],
    mask_bits: np.ndarray,
) -> None:
    """Save all runs to data.npz."""
    atomic_save_npz(
        data_path,
        run_idx=np.array(run_idx, dtype=np.int32),
        k=np.array(k, dtype=np.int32),
        seed=np.array(seed, dtype=np.int32),
        best_value=np.array(best_value, dtype=np.float64),
        elapsed_s=np.array(elapsed_s, dtype=np.float64),
        n_obj_evals=np.array(n_obj_evals, dtype=np.int32),
        mask_bits=mask_bits,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _stable_list_hash(items: List[str], n_chars: int = 16) -> str:
    """Hash a list of strings for fingerprinting."""
    payload = "\n".join(items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n_chars]


def _stable_pauliword_mask_hash(words: List[Any], *, N: int, n_chars: int = 16) -> str:
    """Fast stable hash for PauliWord-like objects.

    We hash the (x_mask, z_mask) integer bitmasks for each word.
    This avoids expensive `str(word)` conversion when the adding set is huge
    (e.g. end_level=NPA4).
    """
    h = hashlib.sha256()
    h.update(struct.pack("<I", int(N)))

    # Fixed-width encoding based on N bits.
    n_bytes = max(1, (int(N) + 7) // 8)
    for w in words:
        x = int(getattr(w, "x_mask"))
        z = int(getattr(w, "z_mask"))
        h.update(x.to_bytes(n_bytes, byteorder="little", signed=False))
        h.update(z.to_bytes(n_bytes, byteorder="little", signed=False))

    return h.hexdigest()[:n_chars]


def _pack_masks(masks: List[np.ndarray]) -> np.ndarray:
    """Bit-pack a list of boolean/int masks into a uint8 array.
    
    Args:
        masks: List of 1D arrays of shape (L,) with 0/1 values.
    
    Returns:
        2D uint8 array of shape (n_masks, ceil(L/8)).
    """
    if not masks:
        return np.array([], dtype=np.uint8).reshape(0, 0)
    
    L = len(masks[0])
    packed = []
    for m in masks:
        bits = np.packbits(m.astype(np.uint8))
        packed.append(bits)
    return np.stack(packed, axis=0)


def _unpack_masks(packed: np.ndarray, L: int) -> np.ndarray:
    """Unpack bit-packed masks back to boolean array.
    
    Args:
        packed: 2D uint8 array of shape (n_masks, ceil(L/8)).
        L: Original mask length.
    
    Returns:
        2D bool array of shape (n_masks, L).
    """
    if packed.size == 0:
        return np.array([], dtype=bool).reshape(0, L)
    
    unpacked = []
    for row in packed:
        bits = np.unpackbits(row)[:L]
        unpacked.append(bits.astype(bool))
    return np.stack(unpacked, axis=0)




# -----------------------------------------------------------------------------
# Objective function wrapper
# -----------------------------------------------------------------------------

class OptimizationObjective:
    """Objective function for optimization.
    
    The objective takes a binary mask (0/1 array of length L=len(adding_set))
    and returns the SDP relaxation value for the corresponding basis.
    """
    def __init__(
        self,
        starting_set: List,
        adding_set: List,
        hamiltonian_dict: Dict,
        symmetry_manager: SymmetryManager,
        mosek_tol: float = 1e-9,
    ):
        self.starting_set = starting_set
        self.adding_set = adding_set
        self.hamiltonian_dict = hamiltonian_dict
        self.symmetry_manager = symmetry_manager
        self.mosek_tol = mosek_tol
    
    def __call__(self, mask: Any) -> float:
        # Accept either:
        #   (A) a full 0/1 mask array of length L
        #   (B) an iterable of selected indices (size k)
        # For large L (e.g. end_level=NPA4), (B) avoids an O(L) scan per objective eval.
        if isinstance(mask, np.ndarray):
            mask_arr = mask
            chosen_indices = np.flatnonzero(mask_arr)
            chosen = [self.adding_set[int(i)] for i in chosen_indices]
        else:
            chosen = [self.adding_set[int(i)] for i in mask]
        basis = self.starting_set + chosen
        
        lb = solve_pauli_relaxation(
            basis,
            self.hamiltonian_dict,
            symmetry_manager=self.symmetry_manager,
            sense="min",
            mosek_tol=self.mosek_tol,
            verbose=False,
        )

        # optimalsdp's Monte Carlo routines are minimizers.
        # Minimize loss = -LB to maximize the lower bound.
        return -float(lb)


def make_objective_function(
    starting_set: List,
    adding_set: List,
    hamiltonian_dict: Dict,
    symmetry_manager: SymmetryManager,
    mosek_tol: float = 1e-9,
) -> OptimizationObjective:
    """Create the objective function for optimization."""
    return OptimizationObjective(
        starting_set=starting_set,
        adding_set=adding_set,
        hamiltonian_dict=hamiltonian_dict,
        symmetry_manager=symmetry_manager,
        mosek_tol=mosek_tol,
    )


# -----------------------------------------------------------------------------
# Optimization runner
# -----------------------------------------------------------------------------

def run_single_optimization(
    obj_func: callable,
    L: int,
    k: int,
    seed: int,
    method: str,
    method_params: Dict[str, Any],
    initial_guess: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run a single optimization and return results.
    
    Args:
        obj_func: Objective function (mask -> float).
        L: Length of mask (= len(adding_set)).
        k: Number of monomials to select (Hamming weight).
        seed: Random seed.
        method: "sa", "pt", "bo", or "random".
        method_params: Method-specific parameters.
        initial_guess: Optional mask from a previous k to warm-start from.
            SA and BO use it directly; PT uses it to seed the coldest chain.
    
    Returns:
        Dict with keys: best_value, mask, elapsed_s, n_obj_evals
    """
    
    t0 = time.perf_counter()
    
    if method == "sa":
        result = simulated_annealing(
            obj_func=obj_func,
            N=L,
            k=k,
            initial_guess=initial_guess,
            steps=method_params.get("steps", 100),
            T_start=method_params.get("T_start", 2.0),
            alpha=method_params.get("alpha", 0.95),
            record_history=False,
            seed=seed,
            verbose=False,
            obj_uses_indices=True,
        )
        n_obj_evals = method_params.get("steps", 100) + 1  # +1 for initial eval
        
    elif method == "pt":
        pt_num_chains = method_params.get("num_chains", None)

        result = parallel_tempering(
            obj_func=obj_func,
            N=L,
            k=k,
            num_chains=pt_num_chains,
            num_epochs=method_params.get("num_epochs", 10),
            steps_per_epoch=method_params.get("steps_per_epoch", 50),
            T_min=method_params.get("T_min", 0.01),
            T_max=method_params.get("T_max", 2.0),
            initial_guess=initial_guess,
            seed=seed,
            verbose=False,
            obj_uses_indices=True,
        )
        # Approximate: chains * epochs * steps_per_epoch + initial evals
        n_chains_eval = pt_num_chains
        if n_chains_eval is None or int(n_chains_eval) <= 0:
            n_chains_eval = os.cpu_count() or 1
        else:
            n_chains_eval = int(n_chains_eval)

        n_epochs = method_params.get("num_epochs", 10)
        steps_per = method_params.get("steps_per_epoch", 50)
        n_obj_evals = n_chains_eval * n_epochs * steps_per + n_chains_eval

    elif method == "bo":
        # Bayesian optimization

        try:
            from src.optimalsdp.bayesian import bayesian as bayesian_optimization
        except ImportError as e:
            raise ImportError(
                "Bayesian optimization requires scikit-learn. Install with `pip install scikit-learn` "
                "(or `pip install -r requirements-dev.txt`), or use --method sa/pt/random."
            ) from e

        n_init = int(method_params.get("n_init", 20))
        n_iter = int(method_params.get("n_iter", 50))
        candidates_per_iter = int(method_params.get("candidates_per_iter", 100))
        beta = float(method_params.get("beta", 1.0))

        bo_result = bayesian_optimization(
            obj_func=obj_func,
            N=L,
            k=k,
            beta=beta,
            n_init=n_init,
            n_iter=n_iter,
            candidates_per_iter=candidates_per_iter,
            previous_best=initial_guess,
            seed=seed,
            verbose=False,
            obj_uses_indices=True,
        )

        # BO evaluates the objective once per initial sample and once per iteration
        n_obj_evals = n_init + n_iter

    elif method == "rbm":
        # RBM-based REINFORCE optimization
        try:
            from src.optimalsdp.rbm import RBMTrainer
        except ImportError as e:
            raise ImportError(
                "RBM optimization requires JAX, Equinox, and Optax. "
                "Install with `pip install jax equinox optax` "
                "(or `pip install -r requirements-dev.txt`), or use --method sa/pt/bo/random."
            ) from e

        import jax.numpy as jnp

        rbm_steps = int(method_params.get("steps", 100))

        # Wrap obj_func so that JAX arrays are converted to NumPy before
        # reaching OptimizationObjective (which uses isinstance(mask, np.ndarray)).
        _raw_obj = obj_func
        def _numpy_obj(v):
            return _raw_obj(np.asarray(v))

        trainer = RBMTrainer(
            obj_func=_numpy_obj,
            N=L,
            hamming_weight=k,
            steps=rbm_steps,
            seed=seed if seed is not None else 42,
        )
        trainer.train(num_steps=rbm_steps, verbose=False)

        best_mask = np.asarray(trainer.current_vec, dtype=np.int32)
        best_loss = float(trainer.current_cost)
        best_lb = -best_loss
        elapsed = time.perf_counter() - t0

        return {
            "best_value": best_lb,
            "mask": best_mask,
            "elapsed_s": elapsed,
            "n_obj_evals": rbm_steps + 1,  # +1 for initial eval
        }

    elif method == "random":
        # Random sampling baseline: just pick k random positions and evaluate once.
        # No optimization - this is a baseline for comparison.
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Generate random mask with exactly k ones
        mask = np.zeros(L, dtype=np.int32)
        if 0 < k <= L:
            chosen_indices = random.sample(range(L), k)
            mask[chosen_indices] = 1

        # Evaluate objective (returns negative of lower bound)
        loss = obj_func(mask)
        best_lb = -float(loss)

        elapsed = time.perf_counter() - t0

        return {
            "best_value": best_lb,
            "mask": mask,
            "elapsed_s": elapsed,
            "n_obj_evals": 1,  # Single evaluation for random sampling
        }

    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed = time.perf_counter() - t0

    if method in {"sa", "pt"}:
        best_loss = float(result["best"]["value"])
        best_mask = np.asarray(result["best"]["selection"], dtype=np.int32)
    elif method == "bo":
        best_loss = float(bo_result["best_value"])
        best_mask = np.asarray(bo_result["best_selection"], dtype=np.int32)
    # Note: "random" returns early above, so we don't handle it here

    best_lb = -best_loss

    return {
        "best_value": float(best_lb),
        "mask": best_mask,
        "elapsed_s": elapsed,
        "n_obj_evals": n_obj_evals,
    }


# -----------------------------------------------------------------------------
# Main compute and save
# -----------------------------------------------------------------------------

def compute_and_save(
    *,
    N: int,
    model_name: str,
    model_params: Dict[str, float],
    boundary: str,
    start_level: int,
    end_level: int,
    end_basis: str = "npa",
    method: str,
    method_params: Dict[str, Any],
    k_values: List[int],
    seeds: List[int],
    symmetry_config: Dict[str, bool],
    mosek_tol: float,
    out_root: Path,
    resume: bool,
    force: bool,
    verbose: bool,
    feedback: bool = False,
) -> Path:
    """Run optimization sweep and save results.
    
    Args:
        N: Number of sites.
        model_name: "ising", "heisenberg", or "heisenberg_j2".
        model_params: Model-specific parameters.
        boundary: "open" or "periodic".
        start_level: NPA level for starting set.
        end_level: NPA level for final set.
        method: "sa" or "pt".
        method_params: Optimization hyperparameters.
        k_values: List of k values to sweep.
        seeds: List of seeds for repetitions.
        symmetry_config: Symmetry configuration dict (excludes N).
        mosek_tol: Solver tolerance.
        out_root: Root directory for results.
        resume: If True, skip already-computed runs.
        force: If True, recompute even if present.
        verbose: Print progress.
        feedback: If True, chain k-values per seed: the best mask found at
            k_i is passed as initial_guess to k_{i+1}.
    
    Returns:
        Path to the run directory.
    """
    starting_set, adding_set, final_set = build_basis_sets(
        N=N, start_level=start_level, end_basis=end_basis, end_level=end_level,
    )
    L = len(adding_set)
    
    if verbose:
        print(f"Model: {model_name}, N={N}, boundary={boundary}, end_basis={end_basis}")
        print(f"Starting set size: {len(starting_set)}, Adding set size: {L}, Final set size: {len(final_set)}")
        if feedback:
            print("Feedback mode: ON (warm-start chaining across k values)")

    # Create fingerprint for adding_set (avoid slow stringification for large L)
    adding_set_hash = _stable_pauliword_mask_hash(adding_set, N=N)
    
    # Build config for hashing
    config: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT_NAME,
        "model": model_name,
        "model_params": dict(sorted(model_params.items())),
        "N": int(N),
        "boundary": boundary,
        "start_level": int(start_level),
        "end_level": int(end_level),
        "end_basis": end_basis,
        "starting_set_size": len(starting_set),
        "adding_set_size": L,
        "adding_set_hash": adding_set_hash,
        "method": method,
        "method_params": dict(sorted(method_params.items())),
        "symmetry": symmetry_config,
        "mosek_tol": float(mosek_tol),
        "feedback": bool(feedback),
    }
    
    cfg_hash = config_hash(config)
    run_dir = out_root / ARTIFACT_NAME / f"v{SCHEMA_VERSION}" / cfg_hash
    meta_path = run_dir / "meta.json"
    data_path = run_dir / "data.npz"
    
    # Load existing runs
    # - `force=True` should *recompute* the requested grid and overwrite the artifact,
    #   not append duplicates (which makes "Total runs" misleading).
    if force:
        completed, existing_arrays = {}, {}
    else:
        completed, existing_arrays = _load_existing_runs(data_path) if resume else ({}, {})
    
    # Determine which (k, seed) pairs to compute
    #
    # When feedback=True the jobs for each seed MUST be processed in ascending k
    # order so that the best mask at k_i can warm-start k_{i+1}.  Resuming a
    # partially-completed feedback sweep is supported: we skip (k, seed) pairs
    # already present, but we still need to replay the mask chain for the
    # earlier k values.  We achieve this cheaply by loading the stored mask for
    # already-completed points but not re-solving the SDP.

    sorted_k_values = sorted(k_values)
    all_jobs = [(kv, s) for kv in sorted_k_values for s in seeds]
    if force:
        missing_jobs_set: Set[Tuple[int, int]] = set(all_jobs)
    else:
        missing_jobs_set = {(kv, s) for kv, s in all_jobs if (kv, s) not in completed}
    
    if verbose:
        print(
            f"Total jobs requested: {len(all_jobs)}, "
            f"Already in artifact: {len(completed)}, "
            f"To compute now: {len(missing_jobs_set)}"
        )
    
    if not missing_jobs_set:
        if verbose:
            print("All jobs already completed. Nothing to do.")
        return run_dir
    
    # Build Hamiltonian and objective
    H_dict_fn = hamiltonian_dict_fn(model_name)
    hamiltonian_dict = H_dict_fn(N=N, boundary=boundary, **model_params)
    
    sym_manager = SymmetryManager(N=N, **symmetry_config)
    obj_func = make_objective_function(
        starting_set=starting_set,
        adding_set=adding_set,
        hamiltonian_dict=hamiltonian_dict,
        symmetry_manager=sym_manager,
        mosek_tol=mosek_tol,
    )
    
    # Initialize result arrays from existing data
    if existing_arrays:
        run_idx_list = list(existing_arrays.get("run_idx", []))
        k_list = list(existing_arrays.get("k", []))
        seed_list = list(existing_arrays.get("seed", []))
        best_value_list = list(existing_arrays.get("best_value", []))
        elapsed_s_list = list(existing_arrays.get("elapsed_s", []))
        n_obj_evals_list = list(existing_arrays.get("n_obj_evals", []))
        mask_bits_list = list(existing_arrays.get("mask_bits", []))
        next_run_idx = max(run_idx_list) + 1 if run_idx_list else 0
    else:
        run_idx_list = []
        k_list = []
        seed_list = []
        best_value_list = []
        elapsed_s_list = []
        n_obj_evals_list = []
        mask_bits_list = []
        next_run_idx = 0
    
    # Run missing jobs
    n_computed = 0
    total_to_compute = len(missing_jobs_set)
    pbar = tqdm(total=total_to_compute, desc="Optimization sweep", disable=not verbose)

    # Build metadata template once; we'll update counters and timestamps as we go.
    meta: Dict[str, Any] = {
        **config,
        "config_hash": cfg_hash,
        "k_values_requested": [int(x) for x in sorted(k_values)],
        "seeds_requested": [int(x) for x in sorted(seeds)],
        "k_values_present": [],
        "seeds_present": [],
        "total_runs": len(run_idx_list),
        "L": int(L),  # For decoding mask_bits
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }

    # Ensure meta.json exists early and is always updated
    # even if interrupted. Data is checkpointed every completed run.
    upsert_meta_json(meta_path, meta)

    # Helper: load a previously-computed mask from existing_arrays for feedback
    # chaining (avoid re-solving the SDP).
    def _load_existing_mask(kv: int, s: int) -> Optional[np.ndarray]:
        """Return the stored mask for an already-completed (k, seed) pair."""
        if not existing_arrays or "mask_bits" not in existing_arrays:
            return None
        run_idx_target = completed.get((kv, s))
        if run_idx_target is None:
            return None
        # Find position of that run_idx in existing arrays
        for pos, rid in enumerate(existing_arrays["run_idx"]):
            if int(rid) == run_idx_target:
                packed_row = existing_arrays["mask_bits"][pos]
                if isinstance(packed_row, np.ndarray):
                    return np.unpackbits(packed_row)[:L].astype(np.int32)
                return None
        return None

    def _checkpoint() -> None:
        """Atomically save all accumulated results to disk."""
        if mask_bits_list:
            mb = np.stack(mask_bits_list, axis=0)
        else:
            mb = np.array([], dtype=np.uint8).reshape(0, 0)
        _save_runs(
            data_path=data_path,
            run_idx=run_idx_list,
            k=k_list,
            seed=seed_list,
            best_value=best_value_list,
            elapsed_s=elapsed_s_list,
            n_obj_evals=n_obj_evals_list,
            mask_bits=mb,
        )

    try:
        if feedback:
            # --- Feedback mode: iterate by seed, then by ascending k ---
            for s in seeds:
                prev_mask: Optional[np.ndarray] = None
                for kv in sorted_k_values:
                    if (kv, s) in missing_jobs_set:
                        pbar.set_postfix({"k": kv, "seed": s})
                        result = run_single_optimization(
                            obj_func=obj_func,
                            L=L,
                            k=kv,
                            seed=s,
                            method=method,
                            method_params=method_params,
                            initial_guess=prev_mask,
                        )
                        prev_mask = result["mask"]

                        packed = np.packbits(result["mask"].astype(np.uint8))
                        run_idx_list.append(next_run_idx)
                        k_list.append(kv)
                        seed_list.append(s)
                        best_value_list.append(result["best_value"])
                        elapsed_s_list.append(result["elapsed_s"])
                        n_obj_evals_list.append(result["n_obj_evals"])
                        mask_bits_list.append(packed)
                        next_run_idx += 1
                        n_computed += 1
                        pbar.update(1)

                        _checkpoint()
                    else:
                        # Already completed — load mask for chaining
                        prev_mask = _load_existing_mask(kv, s)
        else:
            # --- Independent mode: process jobs in given order ---
            for kv, s in [(kv, s) for kv in sorted_k_values for s in seeds]:
                if (kv, s) not in missing_jobs_set:
                    continue
                pbar.set_postfix({"k": kv, "seed": s})

                result = run_single_optimization(
                    obj_func=obj_func,
                    L=L,
                    k=kv,
                    seed=s,
                    method=method,
                    method_params=method_params,
                )

                packed = np.packbits(result["mask"].astype(np.uint8))
                run_idx_list.append(next_run_idx)
                k_list.append(kv)
                seed_list.append(s)
                best_value_list.append(result["best_value"])
                elapsed_s_list.append(result["elapsed_s"])
                n_obj_evals_list.append(result["n_obj_evals"])
                mask_bits_list.append(packed)
                next_run_idx += 1
                n_computed += 1
                pbar.update(1)

                _checkpoint()
    finally:
        pbar.close()
        # Update metadata to reflect whatever is safely on disk.
        meta["k_values_present"] = sorted(set(int(x) for x in k_list))
        meta["seeds_present"] = sorted(set(int(x) for x in seed_list))
        meta["total_runs"] = int(len(run_idx_list))
        meta["updated_at"] = utc_now_iso()
        upsert_meta_json(meta_path, meta)
    
    if verbose:
        print(f"\nResults saved to: {run_dir}")
        print(
            f"Total runs in artifact: {len(run_idx_list)} "
            f"(requested this call: {len(all_jobs)}, computed this call: {n_computed})"
        )
    
    return run_dir


# -----------------------------------------------------------------------------
# Result loading utilities
# -----------------------------------------------------------------------------

def load_optimization_results(
    run_dir: Path | str,
    unpack_masks: bool = True,
) -> Dict[str, Any]:
    """Load optimization results from a run directory.
    
    Args:
        run_dir: Path to the run directory.
        unpack_masks: If True, unpack bit-packed masks to bool arrays.
    
    Returns:
        Dict with:
            - meta: metadata dict
            - data: dict of arrays (run_idx, k, seed, best_value, elapsed_s, n_obj_evals, masks)
    """
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    data_path = run_dir / "data.npz"
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    data = dict(np.load(data_path, allow_pickle=False))
    
    if unpack_masks and "mask_bits" in data:
        L = meta.get("L", meta.get("adding_set_size"))
        data["masks"] = _unpack_masks(data["mask_bits"], L)
        del data["mask_bits"]
    
    return {"meta": meta, "data": data}


def results_to_dataframe(run_dir: Path | str) -> "pd.DataFrame":
    """Load results as a pandas DataFrame.
    
    Args:
        run_dir: Path to the run directory.
    
    Returns:
        DataFrame with columns: run_idx, k, seed, best_value, elapsed_s, n_obj_evals
        (masks not included for simplicity)
    """
    results = load_optimization_results(run_dir, unpack_masks=False)
    data = results["data"]

    pd = _require_pandas()
    
    df = pd.DataFrame({
        "run_idx": data["run_idx"],
        "k": data["k"],
        "seed": data["seed"],
        "best_value": data["best_value"],
        "elapsed_s": data["elapsed_s"],
        "n_obj_evals": data["n_obj_evals"],
    })
    
    return df


def get_best_per_k(run_dir: Path | str) -> Dict[int, Dict[str, Any]]:
    """Get the best result for each k value.
    
    Args:
        run_dir: Path to the run directory.
    
    Returns:
        Dict mapping k -> {best_value, seed, mask, run_idx}
    """
    results = load_optimization_results(run_dir, unpack_masks=True)
    data = results["data"]
    
    best_by_k = {}
    
    for i in range(len(data["k"])):
        k = int(data["k"][i])
        val = float(data["best_value"][i])
        
        if k not in best_by_k or val > best_by_k[k]["best_value"]:
            best_by_k[k] = {
                "best_value": val,
                "seed": int(data["seed"][i]),
                "mask": data["masks"][i] if "masks" in data else None,
                "run_idx": int(data["run_idx"][i]),
            }
    
    return best_by_k


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model parameters
    p.add_argument("--model", choices=["ising", "heisenberg", "heisenberg_j2"], default="heisenberg")
    p.add_argument("--N", type=int, required=True, help="Number of sites")
    p.add_argument("--boundary", choices=["open", "periodic"], default="periodic")
    
    # Ising params
    p.add_argument("--J", type=float, default=1.0, help="Ising J parameter")
    p.add_argument("--h", type=float, default=0.0, help="Ising h parameter")
    p.add_argument("--k-ising", type=float, default=0.0, dest="k_ising", help="Ising k parameter")
    
    # Heisenberg J2 param
    p.add_argument("--J2", type=float, default=0.0, help="Heisenberg J2 parameter")
    
    # Basis parameters
    p.add_argument("--start-level", type=int, default=1, help="NPA level for starting set")
    p.add_argument("--end-level", type=int, default=2, help="NPA level for final set (only used when --end-basis=npa)")
    p.add_argument(
        "--end-basis",
        type=str,
        default="npa",
        choices=["npa", "heisenberg_simple", "heisenberg_j2_weak", "heisenberg_j2_strong"],
        help="Basis type for the final (full) set. 'npa' uses --end-level; others use named generators.",
    )
    
    # Optimization method
    p.add_argument(
        "--method",
        choices=["sa", "pt", "bo", "rbm", "random"],
        default="sa",
        help="sa=simulated annealing, pt=parallel tempering, bo=bayesian optimization, rbm=RBM REINFORCE, random=random sampling baseline",
    )
    
    # SA parameters
    p.add_argument("--sa-steps", type=int, default=100, help="SA: number of steps")
    p.add_argument("--sa-T-start", type=float, default=2.0, help="SA: starting temperature")
    p.add_argument("--sa-alpha", type=float, default=0.95, help="SA: temperature decay factor")
    
    # PT parameters
    p.add_argument(
        "--pt-chains",
        type=int,
        default=0,
        help="PT: number of chains (0 = auto = use all CPUs)",
    )
    p.add_argument("--pt-epochs", type=int, default=5, help="PT: number of epochs")
    p.add_argument("--pt-steps-per-epoch", type=int, default=40, help="PT: steps per epoch")
    p.add_argument("--pt-T-min", type=float, default=0.1, help="PT: minimum temperature")
    p.add_argument("--pt-T-max", type=float, default=2.0, help="PT: maximum temperature")

    # BO parameters
    p.add_argument("--bo-beta", type=float, default=1.0, help="BO: UCB exploration parameter (beta)")
    p.add_argument("--bo-n-init", type=int, default=20, help="BO: number of initial random samples")
    p.add_argument("--bo-n-iter", type=int, default=50, help="BO: number of BO iterations")
    p.add_argument(
        "--bo-candidates-per-iter",
        type=int,
        default=100,
        help="BO: number of candidate masks scored by the surrogate per iteration",
    )

    # RBM parameters
    p.add_argument("--rbm-steps", type=int, default=100, help="RBM: number of REINFORCE training steps")

    # Feedback (warm-start chaining)
    p.add_argument(
        "--feedback",
        action="store_true",
        default=False,
        help="Chain k-values: use best mask at k_i as initial_guess for k_{i+1}",
    )

    # Sweep parameters
    k_group = p.add_mutually_exclusive_group(required=True)
    k_group.add_argument("--ks", nargs="+", type=int, help="Explicit list of k values")
    k_group.add_argument("--k-max", type=int, help="Sweep k from 0 to k-max (inclusive)")
    
    seed_group = p.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seeds", nargs="+", type=int, help="Explicit list of seeds")
    seed_group.add_argument("--num-seeds", type=int, help="Number of seeds (starting from 42)")
    
    # Solver parameters
    p.add_argument("--mosek-tol", type=float, default=1e-9)

    add_symmetry_args(p)
    
    # Output parameters
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Root directory for results (default: spins_sdp/results)",
    )
    
    # Resume/force
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true", help="Recompute all runs")
    p.add_argument("--verbose", action="store_true", default=True)
    
    args = p.parse_args(argv)
    
    # Build k values
    if args.ks is not None:
        k_values = sorted(set(args.ks))
    else:
        k_values = list(range(args.k_max + 1))
    
    # Build seeds
    if args.seeds is not None:
        seeds = sorted(set(args.seeds))
    else:
        seeds = list(range(42, 42 + args.num_seeds))
    
    # Build method params
    if args.method == "sa":
        method_params = {
            "steps": args.sa_steps,
            "T_start": args.sa_T_start,
            "alpha": args.sa_alpha,
        }
    elif args.method == "pt":
        pt_chains = None if int(args.pt_chains) <= 0 else int(args.pt_chains)
        method_params = {
            "num_chains": pt_chains,
            "num_epochs": args.pt_epochs,
            "steps_per_epoch": args.pt_steps_per_epoch,
            "T_min": args.pt_T_min,
            "T_max": args.pt_T_max,
        }
    elif args.method == "bo":
        method_params = {
            "beta": args.bo_beta,
            "n_init": args.bo_n_init,
            "n_iter": args.bo_n_iter,
            "candidates_per_iter": args.bo_candidates_per_iter,
        }
    elif args.method == "rbm":
        method_params = {
            "steps": args.rbm_steps,
        }
    else:
        # random method has no hyperparameters
        method_params = {}
    
    model_params = model_params_from_args(args)
    symmetry_config = symmetry_config_from_args(args)

    run_dir = compute_and_save(
        N=args.N,
        model_name=args.model,
        model_params=model_params,
        boundary=args.boundary,
        start_level=args.start_level,
        end_level=args.end_level,
        end_basis=args.end_basis,
        method=args.method,
        method_params=method_params,
        k_values=k_values,
        seeds=seeds,
        symmetry_config=symmetry_config,
        mosek_tol=args.mosek_tol,
        out_root=args.out_root,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
        feedback=getattr(args, "feedback", False),
    )
    
    print(f"Results saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
