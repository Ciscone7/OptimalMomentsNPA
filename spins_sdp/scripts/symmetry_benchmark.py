"""Benchmark SDP solve time with different symmetry configurations.

This script measures how different symmetry options affect SDP solve time
for spin-chain models. It's designed to answer: "How much speedup do I get
from each symmetry for a given model and system size?"

Results are saved in a resumable way:
  - Results are grouped by a hash of the configuration.
  - If data.npz already exists, only missing (N, sym_level) pairs are computed.

On-disk layout (default):
    spins_sdp/results/spin_symmetry_benchmark/v1/<config_hash>/
    ├── meta.json
    └── data.npz

`data.npz` contains arrays:
  - N (int)                   # system size
  - sym_level (int)           # symmetry level (0-6)
  - basis_size (int)          # number of basis words
  - n_vars (int)              # number of SDP variables after symmetry
  - energy (float)            # SDP objective value
  - t_total (float)           # total solve time (seconds)
  - t_best (float)            # best of repeats
  - t_avg (float)             # average of repeats

Symmetry levels:
  0: No symmetries
  1: +Real operator
  2: +Rotation (signature block diagonal)
  3: +Sign symmetry
  4: +Permutation (XYZ relabeling)
  5: +Translation
  6: +Mirror (all symmetries)

Examples:
    python -m spins_sdp.scripts.symmetry_benchmark \\
        --model heisenberg --npa-level 2 \\
        --N-min 2 --N-max 20 --repeats 3
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from spins_sdp.basis_builder import (
    generate_npa_basis,
    generate_heisenberg_paper_basis,
    generate_heisenberg_j2_basis_weak,
    generate_heisenberg_j2_basis_strong,
)
from spins_sdp.models import heisenberg_hamiltonian_dict, ising_hamiltonian_dict
from spins_sdp.sdp import solve_pauli_relaxation, build_block_reps
from spins_sdp.symmetry import SymmetryManager
from spins_sdp.scripts._artifact_io import (
    config_hash,
    atomic_save_npz,
    upsert_meta_json,
    utc_now_iso,
)
from spins_sdp.scripts._common import (
    model_params_from_args,
    parse_ns_from_args,
    time_best_avg,
)


SCHEMA_VERSION = 1
ARTIFACT_NAME = "spin_symmetry_benchmark"

# Symmetry level definitions (cumulative)
SYM_LEVEL_NAMES = {
    0: "None",
    1: "+Real",
    2: "+Rotation",
    3: "+Sign",
    4: "+Permutation",
    5: "+Translation",
    6: "+Mirror (All)",
}

# Max N thresholds for each symmetry level (empirically determined)
DEFAULT_MAX_N = {0: 8, 1: 9, 2: 12, 3: 12, 4: 16, 5: 20, 6: 30}

# Supported basis names
SUPPORTED_BASES = ["npa", "heisenberg_simple", "heisenberg_j2_weak", "heisenberg_j2_strong"]


def generate_basis(basis_name: str, N: int, npa_level: int, boundary: str) -> List:
    """Generate basis for given configuration.
    
    Args:
        basis_name: One of 'npa', 'heisenberg_simple', 'heisenberg_j2_weak', 'heisenberg_j2_strong'.
        N: System size.
        npa_level: NPA level (only used if basis_name == 'npa').
        boundary: Boundary conditions ('open' or 'periodic').
    
    Returns:
        List of basis elements (PauliWords).
    """
    # Heisenberg paper bases require periodic boundary
    if basis_name in {"heisenberg_simple", "heisenberg_j2_weak", "heisenberg_j2_strong"} and boundary != "periodic":
        raise ValueError(f"basis={basis_name} requires periodic boundary conditions")
    
    if basis_name == "npa":
        return generate_npa_basis(N=N, k=npa_level).words
    elif basis_name == "heisenberg_simple":
        return generate_heisenberg_paper_basis(N=N)
    elif basis_name == "heisenberg_j2_weak":
        return generate_heisenberg_j2_basis_weak(N=N)
    elif basis_name == "heisenberg_j2_strong":
        return generate_heisenberg_j2_basis_strong(N=N)
    else:
        raise ValueError(f"Unknown basis: {basis_name}. Supported: {SUPPORTED_BASES}")


def get_symmetry_manager(N: int, level: int) -> SymmetryManager:
    """Get SymmetryManager for a given cumulative symmetry level."""
    if level == 0:
        return SymmetryManager(N=N)
    elif level == 1:
        return SymmetryManager(N=N, use_real_operator=True)
    elif level == 2:
        return SymmetryManager(N=N, use_real_operator=True, use_rotation=True)
    elif level == 3:
        return SymmetryManager(N=N, use_real_operator=True, use_rotation=True, use_sign_symmetry=True)
    elif level == 4:
        return SymmetryManager(N=N, use_real_operator=True, use_rotation=True, use_sign_symmetry=True, use_permutation=True)
    elif level == 5:
        return SymmetryManager(N=N, use_real_operator=True, use_rotation=True, use_sign_symmetry=True, use_permutation=True, use_translation=True)
    else:  # level == 6 or higher
        return SymmetryManager.default_for_heisenberg(N)


def count_sdp_vars(basis: List, sym_manager: SymmetryManager) -> int:
    """Count the number of independent SDP variables after symmetry reduction."""
    _, global_index = build_block_reps(basis, sym_manager)
    return len(set(global_index.values()))


def compute_and_save(
    *,
    Ns: Iterable[int],
    basis_name: str,
    npa_level: int,
    model_name: str,
    model_params: Dict[str, float],
    boundary: str,
    sym_levels: List[int],
    max_N_per_level: Dict[int, int],
    repeats: int,
    mosek_tol: float,
    out_root: Path,
    resume: bool,
    force: bool,
    verbose: bool,
) -> Path:
    """Run symmetry benchmark and save results."""
    
    config: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT_NAME,
        "model": model_name,
        "params": dict(model_params),
        "boundary": boundary,
        "basis": basis_name,
        "npa_level": int(npa_level) if basis_name == "npa" else None,
        "sym_levels": sorted(sym_levels),
        "mosek_tol": float(mosek_tol),
        "repeats": int(repeats),
    }
    
    # Remove npa_level from config if not using NPA basis
    if basis_name != "npa":
        config.pop("npa_level", None)
    
    cfg_hash = config_hash(config)
    run_dir = out_root / ARTIFACT_NAME / f"v{SCHEMA_VERSION}" / cfg_hash
    meta_path = run_dir / "meta.json"
    data_path = run_dir / "data.npz"
    
    # Load existing data
    existing: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if resume and data_path.exists() and not force:
        with np.load(data_path, allow_pickle=False) as data:
            if "N" in data and "sym_level" in data:
                for i in range(len(data["N"])):
                    key = (int(data["N"][i]), int(data["sym_level"][i]))
                    existing[key] = {
                        "basis_size": int(data["basis_size"][i]),
                        "n_vars": int(data["n_vars"][i]),
                        "energy": float(data["energy"][i]),
                        "t_total": float(data["t_total"][i]),
                        "t_best": float(data["t_best"][i]),
                        "t_avg": float(data["t_avg"][i]),
                    }
    
    # Determine jobs
    Ns_list = sorted(set(int(n) for n in Ns))
    all_jobs = [(N, lvl) for N in Ns_list for lvl in sym_levels]
    missing_jobs = [(N, lvl) for N, lvl in all_jobs 
                    if (N, lvl) not in existing or force]
    
    # Filter by max_N thresholds
    missing_jobs = [(N, lvl) for N, lvl in missing_jobs 
                    if N <= max_N_per_level.get(lvl, 999)]
    
    if verbose:
        basis_desc = f"NPA level {npa_level}" if basis_name == "npa" else basis_name
        print(f"Model: {model_name}, Basis: {basis_desc}, boundary: {boundary}")
        print(f"Total jobs: {len(all_jobs)}, Already done: {len(existing)}, To compute: {len(missing_jobs)}")
    
    # Get Hamiltonian function
    if model_name == "heisenberg":
        H_fn = heisenberg_hamiltonian_dict
    elif model_name == "ising":
        H_fn = ising_hamiltonian_dict
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Build metadata
    meta: Dict[str, Any] = {
        **config,
        "config_hash": cfg_hash,
        "max_N_per_level": max_N_per_level,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "Ns_present": sorted(set(k[0] for k in existing.keys())),
    }
    
    run_dir.mkdir(parents=True, exist_ok=True)
    upsert_meta_json(meta_path, meta)
    
    try:
        pbar = tqdm(missing_jobs, desc="Symmetry benchmark", disable=False)
        for N, sym_level in pbar:
            #if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({"N": N, "sym": sym_level})
            
            # Generate basis and Hamiltonian
            basis = generate_basis(basis_name, N, npa_level, boundary)
            ham = H_fn(N=N, boundary=boundary, **model_params)
            sym_manager = get_symmetry_manager(N, sym_level)
            
            # Count variables
            n_vars = count_sdp_vars(basis, sym_manager)
            
            # Time the solve
            def run_one():
                return solve_pauli_relaxation(
                    basis, ham,
                    symmetry_manager=sym_manager,
                    sense="min",
                    mosek_tol=mosek_tol,
                    verbose=False,
                )
            
            energy, t_best, t_avg = time_best_avg(run_one, repeats=repeats)
            
            existing[(N, sym_level)] = {
                "basis_size": len(basis),
                "n_vars": n_vars,
                "energy": float(energy),
                "t_total": t_avg,  # use avg as default
                "t_best": t_best,
                "t_avg": t_avg,
            }
            
            # Checkpoint
            _save_results(data_path, existing)
            
            meta["Ns_present"] = sorted(set(k[0] for k in existing.keys()))
            meta["updated_at"] = utc_now_iso()
            upsert_meta_json(meta_path, meta)
            
    finally:
        meta["Ns_present"] = sorted(set(k[0] for k in existing.keys()))
        meta["updated_at"] = utc_now_iso()
        upsert_meta_json(meta_path, meta)
    
    return run_dir


def _save_results(data_path: Path, records: Dict[Tuple[int, int], Dict[str, Any]]) -> None:
    """Save results to NPZ file."""
    if not records:
        return
    
    keys = sorted(records.keys())
    N_arr = np.array([k[0] for k in keys], dtype=np.int32)
    sym_level_arr = np.array([k[1] for k in keys], dtype=np.int32)
    basis_size_arr = np.array([records[k]["basis_size"] for k in keys], dtype=np.int64)
    n_vars_arr = np.array([records[k]["n_vars"] for k in keys], dtype=np.int64)
    energy_arr = np.array([records[k]["energy"] for k in keys], dtype=np.float64)
    t_total_arr = np.array([records[k]["t_total"] for k in keys], dtype=np.float64)
    t_best_arr = np.array([records[k]["t_best"] for k in keys], dtype=np.float64)
    t_avg_arr = np.array([records[k]["t_avg"] for k in keys], dtype=np.float64)
    
    atomic_save_npz(
        data_path,
        N=N_arr,
        sym_level=sym_level_arr,
        basis_size=basis_size_arr,
        n_vars=n_vars_arr,
        energy=energy_arr,
        t_total=t_total_arr,
        t_best=t_best_arr,
        t_avg=t_avg_arr,
    )


def load_symmetry_benchmark(run_dir: Path) -> Dict[str, Any]:
    """Load symmetry benchmark results from a run directory."""
    import json
    
    meta_path = run_dir / "meta.json"
    data_path = run_dir / "data.npz"
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    data = dict(np.load(data_path, allow_pickle=False))
    
    return {"meta": meta, "data": data}


def find_symmetry_benchmark(
    *,
    model: str = "heisenberg",
    basis: str = "npa",
    npa_level: Optional[int] = 2,
    boundary: str = "periodic",
    results_root: Optional[Path] = None,
) -> Optional[Path]:
    """Find a symmetry benchmark run matching the given parameters.
    
    Args:
        model: Model name ('heisenberg' or 'ising').
        basis: Basis name ('npa', 'heisenberg_simple', 'heisenberg_j2_weak', 'heisenberg_j2_strong').
        npa_level: NPA level (only used when basis='npa').
        boundary: Boundary condition ('open' or 'periodic').
        results_root: Root directory for results.
    
    Returns:
        Path to run directory if found, None otherwise.
    """
    if results_root is None:
        results_root = Path(__file__).resolve().parents[1] / "results"
    
    base = results_root / ARTIFACT_NAME / f"v{SCHEMA_VERSION}"
    if not base.exists():
        return None
    
    import json
    
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        # Match model and boundary
        if meta.get("model") != model or meta.get("boundary") != boundary:
            continue
        
        # Match basis
        if meta.get("basis") != basis:
            continue
        
        # For NPA basis, also match npa_level
        if basis == "npa" and meta.get("npa_level") != npa_level:
            continue
        
        return run_dir
    
    return None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # N values
    n_group = p.add_mutually_exclusive_group(required=True)
    n_group.add_argument("--Ns", nargs="+", type=int, help="Explicit list of N values")
    n_group.add_argument("--N-min", dest="N_min", type=int, help="Minimum N (inclusive)")
    p.add_argument("--N-max", dest="N_max", type=int, help="Maximum N (inclusive)")
    
    # Model and basis
    p.add_argument("--model", choices=["ising", "heisenberg"], default="heisenberg")
    p.add_argument("--basis", choices=SUPPORTED_BASES, default="npa",
                   help="Basis type: 'npa' (with --npa-level), 'heisenberg_simple', 'heisenberg_j2_weak', 'heisenberg_j2_strong'")
    p.add_argument("--npa-level", type=int, default=2, help="NPA level (only used when --basis=npa)")
    p.add_argument("--boundary", choices=["open", "periodic"], default="periodic")
    
    # Ising params
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--k", type=float, default=0.0)
    
    # Symmetry levels
    p.add_argument("--sym-levels", nargs="+", type=int, default=list(range(7)),
                   help="Symmetry levels to test (0-6)")
    
    # Max N per level (can override defaults)
    p.add_argument("--max-N-0", type=int, default=DEFAULT_MAX_N[0])
    p.add_argument("--max-N-1", type=int, default=DEFAULT_MAX_N[1])
    p.add_argument("--max-N-2", type=int, default=DEFAULT_MAX_N[2])
    p.add_argument("--max-N-3", type=int, default=DEFAULT_MAX_N[3])
    p.add_argument("--max-N-4", type=int, default=DEFAULT_MAX_N[4])
    p.add_argument("--max-N-5", type=int, default=DEFAULT_MAX_N[5])
    p.add_argument("--max-N-6", type=int, default=DEFAULT_MAX_N[6])
    
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--mosek-tol", type=float, default=1e-9)
    
    p.add_argument("--out-root", type=Path,
                   default=Path(__file__).resolve().parents[1] / "results")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true", default=True)
    
    args = p.parse_args(argv)
    
    Ns = parse_ns_from_args(args)
    model_params = model_params_from_args(args)
    
    max_N_per_level = {
        0: args.max_N_0,
        1: args.max_N_1,
        2: args.max_N_2,
        3: args.max_N_3,
        4: args.max_N_4,
        5: args.max_N_5,
        6: args.max_N_6,
    }
    
    run_dir = compute_and_save(
        Ns=Ns,
        basis_name=args.basis,
        npa_level=args.npa_level,
        model_name=args.model,
        model_params=model_params,
        boundary=args.boundary,
        sym_levels=args.sym_levels,
        max_N_per_level=max_N_per_level,
        repeats=args.repeats,
        mosek_tol=args.mosek_tol,
        out_root=args.out_root,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
    )
    
    print(f"Results saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
