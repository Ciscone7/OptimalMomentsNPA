"""Fixed-fraction scaling: energy gap vs system size at a fixed monomial fraction.

For each system size N, selects k = floor(p * L) monomials from the adding set
(L = |adding_set|) via optimisation, computes the full relaxation E_full with all
L monomials, and records the gap.  Sweeping N reveals how the partial-relaxation
quality scales with system size.

Results are saved in a resumable way — only missing (N, seed) pairs are computed.

On-disk layout (default)::

    results/spin_fraction_scaling/v1/<config_hash>/
    ├── meta.json
    └── data.npz

``data.npz`` arrays:

  - run_idx (int)
  - N (int)
  - seed (int)
  - adding_set_size (int)     # L = |adding_set| for this N
  - k (int)                   # floor(fraction * L)
  - E_full (float)            # full relaxation (all L monomials)
  - E_partial (float)         # optimised partial relaxation (this seed)
  - gap (float)               # E_full − E_partial  (≥ 0)
  - t_full (float)            # wall-clock time for full relaxation (s)
  - t_partial (float)         # wall-clock time for this seed's optimisation (s)
  - n_obj_evals (int)         # SDP evaluations in this optimisation

Examples::

    python -m scripts.spins.fraction_scaling \\
        --model heisenberg --N-min 4 --N-max 12 \\
        --fraction 0.5 --method pt --num-seeds 3 \\
        --use-all-symmetries

    python -m scripts.spins.fraction_scaling \\
        --model heisenberg --Ns 4 6 8 10 \\
        --fraction 0.3 --method pt --pt-epochs 5 \\
        --num-seeds 5 --use-all-symmetries
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Ensure prints are flushed immediately (important when output is
# redirected to a log file via nohup).
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
except AttributeError:
    pass  # Python < 3.7 fallback: run with python -u

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))          # for _common
sys.path.insert(0, str(_SCRIPTS_DIR))          # for config_utils
sys.path.insert(0, str(_PROJECT_ROOT))         # for project packages

from artifact_manager import config_hash, utc_now_iso, stable_json_dumps
from _common import (
    add_symmetry_args,
    build_basis_sets,
    hamiltonian_dict_fn,
    parse_ns_from_args,
    symmetry_config_from_args,
)
from spins.spins_sdp import solve_pauli_relaxation
from spins.spins_optimize import make_spin_objective, run_single_optimization
from spins.symmetry import SymmetryManager


SCHEMA_VERSION = 1
ARTIFACT_NAME = "spin_fraction_scaling"

_SCALAR_FIELDS = {
    "N": np.dtype("int32"),
    "seed": np.dtype("int32"),
    "adding_set_size": np.dtype("int32"),
    "k": np.dtype("int32"),
    "E_full": np.dtype("float64"),
    "E_partial": np.dtype("float64"),
    "gap": np.dtype("float64"),
    "t_full": np.dtype("float64"),
    "t_partial": np.dtype("float64"),
    "n_obj_evals": np.dtype("int32"),
}


# ---------------------------------------------------------------------------
# I/O helpers  (compatible with the old on-disk format)
# ---------------------------------------------------------------------------

def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically write an .npz (write to temp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp, **arrays)
    tmp_npz = Path(str(tmp) + ".npz")
    os.replace(tmp_npz, path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(stable_json_dumps(obj).encode("utf-8"))
    os.replace(tmp, path)


def _upsert_meta_json(meta_path: Path, meta: Dict[str, Any]) -> None:
    """Write meta.json, preserving created_at if it exists."""
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(old, dict) and "created_at" in old:
                meta["created_at"] = old["created_at"]
        except Exception:
            pass
    _atomic_write_json(meta_path, meta)


def _load_existing_runs(
    data_path: Path,
) -> Tuple[Dict[Tuple[int, int], int], Dict[str, list]]:
    """Load existing (N, seed) → run_idx mapping and raw column lists."""
    if not data_path.exists():
        return {}, {}
    data = np.load(data_path, allow_pickle=False)
    completed: Dict[Tuple[int, int], int] = {}
    arrays: Dict[str, list] = {name: list(data[name]) for name in data}
    for i in range(len(data["N"])):
        completed[(int(data["N"][i]), int(data["seed"][i]))] = int(
            data["run_idx"][i]
        )
    return completed, arrays


def _save_runs(data_path: Path, **arrays: list) -> None:
    """Atomically save all result arrays to NPZ."""
    np_arrays = {}
    for name, values in arrays.items():
        dtype = _SCALAR_FIELDS.get(name, np.dtype("int32"))
        np_arrays[name] = np.array(values, dtype=dtype)
    _atomic_save_npz(data_path, **np_arrays)


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------

def compute_and_save(
    *,
    Ns: List[int],
    fraction: float,
    model_name: str,
    model_params: Dict[str, float],
    boundary: str,
    start_level: int,
    end_level: int,
    end_basis: str,
    method: str,
    method_params: Dict[str, Any],
    seeds: List[int],
    symmetry_config: Dict[str, bool],
    mosek_tol: float,
    out_root: Path,
    resume: bool,
    force: bool,
    verbose: bool,
) -> Path:
    """Run the fraction-scaling experiment and save results.

    For each N in *Ns* and each seed in *seeds*:

    1. Compute the full-basis relaxation ``E_full`` (cached per N).
    2. Run optimisation at ``k = floor(fraction * L)`` to get ``E_partial``.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    # ---- config & hashing ----
    config: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT_NAME,
        "model": model_name,
        "model_params": dict(sorted(model_params.items())),
        "boundary": boundary,
        "start_level": int(start_level),
        "end_level": int(end_level),
        "end_basis": end_basis,
        "fraction": float(fraction),
        "method": method,
        "method_params": dict(sorted(
            (k, v) for k, v in method_params.items()
        )),
        "symmetry": symmetry_config,
        "mosek_tol": float(mosek_tol),
    }

    cfg_hash = config_hash(config)
    run_dir = out_root / ARTIFACT_NAME / f"v{SCHEMA_VERSION}" / cfg_hash
    meta_path = run_dir / "meta.json"
    data_path = run_dir / "data.npz"

    # ---- resume / existing data ----
    if force:
        completed, existing_arrays = {}, {}
    else:
        completed, existing_arrays = (
            _load_existing_runs(data_path) if resume else ({}, {})
        )

    all_jobs = [(N, s) for N in sorted(Ns) for s in seeds]
    missing = [(N, s) for N, s in all_jobs if (N, s) not in completed]

    if verbose:
        print(f"Fraction scaling: p={fraction}, method={method}")
        print(
            f"  Model: {model_name}, boundary={boundary}, "
            f"end_basis={end_basis}, start_level={start_level}"
        )
        print(f"  N values: {sorted(Ns)}, seeds: {seeds}")
        print(
            f"  Total jobs: {len(all_jobs)}, cached: {len(completed)}, "
            f"to compute: {len(missing)}"
        )

    if not missing:
        if verbose:
            print("  All jobs already completed.")
        return run_dir

    # ---- initialise arrays from existing data ----
    field_names = ["run_idx"] + list(_SCALAR_FIELDS.keys())
    arrays: Dict[str, list] = {
        name: list(existing_arrays.get(name, [])) for name in field_names
    }
    next_idx = max(arrays["run_idx"], default=-1) + 1

    # Pre-populate full-relaxation cache from existing data.
    full_relax_cache: Dict[int, Tuple[float, float]] = {}
    for i in range(len(arrays.get("N", []))):
        n = int(arrays["N"][i])
        if n not in full_relax_cache:
            full_relax_cache[n] = (
                float(arrays["E_full"][i]),
                float(arrays["t_full"][i]),
            )

    H_dict_fn = hamiltonian_dict_fn(model_name)

    # ---- metadata (written early for discoverability) ----
    meta: Dict[str, Any] = {
        **config,
        "config_hash": cfg_hash,
        "seeds_requested": sorted(seeds),
        "Ns_requested": sorted(Ns),
        "Ns_present": sorted(set(int(x) for x in arrays.get("N", []))),
        "total_runs": len(arrays["run_idx"]),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }
    _upsert_meta_json(meta_path, meta)

    def _checkpoint() -> None:
        _save_runs(data_path, **arrays)
        meta["Ns_present"] = sorted(set(int(x) for x in arrays["N"]))
        meta["total_runs"] = len(arrays["run_idx"])
        meta["updated_at"] = utc_now_iso()
        _upsert_meta_json(meta_path, meta)

    # ---- main loop ----
    prev_N: Optional[int] = None
    obj_func = None
    L = 0
    k = 0
    E_full = 0.0
    t_full = 0.0

    pbar = tqdm(total=len(missing), desc="Fraction scaling", disable=not verbose)
    try:
        for N_val, seed in missing:
            pbar.set_postfix({"N": N_val, "seed": seed})

            # ---- per-N setup (only when N changes) ----
            if N_val != prev_N:
                starting_set, adding_set, final_set = build_basis_sets(
                    N=N_val,
                    start_level=start_level,
                    end_basis=end_basis,
                    end_level=end_level,
                )
                L = len(adding_set)
                k = min(int(math.floor(fraction * L)), L)

                hamiltonian = H_dict_fn(
                    N=N_val, boundary=boundary, **model_params
                )
                sym_mgr = SymmetryManager(N=N_val, **symmetry_config)

                # Full relaxation (cached per N)
                if N_val not in full_relax_cache:
                    t0 = time.perf_counter()
                    E_full = solve_pauli_relaxation(
                        final_set,
                        hamiltonian,
                        sym_mgr,
                        sense="min",
                        mosek_tol=mosek_tol,
                    )
                    t_full = time.perf_counter() - t0
                    full_relax_cache[N_val] = (E_full, t_full)
                    if verbose:
                        tqdm.write(
                            f"  N={N_val}: L={L}, k={k}, "
                            f"E_full={E_full:.10f} ({t_full:.1f}s)"
                        )
                else:
                    E_full, t_full = full_relax_cache[N_val]

                obj_func = make_spin_objective(
                    starting_set=starting_set,
                    adding_set=adding_set,
                    operator=hamiltonian,
                    symmetry_manager=sym_mgr,
                    mosek_tol=mosek_tol,
                )
                prev_N = N_val

            # ---- optimisation for this seed ----
            result = run_single_optimization(
                obj_func=obj_func,
                L=L,
                k=k,
                seed=seed,
                method=method,
                method_params=method_params,
            )
            E_partial = result.best_value
            gap = E_full - E_partial  # ≥ 0

            arrays["run_idx"].append(next_idx)
            arrays["N"].append(N_val)
            arrays["seed"].append(seed)
            arrays["adding_set_size"].append(L)
            arrays["k"].append(k)
            arrays["E_full"].append(E_full)
            arrays["E_partial"].append(E_partial)
            arrays["gap"].append(gap)
            arrays["t_full"].append(t_full)
            arrays["t_partial"].append(result.elapsed_s)
            arrays["n_obj_evals"].append(result.n_obj_evals)
            next_idx += 1

            _checkpoint()
            pbar.update(1)

    finally:
        pbar.close()
        meta["Ns_present"] = sorted(set(int(x) for x in arrays["N"]))
        meta["total_runs"] = len(arrays["run_idx"])
        meta["updated_at"] = utc_now_iso()
        _upsert_meta_json(meta_path, meta)

    if verbose:
        print(f"\nResults saved to: {run_dir}")

    return run_dir


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def load_results(run_dir) -> Dict[str, Any]:
    """Load fraction-scaling results from a run directory.

    Returns:
        Dict with ``meta`` and ``data`` keys.
    """
    run_dir = Path(run_dir)
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    data = dict(np.load(run_dir / "data.npz", allow_pickle=False))
    return {"meta": meta, "data": data}


def get_best_per_N(run_dir) -> Dict[int, Dict[str, Any]]:
    """Get the best result (highest ``E_partial``) for each N.

    Returns:
        Dict mapping ``N → {E_full, E_partial, gap, k, ...}``.
    """
    results = load_results(run_dir)
    data = results["data"]
    best: Dict[int, Dict[str, Any]] = {}
    for i in range(len(data["N"])):
        N = int(data["N"][i])
        E_partial = float(data["E_partial"][i])
        if N not in best or E_partial > best[N]["E_partial"]:
            best[N] = {
                "N": N,
                "seed": int(data["seed"][i]),
                "adding_set_size": int(data["adding_set_size"][i]),
                "k": int(data["k"][i]),
                "E_full": float(data["E_full"][i]),
                "E_partial": E_partial,
                "gap": float(data["gap"][i]),
                "t_full": float(data["t_full"][i]),
                "t_partial": float(data["t_partial"][i]),
            }
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- N values ----
    n_group = p.add_mutually_exclusive_group(required=True)
    n_group.add_argument("--Ns", nargs="+", type=int, help="Explicit N values")
    n_group.add_argument("--N-min", dest="N_min", type=int)
    p.add_argument("--N-max", dest="N_max", type=int)

    # ---- model ----
    p.add_argument(
        "--model",
        choices=["ising", "heisenberg", "heisenberg_j2"],
        default="heisenberg",
    )
    p.add_argument(
        "--boundary", choices=["open", "periodic"], default="periodic",
    )
    p.add_argument("--J", type=float, default=1.0, help="Ising J")
    p.add_argument("--h", type=float, default=0.0, help="Ising h")
    p.add_argument(
        "--k-ising", type=float, default=0.0, dest="k_ising",
        help="Ising k parameter",
    )
    p.add_argument("--J2", type=float, default=0.0, help="Heisenberg J2")

    # ---- basis ----
    p.add_argument("--start-level", type=int, default=1)
    p.add_argument("--end-level", type=int, default=2)
    p.add_argument(
        "--end-basis",
        default="heisenberg_simple",
        choices=[
            "npa",
            "heisenberg_simple",
            "heisenberg_j2_weak",
            "heisenberg_j2_strong",
        ],
    )

    # ---- fraction ----
    p.add_argument(
        "--fraction", type=float, required=True,
        help="Fraction p in [0, 1] of adding-set monomials to select",
    )

    # ---- optimisation method & parameters ----
    p.add_argument(
        "--method",
        choices=["sa", "pt", "bo", "rbm", "random"],
        default="pt",
    )
    # SA
    p.add_argument("--sa-steps", type=int, default=100)
    p.add_argument("--sa-T-start", type=float, default=2.0)
    p.add_argument("--sa-alpha", type=float, default=0.95)
    # PT
    p.add_argument("--pt-chains", type=int, default=0)
    p.add_argument("--pt-epochs", type=int, default=5)
    p.add_argument("--pt-steps-per-epoch", type=int, default=40)
    p.add_argument("--pt-T-min", type=float, default=0.1)
    p.add_argument("--pt-T-max", type=float, default=2.0)
    # BO
    p.add_argument("--bo-beta", type=float, default=1.0)
    p.add_argument("--bo-n-init", type=int, default=20)
    p.add_argument("--bo-n-iter", type=int, default=50)
    p.add_argument("--bo-candidates-per-iter", type=int, default=100)
    # RBM
    p.add_argument("--rbm-steps", type=int, default=100)

    # ---- seeds ----
    seed_group = p.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seeds", nargs="+", type=int)
    seed_group.add_argument("--num-seeds", type=int)

    # ---- solver ----
    p.add_argument("--mosek-tol", type=float, default=1e-9)

    # ---- symmetry ----
    add_symmetry_args(p)

    # ---- output ----
    p.add_argument(
        "--out-root", type=Path,
        default=_PROJECT_ROOT / "results",
    )
    p.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true", default=True)

    args = p.parse_args(argv)

    # ---- derived values ----
    Ns = parse_ns_from_args(args)

    # Model params (build manually to avoid --k conflict with ising)
    if args.model == "ising":
        model_params: Dict[str, float] = {
            "J": args.J, "h": args.h, "k": args.k_ising,
        }
    elif args.model == "heisenberg_j2":
        model_params = {"J2": args.J2}
    else:
        model_params = {}

    symmetry_config = symmetry_config_from_args(args)

    seeds = (
        sorted(set(args.seeds))
        if args.seeds is not None
        else list(range(42, 42 + args.num_seeds))
    )

    # Method params (mirrors optimization_sweep.py)
    if args.method == "sa":
        method_params: Dict[str, Any] = {
            "steps": args.sa_steps,
            "T_start": args.sa_T_start,
            "alpha": args.sa_alpha,
        }
    elif args.method == "pt":
        pt_chains = None if args.pt_chains <= 0 else args.pt_chains
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
        method_params = {"steps": args.rbm_steps}
    else:
        method_params = {}

    run_dir = compute_and_save(
        Ns=Ns,
        fraction=args.fraction,
        model_name=args.model,
        model_params=model_params,
        boundary=args.boundary,
        start_level=args.start_level,
        end_level=args.end_level,
        end_basis=args.end_basis,
        method=args.method,
        method_params=method_params,
        seeds=seeds,
        symmetry_config=symmetry_config,
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
