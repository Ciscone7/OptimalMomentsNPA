"""Fixed-fraction scaling for observable gap vs system size.

Same idea as ``fraction_scaling.py`` but the quantity being optimised is the
**observable gap** (ub − lb) rather than the energy lower bound.

For each system size N:
1. Build starting_set / adding_set (same as energy version).
2. Compute the **full-basis** observable bounds via ``bound_observable``.
3. Run optimisation at k = floor(p · L) using :class:`ObservableGapObjective`
   to find the subset that *minimises* the gap (ub − lb).
4. Record both full and partial gaps and individual bounds.

The energy window ``[E_lb, E_ub]`` is obtained from previously saved artifacts
(NPA energy LB + DMRG energy UB), the same sources used everywhere else.

On-disk layout::

    results/spin_fraction_scaling_observable/v1/<config_hash>/
    ├── meta.json
    └── data.npz

``data.npz`` arrays:

  - run_idx (int)
  - N (int)
  - seed (int)
  - adding_set_size (int)       # L = |adding_set| for this N
  - k (int)                     # floor(fraction * L)
  - obs_gap_full (float)        # ub − lb from full basis
  - obs_lb_full (float)
  - obs_ub_full (float)
  - obs_gap_partial (float)     # best gap from optimised partial basis
  - obs_lb_partial (float)
  - obs_ub_partial (float)
  - energy_lb (float)           # energy window used
  - energy_ub (float)
  - t_full (float)              # wall-clock seconds for full observable bound
  - t_partial (float)           # wall-clock seconds for optimisation (this seed)
  - n_obj_evals (int)           # SDP evaluations in optimisation

Examples::

    python -m scripts.spins.fraction_scaling_observable \\
        --model heisenberg --N-min 4 --N-max 18 \\
        --fraction 0.5 --method pt --num-seeds 1 \\
        --use-all-symmetries

    python -m scripts.spins.fraction_scaling_observable \\
        --model heisenberg --Ns 4 6 8 10 12 \\
        --fraction 0.3 --method pt --pt-epochs 3 \\
        --num-seeds 3 --use-all-symmetries
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

try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
except AttributeError:
    pass

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from artifact_manager import config_hash, utc_now_iso, stable_json_dumps
from _common import (
    add_symmetry_args,
    build_basis_sets,
    hamiltonian_dict_fn,
    parse_ns_from_args,
    symmetry_config_from_args,
)
from spins.spins_sdp import bound_observable
from spins.spins_optimize import ObservableGapObjective, run_single_optimization
from spins.symmetry import SymmetryManager
from spins.pauli_logic import PauliWord


SCHEMA_VERSION = 1
ARTIFACT_NAME = "spin_fraction_scaling_observable"

_SCALAR_FIELDS = {
    "N": np.dtype("int32"),
    "seed": np.dtype("int32"),
    "adding_set_size": np.dtype("int32"),
    "k": np.dtype("int32"),
    "obs_gap_full": np.dtype("float64"),
    "obs_lb_full": np.dtype("float64"),
    "obs_ub_full": np.dtype("float64"),
    "obs_gap_partial": np.dtype("float64"),
    "obs_lb_partial": np.dtype("float64"),
    "obs_ub_partial": np.dtype("float64"),
    "energy_lb": np.dtype("float64"),
    "energy_ub": np.dtype("float64"),
    "t_full": np.dtype("float64"),
    "t_partial": np.dtype("float64"),
    "n_obj_evals": np.dtype("int32"),
}


# ---------------------------------------------------------------------------
# Observable builder (same as observable_bound_sweep.py)
# ---------------------------------------------------------------------------

def build_half_chain_corr_x(N: int) -> Dict[PauliWord, complex]:
    r"""Translation-averaged X-correlation at distance N/2.

    .. math::
        \bar{C}_{N/2} = \frac{1}{N}\sum_{i=0}^{N-1}
        \frac{1}{4}\, X_i\, X_{(i+N/2)\bmod N}
    """
    d = N // 2
    op: dict = {}
    for i in range(N):
        j = (i + d) % N
        w = PauliWord((1 << i) | (1 << j), 0)  # X_i X_j
        op[w] = op.get(w, 0.0) + 1.0 / (4 * N)
    return op


# ---------------------------------------------------------------------------
# Energy bound loaders (from previously saved artifacts)
# ---------------------------------------------------------------------------

def _load_energy_lb_data(results_dir: Path, npa_level: float) -> Dict[int, float]:
    """Load SDP energy lower bounds for the given NPA level."""
    tag = f"npa{npa_level:g}"
    path = results_dir / f"spin_moment_energy_lb/v1/heisenberg_{tag}_periodic_N_sweep/data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Energy LB data not found: {path}")
    d = np.load(path)
    return dict(zip(d["N"].tolist(), d["E_lb"].tolist()))


def _load_energy_lb_paper(results_dir: Path) -> Dict[int, float]:
    """Load SDP energy lower bounds for the paper (heisenberg_simple) basis."""
    path = results_dir / "spin_moment_energy_lb/v1/heisenberg_paper_basis_periodic_N_sweep/data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Paper basis energy LB data not found: {path}")
    d = np.load(path)
    return dict(zip(d["N"].tolist(), d["E_lb"].tolist()))


def _load_dmrg_energy_ub(results_dir: Path) -> Dict[int, float]:
    """Load DMRG energy upper bounds."""
    path = results_dir / "spin_dmrg_energy_ub/v1/heisenberg_pbc_J1_chi256_consSz/data.npz"
    if not path.exists():
        raise FileNotFoundError(f"DMRG data not found: {path}")
    d = np.load(path)
    return dict(zip(d["N"].tolist(), d["E"].tolist()))


def _get_energy_bounds(
    results_dir: Path,
    energy_npa_level: float,
    energy_basis: str,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Load energy LB and UB maps."""
    if energy_basis == "paper":
        elb_map = _load_energy_lb_paper(results_dir)
    else:
        elb_map = _load_energy_lb_data(results_dir, energy_npa_level)
    eub_map = _load_dmrg_energy_ub(results_dir)
    return elb_map, eub_map


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
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
    observable_name: str,
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
    energy_npa_level: float,
    energy_basis: str,
    mosek_tol: float,
    out_root: Path,
    energy_results_dir: Path,
    resume: bool,
    force: bool,
    verbose: bool,
) -> Path:
    """Run the observable fraction-scaling experiment and save results."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    # ---- energy bounds ----
    elb_map, eub_map = _get_energy_bounds(energy_results_dir, energy_npa_level, energy_basis)

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
        "observable": observable_name,
        "method": method,
        "method_params": dict(sorted(
            (k, v) for k, v in method_params.items()
        )),
        "symmetry": symmetry_config,
        "energy_npa_level": float(energy_npa_level),
        "energy_basis": energy_basis,
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

    # Filter out N values without energy bounds
    skipped = set()
    filtered_missing = []
    for N_val, s in missing:
        if N_val not in elb_map:
            if N_val not in skipped:
                print(f"  [skip] N={N_val}: no energy LB data (NPA{energy_npa_level})")
                skipped.add(N_val)
            continue
        if N_val not in eub_map:
            if N_val not in skipped:
                print(f"  [skip] N={N_val}: no DMRG energy UB data")
                skipped.add(N_val)
            continue
        filtered_missing.append((N_val, s))
    missing = filtered_missing

    if verbose:
        print(f"Observable fraction scaling: p={fraction}, method={method}")
        print(
            f"  Model: {model_name}, boundary={boundary}, "
            f"end_basis={end_basis}, start_level={start_level}"
        )
        print(f"  Observable: {observable_name}")
        print(f"  Energy bounds: NPA{energy_npa_level} LB + DMRG UB")
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
    full_relax_cache: Dict[int, Dict[str, float]] = {}
    for i in range(len(arrays.get("N", []))):
        n = int(arrays["N"][i])
        if n not in full_relax_cache:
            full_relax_cache[n] = {
                "obs_gap_full": float(arrays["obs_gap_full"][i]),
                "obs_lb_full": float(arrays["obs_lb_full"][i]),
                "obs_ub_full": float(arrays["obs_ub_full"][i]),
                "t_full": float(arrays["t_full"][i]),
            }

    H_dict_fn = hamiltonian_dict_fn(model_name)

    # Build observable
    if observable_name == "half_chain_corr_x":
        build_obs_fn = build_half_chain_corr_x
    else:
        raise ValueError(f"Unknown observable: {observable_name!r}")

    # ---- metadata ----
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
    e_lb = 0.0
    e_ub = 0.0

    pbar = tqdm(total=len(missing), desc="Obs fraction scaling", disable=not verbose)
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
                observable = build_obs_fn(N_val)
                sym_mgr = SymmetryManager(N=N_val, **symmetry_config)

                # Energy bounds for this N
                e_lb = elb_map[N_val]
                e_ub = eub_map[N_val]
                if e_lb > e_ub - 1e-8:
                    mid = 0.5 * (e_lb + e_ub)
                    e_lb = mid - 1e-7
                    e_ub = mid + 1e-7

                # Full observable bound (cached per N)
                if N_val not in full_relax_cache:
                    t0 = time.perf_counter()
                    full_result = bound_observable(
                        basis=final_set,
                        hamiltonian=hamiltonian,
                        observable=observable,
                        energy_lb=e_lb,
                        energy_ub=e_ub,
                        symmetry_manager=sym_mgr,
                        mosek_tol=mosek_tol,
                    )
                    t_full = time.perf_counter() - t0
                    full_cache = {
                        "obs_gap_full": full_result.ub - full_result.lb,
                        "obs_lb_full": float(full_result.lb),
                        "obs_ub_full": float(full_result.ub),
                        "t_full": t_full,
                    }
                    full_relax_cache[N_val] = full_cache
                    if verbose:
                        tqdm.write(
                            f"  N={N_val}: L={L}, k={k}, "
                            f"obs_gap_full={full_cache['obs_gap_full']:.8f} "
                            f"[{full_cache['obs_lb_full']:.8f}, {full_cache['obs_ub_full']:.8f}] "
                            f"({t_full:.1f}s)"
                        )

                obj_func = ObservableGapObjective(
                    starting_set=starting_set,
                    adding_set=adding_set,
                    hamiltonian=hamiltonian,
                    observable=observable,
                    energy_lb=e_lb,
                    energy_ub=e_ub,
                    symmetry_manager=sym_mgr,
                    mosek_tol=mosek_tol,
                )
                prev_N = N_val

            fc = full_relax_cache[N_val]

            # ---- optimisation for this seed ----
            result = run_single_optimization(
                obj_func=obj_func,
                L=L,
                k=k,
                seed=seed,
                method=method,
                method_params=method_params,
            )
            # result.best_value = -gap  (optimizer minimises gap → stores -gap)
            obs_gap_partial = -result.best_value

            # Re-evaluate the best selection to get individual LB/UB
            best_basis = starting_set + [adding_set[i] for i in result.best_indices]
            hamiltonian = H_dict_fn(N=N_val, boundary=boundary, **model_params)
            observable = build_obs_fn(N_val)
            sym_mgr = SymmetryManager(N=N_val, **symmetry_config)
            partial_result = bound_observable(
                basis=best_basis,
                hamiltonian=hamiltonian,
                observable=observable,
                energy_lb=e_lb,
                energy_ub=e_ub,
                symmetry_manager=sym_mgr,
                mosek_tol=mosek_tol,
            )

            arrays["run_idx"].append(next_idx)
            arrays["N"].append(N_val)
            arrays["seed"].append(seed)
            arrays["adding_set_size"].append(L)
            arrays["k"].append(k)
            arrays["obs_gap_full"].append(fc["obs_gap_full"])
            arrays["obs_lb_full"].append(fc["obs_lb_full"])
            arrays["obs_ub_full"].append(fc["obs_ub_full"])
            arrays["obs_gap_partial"].append(partial_result.ub - partial_result.lb)
            arrays["obs_lb_partial"].append(float(partial_result.lb))
            arrays["obs_ub_partial"].append(float(partial_result.ub))
            arrays["energy_lb"].append(e_lb)
            arrays["energy_ub"].append(e_ub)
            arrays["t_full"].append(fc["t_full"])
            arrays["t_partial"].append(result.elapsed_s)
            arrays["n_obj_evals"].append(result.n_obj_evals)
            next_idx += 1

            if verbose:
                tqdm.write(
                    f"  N={N_val}, seed={seed}: "
                    f"gap_partial={partial_result.ub - partial_result.lb:.8f} "
                    f"(full={fc['obs_gap_full']:.8f}) "
                    f"t={result.elapsed_s:.1f}s"
                )

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
    run_dir = Path(run_dir)
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    data = dict(np.load(run_dir / "data.npz", allow_pickle=False))
    return {"meta": meta, "data": data}


def get_best_per_N(run_dir) -> Dict[int, Dict[str, Any]]:
    """Get the best result (smallest obs_gap_partial) for each N."""
    results = load_results(run_dir)
    data = results["data"]
    best: Dict[int, Dict[str, Any]] = {}
    for i in range(len(data["N"])):
        N = int(data["N"][i])
        gap = float(data["obs_gap_partial"][i])
        if N not in best or gap < best[N]["obs_gap_partial"]:
            best[N] = {
                "N": N,
                "seed": int(data["seed"][i]),
                "adding_set_size": int(data["adding_set_size"][i]),
                "k": int(data["k"][i]),
                "obs_gap_full": float(data["obs_gap_full"][i]),
                "obs_lb_full": float(data["obs_lb_full"][i]),
                "obs_ub_full": float(data["obs_ub_full"][i]),
                "obs_gap_partial": gap,
                "obs_lb_partial": float(data["obs_lb_partial"][i]),
                "obs_ub_partial": float(data["obs_ub_partial"][i]),
                "energy_lb": float(data["energy_lb"][i]),
                "energy_ub": float(data["energy_ub"][i]),
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

    # ---- observable ----
    p.add_argument(
        "--observable",
        default="half_chain_corr_x",
        choices=["half_chain_corr_x"],
        help="Observable to bound",
    )

    # ---- fraction ----
    p.add_argument(
        "--fraction", type=float, required=True,
        help="Fraction p in [0, 1] of adding-set monomials to select",
    )

    # ---- energy bounds ----
    p.add_argument(
        "--energy-npa-level", type=float, default=2,
        help="NPA level used for the energy lower bound (default: 2)",
    )
    p.add_argument(
        "--energy-basis", default="npa",
        choices=["npa", "paper"],
        help="Basis used for the energy lower bound",
    )
    p.add_argument(
        "--energy-results-dir", type=Path,
        default=_SCRIPTS_DIR / "results",
        help="Directory containing energy bound artifacts",
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
        observable_name=args.observable,
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
        energy_npa_level=args.energy_npa_level,
        energy_basis=args.energy_basis,
        mosek_tol=args.mosek_tol,
        out_root=args.out_root,
        energy_results_dir=args.energy_results_dir,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
    )

    print(f"Results saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
