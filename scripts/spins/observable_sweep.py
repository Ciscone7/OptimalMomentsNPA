"""Run observable-gap optimization sweeps and save results.

CLI tool for optimising monomial selection to minimise the gap
between upper and lower bounds on a general ground-state observable.
The core optimisation logic lives in :mod:`spins.spins_optimize`;
this script adds persistence, resumability, and a CLI.

On-disk layout::

    results/spin_observable_sweep/v1/<run_name>/
        meta.json      <- config, provenance, timestamps
        data.npz       <- run_idx, k, seed, gap, lb, ub, elapsed_s,
                          n_obj_evals, mask_bits

Examples::

    python scripts/spins/observable_sweep.py \\
        --model heisenberg --N 6 --boundary periodic \\
        --observable staggered_mz \\
        --exact-energy -2.8028 \\
        --start-level 1 --end-level 2 --method sa \\
        --ks 0 1 2 3 --seeds 42 43

    python scripts/spins/observable_sweep.py \\
        --config experiments/spins_observable.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure prints are flushed immediately (important when output is
# redirected to a log file via nohup).
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
except AttributeError:
    pass  # Python < 3.7 fallback: run with python -u

if TYPE_CHECKING:
    import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))          # for _common
sys.path.insert(0, str(_SCRIPTS_DIR))          # for config_utils
sys.path.insert(0, str(_PROJECT_ROOT))         # for project packages

from artifact_manager import ArtifactManager, RunDir
from config_utils import add_config_arg, parse_with_config
from _common import (
    add_symmetry_args,
    build_basis_sets,
    hamiltonian_dict_fn,
    hamiltonian_exact_fn,
    model_params_from_args,
    symmetry_config_from_args,
)
from spins.spins_optimize import (
    ObservableOptimizationResult,
    precompute_energy_bounds,
    sweep_observable_k_values,
)
from spins.symmetry import SymmetryManager
from spins.models import (
    magnetization_z_dict,
    staggered_magnetization_z_dict,
    nearest_neighbor_correlation_dict,
    two_point_correlation_dict,
)


ARTIFACT = "spin_observable_sweep"


# ---------------------------------------------------------------------------
# Observable builders
# ---------------------------------------------------------------------------

def _build_half_chain_corr_x(N: int, **_kw):
    r"""Translation-averaged staggered X-correlation at distance N/2.

    .. math::
        \bar{C}_{N/2} = \frac{1}{N}\sum_{i=0}^{N-1}
        \frac{1}{4}\, X_i\, X_{(i+N/2)\bmod N}

    This preserves translation symmetry in the SDP.
    """
    from spins.pauli_logic import PauliWord
    d = N // 2
    op: dict = {}
    for i in range(N):
        j = (i + d) % N
        w = PauliWord((1 << i) | (1 << j), 0)  # X_i X_j
        op[w] = op.get(w, 0.0) + 1.0 / (4 * N)
    return op


_OBSERVABLE_BUILDERS = {
    "magnetization_z": lambda N, boundary, **_kw: magnetization_z_dict(N),
    "staggered_mz": lambda N, boundary, **_kw: staggered_magnetization_z_dict(N),
    "nn_correlation_z": lambda N, boundary, **_kw: nearest_neighbor_correlation_dict(N, axis="z", boundary=boundary),
    "nn_correlation_x": lambda N, boundary, **_kw: nearest_neighbor_correlation_dict(N, axis="x", boundary=boundary),
    "two_point_z": lambda N, i, j, **_kw: two_point_correlation_dict(N, i=i, j=j, axis="z"),
    "half_chain_corr_x": lambda N, boundary, **_kw: _build_half_chain_corr_x(N),
}


def _build_observable(name: str, N: int, boundary: str, **kwargs):
    """Build an observable dict from a CLI-friendly name."""
    if name not in _OBSERVABLE_BUILDERS:
        raise ValueError(
            f"Unknown observable {name!r}. "
            f"Choose from: {', '.join(sorted(_OBSERVABLE_BUILDERS))}"
        )
    return _OBSERVABLE_BUILDERS[name](N=N, boundary=boundary, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_pauliword_mask_hash(words: List[Any], *, N: int, n_chars: int = 16) -> str:
    """Fast stable hash for PauliWord-like objects using (x_mask, z_mask)."""
    h = hashlib.sha256()
    h.update(struct.pack("<I", int(N)))
    n_bytes = max(1, (int(N) + 7) // 8)
    for w in words:
        x = int(getattr(w, "x_mask"))
        z = int(getattr(w, "z_mask"))
        h.update(x.to_bytes(n_bytes, byteorder="little", signed=False))
        h.update(z.to_bytes(n_bytes, byteorder="little", signed=False))
    return h.hexdigest()[:n_chars]


def _default_name(
    model: str, N: int, boundary: str,
    method: str, start_level: int, end_level: int,
    end_basis: str, observable: str, feedback: bool,
    model_params: Dict[str, float],
) -> str:
    parts = [model, f"N{N}", boundary, observable]
    for pk, pv in sorted(model_params.items()):
        parts.append(f"{pk}{pv:g}")
    if end_basis == "npa":
        parts.append(f"lv{start_level}to{end_level}")
    else:
        parts.append(f"lv{start_level}to_{end_basis}")
    parts.append(method)
    if feedback:
        parts.append("fb")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_existing_runs(run: RunDir, L: int) -> Tuple[Dict[Tuple[int, int], int], Dict[str, list]]:
    """Load existing runs from a RunDir."""
    tbl = run.load_table()
    if not tbl:
        return {}, {}
    completed: Dict[Tuple[int, int], int] = {}
    ks = tbl["k"]
    seeds = tbl["seed"]
    run_idxs = tbl["run_idx"]
    for i in range(len(ks)):
        completed[(int(ks[i]), int(seeds[i]))] = int(run_idxs[i])
    arrays = {key: list(arr) for key, arr in tbl.items()}
    return completed, arrays


def _checkpoint(
    run: RunDir,
    run_idx: List[int],
    k: List[int],
    seed: List[int],
    gap: List[float],
    lb: List[float],
    ub: List[float],
    elapsed_s: List[float],
    n_obj_evals: List[int],
    mask_bits_list: List[np.ndarray],
) -> None:
    """Atomically save all accumulated results."""
    mb = np.stack(mask_bits_list, axis=0) if mask_bits_list else np.array([], dtype=np.uint8).reshape(0, 0)
    run.save_table(
        run_idx=np.array(run_idx, dtype=np.int32),
        k=np.array(k, dtype=np.int32),
        seed=np.array(seed, dtype=np.int32),
        gap=np.array(gap, dtype=np.float64),
        lb=np.array(lb, dtype=np.float64),
        ub=np.array(ub, dtype=np.float64),
        elapsed_s=np.array(elapsed_s, dtype=np.float64),
        n_obj_evals=np.array(n_obj_evals, dtype=np.int32),
        mask_bits=mb,
    )


# ---------------------------------------------------------------------------
# Main compute and save
# ---------------------------------------------------------------------------

def compute_and_save(
    *,
    N: int,
    model_name: str,
    model_params: Dict[str, float],
    boundary: str,
    observable_name: str,
    observable_kwargs: Dict[str, Any],
    energy_lb: float,
    energy_ub: float,
    start_level: int,
    end_level: int,
    end_basis: str,
    method: str,
    method_params: Dict[str, Any],
    k_values: List[int],
    seeds: List[int],
    symmetry_config: Dict[str, bool],
    mosek_tol: float,
    out_root: Path,
    name: str,
    resume: bool,
    force: bool,
    verbose: bool,
    feedback: bool = False,
) -> str:
    """Run observable sweep and save results."""
    starting_set, adding_set, final_set = build_basis_sets(
        N=N, start_level=start_level, end_basis=end_basis, end_level=end_level,
    )
    L = len(adding_set)

    if verbose:
        print(f"Model: {model_name}, N={N}, boundary={boundary}")
        print(f"Observable: {observable_name}")
        print(f"Energy window: [{energy_lb:.6f}, {energy_ub:.6f}]")
        print(f"Starting set: {len(starting_set)}, Adding set: {L}, Final set: {len(final_set)}")
        if feedback:
            print("Feedback mode: ON (warm-start chaining across k values)")

    adding_set_hash = _stable_pauliword_mask_hash(adding_set, N=N)

    # Build observable
    observable_dict = _build_observable(
        observable_name, N=N, boundary=boundary, **observable_kwargs,
    )

    config: Dict[str, Any] = {
        "model": model_name,
        "model_params": dict(sorted(model_params.items())),
        "N": int(N),
        "boundary": boundary,
        "observable": observable_name,
        "observable_kwargs": observable_kwargs,
        "energy_lb": float(energy_lb),
        "energy_ub": float(energy_ub),
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

    am = ArtifactManager(out_root)
    run = am.create_run(artifact=ARTIFACT, name=name, config=config)

    # ------------------------------------------------------------------
    # Load existing checkpoint
    # ------------------------------------------------------------------
    existing_results: Dict[Tuple[int, int], ObservableOptimizationResult] = {}
    if not force:
        completed, existing_arrays = (
            _load_existing_runs(run, L) if resume else ({}, {})
        )
        for (kv, s), run_idx in completed.items():
            pos = list(existing_arrays["run_idx"]).index(run_idx)
            mask: Optional[np.ndarray] = None
            if "mask_bits" in existing_arrays:
                packed = existing_arrays["mask_bits"][pos]
                if isinstance(packed, np.ndarray):
                    mask = np.unpackbits(packed)[:L].astype(np.int32)
            if mask is None:
                mask = np.zeros(L, dtype=np.int32)
            existing_results[(kv, s)] = ObservableOptimizationResult(
                gap=float(existing_arrays["gap"][pos]),
                lb=float(existing_arrays["lb"][pos]),
                ub=float(existing_arrays["ub"][pos]),
                best_indices=sorted(int(i) for i in np.flatnonzero(mask)),
                mask=mask,
                elapsed_s=float(existing_arrays["elapsed_s"][pos]),
                n_obj_evals=int(existing_arrays["n_obj_evals"][pos]),
                method=method,
                k=kv,
                seed=s,
                energy_lb=energy_lb,
                energy_ub=energy_ub,
            )

    if verbose:
        all_jobs = [(kv, s) for kv in sorted(k_values) for s in seeds]
        n_todo = sum(1 for j in all_jobs if j not in existing_results)
        print(f"Total jobs: {len(all_jobs)}, Already done: {len(existing_results)}, To compute: {n_todo}")

    if not any(
        (kv, s) not in existing_results
        for kv in k_values for s in seeds
    ):
        if verbose:
            print("All jobs already completed. Nothing to do.")
        return str(run.path)

    # ------------------------------------------------------------------
    # Accumulator + checkpoint callback
    # ------------------------------------------------------------------
    _acc_run_idx: List[int] = []
    _acc_k: List[int] = []
    _acc_seed: List[int] = []
    _acc_gap: List[float] = []
    _acc_lb: List[float] = []
    _acc_ub: List[float] = []
    _acc_elapsed_s: List[float] = []
    _acc_n_obj_evals: List[int] = []
    _acc_mask_bits: List[np.ndarray] = []

    if not force and resume:
        completed_loaded, arr = _load_existing_runs(run, L)
        if arr:
            _acc_run_idx = list(arr.get("run_idx", []))
            _acc_k = list(arr.get("k", []))
            _acc_seed = list(arr.get("seed", []))
            _acc_gap = list(arr.get("gap", []))
            _acc_lb = list(arr.get("lb", []))
            _acc_ub = list(arr.get("ub", []))
            _acc_elapsed_s = list(arr.get("elapsed_s", []))
            _acc_n_obj_evals = list(arr.get("n_obj_evals", []))
            _acc_mask_bits = list(arr.get("mask_bits", []))
    _next_run_idx = (max(int(x) for x in _acc_run_idx) + 1) if _acc_run_idx else 0

    def _on_result(kv: int, s: int, res: ObservableOptimizationResult) -> None:
        nonlocal _next_run_idx
        packed = np.packbits(res.mask.astype(np.uint8))
        _acc_run_idx.append(_next_run_idx)
        _acc_k.append(kv)
        _acc_seed.append(s)
        _acc_gap.append(res.gap)
        _acc_lb.append(res.lb)
        _acc_ub.append(res.ub)
        _acc_elapsed_s.append(res.elapsed_s)
        _acc_n_obj_evals.append(res.n_obj_evals)
        _acc_mask_bits.append(packed)
        _next_run_idx += 1
        _checkpoint(run, _acc_run_idx, _acc_k, _acc_seed,
                    _acc_gap, _acc_lb, _acc_ub,
                    _acc_elapsed_s, _acc_n_obj_evals, _acc_mask_bits)

    # ------------------------------------------------------------------
    # Build Hamiltonian & delegate to sweep_observable_k_values
    # ------------------------------------------------------------------
    H_dict_fn = hamiltonian_dict_fn(model_name)
    hamiltonian_dict = H_dict_fn(N=N, boundary=boundary, **model_params)
    sym_manager = SymmetryManager(N=N, **symmetry_config)

    run.update_meta(
        k_values_requested=sorted(k_values),
        seeds_requested=sorted(seeds),
        L=int(L),
    )

    try:
        results = sweep_observable_k_values(
            N=N,
            hamiltonian=hamiltonian_dict,
            observable=observable_dict,
            energy_lb=energy_lb,
            energy_ub=energy_ub,
            symmetry_manager=sym_manager,
            starting_set=starting_set,
            adding_set=adding_set,
            k_values=k_values,
            method=method,
            method_params=method_params,
            mosek_tol=mosek_tol,
            seeds=seeds,
            feedback=feedback,
            existing_results=existing_results,
            on_result=_on_result,
            verbose=verbose,
        )
    finally:
        run.update_meta(
            k_values_present=sorted(set(int(x) for x in _acc_k)),
            seeds_present=sorted(set(int(x) for x in _acc_seed)),
            total_runs=len(_acc_run_idx),
        )

    n_computed = len(_acc_run_idx) - len(existing_results)
    if verbose:
        print(f"\nResults -> {run.path}")
        print(f"Total runs: {len(_acc_run_idx)} (computed this call: {n_computed})")

    return str(run.path)


# ---------------------------------------------------------------------------
# Result loading utilities
# ---------------------------------------------------------------------------

def load_observable_results(
    run_dir: Path | str,
) -> Dict[str, Any]:
    """Load observable sweep results from a run directory."""
    run_dir = Path(run_dir)
    am = ArtifactManager(run_dir.parents[2])
    run = am.open_run(artifact=ARTIFACT, name=run_dir.name)
    meta = run.load_meta()
    data = run.load_table()
    return {"meta": meta, "data": data}


def get_best_per_k(run_dir: Path | str) -> Dict[int, Dict[str, Any]]:
    """Get the best (smallest gap) result for each k value."""
    results = load_observable_results(run_dir)
    data = results["data"]
    best_by_k: Dict[int, Dict[str, Any]] = {}
    for i in range(len(data["k"])):
        k = int(data["k"][i])
        g = float(data["gap"][i])
        if k not in best_by_k or g < best_by_k[k]["gap"]:
            best_by_k[k] = {
                "gap": g,
                "lb": float(data["lb"][i]),
                "ub": float(data["ub"][i]),
                "seed": int(data["seed"][i]),
                "run_idx": int(data["run_idx"][i]),
            }
    return best_by_k


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    add_config_arg(p)

    # Model parameters
    p.add_argument("--model", choices=["ising", "heisenberg", "heisenberg_j2"],
                   default="heisenberg")
    p.add_argument("--N", type=int, default=None, help="Number of sites")
    p.add_argument("--boundary", choices=["open", "periodic"], default="periodic")

    p.add_argument("--J", type=float, default=1.0, help="Coupling strength J")
    p.add_argument("--h", type=float, default=0.0, help="Ising transverse field h")
    p.add_argument("--k-ising", type=float, default=0.0, dest="k_ising",
                   help="Ising longitudinal field k")
    p.add_argument("--J2", type=float, default=0.0,
                   help="Heisenberg J2 next-nearest neighbor coupling")

    # Observable
    p.add_argument("--observable", type=str, default=None,
                   choices=sorted(_OBSERVABLE_BUILDERS.keys()),
                   help="Observable to bound")
    p.add_argument("--obs-i", type=int, default=0,
                   help="Site i for two_point observables")
    p.add_argument("--obs-j", type=int, default=1,
                   help="Site j for two_point observables")

    # Energy window
    p.add_argument("--exact-energy", type=float, default=None,
                   help="Exact ground-state energy (upper bound on E_0)")
    p.add_argument("--energy-lb", type=float, default=None,
                   help="Manual energy lower bound (skip SDP pre-compute)")
    p.add_argument("--energy-ub", type=float, default=None,
                   help="Manual energy upper bound (overrides --exact-energy)")

    # Basis parameters
    p.add_argument("--start-level", type=int, default=1)
    p.add_argument("--end-level", type=int, default=2)
    p.add_argument("--end-basis", type=str, default="npa",
                   choices=["npa", "heisenberg_simple", "heisenberg_j2_weak",
                            "heisenberg_j2_strong"])

    # Optimization method
    p.add_argument("--method", choices=["sa", "pt", "bo", "rbm", "random"],
                   default="sa")

    # SA parameters
    p.add_argument("--sa-steps", type=int, default=100)
    p.add_argument("--sa-T-start", type=float, default=2.0)
    p.add_argument("--sa-alpha", type=float, default=0.95)

    # PT parameters
    p.add_argument("--pt-chains", type=int, default=0)
    p.add_argument("--pt-epochs", type=int, default=5)
    p.add_argument("--pt-steps-per-epoch", type=int, default=40)
    p.add_argument("--pt-T-min", type=float, default=0.1)
    p.add_argument("--pt-T-max", type=float, default=2.0)

    # BO parameters
    p.add_argument("--bo-beta", type=float, default=1.0)
    p.add_argument("--bo-n-init", type=int, default=20)
    p.add_argument("--bo-n-iter", type=int, default=50)
    p.add_argument("--bo-candidates-per-iter", type=int, default=100)

    # RBM parameters
    p.add_argument("--rbm-steps", type=int, default=100)

    # Feedback (warm-start chaining)
    p.add_argument("--feedback", action="store_true", default=False)

    # Sweep parameters
    k_group = p.add_mutually_exclusive_group()
    k_group.add_argument("--ks", nargs="+", type=int)
    k_group.add_argument("--k-max", type=int)

    seed_group = p.add_mutually_exclusive_group()
    seed_group.add_argument("--seeds", nargs="+", type=int)
    seed_group.add_argument("--num-seeds", type=int)

    p.add_argument("--mosek-tol", type=float, default=1e-9)
    add_symmetry_args(p)

    p.add_argument("--out-root", type=Path,
                   default=Path(__file__).resolve().parents[1] / "results")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true", default=True)

    args, _sources = parse_with_config(p, argv)

    # Post-parse validation
    if args.N is None:
        p.error("--N is required (via CLI or config)")
    if args.observable is None:
        p.error("--observable is required (via CLI or config)")
    if args.ks is None and args.k_max is None:
        p.error("--ks or --k-max is required (via CLI or config)")
    if args.seeds is None and args.num_seeds is None:
        p.error("--seeds or --num-seeds is required (via CLI or config)")

    # Build k values
    k_values = sorted(set(args.ks)) if args.ks is not None else list(range(args.k_max + 1))
    seeds = sorted(set(args.seeds)) if args.seeds is not None else list(range(42, 42 + args.num_seeds))

    # Build method params
    if args.method == "sa":
        method_params = {"steps": args.sa_steps, "T_start": args.sa_T_start, "alpha": args.sa_alpha}
    elif args.method == "pt":
        pt_chains = None if int(args.pt_chains) <= 0 else int(args.pt_chains)
        method_params = {
            "num_chains": pt_chains,
            "num_epochs": args.pt_epochs,
            "steps_per_epoch": args.pt_steps_per_epoch,
            "T_min": args.pt_T_min, "T_max": args.pt_T_max,
        }
    elif args.method == "bo":
        method_params = {
            "beta": args.bo_beta, "n_init": args.bo_n_init,
            "n_iter": args.bo_n_iter,
            "candidates_per_iter": args.bo_candidates_per_iter,
        }
    elif args.method == "rbm":
        method_params = {"steps": args.rbm_steps}
    else:
        method_params = {}

    model_params = model_params_from_args(args)
    symmetry_config = symmetry_config_from_args(args)

    # Observable kwargs (for two_point correlators)
    observable_kwargs: Dict[str, Any] = {}
    if args.observable == "two_point_z":
        observable_kwargs = {"i": args.obs_i, "j": args.obs_j}

    # Energy bounds
    H_dict_fn = hamiltonian_dict_fn(args.model)
    hamiltonian_dict = H_dict_fn(N=args.N, boundary=args.boundary, **model_params)
    sym_manager = SymmetryManager(N=args.N, **symmetry_config)

    if args.energy_lb is not None and args.energy_ub is not None:
        e_lb, e_ub = args.energy_lb, args.energy_ub
    else:
        starting_set, adding_set, _ = build_basis_sets(
            N=args.N, start_level=args.start_level,
            end_basis=args.end_basis, end_level=args.end_level,
        )
        e_lb, e_ub = precompute_energy_bounds(
            N=args.N,
            hamiltonian_dict=hamiltonian_dict,
            symmetry_manager=sym_manager,
            starting_set=starting_set,
            adding_set=adding_set,
            exact_energy=args.exact_energy or args.energy_ub,
            mosek_tol=args.mosek_tol,
            verbose=args.verbose,
        )
        if args.energy_lb is not None:
            e_lb = args.energy_lb
        if args.energy_ub is not None:
            e_ub = args.energy_ub

    name = args.name or _default_name(
        args.model, args.N, args.boundary, args.method,
        args.start_level, args.end_level, args.end_basis,
        args.observable, getattr(args, "feedback", False), model_params,
    )

    run_dir = compute_and_save(
        N=args.N,
        model_name=args.model,
        model_params=model_params,
        boundary=args.boundary,
        observable_name=args.observable,
        observable_kwargs=observable_kwargs,
        energy_lb=e_lb,
        energy_ub=e_ub,
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
        name=name,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
        feedback=getattr(args, "feedback", False),
    )

    print(f"Results -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
