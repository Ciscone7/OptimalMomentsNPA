"""Compute DMRG upper bounds for spin-chain models and save results.

This script runs DMRG (via TeNPy) to compute variational upper bounds for the ground
state energy. Results are saved in a resumable way using config hashing, mirroring
the pattern of the other artifact scripts.

On-disk layout (default):
    spins_sdp/results/spin_dmrg_energy_ub/v1/<config_hash>/
    ├── meta.json
    └── data.npz

`data.npz` contains arrays keyed by:
  - N (int)
  - E (float)              # total energy (variational upper bound)
  - e (float)              # energy per site (E / N)
  - chi_final (int)        # final bond dimension
  - elapsed_s (float)      # wall-clock time (seconds)

Examples:
    python -m spins_sdp.scripts.dmrg --model heisenberg --N-min 4 --N-max 12 --chi-max 256 --boundary periodic
    python -m spins_sdp.scripts.dmrg --model ising --Ns 6 8 10 --J 1.0 --h 0.5 --chi-max 128 --boundary periodic
    python -m spins_sdp.scripts.dmrg --model heisenberg_j2 --Ns 8 --J2 0.5 --chi-max 512 --mixer --boundary periodic
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from spins_sdp.scripts._artifact_io import (
    config_hash,
    load_records_npz,
    save_records_npz,
    upsert_meta_json,
    utc_now_iso,
)
from spins_sdp.scripts._common import (
    model_params_from_args,
    parse_ns_from_args,
)

# Import model builders and DMRG runner from variational module
from spins_sdp.variational import (
    build_heisenberg_pbc_model,
    build_heisenberg_j1j2_pbc_model,
    build_ising_pbc_model,
    initial_product_state,
    dmrg_upper_bound,
)


SCHEMA_VERSION = 1
ARTIFACT_NAME = "spin_dmrg_energy_ub"

# Minimum N for two-site DMRG with PBC (TeNPy requires L > n_active_sites)
MIN_N_DMRG = 4


_FIELDS = {
    "E": np.dtype("float64"),
    "e": np.dtype("float64"),
    "chi_final": np.dtype("int64"),
    "elapsed_s": np.dtype("float64"),
}


def _build_model(
    model_name: str,
    N: int,
    model_params: Dict[str, float],
    conserve: Optional[str],
):
    """Build a TeNPy model for DMRG.
    
    Note: All models use periodic boundary conditions for the Hamiltonian.
    """
    if model_name == "heisenberg":
        return build_heisenberg_pbc_model(N, J=model_params.get("J", 1.0), conserve=conserve)
    elif model_name == "heisenberg_j2":
        return build_heisenberg_j1j2_pbc_model(
            N, J1=model_params.get("J", 1.0), J2=model_params.get("J2", 0.0), conserve=conserve
        )
    elif model_name == "ising":
        return build_ising_pbc_model(
            N,
            J=model_params.get("J", 1.0),
            h=model_params.get("h", 0.0),
            k=model_params.get("k", 0.0),
            conserve=conserve,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _resolve_conserve(model_name: str, conserve_arg: str, model_params: Dict[str, float]) -> Optional[str]:
    """Resolve the symmetry conservation setting.
    
    For Ising with transverse field (h != 0), Sz conservation must be disabled.
    """
    if conserve_arg.lower() == "none":
        return None
    
    conserve = conserve_arg
    
    if model_name == "ising" and conserve in {"Sz", "best"}:
        if abs(model_params.get("h", 0.0)) > 0:
            # Transverse field breaks Sz conservation
            return None
    
    return conserve


def compute_and_save(
    *,
    Ns: List[int],
    model_name: str,
    model_params: Dict[str, float],
    conserve: str,
    init_state: str,
    chi_max: int,
    svd_min: float,
    max_E_err: float,
    dchi: int,
    nsweeps: int,
    mixer: bool,
    combine: bool,
    out_root: Path,
    resume: bool,
    force: bool,
    verbose: bool,
) -> Path:
    """Run DMRG sweeps and save results.
    
    Args:
        Ns: List of system sizes to compute.
        model_name: "ising", "heisenberg", or "heisenberg_j2".
        model_params: Model-specific parameters (J, h, k, J2).
        conserve: Symmetry sector ("Sz", "None").
        init_state: Initial state kind ("neel", "all_up").
        chi_max: Maximum bond dimension.
        svd_min: SVD cutoff for truncation.
        max_E_err: Energy convergence threshold.
        dchi: Bond dimension increment per sweep.
        nsweeps: Number of sweeps per chi increment.
        mixer: Whether to use DMRG mixer.
        combine: Whether to use combined two-site update.
        out_root: Root directory for results.
        resume: If True, skip already-computed N values.
        force: If True, recompute even if present.
        verbose: Print progress.
    
    Returns:
        Path to the run directory.
    """
    # Build config for hashing
    config: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT_NAME,
        "model": model_name,
        "method": "dmrg",
        "params": dict(sorted(model_params.items())),
        "boundary": "periodic",  # All our DMRG models use PBC
        "conserve": conserve,
        "init_state": init_state,
        "chi_max": int(chi_max),
        "svd_min": float(svd_min),
        "max_E_err": float(max_E_err),
        "dchi": int(dchi),
        "nsweeps": int(nsweeps),
        "mixer": bool(mixer),
        "combine": bool(combine),
    }
    
    cfg_hash = config_hash(config)
    run_dir = out_root / ARTIFACT_NAME / f"v{SCHEMA_VERSION}" / cfg_hash
    meta_path = run_dir / "meta.json"
    data_path = run_dir / "data.npz"
    
    # Load existing data if resuming
    existing = load_records_npz(data_path, key="N", fields=_FIELDS) if resume else {}
    
    requested = sorted(set(int(n) for n in Ns))
    
    # Filter out N values too small for two-site DMRG with PBC
    too_small = [n for n in requested if n < MIN_N_DMRG]
    if too_small and verbose:
        print(f"Warning: Skipping N={too_small} (DMRG with PBC requires N >= {MIN_N_DMRG})")
    requested = [n for n in requested if n >= MIN_N_DMRG]
    
    if not requested:
        raise ValueError(f"No valid N values to compute. DMRG with PBC requires N >= {MIN_N_DMRG}.")
    
    missing = [n for n in requested if (n not in existing) or force]
    
    if verbose:
        print(f"Model: {model_name}, boundary=periodic")
        print(f"DMRG params: chi_max={chi_max}, mixer={mixer}, combine={combine}")
        print(f"Requested N: {requested}")
        print(f"Already computed: {sorted(existing.keys())}")
        print(f"To compute: {missing}")
    
    # Prepare metadata
    meta: Dict[str, Any] = {
        **config,
        "config_hash": cfg_hash,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "Ns_present": sorted(existing.keys()),
    }
    
    # Write initial metadata
    upsert_meta_json(meta_path, meta)
    
    # Resolve conserve setting
    conserve_resolved = _resolve_conserve(model_name, conserve, model_params)
    
    try:
        pbar = tqdm(missing, desc="DMRG sweep", disable=not verbose)
        for N in pbar:
            pbar.set_postfix({"N": int(N), "done": len(existing), "total": len(requested)})
            
            # Build model and initial state
            model = _build_model(model_name, N, model_params, conserve_resolved)
            psi = initial_product_state(model, kind=init_state)
            
            # Run DMRG
            E, psi_out, info, elapsed_s = dmrg_upper_bound(
                model, psi,
                chi_max=chi_max,
                svd_min=svd_min,
                max_E_err=max_E_err,
                mixer=mixer,
                dchi=dchi,
                nsweeps=nsweeps,
                combine=combine,
            )
            
            # Record result
            existing[int(N)] = {
                "E": float(E),
                "e": float(E) / N,
                "chi_final": int(max(psi_out.chi)) if hasattr(psi_out.chi, '__iter__') else int(psi_out.chi),
                "elapsed_s": float(elapsed_s),
            }
            
            # Checkpoint immediately: atomic save
            save_records_npz(data_path, key="N", fields=_FIELDS, records=existing)
            
            # Update metadata
            meta["Ns_present"] = sorted(existing.keys())
            meta["updated_at"] = utc_now_iso()
            upsert_meta_json(meta_path, meta)
            
    finally:
        # Final metadata update
        meta["Ns_present"] = sorted(existing.keys())
        meta["updated_at"] = utc_now_iso()
        upsert_meta_json(meta_path, meta)
    
    if verbose:
        print(f"\nResults saved to: {run_dir}")
        print(f"Total N values: {len(existing)}")
    
    return run_dir


# -----------------------------------------------------------------------------
# Result loading utilities
# -----------------------------------------------------------------------------

def load_dmrg_results(run_dir: Path) -> Dict[str, Any]:
    """Load DMRG results from a run directory.
    
    Args:
        run_dir: Path to the run directory.
    
    Returns:
        Dict with:
            - meta: metadata dict
            - data: dict keyed by N -> {E, e, chi_final, elapsed_s}
    """
    import json
    
    meta_path = run_dir / "meta.json"
    data_path = run_dir / "data.npz"
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    data = load_records_npz(data_path, key="N", fields=_FIELDS)
    
    return {"meta": meta, "data": data}


def list_dmrg_runs(
    root: Optional[Path] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List available DMRG runs.
    
    Args:
        root: Root directory (default: spins_sdp/results).
        model: Filter by model name.
    
    Returns:
        List of metadata dicts for matching runs.
    """
    import json
    
    if root is None:
        root = Path(__file__).resolve().parents[1] / "results"
    
    artifact_dir = root / ARTIFACT_NAME / f"v{SCHEMA_VERSION}"
    
    if not artifact_dir.exists():
        return []
    
    runs = []
    for run_dir in artifact_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            
            if model is not None and meta.get("model") != model:
                continue
            
            meta["_run_dir"] = str(run_dir)
            runs.append(meta)
        except Exception:
            continue
    
    return runs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # N values
    n_group = p.add_mutually_exclusive_group(required=True)
    n_group.add_argument("--Ns", nargs="+", type=int, help="Explicit list of N values")
    n_group.add_argument("--N-min", dest="N_min", type=int, help="Minimum N (inclusive)")
    p.add_argument("--N-max", dest="N_max", type=int, help="Maximum N (inclusive, required with --N-min)")
    
    # Model selection
    p.add_argument("--model", choices=["ising", "heisenberg", "heisenberg_j2"], default="heisenberg")
    
    # Ising params
    p.add_argument("--J", type=float, default=1.0, help="Coupling strength J")
    p.add_argument("--h", type=float, default=0.0, help="Ising transverse field h")
    p.add_argument("--k", type=float, default=0.0, dest="k_ising", help="Ising longitudinal field k")
    
    # Heisenberg J2 param
    p.add_argument("--J2", type=float, default=0.0, help="Heisenberg J2 next-nearest neighbor coupling")
    
    # Symmetry and initial state
    p.add_argument("--conserve", type=str, default="Sz",
                   help="Symmetry conservation: 'Sz' for U(1) or 'None' to disable")
    p.add_argument("--init", type=str, default="neel", choices=["neel", "all_up"],
                   help="Initial product state")
    
    # DMRG accuracy/performance
    p.add_argument("--chi-max", type=int, default=256, help="Maximum bond dimension")
    p.add_argument("--svd-min", type=float, default=1e-10, help="SVD cutoff for truncation")
    p.add_argument("--max-E-err", type=float, default=1e-10, help="Energy convergence threshold")
    p.add_argument("--dchi", type=int, default=64, help="Bond dimension increment per sweep")
    p.add_argument("--nsweeps", type=int, default=2, help="Number of sweeps per chi increment")
    p.add_argument("--mixer", action="store_true", help="Enable DMRG mixer (helps avoid local minima)")
    p.add_argument("--combine", action="store_true", help="Use combined two-site update")
    
    # Output
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Root directory for results (default: spins_sdp/results)",
    )
    
    # Resume/force
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true", help="Recompute all N values")
    p.add_argument("--verbose", action="store_true", default=True)
    
    args = p.parse_args(argv)
    
    Ns = parse_ns_from_args(args)
    
    # Build model params - handle the k/k_ising naming
    if args.model == "ising":
        model_params = {"J": float(args.J), "h": float(args.h), "k": float(args.k_ising)}
    elif args.model == "heisenberg":
        model_params = {"J": float(args.J)}
    elif args.model == "heisenberg_j2":
        model_params = {"J": float(args.J), "J2": float(args.J2)}
    else:
        model_params = model_params_from_args(args)
    
    run_dir = compute_and_save(
        Ns=Ns,
        model_name=args.model,
        model_params=model_params,
        conserve=args.conserve,
        init_state=args.init,
        chi_max=args.chi_max,
        svd_min=args.svd_min,
        max_E_err=args.max_E_err,
        dchi=args.dchi,
        nsweeps=args.nsweeps,
        mixer=args.mixer,
        combine=args.combine,
        out_root=args.out_root,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
    )
    
    print(f"Results saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
