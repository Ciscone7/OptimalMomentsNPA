"""Compute a moment-relaxation lower bound for spin-chain models and save results.

This script is designed to decouple expensive SDP solves from plotting.
It writes results in a resumable way:
  - Results are grouped by a hash of the model+relaxation configuration.
  - If a `data.npz` already exists for the config, only missing N values are computed.

On-disk layout (default):
    spins_sdp/results/spin_moment_energy_lb/v1/<config_hash>/
    meta.json
    data.npz

`data.npz` contains arrays keyed by:
  - N (int)
  - basis_size (int)          # number of basis words
  - E_lb (float)              # relaxation optimum (lower bound when sense='min')
  - t_best (float)            # best timing over repeats (seconds)
  - t_avg (float)             # avg timing over repeats (seconds)

Examples:
    python -m spins_sdp.scripts.ising_npa_energy_lb --model ising --basis npa --npa-level 2 --N-min 5 --N-max 10 --J 1 --h 0.7 --k 0.3 --boundary periodic --solver MOSEK
    python spins_sdp/scripts/ising_npa_energy_lb.py --model heisenberg --basis heisenberg_simple --Ns 8 10 12 --boundary periodic --solver MOSEK
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from spins_sdp.basis_builder import (
    generate_heisenberg_j2_basis_strong,
    generate_heisenberg_j2_basis_weak,
    generate_heisenberg_paper_basis,
    generate_npa_basis,
)
from spins_sdp.scripts._artifact_io import (
    config_hash,
    load_records_npz,
    save_records_npz,
    upsert_meta_json,
    utc_now_iso,
)
from spins_sdp.scripts._common import (
    add_symmetry_args,
    hamiltonian_dict_fn,
    model_params_from_args,
    parse_ns_from_args,
    symmetry_config_from_args,
    time_best_avg,
)
from spins_sdp.sdp import solve_pauli_relaxation
from spins_sdp.symmetry import SymmetryManager


SCHEMA_VERSION = 1
ARTIFACT_NAME = "spin_moment_energy_lb"


_FIELDS = {
    "basis_size": np.dtype("int64"),
    "E_lb": np.dtype("float64"),
    "t_best": np.dtype("float64"),
    "t_avg": np.dtype("float64"),
}


def _basis_words(
    *,
    basis_name: str,
    N: int,
    level: int,
    boundary: str,
) -> List[Any]:
    # NOTE: the Heisenberg "paper"/J2 bases currently implement periodic boundary conditions only.
    if basis_name in {"heisenberg_simple", "heisenberg_j2_weak", "heisenberg_j2_strong"} and boundary != "periodic":
        raise SystemExit(f"basis={basis_name} requires --boundary periodic")

    if basis_name == "npa":
        return generate_npa_basis(N=N, k=level).words
    if basis_name == "heisenberg_simple":
        return generate_heisenberg_paper_basis(N=N)
    if basis_name == "heisenberg_j2_weak":
        return generate_heisenberg_j2_basis_weak(N=N)
    if basis_name == "heisenberg_j2_strong":
        return generate_heisenberg_j2_basis_strong(N=N)
    raise SystemExit(f"Unknown basis: {basis_name}")



def compute_and_save(
    *,
    Ns: Iterable[int],
    npa_level: int,
    model_name: str,
    model_params: Dict[str, float],
    basis_name: str,
    boundary: str,
    symmetry_config: Dict[str, bool],
    mosek_tol: float,
    repeats: int,
    out_root: Path,
    resume: bool,
    force: bool,
    verbose: bool,
) -> Path:
    config: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT_NAME,
        "model": model_name,
        "method": "pauli_moment_relaxation",
        "params": dict(model_params),
        "boundary": boundary,
        "basis": basis_name,
        "npa_level": int(npa_level) if basis_name == "npa" else None,
        "sense": "min",
        "mosek_tol": float(mosek_tol),
        "repeats": int(repeats),
        "symmetry": symmetry_config,
    }

    if basis_name != "npa":
        config.pop("npa_level", None)

    cfg_hash = config_hash(config)
    run_dir = out_root / ARTIFACT_NAME / f"v{SCHEMA_VERSION}" / cfg_hash
    meta_path = run_dir / "meta.json"
    data_path = run_dir / "data.npz"

    existing = load_records_npz(data_path, key="N", fields=_FIELDS) if resume else {}

    requested = sorted(set(int(n) for n in Ns))
    missing = [n for n in requested if (n not in existing) or force]

    H_dict_fn = hamiltonian_dict_fn(model_name)

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

    # Ensure meta.json exists early and is always updated.
    upsert_meta_json(meta_path, meta)

    try:
        pbar = tqdm(missing, desc="Moment relaxation sweep")
        for N in pbar:
            pbar.set_postfix({"N": int(N), "done": len(existing), "total": len(requested)})
            basis = _basis_words(basis_name=basis_name, N=N, level=npa_level, boundary=boundary)
            operator = H_dict_fn(N=N, boundary=boundary, **model_params)
            sym_manager = SymmetryManager(N=N, **symmetry_config)

            def run_one() -> float:
                return solve_pauli_relaxation(
                    basis,
                    operator,
                    symmetry_manager=sym_manager,
                    sense="min",
                    mosek_tol=mosek_tol,
                    verbose=verbose,
                )

            val, tb, ta = time_best_avg(run_one, repeats=repeats)
            existing[int(N)] = {
                "basis_size": int(len(basis)),
                "E_lb": float(val),
                "t_best": float(tb),
                "t_avg": float(ta),
            }

            # Checkpoint immediately: rewrite full NPZ atomically (temp + replace).
            save_records_npz(data_path, key="N", fields=_FIELDS, records=existing)

            # Also refresh metadata so partial runs are discoverable.
            meta["Ns_present"] = sorted(existing.keys())
            meta["updated_at"] = utc_now_iso()
            upsert_meta_json(meta_path, meta)
    finally:
        meta["Ns_present"] = sorted(existing.keys())
        meta["updated_at"] = utc_now_iso()
        upsert_meta_json(meta_path, meta)

    return run_dir


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)

    n_group = p.add_mutually_exclusive_group(required=True)
    n_group.add_argument("--Ns", nargs="+", type=int, help="Explicit list of N values")
    n_group.add_argument("--N-min", dest="N_min", type=int, help="Minimum N (inclusive)")
    p.add_argument("--N-max", dest="N_max", type=int, help="Maximum N (inclusive) (required with --N-min)")

    p.add_argument("--model", choices=["ising", "heisenberg", "heisenberg_j2"], default="ising")
    p.add_argument(
        "--basis",
        choices=[
            "npa",
            "heisenberg_simple",
            "heisenberg_j2_weak",
            "heisenberg_j2_strong"
        ],
        default="npa",
    )
    p.add_argument("--npa-level", type=int, default=1, help="NPA/moment level k (used when --basis npa)")

    # Ising params
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--k", type=float, default=0.0)

    # Heisenberg J2 param
    p.add_argument("--J2", type=float, default=0.0)
    p.add_argument("--boundary", choices=["open", "periodic"], default="open")

    p.add_argument("--mosek-tol", type=float, default=1e-9)

    add_symmetry_args(p)

    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--verbose", action="store_true")

    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Root directory to write results (default: spins_sdp/results)",
    )

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")

    args = p.parse_args(argv)
    Ns = parse_ns_from_args(args)
    model_params = model_params_from_args(args)
    symmetry_config = symmetry_config_from_args(args)

    out_dir = compute_and_save(
        Ns=Ns,
        npa_level=args.npa_level,
        model_name=args.model,
        model_params=model_params,
        basis_name=args.basis,
        boundary=args.boundary,
        symmetry_config=symmetry_config,
        mosek_tol=args.mosek_tol,
        repeats=args.repeats,
        out_root=args.out_root,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
    )

    print(f"Wrote results to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
