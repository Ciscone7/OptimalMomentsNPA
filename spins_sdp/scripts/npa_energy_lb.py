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
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from spins_sdp import models
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
from spins_sdp.sdp import solve_pauli_relaxation


SCHEMA_VERSION = 1
ARTIFACT_NAME = "spin_moment_energy_lb"


_FIELDS = {
    "basis_size": np.dtype("int64"),
    "E_lb": np.dtype("float64"),
    "t_best": np.dtype("float64"),
    "t_avg": np.dtype("float64"),
}


def _hamiltonian_dict_fn(model_name: str):
    if model_name == "ising":
        return models.ising_hamiltonian_dict
    if model_name == "heisenberg":
        return models.heisenberg_hamiltonian_dict
    if model_name == "heisenberg_j2":
        return models.heisenberg_j2_hamiltonian_dict
    raise SystemExit(f"Unknown model: {model_name}")


def _model_params_from_args(args: argparse.Namespace) -> Dict[str, float]:
    if args.model == "ising":
        return {"J": float(args.J), "h": float(args.h), "k": float(args.k)}
    if args.model == "heisenberg":
        return {}
    return {"J2": float(args.J2)}


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


def _time_best_avg(fn, repeats: int) -> Tuple[float, float, float]:
    times: List[float] = []
    out: Optional[float] = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    assert out is not None
    return out, float(min(times)), float(sum(times) / len(times))


def _parse_ns_from_args(args: argparse.Namespace) -> List[int]:
    if args.Ns is not None and len(args.Ns) > 0:
        Ns = list(args.Ns)
    else:
        if args.N_min is None or args.N_max is None:
            raise SystemExit("Provide either --Ns or both --N-min and --N-max")
        if args.N_min > args.N_max:
            raise SystemExit("--N-min must be <= --N-max")
        Ns = list(range(args.N_min, args.N_max + 1))

    Ns = [int(n) for n in Ns]
    if any(n <= 0 for n in Ns):
        raise SystemExit("All N must be >= 1")
    return sorted(set(Ns))

def compute_and_save(
    *,
    Ns: Iterable[int],
    npa_level: int,
    model_name: str,
    model_params: Dict[str, float],
    basis_name: str,
    boundary: str,
    solver: str,
    mosek_tol: float,
    solver_opts_json: Optional[str],
    repeats: int,
    out_root: Path,
    resume: bool,
    force: bool,
    verbose: bool,
) -> Path:
    # Allow passing solver opts as a JSON string from CLI.
    solver_opts: Optional[Dict[str, Any]] = None
    if solver_opts_json:
        try:
            parsed = json.loads(solver_opts_json)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--solver-opts-json must be valid JSON: {e}")
        if not isinstance(parsed, dict):
            raise SystemExit("--solver-opts-json must decode to a JSON object (dict)")
        solver_opts = parsed

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
        "solver": solver,
        "mosek_tol": float(mosek_tol),
        "solver_opts": solver_opts or None,
        "repeats": int(repeats),
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

    H_dict_fn = _hamiltonian_dict_fn(model_name)

    for N in missing:
        basis = _basis_words(basis_name=basis_name, N=N, level=npa_level, boundary=boundary)
        operator = H_dict_fn(N=N, boundary=boundary, **model_params)

        def run_one() -> float:
            return solve_pauli_relaxation(
                basis,
                operator,
                sense="min",
                solver=solver,
                mosek_tol=mosek_tol,
                solver_opts=solver_opts,
                verbose=verbose,
            )

        val, tb, ta = _time_best_avg(run_one, repeats=repeats)
        existing[int(N)] = {
            "basis_size": int(len(basis)),
            "E_lb": float(val),
            "t_best": float(tb),
            "t_avg": float(ta),
        }

    save_records_npz(data_path, key="N", fields=_FIELDS, records=existing)

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

    p.add_argument("--solver", type=str, default="MOSEK")
    p.add_argument("--mosek-tol", type=float, default=1e-9)
    p.add_argument(
        "--solver-opts-json",
        type=str,
        default=None,
        help="Optional JSON dict to pass through as solver_opts (merged on top of defaults in solve_pauli_relaxation)",
    )

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
    Ns = _parse_ns_from_args(args)

    model_params = _model_params_from_args(args)

    out_dir = compute_and_save(
        Ns=Ns,
        npa_level=args.npa_level,
        model_name=args.model,
        model_params=model_params,
        basis_name=args.basis,
        boundary=args.boundary,
        solver=args.solver,
        mosek_tol=args.mosek_tol,
        solver_opts_json=args.solver_opts_json,
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
