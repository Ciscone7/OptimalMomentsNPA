"""Compute exact ground-state energy for spin-chain models and save results.

This script writes results in a resumable way:
  - Results are grouped by a hash of the model+solver configuration.
  - If a `data.npz` already exists for the config, only missing N values are computed.

On-disk layout (default):
    spins_sdp/results/spin_exact_ground_energy/v1/<config_hash>/
    meta.json
    data.npz

`data.npz` contains arrays keyed by:
  - N (int)
  - dim (int)                 # 2**N
  - E0 (float)                # ground state energy
  - t_best (float)            # best timing over repeats (seconds)
  - t_avg (float)             # avg timing over repeats (seconds)

Examples:
    python spins_sdp/scripts/ising_exact_ground_energy.py --model ising --N-min 5 --N-max 10 --J 1 --h 0.7 --k 0.3 --boundary periodic
    python spins_sdp/scripts/ising_exact_ground_energy.py --model heisenberg --Ns 8 10 12 --boundary periodic
    python spins_sdp/scripts/ising_exact_ground_energy.py --model heisenberg_j2 --J2 0.7 --N-min 4 --N-max 8
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from spins_sdp.scripts._artifact_io import (
    config_hash,
    load_records_npz,
    save_records_npz,
    upsert_meta_json,
    utc_now_iso,
)
from spins_sdp.scripts._common import (
    hamiltonian_exact_fn,
    model_params_from_args,
    parse_ns_from_args,
    time_best_avg,
)


SCHEMA_VERSION = 1
ARTIFACT_NAME = "spin_exact_ground_energy"


_FIELDS = {
    "dim": np.dtype("int64"),
    "E0": np.dtype("float64"),
    "t_best": np.dtype("float64"),
    "t_avg": np.dtype("float64"),
}


def compute_and_save(
    *,
    Ns: Iterable[int],
    model_name: str,
    model_params: Dict[str, float],
    boundary: str,
    repeats: int,
    out_root: Path,
    resume: bool,
    force: bool,
) -> Path:
    config: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT_NAME,
        "model": model_name,
        "method": "exact_diagonalization",
        "params": dict(model_params),
        "boundary": boundary,
        "repeats": int(repeats),
    }

    cfg_hash = config_hash(config)
    run_dir = out_root / ARTIFACT_NAME / f"v{SCHEMA_VERSION}" / cfg_hash
    meta_path = run_dir / "meta.json"
    data_path = run_dir / "data.npz"

    existing = load_records_npz(data_path, key="N", fields=_FIELDS) if resume else {}

    requested = sorted(set(int(n) for n in Ns))
    missing = [n for n in requested if (n not in existing) or force]

    # Compute missing
    H_fn = hamiltonian_exact_fn(model_name)
    for N in missing:
        def run_one() -> float:
            H = H_fn(N, boundary=boundary, **model_params)
            evals = H.eigenenergies(eigvals=1)
            return float(np.asarray(evals)[0])

        E0, tb, ta = time_best_avg(run_one, repeats=repeats)
        existing[int(N)] = {
            "dim": int(2**N),
            "E0": float(E0),
            "t_best": float(tb),
            "t_avg": float(ta),
        }

    # Persist
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

    # Ising params
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--k", type=float, default=0.0)

    # Heisenberg J2 param
    p.add_argument("--J2", type=float, default=0.0)
    p.add_argument("--boundary", choices=["open", "periodic"], default="periodic")

    p.add_argument("--repeats", type=int, default=1, help="Timing repeats per N")

    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Root directory to write results (default: spins_sdp/results)",
    )

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Skip already-computed N")
    p.add_argument("--force", action="store_true", help="Recompute requested N even if present")

    args = p.parse_args(argv)
    Ns = parse_ns_from_args(args)

    model_params = model_params_from_args(args)

    out_dir = compute_and_save(
        Ns=Ns,
        model_name=args.model,
        model_params=model_params,
        boundary=args.boundary,
        repeats=args.repeats,
        out_root=args.out_root,
        resume=args.resume,
        force=args.force,
    )

    print(f"Wrote results to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
