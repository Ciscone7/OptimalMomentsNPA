"""Compute observable (half_chain_corr_x) bounds via moment relaxation for Heisenberg model.

For each system size N and NPA level, this script:
  1. Loads energy bounds: SDP lower bound (from NPA energy data) and DMRG upper bound.
  2. Builds the half_chain_corr_x observable.
  3. Builds the NPA basis and calls bound_observable() to get certified LB/UB on <O>.
  4. Saves results atomically (resumable).

On-disk layout::

    scripts/results/spin_observable_bound/v1/<run_name>/
        meta.json      <- config, provenance, timestamps
        data.npz       <- N, basis_size, obs_lb, obs_ub, energy_lb, energy_ub, t_elapsed

Usage::

    python scripts/spins/observable_bound_sweep.py --npa-level 1 --N-min 4 --N-max 25
    python scripts/spins/observable_bound_sweep.py --npa-level 2 --N-min 4 --N-max 25
    python scripts/spins/observable_bound_sweep.py --npa-level 3 --N-min 4 --N-max 15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from artifact_manager import ArtifactManager
from spins.basis_builder import generate_npa_basis, generate_heisenberg_paper_basis
from spins.models import heisenberg_hamiltonian_dict
from spins.pauli_logic import PauliWord
from spins.spins_sdp import bound_observable
from spins.symmetry import SymmetryManager


ARTIFACT = "spin_observable_bound"
RESULTS_DIR = _SCRIPTS_DIR / "results"

FIELDS = {
    "basis_size": np.dtype("int64"),
    "obs_lb":     np.dtype("float64"),
    "obs_ub":     np.dtype("float64"),
    "energy_lb":  np.dtype("float64"),
    "energy_ub":  np.dtype("float64"),
    "t_elapsed":  np.dtype("float64"),
}

# ---------------------------------------------------------------------------
# Observable builder
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
# Energy bound loaders
# ---------------------------------------------------------------------------

def _load_energy_lb_data(npa_level: float) -> Dict[int, float]:
    """Load SDP energy lower bounds for the given NPA level."""
    tag = f"npa{npa_level:g}"
    path = RESULTS_DIR / f"spin_moment_energy_lb/v1/heisenberg_{tag}_periodic_N_sweep/data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Energy LB data not found: {path}")
    d = np.load(path)
    return dict(zip(d["N"].tolist(), d["E_lb"].tolist()))


def _load_energy_lb_paper() -> Dict[int, float]:
    """Load SDP energy lower bounds for the paper (heisenberg_simple) basis."""
    path = RESULTS_DIR / "spin_moment_energy_lb/v1/heisenberg_paper_basis_periodic_N_sweep/data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Paper basis energy LB data not found: {path}")
    d = np.load(path)
    return dict(zip(d["N"].tolist(), d["E_lb"].tolist()))


def _load_dmrg_energy_ub() -> Dict[int, float]:
    """Load DMRG energy upper bounds."""
    path = RESULTS_DIR / "spin_dmrg_energy_ub/v1/heisenberg_pbc_J1_chi256_consSz/data.npz"
    if not path.exists():
        raise FileNotFoundError(f"DMRG data not found: {path}")
    d = np.load(path)
    return dict(zip(d["N"].tolist(), d["E"].tolist()))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def compute_and_save(
    *,
    Ns: List[int],
    npa_level: float,
    basis_name: str = "npa",
    mosek_tol: float = 1e-6,
    resume: bool = True,
    force: bool = False,
    verbose: bool = False,
) -> str:
    """Run the observable bound sweep and save results."""

    # Load energy bounds
    if basis_name == "paper":
        elb_map = _load_energy_lb_paper()
    else:
        elb_map = _load_energy_lb_data(npa_level)
    eub_map = _load_dmrg_energy_ub()

    if basis_name == "paper":
        name = "heisenberg_paper_basis_periodic_half_chain_corr_x"
    else:
        name = f"heisenberg_npa{npa_level:g}_periodic_half_chain_corr_x"
    config = {
        "model": "heisenberg",
        "boundary": "periodic",
        "observable": "half_chain_corr_x",
        "npa_level": npa_level,
        "mosek_tol": mosek_tol,
        "energy_lb_source": f"npa{npa_level:g}_sdp",
        "energy_ub_source": "dmrg_chi256",
    }

    am = ArtifactManager(RESULTS_DIR)
    run = am.create_run(artifact=ARTIFACT, name=name, config=config)

    existing = run.load_records(key="N", fields=FIELDS) if resume else {}
    requested = sorted(set(int(n) for n in Ns))

    # Filter N values with available energy bounds
    valid = []
    for n in requested:
        if n not in elb_map:
            print(f"  [skip] N={n}: no NPA{npa_level} energy LB data")
            continue
        if n not in eub_map:
            print(f"  [skip] N={n}: no DMRG energy UB data")
            continue
        if (n in existing) and not force:
            continue
        valid.append(n)

    if not valid:
        print("Nothing to compute (all N values already done or skipped).")
        return str(run.path)

    print(f"Observable bound sweep: NPA{npa_level}, N={valid}")

    try:
        pbar = tqdm(valid, desc=f"NPA{npa_level} obs bound")
        for N in pbar:
            pbar.set_postfix({"N": N})

            energy_lb = elb_map[N]
            energy_ub = eub_map[N]

            # When bounds nearly coincide (exact convergence at small N),
            # widen slightly to avoid an empty feasible region.
            if energy_lb > energy_ub - 1e-8:
                mid = 0.5 * (energy_lb + energy_ub)
                energy_lb = mid - 1e-7
                energy_ub = mid + 1e-7

            hamiltonian = heisenberg_hamiltonian_dict(N, boundary="periodic")
            observable = build_half_chain_corr_x(N)
            if basis_name == "paper":
                basis = generate_heisenberg_paper_basis(N)
            else:
                basis = generate_npa_basis(N, k=int(npa_level)).words
            sym = SymmetryManager.default_for_heisenberg(N)

            t0 = time.perf_counter()
            result = bound_observable(
                basis=basis,
                hamiltonian=hamiltonian,
                observable=observable,
                energy_lb=energy_lb,
                energy_ub=energy_ub,
                symmetry_manager=sym,
                mosek_tol=mosek_tol,
                verbose=verbose,
            )
            elapsed = time.perf_counter() - t0

            existing[N] = {
                "basis_size": len(basis),
                "obs_lb":     float(result.lb),
                "obs_ub":     float(result.ub),
                "energy_lb":  float(energy_lb),
                "energy_ub":  float(energy_ub),
                "t_elapsed":  float(elapsed),
            }

            pbar.write(
                f"  N={N:3d}  basis={len(basis):5d}  "
                f"obs=[{result.lb:.8f}, {result.ub:.8f}]  "
                f"E=[{energy_lb:.6f}, {energy_ub:.6f}]  "
                f"t={elapsed:.2f}s"
            )

            # Atomic checkpoint
            run.save_records(key="N", fields=FIELDS, records=existing)
            run.update_meta(Ns_present=sorted(existing.keys()))

    finally:
        run.update_meta(Ns_present=sorted(existing.keys()))

    print(f"Results -> {run.path}")
    return str(run.path)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--npa-level", type=float, default=1,
                   help="NPA level k for the moment relaxation basis (supports 1.5)")
    p.add_argument("--basis", choices=["npa", "paper"], default="npa",
                   help="Basis type: 'npa' for NPA hierarchy, 'paper' for PRX local basis")

    n_group = p.add_mutually_exclusive_group()
    n_group.add_argument("--Ns", nargs="+", type=int)
    n_group.add_argument("--N-min", dest="N_min", type=int, default=4)
    p.add_argument("--N-max", dest="N_max", type=int, default=25)

    p.add_argument("--mosek-tol", type=float, default=1e-6)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")

    args = p.parse_args(argv)

    if args.Ns:
        Ns = list(args.Ns)
    else:
        Ns = list(range(args.N_min, args.N_max + 1))

    compute_and_save(
        Ns=Ns,
        npa_level=args.npa_level,
        basis_name=args.basis,
        mosek_tol=args.mosek_tol,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
