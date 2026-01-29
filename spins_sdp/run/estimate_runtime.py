#!/usr/bin/env python3
"""Estimate total SDP solve time from an experiments config.

This is a *rough* estimator based on a fitted single-SDP solve-time model:

    t_sdp(L) = a * L^b

where L is the SDP basis size (number of monomials / basis words).

We only estimate experiments that run SDPs:
  - full_relaxation (one SDP per N)
  - optimization (many SDPs; one per objective evaluation)

We explicitly ignore:
  - exact diagonalization
  - DMRG

Usage:
  python3 -m spins_sdp.run.estimate_runtime --config spins_sdp/run/experiments_config.json
  python3 -m spins_sdp.run.estimate_runtime --test
  python3 -m spins_sdp.run.estimate_runtime --dry-run  # prints counts + equations only

Notes:
- The default (a, b) comes from the power law benchmark fit in benchmark.ipynb.
- Extrapolating beyond the fit window may be very wrong.
- The fit is based on my ICFO computer: 12th gen i5-12500, 16GB RAM
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple


# Allow running as a script without installing the package:
#   python3 spins_sdp/run/estimate_runtime.py ...
if __package__ is None:  # pragma: no cover
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))


try:
    from spins_sdp.basis_builder import generate_heisenberg_paper_basis, generate_npa_basis
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing runtime dependencies for the estimator.\n"
        "\n"
        "If you're in a fresh environment, create a venv and install deps, e.g.:\n"
        "  python3 -m venv venv && source venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "\n"
        f"Original error: {e}"
    )


# -----------------------------------------------------------------------------
# Fit model
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FitModel:
    a: float = 1.0283544290894659e-07
    b: float = 3.85611475429724
    fit_L_min: int = 20
    fit_L_max: int = 200

    def t_sdp_seconds(self, basis_size: int) -> float:
        return float(self.a) * float(basis_size) ** float(self.b)


def _human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.2f}h"
    days = hours / 24
    return f"{days:.2f}d"


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def pick_mode_block(config: Dict[str, Any], *, test: bool) -> Dict[str, Any]:
    return config["test_mode"] if test else config["experiments"]


# -----------------------------------------------------------------------------
# Basis-size estimators
# -----------------------------------------------------------------------------

def basis_size_for_relaxation(*, basis: str, N: int, npa_level: int | None) -> int:
    """Return basis size used by npa_energy_lb for the given config."""
    # Map user-friendly names -> actual basis builder behavior.
    # In spins_sdp/scripts/npa_energy_lb.py the paper basis is "heisenberg_simple".
    basis_norm = str(basis).strip().lower()

    if basis_norm == "heisenberg_simple":
        return len(generate_heisenberg_paper_basis(N=N))

    if basis_norm == "npa":
        if npa_level is None:
            raise ValueError("basis='npa' requires npa_level in config")
        return len(generate_npa_basis(N=N, k=int(npa_level)).words)

    raise ValueError(f"Unknown relaxation basis '{basis}'. Use 'heisenberg_simple' or 'npa'.")


def build_npa_basis_sets(N: int, start_level: int, end_level: int) -> Tuple[List[Any], List[Any], List[Any]]:
    """Local copy of spins_sdp.scripts._common.build_npa_basis_sets (to avoid heavy imports)."""
    full_basis = generate_npa_basis(N=N, k=end_level)

    starting_set: List[Any] = []
    for level_idx in range(min(start_level + 1, len(full_basis.levels))):
        starting_set.extend(full_basis.levels[level_idx])

    adding_set: List[Any] = []
    for level_idx in range(start_level + 1, min(end_level + 1, len(full_basis.levels))):
        adding_set.extend(full_basis.levels[level_idx])

    final_set = full_basis.words
    return starting_set, adding_set, final_set


# -----------------------------------------------------------------------------
# Counting SDPs
# -----------------------------------------------------------------------------

def n_obj_evals_for_method(method: str, method_cfg: Dict[str, Any]) -> int:
    method = method.lower()

    if method == "random":
        return 1

    if method == "sa":
        steps = int(method_cfg.get("steps", 100))
        return steps + 1

    if method == "pt":
        # Matches optimization_sweep.py's accounting:
        # n_obj_evals = chains * epochs * steps_per_epoch + chains
        chains = int(method_cfg.get("chains", method_cfg.get("num_chains", 4)))
        epochs = int(method_cfg.get("epochs", method_cfg.get("num_epochs", 10)))
        steps_per_epoch = int(method_cfg.get("steps_per_epoch", 50))
        return chains * epochs * steps_per_epoch + chains

    if method == "bo":
        n_init = int(method_cfg.get("n_init", 10))
        n_iter = int(method_cfg.get("n_iter", 30))
        return n_init + n_iter

    raise ValueError(f"Unknown method '{method}'")


# -----------------------------------------------------------------------------
# Estimation
# -----------------------------------------------------------------------------

@dataclass
class Estimate:
    name: str
    n_sdps: int
    total_seconds: float
    notes: str = ""


def estimate_full_relaxation(block: Dict[str, Any], fit: FitModel) -> Estimate | None:
    cfg = block.get("full_relaxation", {})
    if not cfg.get("enabled", False):
        return None

    Ns = list(cfg["Ns"])
    basis = cfg.get("basis", "heisenberg_simple")
    npa_level = cfg.get("npa_level")

    total = 0.0
    out_of_fit = 0

    for N in Ns:
        L = basis_size_for_relaxation(basis=basis, N=int(N), npa_level=npa_level)
        if L < fit.fit_L_min or L > fit.fit_L_max:
            out_of_fit += 1
        total += fit.t_sdp_seconds(L)

    notes = ""
    if out_of_fit:
        notes = f"{out_of_fit}/{len(Ns)} basis sizes outside fit window [{fit.fit_L_min},{fit.fit_L_max}]"

    return Estimate(
        name="full_relaxation",
        n_sdps=len(Ns),
        total_seconds=total,
        notes=notes,
    )


def estimate_optimization(block: Dict[str, Any], fit: FitModel) -> Estimate | None:
    cfg = block.get("optimization", {})
    if not cfg.get("enabled", False):
        return None

    Ns = list(cfg["Ns"])
    start_level = int(cfg["start_level"])
    end_level = int(cfg["end_level"])
    k_max = int(cfg["k_max"])

    methods = cfg.get("methods", {})

    total_seconds = 0.0
    total_sdps = 0
    out_of_fit = 0
    total_basis_calls = 0

    for N in Ns:
        starting_set, adding_set, _final = build_npa_basis_sets(N=int(N), start_level=start_level, end_level=end_level)
        starting_size = len(starting_set)
        adding_size = len(adding_set)

        # The script sweeps k from 0..k_max (inclusive) but cannot exceed adding_set length.
        k_max_eff = min(k_max, adding_size)
        k_values = list(range(0, k_max_eff + 1))

        for method_name, method_cfg in methods.items():
            if not method_cfg.get("enabled", False):
                continue

            num_seeds = int(method_cfg.get("num_seeds", 1))
            evals_per_run = n_obj_evals_for_method(method_name, method_cfg)

            for k in k_values:
                basis_size = starting_size + int(k)
                t_one_eval = fit.t_sdp_seconds(basis_size)

                if basis_size < fit.fit_L_min or basis_size > fit.fit_L_max:
                    out_of_fit += 1
                total_basis_calls += 1

                # Each objective evaluation solves one SDP.
                n_sdps_this_k = num_seeds * evals_per_run
                total_sdps += n_sdps_this_k
                total_seconds += n_sdps_this_k * t_one_eval

    notes = ""
    if total_basis_calls and out_of_fit:
        notes = f"{out_of_fit}/{total_basis_calls} basis sizes outside fit window [{fit.fit_L_min},{fit.fit_L_max}]"

    return Estimate(
        name="optimization",
        n_sdps=total_sdps,
        total_seconds=total_seconds,
        notes=notes,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Estimate runtime from experiments_config.json")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "experiments_config.json",
        help="Path to experiments_config.json",
    )
    p.add_argument("--test", action="store_true", help="Estimate test_mode instead of experiments")
    p.add_argument("--dry-run", action="store_true", help="Only print counts and equations")

    p.add_argument("--a", type=float, default=FitModel.a, help="Fit constant a in t=a*L^b")
    p.add_argument("--b", type=float, default=FitModel.b, help="Fit exponent b in t=a*L^b")
    p.add_argument("--fit-L-min", type=int, default=FitModel.fit_L_min, help="Fit window min L")
    p.add_argument("--fit-L-max", type=int, default=FitModel.fit_L_max, help="Fit window max L")

    args = p.parse_args()

    config = load_config(args.config)
    block = pick_mode_block(config, test=args.test)

    fit = FitModel(a=args.a, b=args.b, fit_L_min=args.fit_L_min, fit_L_max=args.fit_L_max)

    mode_name = "TEST" if args.test else "FULL"
    print("=" * 72)
    print(f"SpinsSDP SDP Runtime Estimate ({mode_name} mode)")
    print("=" * 72)
    print(f"Fit: t(L) = {fit.a:.4g} * L^{fit.b:.4g}   (fit window L∈[{fit.fit_L_min},{fit.fit_L_max}])")

    estimates: List[Estimate] = []

    e_relax = estimate_full_relaxation(block, fit)
    if e_relax:
        estimates.append(e_relax)

    e_opt = estimate_optimization(block, fit)
    if e_opt:
        estimates.append(e_opt)

    if not estimates:
        print("No SDP-based experiments enabled (full_relaxation/optimization).")
        return 0

    print("\nEstimates (serial SDP solve time):")
    total_all = 0.0
    total_sdps = 0

    for e in estimates:
        total_all += e.total_seconds
        total_sdps += e.n_sdps
        line = f"- {e.name}: {e.n_sdps} SDPs, ~{_human_time(e.total_seconds)}"
        if e.notes:
            line += f"  ({e.notes})"
        print(line)

    print(f"- TOTAL: {total_sdps} SDPs, ~{_human_time(total_all)}")

    if args.dry_run:
        print("\n(dry-run: not attempting to account for parallelism / scheduling)")
        return 0

    print("\nObs: if we run SDPs in parallel (like running different optimization strats or different Ns for the full sdp lower bound), we can probably cut the total time a bit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
