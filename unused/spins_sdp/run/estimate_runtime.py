#!/usr/bin/env python3
"""Estimate total SDP solve time from an experiments config.

This is a *rough* estimator based on a fitted single-SDP solve-time model:

    t_sdp(L) = a * L^b

where L is the SDP basis size (number of monomials / basis words), OR the
number of independent SDP variables after symmetry reduction.

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
  python3 -m spins_sdp.run.estimate_runtime --from-artifact  # load fit from symmetry benchmark

Notes:
- The default (a, b) comes from the power law benchmark fit in benchmark.ipynb.
- Extrapolating beyond the fit window may be very wrong.
- The fit is based on my ICFO computer: 12th gen i5-12500, 16GB RAM
- Use --from-artifact to load fit from the latest symmetry benchmark (recommended).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import multiprocessing as mp
from typing import Any, Dict, List, Tuple


# Allow running as a script without installing the package:
#   python3 spins_sdp/run/estimate_runtime.py ...
if __package__ is None:  # pragma: no cover
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))


try:
    from spins.basis_builder import generate_heisenberg_paper_basis, generate_npa_basis
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
    use_n_vars: bool = False  # if True, L is n_vars (after symmetry), else basis_size

    def t_sdp_seconds(self, basis_size: int) -> float:
        return float(self.a) * float(basis_size) ** float(self.b)


def load_fit_from_symmetry_benchmark(
    *,
    model: str = "heisenberg",
    basis: str = "heisenberg_simple",
    npa_level: int | None = None,
    boundary: str = "periodic",
    sym_level: int = 6,
    results_root: Path | None = None,
) -> FitModel | None:
    """Try to load fit parameters from a symmetry benchmark artifact.
    
    Returns FitModel if found, None otherwise.
    """
    try:
        # Import here to avoid circular imports
        from plots import fit_time_vs_n_vars, find_symmetry_benchmark
        
        if results_root is None:
            results_root = Path(__file__).resolve().parents[1] / "results"
        
        run_dir = find_symmetry_benchmark(
            model=model, basis=basis, npa_level=npa_level, boundary=boundary, results_root=results_root
        )
        if run_dir is None:
            return None
        
        fit_info = fit_time_vs_n_vars(
            run_dir, model=model, basis=basis, npa_level=npa_level, boundary=boundary,
            sym_level=sym_level,
        )
        
        L_min, L_max = fit_info["L_range"]
        return FitModel(
            a=fit_info["a"],
            b=fit_info["b"],
            fit_L_min=L_min,
            fit_L_max=L_max,
            use_n_vars=True,
        )
    except Exception:
        return None


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

def _get_symmetry_manager(N: int, global_cfg: Dict[str, Any]):
    """Create a SymmetryManager from global config symmetry settings."""
    try:
        from spins.symmetry import SymmetryManager
        
        sym_cfg = global_cfg.get("symmetry", {})
        return SymmetryManager(
            N=N,
            use_rotation=sym_cfg.get("use_rotation", False),
            use_sign_symmetry=sym_cfg.get("use_sign_symmetry", False),
            use_translation=sym_cfg.get("use_translation", False),
            use_mirror=sym_cfg.get("use_mirror", False),
            use_permutation=sym_cfg.get("use_permutation", False),
            use_real_operator=sym_cfg.get("use_real_operator", False),
        )
    except ImportError:
        return None


def count_n_vars(basis: List[Any], sym_manager) -> int:
    """Count the number of independent SDP variables after symmetry reduction."""
    try:
        from spins.spins_sdp import build_block_reps
        _, global_index = build_block_reps(basis, sym_manager)
        return len(set(global_index.values()))
    except ImportError:
        # Fallback: return basis size
        return len(basis)


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


def get_basis_for_relaxation(*, basis: str, N: int, npa_level: int | None) -> List[Any]:
    """Return the actual basis (list of PauliWords) for the given config."""
    basis_norm = str(basis).strip().lower()

    if basis_norm == "heisenberg_simple":
        return generate_heisenberg_paper_basis(N=N)

    if basis_norm == "npa":
        if npa_level is None:
            raise ValueError("basis='npa' requires npa_level in config")
        return generate_npa_basis(N=N, k=int(npa_level)).words

    raise ValueError(f"Unknown relaxation basis '{basis}'. Use 'heisenberg_simple' or 'npa'.")


def n_vars_for_relaxation(*, basis: str, N: int, npa_level: int | None, global_cfg: Dict[str, Any]) -> int:
    """Return n_vars (after symmetry reduction) for the given config."""
    basis_list = get_basis_for_relaxation(basis=basis, N=N, npa_level=npa_level)
    sym_manager = _get_symmetry_manager(N, global_cfg)
    if sym_manager is None:
        return len(basis_list)
    return count_n_vars(basis_list, sym_manager)


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

    def _int(value, default: int) -> int:
        if value is None:
            return default
        return int(value)

    if method == "random":
        return 1

    if method == "sa":
        steps = int(method_cfg.get("steps", 100))
        return steps + 1

    if method == "pt":
        # Matches optimization_sweep.py's accounting:
        # n_obj_evals = chains * epochs * steps_per_epoch + chains
        raw_chains = method_cfg.get("chains", method_cfg.get("num_chains"))
        chains = _int(raw_chains, 0)
        if chains <= 0:
            chains = int(mp.cpu_count())
        epochs = _int(method_cfg.get("epochs", method_cfg.get("num_epochs")), 10)
        steps_per_epoch = _int(method_cfg.get("steps_per_epoch"), 50)
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
    # Optional: approximate wall-clock seconds if some internal parallelism is assumed.
    n_sdps_wall: int | None = None
    total_seconds_wall: float | None = None
    notes: str = ""


def estimate_full_relaxation(block: Dict[str, Any], global_cfg: Dict[str, Any], fit: FitModel) -> Estimate | None:
    cfg = block.get("full_relaxation", {})
    if not cfg.get("enabled", False):
        return None

    Ns = list(cfg["Ns"])
    basis = cfg.get("basis", "heisenberg_simple")
    npa_level = cfg.get("npa_level")

    total = 0.0
    out_of_fit = 0

    for N in Ns:
        if fit.use_n_vars:
            # Use n_vars (after symmetry reduction)
            L = n_vars_for_relaxation(basis=basis, N=int(N), npa_level=npa_level, global_cfg=global_cfg)
        else:
            L = basis_size_for_relaxation(basis=basis, N=int(N), npa_level=npa_level)
        if L < fit.fit_L_min or L > fit.fit_L_max:
            out_of_fit += 1
        total += fit.t_sdp_seconds(L)

    l_type = "n_vars" if fit.use_n_vars else "basis sizes"
    notes = ""
    if out_of_fit:
        notes = f"{out_of_fit}/{len(Ns)} {l_type} outside fit window [{fit.fit_L_min},{fit.fit_L_max}]"

    return Estimate(
        name="full_relaxation",
        n_sdps=len(Ns),
        total_seconds=total,
        notes=notes,
    )


def estimate_optimization(block: Dict[str, Any], global_cfg: Dict[str, Any], fit: FitModel) -> Estimate | None:
    cfg = block.get("optimization", {})
    if not cfg.get("enabled", False):
        return None

    Ns = list(cfg["Ns"])
    start_level = int(cfg["start_level"])
    end_level = int(cfg["end_level"])
    k_min = int(cfg.get("k_min", 0))
    k_max = int(cfg["k_max"])
    k_step = int(cfg.get("k_step", 1))
    if k_step <= 0:
        raise ValueError("optimization.k_step must be >= 1")

    methods = cfg.get("methods", {})

    total_seconds = 0.0
    total_sdps = 0

    total_seconds_wall = 0.0
    total_sdps_wall = 0
    out_of_fit = 0
    total_basis_calls = 0
    
    # Get symmetry manager if using n_vars mode
    sym_manager_cache: Dict[int, Any] = {}

    for N in Ns:
        starting_set, adding_set, _final = build_npa_basis_sets(N=int(N), start_level=start_level, end_level=end_level)
        starting_size = len(starting_set)
        adding_size = len(adding_set)

        # The script will run for each explicit k in the sweep.
        # Cap to adding_set length to avoid impossible k.
        k_max_eff = min(k_max, adding_size)
        if k_min > k_max_eff:
            continue
        k_values = list(range(k_min, k_max_eff + 1, k_step))
        
        # Get symmetry manager for this N if using n_vars mode
        if fit.use_n_vars and N not in sym_manager_cache:
            sym_manager_cache[N] = _get_symmetry_manager(int(N), global_cfg)

        for method_name, method_cfg in methods.items():
            if not method_cfg.get("enabled", False):
                continue

            num_seeds = int(method_cfg.get("num_seeds", 1))

            # Total objective evaluations across the whole run.
            evals_per_run = n_obj_evals_for_method(method_name, method_cfg)

            # Approximate wall-clock objective evaluations.
            # For most methods this equals total evaluations.
            # For PT we assume the chains run concurrently during the SA bursts, so wall-clock
            # is closer to "one chain's work per epoch" plus the (currently serial) initial
            # cost evaluation phase.
            evals_per_run_wall = evals_per_run
            if method_name.lower() == "pt":
                raw_chains = method_cfg.get("chains", method_cfg.get("num_chains"))
                chains = int(raw_chains) if raw_chains not in (None, "") else 0
                if chains <= 0:
                    chains = int(mp.cpu_count())
                epochs = int(method_cfg.get("epochs", method_cfg.get("num_epochs", 10)))
                steps_per_epoch = int(method_cfg.get("steps_per_epoch", 40))

                # PT implementation evaluates the objective once per chain to initialize costs
                # (currently done serially), then runs SA bursts in parallel.
                evals_per_run_wall = chains + epochs * steps_per_epoch

            for k in k_values:
                basis_size = starting_size + int(k)
                
                # Determine L: use n_vars if fit.use_n_vars, else basis_size
                if fit.use_n_vars:
                    sym_manager = sym_manager_cache.get(N)
                    if sym_manager is not None:
                        # Build the actual basis at this k
                        current_basis = starting_set + adding_set[:int(k)]
                        L = count_n_vars(current_basis, sym_manager)
                    else:
                        L = basis_size
                else:
                    L = basis_size
                
                t_one_eval = fit.t_sdp_seconds(L)

                if L < fit.fit_L_min or L > fit.fit_L_max:
                    out_of_fit += 1
                total_basis_calls += 1

                # Each objective evaluation solves one SDP.
                n_sdps_this_k = num_seeds * evals_per_run
                total_sdps += n_sdps_this_k
                total_seconds += n_sdps_this_k * t_one_eval

                n_sdps_this_k_wall = num_seeds * evals_per_run_wall
                total_sdps_wall += n_sdps_this_k_wall
                total_seconds_wall += n_sdps_this_k_wall * t_one_eval

    l_type = "n_vars" if fit.use_n_vars else "basis sizes"
    notes = ""
    if total_basis_calls and out_of_fit:
        notes = f"{out_of_fit}/{total_basis_calls} {l_type} outside fit window [{fit.fit_L_min},{fit.fit_L_max}]"

    return Estimate(
        name="optimization",
        n_sdps=total_sdps,
        total_seconds=total_seconds,
        n_sdps_wall=total_sdps_wall,
        total_seconds_wall=total_seconds_wall,
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

    p.add_argument("--a", type=float, default=None, help="Fit constant a in t=a*L^b (overrides --from-artifact)")
    p.add_argument("--b", type=float, default=None, help="Fit exponent b in t=a*L^b (overrides --from-artifact)")
    p.add_argument("--fit-L-min", type=int, default=None, help="Fit window min L")
    p.add_argument("--fit-L-max", type=int, default=None, help="Fit window max L")
    
    p.add_argument(
        "--from-artifact", 
        action="store_true", 
        help="Load fit from symmetry benchmark artifact (uses n_vars after symmetry reduction)"
    )
    p.add_argument(
        "--artifact-basis", 
        type=str, 
        default="heisenberg_simple",
        help="Basis for artifact lookup (default: heisenberg_simple)"
    )
    p.add_argument(
        "--artifact-sym-level", 
        type=int, 
        default=6,
        help="Symmetry level for artifact lookup (default: 6 = all symmetries)"
    )

    args = p.parse_args()

    config = load_config(args.config)
    global_cfg = config.get("global_settings", {})
    block = pick_mode_block(config, test=args.test)

    # Determine fit model
    fit = None
    fit_source = "default"
    
    if args.from_artifact:
        # Try to load from symmetry benchmark artifact
        model = global_cfg.get("model", "heisenberg")
        basis = args.artifact_basis
        boundary = "periodic"  # symmetry benchmark uses periodic
        sym_level = args.artifact_sym_level
        
        fit = load_fit_from_symmetry_benchmark(
            model=model, basis=basis, boundary=boundary, sym_level=sym_level
        )
        if fit is not None:
            fit_source = f"symmetry benchmark (basis={basis}, sym_level={sym_level})"
        else:
            print("Warning: Could not load fit from artifact, using defaults.")
    
    if fit is None:
        # Use default or manual overrides
        a = args.a if args.a is not None else FitModel.a
        b = args.b if args.b is not None else FitModel.b
        fit_L_min = args.fit_L_min if args.fit_L_min is not None else FitModel.fit_L_min
        fit_L_max = args.fit_L_max if args.fit_L_max is not None else FitModel.fit_L_max
        fit = FitModel(a=a, b=b, fit_L_min=fit_L_min, fit_L_max=fit_L_max)
    
    # Apply any manual overrides
    if args.a is not None or args.b is not None or args.fit_L_min is not None or args.fit_L_max is not None:
        fit = FitModel(
            a=args.a if args.a is not None else fit.a,
            b=args.b if args.b is not None else fit.b,
            fit_L_min=args.fit_L_min if args.fit_L_min is not None else fit.fit_L_min,
            fit_L_max=args.fit_L_max if args.fit_L_max is not None else fit.fit_L_max,
            use_n_vars=fit.use_n_vars,
        )

    mode_name = "TEST" if args.test else "FULL"
    print("=" * 72)
    print(f"SpinsSDP SDP Runtime Estimate ({mode_name} mode)")
    print("=" * 72)
    l_var_name = "n_vars" if fit.use_n_vars else "L (basis size)"
    print(f"Fit source: {fit_source}")
    print(f"Fit: t({l_var_name}) = {fit.a:.4g} * {l_var_name}^{fit.b:.4g}   (window [{fit.fit_L_min},{fit.fit_L_max}])")

    estimates: List[Estimate] = []

    e_relax = estimate_full_relaxation(block, global_cfg, fit)
    if e_relax:
        estimates.append(e_relax)

    e_opt = estimate_optimization(block, global_cfg, fit)
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

    # Optional wall-clock estimate if available.
    if any(e.total_seconds_wall is not None for e in estimates):
        print("\nApprox wall-clock (assumes PT chains run concurrently):")
        total_all_wall = 0.0
        total_sdps_wall_sum = 0
        for e in estimates:
            if e.total_seconds_wall is None or e.n_sdps_wall is None:
                continue
            total_all_wall += e.total_seconds_wall
            total_sdps_wall_sum += e.n_sdps_wall
            print(f"- {e.name}: ~{e.n_sdps_wall} effective SDPs, ~{_human_time(e.total_seconds_wall)}")
        print(f"- TOTAL (wall): ~{total_sdps_wall_sum} effective SDPs, ~{_human_time(total_all_wall)}")

    if args.dry_run:
        print("\n(dry-run: not attempting to account for parallelism / scheduling)")
        return 0

    print("\nObs: if we run SDPs in parallel (like running different optimization strats or different Ns for the full sdp lower bound), we can probably cut the total time a bit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
