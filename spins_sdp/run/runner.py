#!/usr/bin/env python3
"""Run experiments from JSON configuration file.

Usage:
    python -m spins_sdp.run.runner                    # Full mode
    python -m spins_sdp.run.runner --test             # Test mode
    python -m spins_sdp.run.runner --dry-run          # Preview commands
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _safe_int(value, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _compute_ks(cfg: dict, *, adding_size: int | None) -> list[int]:
    """Compute the list of k values to sweep.

    Supports either:
      - k_min/k_max/k_step (preferred)
      - k_max only (falls back to 0..k_max)
    """
    k_max = int(cfg["k_max"])
    k_min = int(cfg.get("k_min", 0))
    k_step = int(cfg.get("k_step", 1))
    if k_step <= 0:
        raise ValueError("k_step must be >= 1")

    k_max_eff = k_max
    if adding_size is not None:
        k_max_eff = min(k_max_eff, int(adding_size))

    if k_min > k_max_eff:
        return []
    return list(range(k_min, k_max_eff + 1, k_step))


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def run(cmd: list[str], dry_run: bool = False) -> int:
    """Run a command, printing it first."""
    print(f"  → {' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.run(cmd).returncode


def run_exact(cfg: dict, global_cfg: dict, dry_run: bool) -> None:
    """Run exact diagonalization."""
    if not cfg.get("enabled", True):
        return
    
    print("\n[EXACT] Computing exact ground energies...")
    cmd = [
        "python", "-m", "spins_sdp.scripts.exact_ground_energy",
        "--model", global_cfg["model"],
        "--N-min", str(cfg["N_min"]),
        "--N-max", str(cfg["N_max"]),
        "--boundary", "periodic"
    ]
    if global_cfg.get("resume"):
        cmd.append("--resume")
    
    run(cmd, dry_run)


def run_dmrg(cfg: dict, global_cfg: dict, dry_run: bool) -> None:
    """Run DMRG upper bounds."""
    if not cfg.get("enabled", True):
        return
    
    print("\n[DMRG] Computing DMRG upper bounds...")
    cmd = [
        "python", "-m", "spins_sdp.scripts.dmrg",
        "--model", global_cfg["model"],
        "--N-min", str(cfg["N_min"]),
        "--N-max", str(cfg["N_max"]),
        "--chi-max", str(cfg["chi_max"])
    ]
    if cfg.get("mixer"):
        cmd.append("--mixer")
    if global_cfg.get("resume"):
        cmd.append("--resume")
    if global_cfg.get("verbose"):
        cmd.append("--verbose")
    
    run(cmd, dry_run)


def run_relaxation(cfg: dict, global_cfg: dict, dry_run: bool) -> None:
    """Run SDP relaxation."""
    if not cfg.get("enabled", True):
        return
    
    print("\n[RELAXATION] Computing SDP lower bounds...")
    
    Ns = cfg["Ns"]
    basis = cfg.get("basis", "npa")
    npa_level = cfg.get("npa_level")
    
    cmd = [
        "python", "-m", "spins_sdp.scripts.npa_energy_lb",
        "--model", global_cfg["model"],
        "--basis", basis,
        "--Ns", *[str(n) for n in Ns],
        "--boundary", "periodic",
        "--solver", global_cfg["solver"],
    ]
    if npa_level:
        cmd.extend(["--npa-level", str(npa_level)])
    if global_cfg.get("resume"):
        cmd.append("--resume")
    
    run(cmd, dry_run)


def run_optimization(cfg: dict, global_cfg: dict, dry_run: bool) -> None:
    """Run optimization sweeps."""
    if not cfg.get("enabled", True):
        return
    
    print("\n[OPTIMIZATION] Running optimization sweeps...")
    
    Ns = cfg["Ns"]
    start_level = int(cfg["start_level"])
    end_level = int(cfg["end_level"])
    methods = cfg["methods"]

    # If the user specified k_min/k_step, we must pass explicit --ks to the script.
    # We'll also cap k to the length of the adding_set to avoid runtime errors.
    adding_sizes_by_N: dict[int, int] = {}
    try:
        # Local import to keep runner lightweight.
        from spins_sdp.basis_builder import generate_npa_basis

        for N in Ns:
            full_basis = generate_npa_basis(N=int(N), k=end_level)
            adding_set = []
            for level_idx in range(start_level + 1, min(end_level + 1, len(full_basis.levels))):
                adding_set.extend(full_basis.levels[level_idx])
            adding_sizes_by_N[int(N)] = len(adding_set)
    except Exception:
        # If basis generation fails for some reason, proceed without capping.
        adding_sizes_by_N = {}
    
    for N in Ns:
        ks = _compute_ks(cfg, adding_size=adding_sizes_by_N.get(int(N)))
        if not ks:
            print(f"\n  N={N}: no valid k values to run (check k_min/k_max/k_step)")
            continue

        for method_name, method_cfg in methods.items():
            if not method_cfg.get("enabled", True):
                continue
            
            print(f"\n  N={N}, method={method_name}")
            
            cmd = [
                "python", "-m", "spins_sdp.scripts.optimization_sweep",
                "--model", global_cfg["model"],
                "--N", str(N),
                "--boundary", "periodic",
                "--start-level", str(start_level),
                "--end-level", str(end_level),
                "--method", method_name,
                "--ks", *[str(k) for k in ks],
                "--num-seeds", str(method_cfg.get("num_seeds", 10)),
            ]
            
            # Method-specific parameters
            if method_name == "sa":
                cmd.extend(["--sa-steps", str(method_cfg.get("steps", 100))])
            elif method_name == "pt":
                cmd.extend([
                    "--pt-epochs",
                    str(_safe_int(method_cfg.get("epochs"), 10)),
                    "--pt-steps-per-epoch",
                    str(_safe_int(method_cfg.get("steps_per_epoch"), 40)),
                ])
                chains = method_cfg.get("chains")
                if chains is not None and int(chains) > 0:
                    cmd.extend(["--pt-chains", str(int(chains))])
                if method_cfg.get("T_min") is not None:
                    cmd.extend(["--pt-T-min", str(method_cfg["T_min"])])
                if method_cfg.get("T_max") is not None:
                    cmd.extend(["--pt-T-max", str(method_cfg["T_max"])])
            elif method_name == "bo":
                cmd.extend([
                    "--bo-n-init", str(method_cfg.get("n_init", 10)),
                    "--bo-n-iter", str(method_cfg.get("n_iter", 30)),
                    "--bo-candidates-per-iter", str(method_cfg.get("candidates_per_iter", 50)),
                ])
            
            if global_cfg.get("resume"):
                cmd.append("--resume")
            if global_cfg.get("verbose"):
                cmd.append("--verbose")
            
            run(cmd, dry_run)


def main():
    parser = argparse.ArgumentParser(description="Run SpinsSDP experiments")
    parser.add_argument("--config", "-c", type=Path,
                        default=Path(__file__).parent / "experiments_config.json")
    parser.add_argument("--test", "-t", action="store_true",
                        help="Use test mode parameters")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    global_cfg = config["global_settings"]
    
    # Pick experiment params based on mode
    if args.test:
        experiments = config["test_mode"]
        print("=" * 50)
        print("SpinsSDP Experiments - TEST MODE")
        print("=" * 50)
    else:
        experiments = config["experiments"]
        print("=" * 50)
        print("SpinsSDP Experiments - FULL MODE")
        print("=" * 50)
    
    if args.dry_run:
        print("(DRY RUN - commands will not execute)\n")
    
    # Run each experiment type
    run_exact(experiments.get("exact", {}), global_cfg, args.dry_run)
    run_dmrg(experiments.get("dmrg", {}), global_cfg, args.dry_run)
    run_relaxation(experiments.get("full_relaxation", {}), global_cfg, args.dry_run)
    run_optimization(experiments.get("optimization", {}), global_cfg, args.dry_run)
    
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
