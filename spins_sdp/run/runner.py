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
    k_max = cfg["k_max"]
    start_level = cfg["start_level"]
    end_level = cfg["end_level"]
    methods = cfg["methods"]
    
    for N in Ns:
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
                "--k-max", str(k_max),
                "--num-seeds", str(method_cfg.get("num_seeds", 10)),
            ]
            
            # Method-specific parameters
            if method_name == "sa":
                cmd.extend(["--sa-steps", str(method_cfg.get("steps", 100))])
            elif method_name == "pt":
                cmd.extend([
                    "--pt-chains", str(method_cfg.get("chains", 4)),
                    "--pt-epochs", str(method_cfg.get("epochs", 5)),
                    "--pt-steps-per-epoch", str(method_cfg.get("steps_per_epoch", 50)),
                ])
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
