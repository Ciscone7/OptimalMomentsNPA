from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any, List, Dict, Literal, Optional, Iterable

import argparse
import json
from pathlib import Path

import numpy as np

from spins_sdp.bell_algebra import generate_npa_basis
from spins_sdp.pauli import Operator, PauliWord
from spins_sdp.models import ising_hamiltonian_exact, ising_hamiltonian_dict
from spins_sdp.sdp import solve_pauli_relaxation



def _time_best_avg(fn, repeats: int):
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, float(min(times)), float(sum(times) / len(times))




# ---------- Timing: exact diagonalization vs N ----------
def benchmark_exact_diagonalization(
    N_values,
    J=1.0, h=0.0, k=0.0,
    boundary="open",
    repeats=3,
):
    """
    Returns a dict with arrays: N, dim, E0, t_best, t_avg.
    """
    Ns = np.array(list(N_values), dtype=int)
    dims = np.array([2**N for N in Ns], dtype=int)

    E0s = np.empty(len(Ns), dtype=float)
    t_best = np.empty(len(Ns), dtype=float)
    t_avg = np.empty(len(Ns), dtype=float)

    for idx, N in enumerate(Ns):
        def run():
            H = ising_hamiltonian_exact(N, J=J, h=h, k=k, boundary=boundary)
            evals, _ = H.eigenstates(eigvals=1)
            E0 = evals[0]
            return float(E0)

        E0, tb, ta = _time_best_avg(run, repeats=repeats)
        E0s[idx] = E0
        t_best[idx] = tb
        t_avg[idx] = ta

    return {
        "N": Ns,
        "dim": dims,
        "E0": E0s,
        "t_best": t_best,
        "t_avg": t_avg,
        "meta": {"J": J, "h": h, "k": k, "boundary": boundary, "repeats": repeats},
    }


# ---------- General relaxation benchmark ----------
def benchmark_relaxation(
    N_values,
    basis_fn: Callable[[int], List[PauliWord]],
    operator_fn: Callable[[int], Operator],
    sense: Literal["min", "max"] = "min",
    solver: str = "MOSEK",
    repeats: int = 1,
    mosek_tol: float = 1e-9,
    solver_opts: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    General benchmark for moment relaxation over varying N.
    
    Args:
        N_values: Iterable of system sizes to benchmark
        basis_fn: Function N -> List[PauliWord] generating the relaxation basis
        operator_fn: Function N -> Operator generating the objective operator
        sense: "min" or "max" optimization
        solver: default "MOSEK"
        repeats: Number of timing repeats (best and average reported)
        mosek_tol: MOSEK conic tolerance
        solver_opts: Override default solver options
        
    Returns:
        Dict with arrays: N, objective_values, t_best, t_avg
    """
    Ns = np.array(list(N_values), dtype=int)
    
    objective_values = np.empty(len(Ns), dtype=float)
    t_best = np.empty(len(Ns), dtype=float)
    t_avg = np.empty(len(Ns), dtype=float)
    
    for idx, N in enumerate(Ns):
        def run():
            # Generate basis and operator for this N
            basis = basis_fn(N)
            operator = operator_fn(N)
            
            # Solve relaxation
            obj_val = solve_pauli_relaxation(
                basis=basis,
                operator=operator,
                sense=sense,
                solver=solver,
                mosek_tol=mosek_tol,
                solver_opts=solver_opts,
                verbose=False,
            )
            return obj_val
        
        obj_val, tb, ta = _time_best_avg(run, repeats=repeats)
        objective_values[idx] = obj_val
        t_best[idx] = tb
        t_avg[idx] = ta
    
    return {
        "N": Ns,
        "objective": objective_values,
        "t_best": t_best,
        "t_avg": t_avg,
        "meta": {
            "solver": solver,
            "repeats": repeats,
            "mosek_tol": mosek_tol,
            "sense": sense,
        },
    }


# ---------- Timing: NPA relaxation vs N (fixed level) ----------
def benchmark_npa_relaxation(
    N_values,
    NPA_level: int,
    J=1.0, h=0.0, k=0.0,
    boundary="open",
    solver="MOSEK",
    repeats=1,
    mosek_tol=1e-9,
    solver_opts: Optional[Dict[str, Any]] = None,
):
    """
    Convenience wrapper for NPA hierarchy benchmarks on 1D Ising model.
    
    Returns a dict with arrays: N, E_lb, t_best, t_avg.
    
    Args:
        mosek_tol: Tolerance for MOSEK conic solver
        solver_opts: Override default MOSEK options with custom dict
    """
    
    # Define basis generator
    def basis_fn(N: int) -> List[PauliWord]:
        return generate_npa_basis(N=N, k=NPA_level).words
    
    # Call general benchmark
    result = benchmark_relaxation(
        N_values=N_values,
        basis_fn=basis_fn,
        operator_fn=lambda N: ising_hamiltonian_dict(N=N, J=J, h=h, k=k, boundary=boundary),
        sense="min",
        solver=solver,
        repeats=repeats,
        mosek_tol=mosek_tol,
        solver_opts=solver_opts,
    )
    
    # Add NPA-specific metadata
    result["meta"].update({
        "NPA_level": NPA_level,
        "J": J,
        "h": h,
        "k": k,
        "boundary": boundary,
    })
    
    # Rename for backward compatibility
    result["E_lb"] = result.pop("objective")
    result["N"] = result["N"]
    
    return result


# --- Helpers for debugging / pretty printing ---

def local_ops(word: PauliWord, N: int) -> str:
    """Return a compact local-operator string like 'X I Z' for small N."""
    out = []
    for k in range(N):
        xb = (word.x_mask >> k) & 1
        zb = (word.z_mask >> k) & 1
        if xb == 0 and zb == 0:
            out.append("I")
        elif xb == 1 and zb == 0:
            out.append("X")
        elif xb == 0 and zb == 1:
            out.append("Z")
        else:
            out.append("Y")
    return " ".join(out)

def X(i: int) -> PauliWord: return PauliWord(1 << i, 0)
def Z(i: int) -> PauliWord: return PauliWord(0, 1 << i)
def Y(i: int) -> PauliWord: 
    b = 1 << i
    return PauliWord(b, b)




def optimization_wrapper_spins(x, starting_set, adding_set, hamiltonian_expression):
  
    chosen_additions = [s for val, s in zip(x, adding_set) if val]
    total_moments = starting_set + chosen_additions

    return solve_pauli_relaxation(total_moments, hamiltonian_expression ,sense = "min")


# ---------- Results/Artifacts: list available hashes ----------

def _default_results_root() -> Path:
    # scripts default to: spins_sdp/results
    return Path(__file__).resolve().parent / "results"


def iter_result_runs(
    *,
    results_root: Optional[Path] = None,
    artifact: Optional[str] = None,
    schema_version: Optional[int] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return a list of available result runs from `spins_sdp/results`.

    Each entry includes:
      - artifact, schema_version, config_hash, run_dir
      - created_at, updated_at
    - model, params, boundary, basis, npa_level, solver (if present)
            - max_N (if data.npz contains an N array)
    """
    root = results_root or _default_results_root()
    runs: List[Dict[str, Any]] = []

    if not root.exists():
        return runs

    for artifact_dir in root.iterdir():
        if not artifact_dir.is_dir():
            continue
        if artifact is not None and artifact_dir.name != artifact:
            continue

        for vdir in artifact_dir.iterdir():
            if not vdir.is_dir() or not vdir.name.startswith("v"):
                continue
            try:
                vnum = int(vdir.name[1:])
            except ValueError:
                continue
            if schema_version is not None and vnum != schema_version:
                continue

            for run_dir in vdir.iterdir():
                if not run_dir.is_dir():
                    continue

                meta_path = run_dir / "meta.json"
                meta: Dict[str, Any] = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}

                meta_model = meta.get("model")
                if model is not None and meta_model is not None and str(meta_model) != str(model):
                    continue

                max_N: Optional[int] = None
                data_path = run_dir / "data.npz"
                if data_path.exists():
                    try:
                        with np.load(data_path, allow_pickle=False) as data:
                            if "N" in data.files:
                                Ns = np.asarray(data["N"], dtype=int)
                                if Ns.size > 0:
                                    max_N = int(np.max(Ns))
                    except Exception:
                        max_N = None

                runs.append(
                    {
                        "artifact": artifact_dir.name,
                        "schema_version": vnum,
                        "config_hash": run_dir.name,
                        "run_dir": str(run_dir),
                        "created_at": meta.get("created_at"),
                        "updated_at": meta.get("updated_at"),
                        "model": meta_model,
                        "params": meta.get("params"),
                        "boundary": meta.get("boundary"),
                        "basis": meta.get("basis"),
                        "npa_level": meta.get("npa_level"),
                        "solver": meta.get("solver"),
                        "max_N": max_N,
                    }
                )

    def sort_key(r: Dict[str, Any]) -> str:
        return str(r.get("updated_at") or r.get("created_at") or "")

    runs.sort(key=sort_key, reverse=True)
    return runs


def format_result_runs_table(runs: Iterable[Dict[str, Any]], *, limit: Optional[int] = None) -> str:
    rows = list(runs)
    if limit is not None:
        rows = rows[: int(limit)]

    cols = [
        ("artifact", 22),
        ("v", 3),
        ("hash", 16),
        ("max_N", 5),
        ("updated_at", 20),
        ("model", 14),
        ("boundary", 9),
        ("basis", 18),
        ("npa_k", 6),
    ]

    def trunc(s: Any, w: int) -> str:
        s2 = "" if s is None else str(s)
        if len(s2) <= w:
            return s2.ljust(w)
        return (s2[: max(0, w - 1)] + "…")

    header = " ".join(trunc(name, w) for name, w in cols)
    sep = "-" * len(header)
    out = [header, sep]

    for r in rows:
        out.append(
            " ".join(
                [
                    trunc(r.get("artifact"), 22),
                    trunc(f"v{r.get('schema_version', '')}", 3),
                    trunc(r.get("config_hash"), 16),
                    trunc(r.get("max_N"), 5),
                    trunc(r.get("updated_at") or r.get("created_at"), 20),
                    trunc(r.get("model"), 14),
                    trunc(r.get("boundary"), 9),
                    trunc(r.get("basis"), 18),
                    trunc(r.get("npa_level"), 6),
                ]
            )
        )

    return "\n".join(out)


def _cli_list_results(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="List available result hashes under spins_sdp/results")
    p.add_argument("--results-root", type=Path, default=_default_results_root())
    p.add_argument("--artifact", type=str, default=None)
    p.add_argument("--schema", type=int, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", dest="as_json", action="store_true")
    args = p.parse_args(argv)

    runs = iter_result_runs(
        results_root=args.results_root,
        artifact=args.artifact,
        schema_version=args.schema,
        model=args.model,
    )

    if args.as_json:
        print(json.dumps(runs, indent=2, ensure_ascii=False))
    else:
        print(format_result_runs_table(runs, limit=args.limit))
    return 0


# ---------- Convenience wrappers for notebooks ----------

def list_exact_runs(*, limit: int = 20, as_df: bool = False):
    """List stored exact ground energy runs."""
    runs = iter_result_runs(artifact="spin_exact_ground_energy")
    if as_df:
        try:
            import pandas as pd
            return pd.DataFrame(runs[:limit])
        except ImportError:
            pass
    print(format_result_runs_table(runs, limit=limit))


def list_lb_runs(*, limit: int = 20, as_df: bool = False):
    """List stored lower-bound (SDP) runs."""
    runs = iter_result_runs(artifact="spin_moment_energy_lb")
    if as_df:
        try:
            import pandas as pd
            return pd.DataFrame(runs[:limit])
        except ImportError:
            pass
    print(format_result_runs_table(runs, limit=limit))


if __name__ == "__main__":
    raise SystemExit(_cli_list_results())




























