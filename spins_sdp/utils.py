from collections.abc import Callable
from time import time
from typing import Any, List, Dict, Literal, Optional

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
            obj_val = solve_relaxation(
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




























