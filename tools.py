import time
import numpy as np
import qutip as qt
import cvxpy as cp
from typing import Final, List, Tuple, Union, Literal, Optional, Dict, Any, Callable

from spins_sdp.exact import ising_hamiltonian
from spins_sdp.pauli_strings_new import (
    PauliWord,
    compile_moment_matrix_rep,
    build_sdp_from_rep,
    Operator,
    MomentMatrixRep,
)

from spins_sdp.basis_builder import generate_npa_basis, generate_heisenberg_j2_basis_weak, generate_heisenberg_j2_basis_strong

BoundaryType = Literal["open", "periodic"]
AxisType = Literal["x", "y", "z"]

_I_POW: Final[Tuple[complex, complex, complex, complex]] = (1+0j, 1j, -1+0j, -1j)

# TODO: Only works for 1D Ising model for now
def npa_lb_energy(
    J: float,
    h: float,
    k: float,
    N: int,
    NPA_level: int,
    solver: str = "MOSEK",
    boundary: BoundaryType = "open",
    solver_opts: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> float:
    """Compute a lower bound to the ground-state energy using the new moment-SDP pipeline.

    Uses the compiled NPA basis from pauli_strings_new and builds an SDP with
    moment variables y corresponding to Pauli words. The objective is the
    Ising Hamiltonian expressed as a Pauli-word operator.
    """

    basis = generate_npa_basis(N=N, k=NPA_level).words
    rep = compile_moment_matrix_rep(basis)

    # Build Ising Hamiltonian as a PauliWord->coeff dict
    op: Operator = {}

    # -J sum Z_i Z_{i+1}
    for i in range(N):
        if i < N - 1 or boundary == "periodic":
            j = (i + 1) % N
            w = PauliWord(0, (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) - J

    # -h sum X_i
    for i in range(N):
        w = PauliWord(1 << i, 0)
        op[w] = op.get(w, 0.0) - h

    # -k sum Z_i
    for i in range(N):
        w = PauliWord(0, 1 << i)
        op[w] = op.get(w, 0.0) - k

    sdp = build_sdp_from_rep(rep, op, sense="min")

    # Allow tighter solver tolerances / custom options to be injected by callers.
    solve_kwargs: Dict[str, Any] = {"solver": solver, "verbose": verbose}
    if solver_opts:
        solve_kwargs.update(solver_opts)

    sdp.problem.solve(**solve_kwargs)

    return float(sdp.problem.value)


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
            H = ising_hamiltonian(N, J=J, h=h, k=k, boundary=boundary)
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


def ising_operator(N: int, J: float, h: float, k: float, boundary: BoundaryType) -> Operator:
    """
    Build the 1D Ising Hamiltonian as a Pauli operator dictionary.
    
    H = -J Σ_i Z_i Z_{i+1} - h Σ_i X_i - k Σ_i Z_i
    
    Args:
        N: Number of spins
        J: Nearest-neighbor ZZ coupling strength
        h: Transverse field strength (X direction)
        k: Longitudinal field strength (Z direction)
        boundary: "open" or "periodic"
        
    Returns:
        Operator dictionary mapping PauliWord -> coefficient
    """
    op: Operator = {}
    
    # -J sum Z_i Z_{i+1}
    for i in range(N):
        if i < N - 1 or boundary == "periodic":
            j = (i + 1) % N
            w = PauliWord(0, (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) - J
    
    # -h sum X_i
    for i in range(N):
        w = PauliWord(1 << i, 0)
        op[w] = op.get(w, 0.0) - h
    
    # -k sum Z_i
    for i in range(N):
        w = PauliWord(0, 1 << i)
        op[w] = op.get(w, 0.0) - k
    
    return op


def heisenberg_operator(N: int, boundary: BoundaryType = "periodic") -> Operator:
    """
    Build the 1D Heisenberg Hamiltonian as a Pauli operator dictionary.
    
    H = (1/4) Σ_i Σ_a∈{x,y,z} σ_i^a σ_{i+1}^a
    
    This matches the paper's definition (case B).
    
    Args:
        N: Number of spins
        boundary: "open" or "periodic"
        
    Returns:
        Operator dictionary mapping PauliWord -> coefficient
    """
    op: Operator = {}
    
    # (1/4) sum_i sum_a σ_i^a σ_{i+1}^a
    for i in range(N):
        if i < N - 1 or boundary == "periodic":
            j = (i + 1) % N
            
            # X_i X_{i+1}
            w = PauliWord((1 << i) | (1 << j), 0)
            op[w] = op.get(w, 0.0) + 0.25
            
            # Y_i Y_{i+1}
            w = PauliWord((1 << i) | (1 << j), (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) + 0.25
            
            # Z_i Z_{i+1}
            w = PauliWord(0, (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) + 0.25
    
    return op


def heisenberg_j2_operator(N: int, J2: float, boundary: BoundaryType = "periodic") -> Operator:
    """
    Build the 1D Heisenberg Hamiltonian with second-neighbor couplings as a Pauli operator dictionary.
    
    H = (1/4) Σ_i Σ_a∈{x,y,z} [σ_i^a σ_{i+1}^a + J2 σ_i^a σ_{i+2}^a]
    
    This matches the paper's definition (case C).
    
    Args:
        N: Number of spins
        J2: Coupling strength for second-neighbor terms
        boundary: "open" or "periodic"
        
    Returns:
        Operator dictionary mapping PauliWord -> coefficient
    """
    op: Operator = {}
    
    # First-neighbor terms: (1/4) sum_i sum_a σ_i^a σ_{i+1}^a
    for i in range(N):
        if i < N - 1 or boundary == "periodic":
            j = (i + 1) % N
            
            # X_i X_{i+1}
            w = PauliWord((1 << i) | (1 << j), 0)
            op[w] = op.get(w, 0.0) + 0.25
            
            # Y_i Y_{i+1}
            w = PauliWord((1 << i) | (1 << j), (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) + 0.25
            
            # Z_i Z_{i+1}
            w = PauliWord(0, (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) + 0.25
    
    # Second-neighbor terms: (J2/4) sum_i sum_a σ_i^a σ_{i+2}^a
    for i in range(N):
        if i < N - 2 or boundary == "periodic":
            j = (i + 2) % N
            
            # X_i X_{i+2}
            w = PauliWord((1 << i) | (1 << j), 0)
            op[w] = op.get(w, 0.0) + 0.25 * J2
            
            # Y_i Y_{i+2}
            w = PauliWord((1 << i) | (1 << j), (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) + 0.25 * J2
            
            # Z_i Z_{i+2}
            w = PauliWord(0, (1 << i) | (1 << j))
            op[w] = op.get(w, 0.0) + 0.25 * J2
    
    return op


def solve_relaxation(
    basis: List[PauliWord],
    operator: Operator,
    sense: Literal["min", "max"] = "min",
    solver: str = "MOSEK",
    mosek_tol: float = 1e-9,
    solver_opts: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> float:
    """
    Solve a moment relaxation SDP for a given basis and operator.
    
    Args:
        basis: List of Pauli words defining the relaxation basis
        operator: Objective operator as a Pauli word dictionary
        sense: "min" or "max" optimization
        solver: default "MOSEK"
        mosek_tol: MOSEK conic tolerance
        solver_opts: Override default solver options
        verbose: Print solver output
        
    Returns:
        Optimal objective value
    """
    
    # print(f"Compiling moment matrix with basis size {len(basis)}...")
    
    # Compile moment matrix representation
    rep = compile_moment_matrix_rep(basis)
    
    # print("Building SDP...")
    
    # Build SDP
    sdp = build_sdp_from_rep(rep, operator, sense=sense)
    
    # Default MOSEK options
    default_solver_opts: Dict[str, Any] = {
        "mosek_params": {
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": mosek_tol,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": mosek_tol,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": mosek_tol,
        }
    }
    
    merged_solver_opts = dict(default_solver_opts)
    if solver_opts:
        merged_solver_opts.update(solver_opts)
    
    # print("Solving SDP...")
    
    # Solve
    sdp.problem.solve(solver=solver, verbose=verbose, **merged_solver_opts)
    # print("SDP solved.")
    
    return float(sdp.problem.value)

def optimization_wrapper_spins(x, starting_set, adding_set, hamiltonian_expression):

    chosen_additions = [s for val, s in zip(x, adding_set) if val]
    total_moments = starting_set + chosen_additions

    return solve_relaxation(total_moments, hamiltonian_expression ,sense = "min")

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
        operator_fn=lambda N: ising_operator(N=N, J=J, h=h, k=k, boundary=boundary),
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







