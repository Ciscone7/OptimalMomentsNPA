import time
import numpy as np
import qutip as qt
import cvxpy as cp
from typing import Final, List, Tuple, Union, Literal

from spins_sdp.exact import ising_hamiltonian
from spins_sdp.pauli_strings_new import (
    PauliWord,
    generate_npa_basis,
    compile_moment_matrix_rep,
    build_sdp_from_rep,
    Operator,
)

BoundaryType = Literal["open", "periodic"]
AxisType = Literal["x", "y", "z"]

_I_POW: Final[Tuple[complex, complex, complex, complex]] = (1+0j, 1j, -1+0j, -1j)

def exact_ground_state_eigenpair(H: qt.Qobj) -> Tuple[float, qt.Qobj]:
    """Compute the exact ground state energy of a Hamiltonian by diagonalization.

    Args:
        H (np.ndarray): Hamiltonian matrix.

    Returns:
        float: Ground state energy.
    """
    evals, evecs = H.eigenstates(eigvals=1)  # just the lowest one
    return (evals[0], evecs[0])


# TODO: Only works for 1D Ising model for now
def npa_lb_energy(
    J: float,
    h: float,
    k: float,
    N: int,
    NPA_level: int,
    solver: str = "SCS",
    boundary: BoundaryType = "open",
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
    sdp.problem.solve(solver=solver, verbose=False)

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
            E0, _ = exact_ground_state_eigenpair(H)
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


# ---------- Timing: NPA relaxation vs N (fixed level) ----------
def benchmark_npa_relaxation(
    N_values,
    NPA_level: int,
    J=1.0, h=0.0, k=0.0,
    boundary="open",
    solver="SCS",
    repeats=1,
    scs_eps=1e-6,
    scs_max_iters=20000,
):
    """
    Returns a dict with arrays: N, E_lb, t_best, t_avg.
    """
    Ns = np.array(list(N_values), dtype=int)

    E_lbs = np.empty(len(Ns), dtype=float)
    t_best = np.empty(len(Ns), dtype=float)
    t_avg = np.empty(len(Ns), dtype=float)

    for idx, N in enumerate(Ns):
        def run():
            # TODO: modify npa_lb_energy to accept eps/max_iters
            E_lb = npa_lb_energy(J=J, h=h, k=k, N=N, NPA_level=NPA_level, solver=solver, boundary=boundary)
            return float(E_lb)

        E_lb, tb, ta = _time_best_avg(run, repeats=repeats)
        E_lbs[idx] = E_lb
        t_best[idx] = tb
        t_avg[idx] = ta

    return {
        "N": Ns,
        "E_lb": E_lbs,
        "t_best": t_best,
        "t_avg": t_avg,
        "meta": {
            "J": J, "h": h, "k": k,
            "boundary": boundary,
            "solver": solver,
            "NPA_level": NPA_level,
            "repeats": repeats,
            "scs_eps": scs_eps,
            "scs_max_iters": scs_max_iters,
        },
    }

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







