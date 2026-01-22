# This whole file will be generalised later when we add the bell scenario

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import cvxpy as cp
import scipy.sparse as sp

from spins_sdp.pauli import compile_moment_matrix_rep, Operator, PauliMomentMatrixRep, PauliWord


# Later we will generalise this to account for the bell scenario
@dataclass(frozen=True, slots=True)
class PauliMomentSDP:
    rep: PauliMomentMatrixRep
    y: cp.Variable
    M_real: cp.Expression
    M_imag: cp.Expression
    PSD_block: cp.Expression
    constraints: List[cp.Constraint]
    objective: cp.Expression
    problem: cp.Problem


Sense = Literal["min", "max"]


def pauli_moment_matrix_real_embedding(rep: PauliMomentMatrixRep, y: cp.Variable
                                       ) -> tuple[cp.Expression, cp.Expression, cp.Expression]:
    """
    Build M = A + iB where A,B are real matrices affine in y (y real),
    then build the real embedding PSD matrix:
        K = [[A, -B],
             [B,  A]]  >= 0
    """
    n = rep.label_idx.shape[0]
    m = len(rep.labels)

    cols = rep.label_idx.reshape(-1, order="F").astype(np.int32)
    dataA = rep.a_coef.reshape(-1, order="F").astype(float)
    dataB = rep.b_coef.reshape(-1, order="F").astype(float)
    rows = np.arange(n * n, dtype=np.int32)

    CA = sp.coo_matrix((dataA, (rows, cols)), shape=(n * n, m)).tocsr()
    CB = sp.coo_matrix((dataB, (rows, cols)), shape=(n * n, m)).tocsr()

    A_vec = cp.Constant(CA) @ y
    B_vec = cp.Constant(CB) @ y

    # Explicit order matching column-major flattening
    A = cp.reshape(A_vec, (n, n), order="F")
    B = cp.reshape(B_vec, (n, n), order="F")

    K = cp.bmat([[A, -B],
                 [B,  A]])
    return A, B, K


def compile_operator_linear_form(rep: PauliMomentMatrixRep, op: Operator, *, tol: float = 1e-12
                                 ) -> np.ndarray:
    """
    Compile <op> = sum_u c_u <u> into a real coefficient vector c over y,
    assuming y_u = <u> are real (u Hermitian Pauli words).

    For a Hermitian operator expressed in Pauli words, coefficients should be real.
    We allow small imaginary parts (numerical noise) and drop them; otherwise we raise.
    """
    m = len(rep.labels)
    c = np.zeros(m, dtype=float)

    for u, coef in op.items():
        if abs(coef.imag) > tol:
            raise ValueError(f"Operator coefficient for {u} has significant imaginary part: {coef}")
        if u not in rep.label_index:
            raise KeyError(f"Operator contains label not present in rep.labels: {u}")
        c[rep.label_index[u]] += float(coef.real)

    return c


def build_sdp_from_rep(rep: PauliMomentMatrixRep,
                       objective_op: Operator,
                       *,
                       sense: Sense = "min",
                       extra_constraints: Optional[List[cp.Constraint]] = None
                       ) -> MomentSDP:
    """
    Build and return a CVXPY+MOSEK-ready SDP:
      optimize  <objective_op>  subject to  M >= 0  and y_I=1.

    - Moments y_u are modeled as real variables.
    - The complex PSD constraint is imposed via the real embedding block matrix.
    """
    m = len(rep.labels)
    y = cp.Variable(m, name="y")  # real vector of moments

    A, B, K = pauli_moment_matrix_real_embedding(rep, y)

    constraints: List[cp.Constraint] = []
    # Normalization: <I> = 1
    constraints.append(y[rep.idx_I] == 1.0)

    # PSD constraint
    constraints.append(K >> 0)

    # Any additional linear constraints
    if extra_constraints:
        constraints.extend(extra_constraints)

    # Objective: <objective_op> = c @ y
    c = compile_operator_linear_form(rep, objective_op)
    obj_expr = c @ y  # real scalar

    objective = cp.Minimize(obj_expr) if sense == "min" else cp.Maximize(obj_expr)
    problem = cp.Problem(objective, constraints)

    return PauliMomentSDP(
        rep=rep,
        y=y,
        M_real=A,
        M_imag=B,
        PSD_block=K,
        constraints=constraints,
        objective=obj_expr,
        problem=problem,
    )


def solve_pauli_relaxation(
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
    
    rep = compile_moment_matrix_rep(basis)
    
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
    
    # Solve
    sdp.problem.solve(solver=solver, verbose=verbose, **merged_solver_opts)
    
    return float(sdp.problem.value)




