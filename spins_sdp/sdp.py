# This whole file will be generalised later when we add the bell scenario

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import cvxpy as cp
import scipy.sparse as sp

from spins_sdp.pauli import compile_moment_matrix_rep, Operator, PauliMomentMatrixRep, PauliWord, multiply_words


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

@dataclass(frozen=True, slots=True)
class PauliMomentDualSDP:
    """Dual (real-embedded) SDP for the moment relaxation."""
    rep: PauliMomentMatrixRep
    S: cp.Variable                 # PSD matrix variable (2n x 2n)
    lam: cp.Variable               # scalar dual variable (objective)
    L: sp.csr_matrix               # linear map vec(K)=L@y
    constraints: List[cp.Constraint]
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
                       ) -> PauliMomentSDP:
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
    formulation: Literal["primal", "dual"] = "primal",
    solver: str = "MOSEK",
    mosek_tol: float = 1e-6,
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
    if formulation == "primal":
        sdp = build_sdp_from_rep(rep, operator, sense=sense)
        problem = sdp.problem
    elif formulation == "dual":
        dual_sdp  = build_dual_sdp_from_rep(rep, operator, sense=sense)
        problem = dual_sdp.problem
    else:
        raise ValueError("formulation must be 'primal' or 'dual'")
    
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
    problem.solve(solver=solver, verbose=verbose, **merged_solver_opts)
    
    # In dual mode with sense='max', we solved min(-c^T y), so max(c^T y) = -lambda*
    if formulation == "dual" and sense == "max":
        return float(-problem.value)
    return float(problem.value)



# ----------------------------------
# DUAL STUFF
# ----------------------------------

def compile_real_embedding_linear_map(rep: PauliMomentMatrixRep) -> sp.csr_matrix:
    """Compile sparse L such that vec(K) = L @ y (column-major vec).

    K = [[A, -B],
         [B,  A]]
    where A_ij = a_ij * y[label_ij], B_ij = b_ij * y[label_ij].
    """
    n = rep.label_idx.shape[0]
    m = len(rep.labels)
    dim = 2 * n  # K is (2n x 2n)

    # Flatten in same order as your primal (column-major)
    lbl = rep.label_idx.reshape(-1, order="F").astype(np.int32)
    a = rep.a_coef.reshape(-1, order="F").astype(np.int8)
    b = rep.b_coef.reshape(-1, order="F").astype(np.int8)

    # Column-major indices for (i,j) over an (n,n) block:
    # i varies fastest, then j
    i = np.tile(np.arange(n, dtype=np.int32), n)
    j = np.repeat(np.arange(n, dtype=np.int32), n)

    # vec index for K (column-major): idx = row + col * dim
    row_TL = i + j * dim
    row_BL = (i + n) + j * dim
    row_TR = i + (j + n) * dim
    row_BR = (i + n) + (j + n) * dim

    maskA = a != 0
    maskB = b != 0

    rows = np.concatenate([row_TL[maskA], row_BR[maskA], row_BL[maskB], row_TR[maskB]])
    cols = np.concatenate([lbl[maskA],    lbl[maskA],    lbl[maskB],    lbl[maskB]])
    data = np.concatenate([a[maskA],      a[maskA],      b[maskB],     -b[maskB]]).astype(float)

    L = sp.coo_matrix((data, (rows, cols)), shape=(dim * dim, m)).tocsr()
    L.sum_duplicates()
    return L


def build_dual_sdp_from_rep(
    rep: PauliMomentMatrixRep,
    objective_op: Operator,
    *,
    sense: Sense = "min",
) -> PauliMomentDualSDP:
    """Build the *dual* of the core relaxation (PSD + y_I=1) in real embedding.

    Primal (min):
        min    c^T y
        s.t.   K(y) >= 0,
               y_I = 1

    Dual:
        max    lam
        s.t.   L^T vec(S) = c - lam * e_I
               S >= 0

    NOTE: If you add extra linear constraints on y, the dual gets extra variables.
    """

    c = compile_operator_linear_form(rep, objective_op)
    # For max, solve min(-c^T y) and flip sign at the end
    if sense == "max":
        c = -c

    n = rep.label_idx.shape[0]
    dim = 2 * n
    m = len(rep.labels)

    L = compile_real_embedding_linear_map(rep)

    S = cp.Variable((dim, dim), PSD=True, name="S")
    lam = cp.Variable(name="lambda")

    eI = np.zeros(m, dtype=float)
    eI[rep.idx_I] = 1.0

    constraints = [
        (cp.Constant(L.T) @ cp.vec(S)) == (c - lam * eI)
    ]
    problem = cp.Problem(cp.Maximize(lam), constraints)

    return PauliMomentDualSDP(rep=rep, S=S, lam=lam, L=L, constraints=constraints, problem=problem)




# ----------------------------------
# Block diagonalization
# ----------------------------------

def build_block_reps(full_basis: List[PauliWord]) -> Tuple[List[PauliMomentMatrixRep], Dict[PauliWord, int]]:
    # Split Basis
    blocks = {}
    for w in full_basis:
        sig = w.signature()
        blocks.setdefault(sig, []).append(w)
    
    # Collect Global Labels
    global_labels_set = {PauliWord(0, 0)} # Always include Identity
    
    for sig, block_basis in blocks.items():
        # Iterate over all pairs in this block
        for i, w1 in enumerate(block_basis):
            for j in range(i, len(block_basis)):  # upper triangle
                w2 = block_basis[j]
                _, u = multiply_words(w1, w2)
                global_labels_set.add(u)
    
    # Build Global Registry
    # Sort them deterministically
    sorted_labels = sorted(list(global_labels_set), key=lambda u: (u.support_size(), u.x_mask, u.z_mask))
    
    # Force Identity to be at index 0
    if sorted_labels[0].x_mask != 0 or sorted_labels[0].z_mask != 0:
        # Find I and move it
        I = PauliWord(0,0)
        sorted_labels.remove(I)
        sorted_labels.insert(0, I)
        
    global_index = {u: i for i, u in enumerate(sorted_labels)}
    
    # Compile Blocks using the Registry
    reps = []
    for sig in sorted(blocks.keys()): # Deterministic order of blocks
        basis_sub = blocks[sig]
        if not basis_sub: continue # Skip empty blocks
        
        rep = compile_moment_matrix_rep(basis_sub, precomputed_label_index=global_index)
        reps.append(rep)
        
    return reps, global_index


def build_block_diagonal_sdp(
    reps: List[PauliMomentMatrixRep],
    objective_op: Operator,
    sense: Sense = "min"
) -> PauliMomentSDP:
    
    # Create ONE variable vector y
    # All reps share the same global labels, so we can check the first one
    m = len(reps[0].labels)
    y = cp.Variable(m, name="y")
    
    constraints = []
    
    # Normalization <I> = 1
    constraints.append(y[reps[0].idx_I] == 1.0)
    
    for rep in reps:
        # Build the embedding K for this specific block
        A, B, K = pauli_moment_matrix_real_embedding(rep, y)
        
        # Add the constraint: Block K must be PSD
        constraints.append(K >> 0)

    # Objective
    # Compile objective using the registry (from any rep)
    c = compile_operator_linear_form(reps[0], objective_op)
    obj_expr = c @ y
    
    objective = cp.Minimize(obj_expr) if sense == "min" else cp.Maximize(obj_expr)
    problem = cp.Problem(objective, constraints)
    
    # Return a slightly modified SDP object (or standard one if you don't need M_real access)
    return PauliMomentSDP(
        rep=reps[0], # Just store one for reference
        y=y,
        M_real=None,
        M_imag=None,
        PSD_block=None,
        constraints=constraints,
        objective=obj_expr,
        problem=problem
    )


def solve_block_diagonal_pauli_relaxation(
    full_basis: List[PauliWord],
    operator: Operator,
    sense: Sense = "min",
    solver: str = "MOSEK",
    mosek_tol: float = 1e-6,
    solver_opts: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> float:
    """
    Solve a moment relaxation SDP for a given full basis and operator exploiting block diagonalization.
    
    Args:
        full_basis: List of Pauli words defining the full relaxation basis
        operator: Objective operator as a Pauli word dictionary
        sense: "min" or "max" optimization
        solver: default "MOSEK"
        mosek_tol: MOSEK conic tolerance
        solver_opts: Override default solver options
        verbose: Print solver output
        
    Returns:
        Optimal objective value
    """
    
    reps, _ = build_block_reps(full_basis)
    
    # Build SDP
    sdp = build_block_diagonal_sdp(reps, operator, sense=sense)
    problem = sdp.problem
    
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
    problem.solve(solver=solver, verbose=verbose, **merged_solver_opts)
    
    return float(problem.value)










