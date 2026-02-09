# This whole file will be generalised later when we add the bell scenario

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import cvxpy as cp
import scipy.sparse as sp

from spins_sdp.pauli import compile_moment_matrix_rep, Operator, PauliMomentMatrixRep, PauliWord, multiply_words
from spins_sdp.symmetry import SymmetryManager

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


def compile_operator_linear_form(rep: PauliMomentMatrixRep, op: Operator, *, tol: float = 1e-12,
                                 symmetry_manager: Optional[SymmetryManager] = None
                                 ) -> np.ndarray:
    """
    Compile <op> = sum_u c_u <u> into a real coefficient vector c over y,
    assuming y_u = <u> are real (u Hermitian Pauli words).

    For a Hermitian operator expressed in Pauli words, coefficients should be real.
    We allow small imaginary parts (numerical noise) and drop them; otherwise we raise.
    
    If symmetry_manager is provided, operator terms are canonicalized before lookup.
    """
    m = len(rep.labels)
    c = np.zeros(m, dtype=float)

    for u, coef in op.items():
        if abs(coef.imag) > tol:
            raise ValueError(f"Operator coefficient for {u} has significant imaginary part: {coef}")
        
        # Canonicalize the operator term if symmetry manager is provided
        lookup_key = u
        if symmetry_manager is not None:
            canonical = symmetry_manager.canonicalize(u)
            if canonical is None:
                # Term is killed by symmetry (e.g., sign symmetry) - skip it
                continue
            lookup_key = canonical
        
        if lookup_key not in rep.label_index:
            raise KeyError(f"Operator contains label not present in rep.labels: {u} (canonical: {lookup_key})")
        c[rep.label_index[lookup_key]] += float(coef.real)

    return c

def solve_pauli_relaxation(
    basis: List[PauliWord],
    operator: Operator,
    symmetry_manager: SymmetryManager,
    sense: Sense = "min",
    mosek_tol: float = 1e-6,
    verbose: bool = False) -> float:
    """
    Solve a moment relaxation SDP for a given basis and operator.
    
    Args:
        basis: List of Pauli words defining the relaxation basis
        operator: Objective operator as a Pauli word dictionary
        sense: "min" or "max" optimization
        mosek_tol: MOSEK conic tolerance
        verbose: Print solver output
        
    Returns:
        Optimal objective value
    """
    # Build block reps with chosen symmetries
    reps, _ = build_block_reps(basis, symmetry_manager)
    
    # Build Block-Diagonal SDP
    sdp = build_block_diagonal_sdp(reps, operator, sense=sense, symmetry_manager=symmetry_manager)
    problem = sdp.problem
    
    default_solver_opts: Dict[str, Any] = {
        "mosek_params": {
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": mosek_tol,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": mosek_tol,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": mosek_tol,
        }
    }
    
    problem.solve(solver="MOSEK", verbose=verbose, **default_solver_opts)
    
    return float(problem.value)
    

# ----------------------------------
# Block diagonalization
# ----------------------------------

def build_block_reps(full_basis: List[PauliWord], symmetry_manager: SymmetryManager) -> Tuple[List[PauliMomentMatrixRep], Dict[PauliWord, int]]:
    blocks = {}
        
    # Split Basis (Rotation Symmetry)
    if symmetry_manager.use_rotation:
        for w in full_basis:
            blocks.setdefault(w.signature(), []).append(w)
    else:
        blocks["all"] = list(full_basis)
    
    # Collect Global Labels from ALL pairs in the FULL basis
    # This ensures that products that canonicalize across blocks are included
    raw_to_canonical = {}
    canonical_set = {PauliWord(0,0)}
    
    # Always map Identity to Identity
    raw_to_canonical[PauliWord(0,0)] = PauliWord(0,0)

    # Iterate over ALL pairs in the full basis (not just within-block pairs)
    # This is necessary because moment matrix entries M[i,j] = <w_i^dag w_j>
    # come from pairs within each block, but their canonical forms must all
    # be present in the global index.
    for sig, block_basis in blocks.items():
        for i, w1 in enumerate(block_basis):
            for w2 in block_basis[i:]:
                _, u = multiply_words(w1, w2)
                
                # Delegate all logic to the manager
                c = symmetry_manager.canonicalize(u)
                
                if c is None:
                    # Sign symmetry killed it
                    continue
                
                # Map the raw moment u to this final canonical form c
                raw_to_canonical[u] = c
                canonical_set.add(c)

    # Build Registry for CANONICAL words only
    sorted_canons = sorted(list(canonical_set), key=lambda u: (u.support_size(), u.x_mask, u.z_mask))
    
    # Ensure I is at 0
    if PauliWord(0,0) in sorted_canons:
        sorted_canons.remove(PauliWord(0,0))
    sorted_canons.insert(0, PauliWord(0,0))
    
    canonical_index = {u: i for i, u in enumerate(sorted_canons)}
    
    # Build the FINAL Global Index
    # This maps every Raw U -> The index of its Canonical Representative
    global_index = {u: canonical_index[c] for u, c in raw_to_canonical.items()}
    
    # Compile Blocks
    # Each block is compiled with the full canonical_index so that all blocks
    # share the same variable indices (important for the combined SDP).
    reps = []
    for sig in sorted(blocks.keys()):
        if blocks[sig]:
            reps.append(compile_moment_matrix_rep(blocks[sig], symmetry_manager, precomputed_label_index=canonical_index))
            
    return reps, global_index


def build_block_diagonal_sdp(
    reps: List[PauliMomentMatrixRep],
    objective_op: Operator,
    sense: Sense = "min",
    symmetry_manager: Optional[SymmetryManager] = None
) -> PauliMomentSDP:
    """
    Constructs a Block-Diagonal SDP from a list of Moment Matrix blocks (reps).
    Handles both Real Symmetric blocks (if rep.b_coef is empty) and 
    Complex Hermitian blocks (via real embedding) automatically.
    """
    if not reps:
        raise ValueError("Must provide at least one block rep.")
    
    # All reps share the same global labels (enforced by the builder), 
    # so we use the first rep to determine size and normalization index.
    rep0 = reps[0]
    m = len(rep0.labels)
    y = cp.Variable(m, name="y")
    
    constraints = []
    
    # Normalization <I> = 1
    constraints.append(y[rep0.idx_I] == 1.0)
    
    # Iterate over Blocks
    for rep in reps:
        n = rep.label_idx.shape[0]
        
        # We map y -> flattened matrix vector (size n*n)
        rows = np.arange(n * n, dtype=np.int32)
        cols = rep.label_idx.reshape(-1, order="F").astype(np.int32)
        
        # --- Build Real Part A ---
        dataA = rep.a_coef.reshape(-1, order="F").astype(float)
        # Construct the linear map CA: y -> vec(A)
        CA = sp.coo_matrix((dataA, (rows, cols)), shape=(n * n, m)).tocsr()
        vec_A = cp.Constant(CA) @ y
        A = cp.reshape(vec_A, (n, n), order="F")
        
        if rep.b_coef is None or rep.b_coef.size == 0:
            # --- CASE 1: Real Symmetric Block ---
            # Im(M) is structurally zero. We just enforce A >= 0.
            constraints.append(A >> 0)
        else:
            # --- CASE 2: Complex Hermitian Block ---
            # Build Imaginary Part B
            dataB = rep.b_coef.reshape(-1, order="F").astype(float)
            CB = sp.coo_matrix((dataB, (rows, cols)), shape=(n * n, m)).tocsr()
            vec_B = cp.Constant(CB) @ y
            B = cp.reshape(vec_B, (n, n), order="F")
            
            # Real Embedding: [[A, -B], [B, A]] >= 0
            K = cp.bmat([[A, -B], [B, A]])
            constraints.append(K >> 0)
    
    # Objective Function
    c = compile_operator_linear_form(rep0, objective_op, symmetry_manager=symmetry_manager)
    obj_expr = c @ y
            
    objective = cp.Minimize(obj_expr) if sense == "min" else cp.Maximize(obj_expr)
    problem = cp.Problem(objective, constraints)
    
    return PauliMomentSDP(
        rep=rep0,          # Reference rep (for label lookup)
        y=y,
        M_real=None,       # Ambiguous for multi-block
        M_imag=None,       # Ambiguous for multi-block
        PSD_block=None,    # Ambiguous for multi-block
        constraints=constraints,
        objective=obj_expr,
        problem=problem
    )



