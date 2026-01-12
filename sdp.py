import cvxpy as cp
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional, Union, Literal

from spin_sdp.pauli_strings import (
    pauli_normal_form,
    multiply_moments,
    dagger,
    op_from_word,
    op_add,
    op_scalar_mul,
    op_multiply,
    OperatorDict,
    PauliString
)

BoundaryType = Literal["open", "periodic"]
# Type alias for CVXPY expression
CvxExpr = Union[cp.Expression, float, complex, int]
LabelToExpr = Dict[str, CvxExpr]

def average_magnetization(axis: str, N: int, label_to_expr: LabelToExpr) -> CvxExpr:
    """
    Return the average magnetization along a given axis ('x', 'y', or 'z')
    as a CVXPY expression.
    """
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y', or 'z'")

    mag = 0
    for i in range(N):
        lbl = f"{axis}{i}"
        if lbl in label_to_expr:
            mag += label_to_expr[lbl]
        # if not present (e.g. at a low NPA level), we just skip it

    return mag / N

def ising_pauli_symbolic_terms(N: int, J: float, h: float, k: float, boundary: BoundaryType) -> OperatorDict:
    """
    Return the Ising Hamiltonian as a dictionary of Pauli-word coefficients.
    
    H = -J Σ_i σ_i^z σ_{i+1}^z - h Σ_i σ_i^x - k Σ_i σ_i^z
    
    Returns:
        H_terms: Mapping from Pauli-word strings (like 'x0', 'z0z1') to coefficients (floats).
    """
    H_terms: OperatorDict = {}

    # -J Σ_i σ_i^z σ_{i+1}^z
    for i in range(N):
        if i < N - 1 or boundary == "periodic":
            w = f"z{i}z{(i+1) % N}"
            H_terms[w] = H_terms.get(w, 0) - J

    # -h Σ_i σ_i^x
    for i in range(N):
        w = f"x{i}"
        H_terms[w] = H_terms.get(w, 0) - h

    # -k Σ_i σ_i^z
    for i in range(N):
        w = f"z{i}"
        H_terms[w] = H_terms.get(w, 0) - k

    return H_terms

def moment_matrix_dict(relaxation: List[PauliString]) -> Dict[Tuple[int, int], Tuple[complex, str]]:
    """
    Construct the symbolic moment matrix from a list of Pauli strings.
    Returns a dict mapping (i,j) -> (coeff, label).
    """
    moments = []
    for string in relaxation:
        moments.append(pauli_normal_form(string))
    
    M_dict = {}
    for i in range(len(moments)):
        for j in range(len(moments)):
            M_dict[(i,j)] = multiply_moments(dagger(moments[i]), moments[j])

    return M_dict

def dict_to_cvxpy_matrix_from_expr(M_dict: Dict[Tuple[int, int], Tuple[complex, str]], label_to_expr: LabelToExpr) -> cp.Expression:
    """
    Convert the symbolic moment matrix dict into a CVXPY matrix expression.
    
    M_dict: (i,j) -> (coef, label) from moment_matrix_dict
    label_to_expr: dict mapping each Pauli word 'label' -> CVXPY Expression
    """
    if not M_dict:
        return np.array([[]]) # empty matrix

    rows = max(i for (i, j) in M_dict.keys()) + 1
    cols = max(j for (i, j) in M_dict.keys()) + 1

    M_entries = [[0 for _ in range(cols)] for _ in range(rows)]

    for (i, j), (coef, label) in M_dict.items():
        if label not in label_to_expr:
             # If the label is not found, it implies it wasn't created as a variable.
             # This might happen if the npa level is low but higher moments appear?
             # Usually label_to_expr should contain everything needed if built correctly.
             # However, for robustness, we can assume 0 or raise error?
             # Based on notebook logic, it assumes it exists.
             pass
        
        expr = coef * label_to_expr[label]  # label_to_expr[label] is affine
        M_entries[i][j] = expr

    return cp.bmat(M_entries)

def build_sdp_variables(x: cp.Variable, z: cp.Variable, zz: cp.Variable, 
                        all_labels: List[str], boundary: str, N: int) -> Tuple[LabelToExpr, Dict[str, cp.Variable]]:
    """
    Map each label -> CVXPY expression (x[i], z[i], zz[b], or new scalar var).
    
    Returns:
        label_to_expr: complete mapping
        extra_moments: dict of new variables created for higher moments
    """
    label_to_expr: LabelToExpr = {}
    extra_moments: Dict[str, cp.Variable] = {}

    for label in all_labels:
        # Identity
        if label == "I":
            label_to_expr[label] = 1
            continue

        # Single-site x_i
        m = re.fullmatch(r"x(\d+)", label)
        if m:
            i = int(m.group(1))
            if 0 <= i < N:
                label_to_expr[label] = x[i]
                continue

        # Single-site z_i
        m = re.fullmatch(r"z(\d+)", label)
        if m:
            i = int(m.group(1))
            if 0 <= i < N:
                label_to_expr[label] = z[i]
                continue

        # Two-site z_i z_j
        m = re.fullmatch(r"z(\d+)z(\d+)", label)
        if m:
            i = int(m.group(1))
            j = int(m.group(2))

            if boundary == "open":
                # bonds (0,1), (1,2), ..., (N-2,N-1)
                # assuming ordered i<j, but pauli_normal_form orders them.
                # if j = i+1
                if j == i + 1 and 0 <= i < N - 1:
                    label_to_expr[label] = zz[i]
                    continue
            else:  # periodic
                # standard neighbors
                if j == i + 1 and 0 <= i < N - 1:
                    label_to_expr[label] = zz[i]
                    continue
                # wrap-around bond (N-1,0) -> "z0z(N-1)" or "z(N-1)z0"?
                # Sorted order means "z0 z(N-1)" is likely "z0z{N-1}" if N-1 > 0.
                # Actually pauli_normal_form sorts by index.
                # So "z(N-1) z0" -> "z0 z(N-1)"
                # If i=0, j=N-1
                if i == 0 and j == N - 1:
                     label_to_expr[label] = zz[N - 1]
                     continue

        # If we reach here, this label is some other moment
        v = cp.Variable(name=f"m_{label}")  # real scalar variable
        extra_moments[label] = v
        label_to_expr[label] = v
        
    return label_to_expr, extra_moments

def ising_energy_expr(J, h, k, boundary, x, z, zz) -> cp.Expression:
    """
    Linearized Ising energy expression.
    H = -J Sum zz - h Sum x - k Sum z
    """
    # energy = -J * cp.sum(zz) - h * cp.sum(x) - k * cp.sum(z)
    # Be careful with sum dimensions if variables are not full vectors
    return -J * cp.sum(zz) - h * cp.sum(x) - k * cp.sum(z)

def op_expectation_expr(op_terms: OperatorDict, label_to_expr: LabelToExpr) -> cp.Expression:
    """
    Given an operator as a dict {word: coeff}, build the CVXPY expression
    for its expectation value: sum_word coeff * <word>.
    """
    expr = 0
    for word, c in op_terms.items():
        if word not in label_to_expr:
            raise KeyError(f"Pauli word '{word}' not in label_to_expr.")
        expr += c * label_to_expr[word]
    # Ensure it's a scalar expression (sum can return scalar?)
    return expr

def commutator_expectation_expr(H_terms: OperatorDict, a_label: str, label_to_expr: LabelToExpr) -> cp.Expression:
    """
    Build the CVXPY expression for <[H, a]>.
    """
    coeffs: OperatorDict = {}

    for w_H, c_H in H_terms.items():
        # H a
        phase1, w1 = pauli_normal_form(w_H + a_label)
        # a H
        phase2, w2 = pauli_normal_form(a_label + w_H)

        # [H,a] = H a - a H
        coeffs[w1] = coeffs.get(w1, 0) + c_H * phase1
        coeffs[w2] = coeffs.get(w2, 0) - c_H * phase2

    return op_expectation_expr(coeffs, label_to_expr)


def build_diagonality_constraints(H_terms: OperatorDict, h_labels: List[str], label_to_expr: LabelToExpr) -> List[cp.Constraint]:
    """
    Build constraints <[H, h_i h_j^dag]> = 0.
    """
    constraints = []

    for i, hi in enumerate(h_labels):
        for j, hj in enumerate(h_labels):
            # For Pauli strings, h_j^dag = h_j.
            # a = h_i h_j^dag
            phase, a_word = pauli_normal_form(hi + hj)

            # [H, a]
            comm_expr_word = commutator_expectation_expr(H_terms, a_word, label_to_expr)
            comm_expr = phase * comm_expr_word

            constraints.append(comm_expr == 0)

    return constraints


def lambda_entry_expr(beta: float, Delta: float, H_terms: OperatorDict, 
                      h_j_label: str, h_k_label: str, label_to_expr: LabelToExpr) -> cp.Expression:
    """
    Build the CVXPY expression for Lambda_{jk}(beta, Delta).
    """
    # Represent basis operators and H as dicts
    h_j     = op_from_word(h_j_label)
    h_k     = op_from_word(h_k_label)
    h_j_dag = h_j      # Pauli strings are Hermitian
    h_k_dag = h_k
    H_op    = H_terms  # dict word -> coeff

    # hh_dag and h_dag_h (for Pauli strings, they coincide up to canonical normalization)
    h_j_h_k_dag = op_multiply(h_j, h_k_dag)   # h_j h_k+
    h_j_dag_h_k = op_multiply(h_j_dag, h_k)   # h_j+ h_k

    # 1) h_j H h_k+
    term1 = op_multiply(op_multiply(h_j, H_op), h_k_dag)

    # 2) -1/2 { h_j h_k+ , H }
    hhH  = op_multiply(h_j_h_k_dag, H_op)
    Hhh  = op_multiply(H_op, h_j_h_k_dag)
    anticomm1 = op_add(hhH, Hhh)          # (hh+)H + H(hh+)
    term2 = op_scalar_mul(anticomm1, -0.5)

    # First bracket:
    op_br1 = op_add(term1, term2)

    # 3) h_j+ H h_k
    term3 = op_multiply(op_multiply(h_j_dag, H_op), h_k)

    # 4) -1/2 { h_j+ h_k , H }
    hdagh  = h_j_dag_h_k
    hdaghH = op_multiply(hdagh, H_op)
    Hhdagh = op_multiply(H_op, hdagh)
    anticomm2 = op_add(hdaghH, Hhdagh)
    term4 = op_scalar_mul(anticomm2, -0.5)

    # Second bracket:
    op_br2 = op_add(term3, term4)

    # 5) Delta( h_j h_k+ - e^{-beta Delta} h_j+ h_k )
    exp_factor = np.exp(-beta * Delta)
    op_br3_inner = op_add(h_j_h_k_dag,
                          op_scalar_mul(h_j_dag_h_k, -exp_factor))
    op_br3 = op_scalar_mul(op_br3_inner, Delta)

    # Combine all pieces:
    op_total = op_add(
        op_add(op_br1, op_scalar_mul(op_br2, exp_factor)),
        op_br3
    )

    # Finally, expectation value:
    return op_expectation_expr(op_total, label_to_expr)


def build_lambda_matrix(beta: float, Delta: float, H_terms: OperatorDict, 
                        h_labels: List[str], label_to_expr: LabelToExpr) -> cp.Expression:
    """
    Construct the matrix Lambda(beta, Delta).
    """
    m = len(h_labels)
    entries = [[0 for _ in range(m)] for _ in range(m)]

    for j, h_j_label in enumerate(h_labels):
        for k, h_k_label in enumerate(h_labels):
            entries[j][k] = lambda_entry_expr(
                beta, Delta, H_terms, h_j_label, h_k_label, label_to_expr
            )

    return cp.bmat(entries)
