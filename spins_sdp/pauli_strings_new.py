from __future__ import annotations
from dataclasses import dataclass
from typing import Final, Iterable, Dict, List, Literal, Tuple, Optional

import numpy as np
import cvxpy as cp
import scipy.sparse as sp

PhaseExp = int  # always interpreted mod 4

# i^p for p in {0,1,2,3}
_I_POW: Final[Tuple[complex, complex, complex, complex]] = (1+0j, 1j, -1+0j, -1j)

# Precompute int real/imag coefficients once
_PHASE_RE: Final[Tuple[int, int, int, int]] = tuple(int(c.real) for c in _I_POW)  # ( 1, 0,-1, 0)
_PHASE_IM: Final[Tuple[int, int, int, int]] = tuple(int(c.imag) for c in _I_POW)  # ( 0, 1, 0,-1)


@dataclass(frozen=True, slots=True)
class PauliWord:
    """
    Pauli word on N qubits encoded by two bitmasks.

    Bit k of x_mask is 1 iff there is an X-component on qubit k.
    Bit k of z_mask is 1 iff there is a Z-component on qubit k.

    Local decoding:
      (0,0)->I, (1,0)->X, (0,1)->Z, (1,1)->Y
    """
    x_mask: int
    z_mask: int

    def __post_init__(self) -> None:
        if self.x_mask < 0 or self.z_mask < 0:
            raise ValueError("Bitmasks must be non-negative integers.")
    
    def support_size(self) -> int:
        """Number of non-identity sites."""
        return (self.x_mask | self.z_mask).bit_count()


@dataclass(frozen=True, slots=True)
class NPABasis:
    """
    Canonical unique basis up to level k (shortest-length representatives).

    words:      all unique words reachable by <= k generator multiplications
    levels:     levels[ℓ] are the words whose minimal length is exactly ℓ
    min_len:    minimal length at which each word appears
    index:      word -> index in `words` (deterministic ordering)
    """
    N: int
    k: int
    words: List[PauliWord]
    levels: List[List[PauliWord]]
    min_len: Dict[PauliWord, int]
    index: Dict[PauliWord, int]


@dataclass(frozen=True, slots=True)
class MomentMatrixRep:
    """
    Represents M_ij = (a_coef_ij + i b_coef_ij) * y[label_idx_ij],
    where y[...] are real moment variables y_u = <u>.
    """
    basis: List[PauliWord]                 # w_0..w_{n-1}
    labels: List[PauliWord]                # u_0..u_{m-1} (moments needed)
    label_index: Dict[PauliWord, int]      # u -> idx
    label_idx: np.ndarray                  # (n,n) int: which moment label is used at entry (i,j)
    a_coef: np.ndarray                     # (n,n) int8: Re coefficient in { -1,0,1 }
    b_coef: np.ndarray                     # (n,n) int8: Im coefficient in { -1,0,1 }
    idx_I: int                             # index of identity label


@dataclass(frozen=True, slots=True)
class MomentSDP:
    rep: MomentMatrixRep
    y: cp.Variable
    M_real: cp.Expression
    M_imag: cp.Expression
    PSD_block: cp.Expression
    constraints: List[cp.Constraint]
    objective: cp.Expression
    problem: cp.Problem


Operator = Dict[PauliWord, complex]
Sense = Literal["min", "max"]


def multiply_words(a: PauliWord, b: PauliWord) -> tuple[PhaseExp, PauliWord]:
    """
    Multiply two Pauli words (binary representation).

    Returns:
      (p, c) where
        a * b = i^p * c
      and c is returned as a PauliWord (up to phase).

    Convention:
      P(x,z) = i^{|x&z|} X^x Z^z
    """
    x1, z1 = a.x_mask, a.z_mask
    x2, z2 = b.x_mask, b.z_mask

    # label part is XOR
    x3 = x1 ^ x2
    z3 = z1 ^ z2

    # popcount helpers
    a_xz = (x1 & z1).bit_count()
    b_xz = (x2 & z2).bit_count()
    c_xz = (x3 & z3).bit_count()
    zx   = (z1 & x2).bit_count()   # counts Z from first commuting past X from second

    # phase exponent mod 4:
    # p = |x1&z1| + |x2&z2| - |x3&z3| + 2|z1&x2|   (mod 4)
    p = (a_xz + b_xz - c_xz + 2 * zx) & 3

    return p, PauliWord(x3, z3)


@dataclass(frozen=True, slots=True)
class PauliTerm:
    coeff: complex
    word: PauliWord


def multiply_terms(t1: PauliTerm, t2: PauliTerm) -> PauliTerm:
    """
    Multiply two Pauli terms (coeff * word), returning a single term in canonical form:
      (c1*w1) * (c2*w2) = (c1*c2*i^p) * w3
    """
    p, w3 = multiply_words(t1.word, t2.word)
    return PauliTerm(t1.coeff * t2.coeff * _I_POW[p], w3)



def _atoms_xyz(N: int) -> List[PauliWord]:
    """
    Atomic generators in a fixed deterministic order: x0,y0,z0,x1,y1,z1,...,x(N-1),y(N-1),z(N-1).
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    atoms: List[PauliWord] = []
    for i in range(N):
        bit = 1 << i
        atoms.append(PauliWord(bit, 0))     # X_i
        atoms.append(PauliWord(bit, bit))   # Y_i
        atoms.append(PauliWord(0, bit))     # Z_i
    return atoms


def generate_npa_basis(N: int, k: int) -> NPABasis:
    """
    Generate the canonical unique basis up to level k.

    Level definition:
      Start from identity I.
      At each step, multiply by any single-site generator in {X_i, Y_i, Z_i}.
      Keep each resulting canonical PauliWord the *first* time it is discovered (minimal length).

    Output ordering:
      sort by (min_length, support_size, x_mask, z_mask).

    Returns an NPABasis containing:
      - words: concatenation of all levels (sorted deterministically)
      - levels: list of per-length frontiers (also sorted deterministically)
      - min_len: shortest length for each word
      - index: mapping word -> row/col index
    """
    if k < 0:
        raise ValueError("k must be >= 0.")
    if N <= 0:
        raise ValueError("N must be >= 1.")

    I = PauliWord(0, 0)
    atoms = _atoms_xyz(N)

    # BFS structures
    levels: List[List[PauliWord]] = [[] for _ in range(k + 1)]
    min_len: Dict[PauliWord, int] = {I: 0}
    frontier: List[PauliWord] = [I]
    levels[0] = [I]

    # BFS by length: each discovered word is assigned its minimal length
    for length in range(1, k + 1):
        next_set: set[PauliWord] = set()
        for w in frontier:
            for a in atoms:
                _, w_new = multiply_words(w, a)   # ignore phase for the basis
                if w_new not in min_len:          # first time discovered => minimal length
                    min_len[w_new] = length
                    next_set.add(w_new)

        # ordering within the level
        next_level = sorted(
            next_set,
            key=lambda ww: (ww.support_size(), ww.x_mask, ww.z_mask),
        )
        levels[length] = next_level
        frontier = next_level

        # early stop if nothing new appears
        if not frontier:
            # trim remaining empty levels for cleanliness
            levels = levels[: length + 1]
            break

    # global ordering
    all_words = list(min_len.keys())
    all_words_sorted = sorted(
        all_words,
        key=lambda w: (min_len[w], w.support_size(), w.x_mask, w.z_mask),
    )

    index = {w: i for i, w in enumerate(all_words_sorted)}

    return NPABasis(
        N=N,
        k=k,
        words=all_words_sorted,
        levels=levels,
        min_len=min_len,
        index=index,
    )


def compile_moment_matrix_rep(basis: List[PauliWord]) -> MomentMatrixRep:
    """
    Given basis monomials W=[w_i], build the compiled representation of the moment matrix:
      M_ij = <w_i^† w_j> = (known phase) * <u_ij>
    Since (reduced) Pauli words are Hermitian, w_i^† = w_i.

    Output contains:
      - the set of required moment labels u (canonical Pauli words),
      - and for each (i,j) the label index plus the re/im phase coefficient.
    """
    n = len(basis)
    if n == 0:
        raise ValueError("Basis must be non-empty.")

    I = PauliWord(0, 0)

    # First pass: collect all labels u_ij that appear in products w_i w_j (upper triangle)
    label_set: set[PauliWord] = {I}
    for i in range(n):
        wi = basis[i]
        for j in range(i, n):
            wj = basis[j]
            _, u = multiply_words(wi, wj)  # dagger is trivial for Pauli words
            label_set.add(u)

    # Label ordering: put I first, then by (support, x_mask, z_mask)
    labels_rest = sorted(
        (u for u in label_set if u != I),
        key=lambda u: (u.support_size(), u.x_mask, u.z_mask),
    )
    labels = [I] + labels_rest
    label_index = {u: k for k, u in enumerate(labels)}
    idx_I = label_index[I]

    # Allocate compiled arrays
    label_idx = np.empty((n, n), dtype=np.int32)
    a_coef = np.empty((n, n), dtype=np.int8)
    b_coef = np.empty((n, n), dtype=np.int8)

    # Second pass: fill upper triangle, mirror using Hermitian structure
    for i in range(n):
        wi = basis[i]
        for j in range(i, n):
            wj = basis[j]
            p, u = multiply_words(wi, wj)
            
            re = _PHASE_RE[p]
            im = _PHASE_IM[p]
            
            k = label_index[u]

            label_idx[i, j] = k
            a_coef[i, j] = re
            b_coef[i, j] = im

            # Hermitian completion: M_ji = conj(M_ij)
            label_idx[j, i] = k
            a_coef[j, i] = re
            b_coef[j, i] = -im

    return MomentMatrixRep(
        basis=basis,
        labels=labels,
        label_index=label_index,
        label_idx=label_idx,
        a_coef=a_coef,
        b_coef=b_coef,
        idx_I=idx_I,
    )


def build_moment_matrix_real_embedding(rep: MomentMatrixRep, y: cp.Variable
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

    A = cp.reshape(A_vec, (n, n)) 
    B = cp.reshape(B_vec, (n, n))

    K = cp.bmat([[A, -B],
                 [B,  A]])
    return A, B, K


def compile_operator_linear_form(rep: MomentMatrixRep, op: Operator, *, tol: float = 1e-12
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


def build_sdp_from_rep(rep: MomentMatrixRep,
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

    A, B, K = build_moment_matrix_real_embedding(rep, y)

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

    return MomentSDP(
        rep=rep,
        y=y,
        M_real=A,
        M_imag=B,
        PSD_block=K,
        constraints=constraints,
        objective=obj_expr,
        problem=problem,
    )














