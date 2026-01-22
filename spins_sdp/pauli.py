from __future__ import annotations
from dataclasses import dataclass
from typing import Final, Dict, List, Literal, Sequence, Tuple

import numpy as np


PhaseExp = int  # always interpreted mod 4

Axis = Literal["x", "y", "z"]

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
class PauliTerm:
    coeff: complex
    word: PauliWord

@dataclass(frozen=True, slots=True)
class PauliMomentMatrixRep:
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

Operator = Dict[PauliWord, complex]


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


def multiply_terms(t1: PauliTerm, t2: PauliTerm) -> PauliTerm:
    """
    Multiply two Pauli terms (coeff * word), returning a single term in canonical form:
      (c1*w1) * (c2*w2) = (c1*c2*i^p) * w3
    """
    p, w3 = multiply_words(t1.word, t2.word)
    return PauliTerm(t1.coeff * t2.coeff * _I_POW[p], w3)


def reduce_monomial(factors: Sequence[PauliWord]) -> PauliWord:
    """
    Multiply a list of factors and return only the *canonical word label*, discarding the global phase.
    """
    w = PauliWord(0, 0)  # identity
    for f in factors:
        _, w = multiply_words(w, f)
    return w


def local_pauli(site: int, axis: Axis) -> PauliWord:
    """
    Return the PauliWord representing σ^axis_site on a single site (0-based indexing).
    Encoding:
      X: (x=1,z=0), Z: (x=0,z=1), Y: (x=1,z=1).
    """
    bit = 1 << site
    if axis == "x":
        return PauliWord(bit, 0)
    if axis == "z":
        return PauliWord(0, bit)
    # axis == "y"
    return PauliWord(bit, bit)



def compile_moment_matrix_rep(basis: List[PauliWord]) -> PauliMomentMatrixRep:
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

    return PauliMomentMatrixRep(
        basis=basis,
        labels=labels,
        label_index=label_index,
        label_idx=label_idx,
        a_coef=a_coef,
        b_coef=b_coef,
        idx_I=idx_I,
    )






