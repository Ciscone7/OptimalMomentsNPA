from __future__ import annotations
from dataclasses import dataclass
from typing import Final, Iterable, Dict, List, Tuple

PhaseExp = int  # always interpreted mod 4


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


# i^p for p in {0,1,2,3}
_I_POW: Final[Tuple[complex, complex, complex, complex]] = (1+0j, 1j, -1+0j, -1j)


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






