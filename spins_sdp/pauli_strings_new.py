from __future__ import annotations
from dataclasses import dataclass
from typing import Final, Tuple

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



# Single-qubit masks:
I = PauliWord(0b0, 0b0)
X = PauliWord(0b1, 0b0)
Z = PauliWord(0b0, 0b1)
Y = PauliWord(0b1, 0b1)

def show(a: PauliWord, b: PauliWord):
    p, c = multiply_words(a, b)
    return _I_POW[p], c

print(show(X, Y))  # should be ( +i, Z )
print(show(Y, X))  # should be ( -i, Z )
print(show(Z, X))  # should be ( +i, Y )
print(show(X, Z))  # should be ( -i, Y )