import re
from functools import lru_cache
from typing import List, Tuple, Dict, Set, Optional, Union

# Type aliases
PauliString = str
Coefficient = complex
PauliTerm = Tuple[Coefficient, PauliString]
OperatorDict = Dict[PauliString, Coefficient]


def npa_level(N: int = 1, NPA_level: int = 1, collapse_rule: bool = False) -> List[PauliString]:
    """
    Generate the set of operator strings (monomials) for the NPA hierarchy.
    """
    
    def _npa1(N: int) -> List[str]:
        out = [""]
        for i in range(N):
            for p in ("x", "y", "z"):
                out.append(f"{p}{i}")
        return out

    if NPA_level < 1:
        raise ValueError("NPA_level must be an integer >= 1")
    elif NPA_level == 1:
        result = _npa1(N)
        result[0] = "I"
        return result
    else:
        x = _npa1(N)

        # Keep token order, ignore the empty string for generation
        tokens = [t for t in x if t]

        # sequences_by_length[L] will store all sequences (as tuples of tokens) of length L
        sequences_by_length = {0: [()]}  # length 0: just the empty sequence

        for length in range(1, NPA_level + 1):
            prev_seqs = sequences_by_length[length - 1]
            curr_seqs = []
            for seq in prev_seqs:
                last = seq[-1] if seq else None
                for tok in tokens:
                    # avoid consecutive identical tokens: this enforces the "collapse" (Xi Xi = I rule)
                    if tok != last:
                        curr_seqs.append(seq + (tok,))
            sequences_by_length[length] = curr_seqs

        # Build the final list of strings, ordered by length: 0, 1, 2, ..., k
        result = [""]
        for length in range(1, NPA_level + 1):
            for seq in sequences_by_length[length]:
                result.append("".join(seq))

        result[0] = "I"
        if not collapse_rule:
            return result
        else:
            unique_labels = []
            seen = set()
            for s in result:
                # ignore coefficient, just take the canonical string
                _, word = pauli_normal_form(s)
                if word not in seen:
                    seen.add(word)
                    unique_labels.append(word)
            return unique_labels


def pauli_normal_form(expr: str) -> PauliTerm: # TODO: check the case of i>=10 for multi-digit
    """
    Take a Pauli string expression built from atomic symbols:
      - 'I'
      - 'xi', 'yi', 'zi' where i is any integer >= 0
    and return its normal form as (coeff, string) where:
      - coeff ∈ {1, -1, 1j, -1j}
      - string is the reordered Pauli string ('x1y2z3' or 'I' if identity)
    """

    expr = expr.replace(" ", "")
    token_pattern = re.compile(r"I|[xyz]\d+")
    tokens = token_pattern.findall(expr)

    if "".join(tokens) != expr:
        raise ValueError(f"Cannot fully parse expression: {expr}")

    # --- single-site multiplication rules ---

    def mul_single_site(a: str, b: str) -> Tuple[str, complex]:
        """Multiply single-site Pauli operators a*b (a,b ∈ {'i','x','y','z'})"""
        if a == "i":
            return b, 1
        if b == "i":
            return a, 1
        if a == b:
            return "i", 1

        # x y = i z,  y z = i x,  z x = i y
        # reversed gives -i times same
        if a == "x" and b == "y":
            return "z", 1j
        if a == "y" and b == "z":
            return "x", 1j
        if a == "z" and b == "x":
            return "y", 1j

        if a == "y" and b == "x":
            return "z", -1j
        if a == "z" and b == "y":
            return "x", -1j
        if a == "x" and b == "z":
            return "y", -1j

        raise RuntimeError(f"Unexpected Pauli pair: {a}, {b}")

    # --- accumulate per-site operators and global phase ---

    site_ops: Dict[int, str] = {}
    phase = 1 + 0j

    for tok in tokens:
        if tok == "I":
            continue
        axis = tok[0]          # 'x', 'y', or 'z'
        idx = int(tok[1:])     # site index

        cur = site_ops.get(idx, "i")
        new_op, local_phase = mul_single_site(cur, axis)
        site_ops[idx] = new_op
        phase *= local_phase

    # --- clean and order by site index ---

    cleaned = {i: op for i, op in site_ops.items() if op != "i"}

    if not cleaned:
        return phase, "I"

    out = "".join(op + str(idx) for idx, op in sorted(cleaned.items()))
    return phase, out

def dagger(item: PauliTerm) -> PauliTerm:
    """Return the conjugate transpose of a Pauli term (coeff, word)."""
    c, w = item
    return (c.conjugate(), w)


def multiply_moments(a: PauliTerm, b: PauliTerm) -> PauliTerm:
    """Multiply two Pauli terms."""
    c_a, w_a = a
    c_b, w_b = b
    c_ab, w_ab = pauli_normal_form(w_a + w_b)
    return (c_a * c_b * c_ab, w_ab)


# --- Operator Dictionary Utilities ---

def op_from_word(label: PauliString) -> OperatorDict:
    """
    Represent a single Pauli word 'label' as an operator dict {word: coeff},
    in canonical form.
    """
    phase, word = pauli_normal_form(label)
    return {word: phase}


def op_add(A: OperatorDict, B: OperatorDict, factor: complex = 1.0) -> OperatorDict:
    """
    Return a new operator representing A + factor * B.
    """
    C = dict(A)  # copy A so we don't mutate it
    for w, c in B.items():
        C[w] = C.get(w, 0) + factor * c
    return C


def op_scalar_mul(op: OperatorDict, scalar: complex) -> OperatorDict:
    """Return scalar * op."""
    return {w: scalar * c for w, c in op.items()}


def op_multiply(A: OperatorDict, B: OperatorDict) -> OperatorDict:
    """
    Operator product A * B.
    """
    result = {}
    for w1, c1 in A.items():
        for w2, c2 in B.items():
            phase, w = pauli_normal_form(w1 + w2)
            result[w] = result.get(w, 0) + c1 * c2 * phase
    return result
