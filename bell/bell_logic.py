"""
Bell NPA hierarchy for bipartite scenarios with projective measurements.

Definitions:
  - Alice projectors: Ax|a for settings x in {0,...,m_A-1}, outcomes a in {0,...,d_A-1}
  - Bob projectors: By|b for settings y in {0,...,m_B-1}, outcomes b in {0,...,d_B-1}

Display:
  BellWord.__repr__ always uses the full form Ax|a / By|b.
  For compact display, use scenario.format_word(w) which
  drops the outcome label for binary outcomes: A0|0 → A0, B1|0 → B1.

Algebra:
  - Idempotence: (Ax|a)^2 = Ax|a, (By|b)^2 = By|b
  - Orthogonality: Ax|a · Ax|a' = 0 for a != a', same for Bob
  - Completeness: sum_a Ax|a = I, sum_b By|b = I
  - Commutation: [Ax|a, By|b] = 0 for all a,b,x,y
  - Hermitian: (Ax|a)^dag = Ax|a, (By|b)^dag = By|b

IMPORTANT: Within a single party, projectors for *different* settings do NOT commute.
  Ax|a · Ax'|a' != Ax'|a' · Ax|a in general when x != x'.
  Words must track the *ordered* sequence of projectors per party.

Storage: ordered tuples of (setting, outcome) pairs per party.
  Canonical form: Alice projectors left, Bob projectors right (by [A,B]=0),
  with within-party sequences reduced only by adjacent same-setting rules.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np


# Sentinel value meaning "this setting is not present in the word"
ABSENT: int = -1


@dataclass(frozen=True, slots=True)
class BellScenario:
    """
    Defines a Bell scenario.

    Attributes:
        m_A: number of settings for Alice
        m_B: number of settings for Bob
        d_A: number of outcomes for each Alice setting
        d_B: number of outcomes for each Bob setting
    """
    m_A: int
    m_B: int
    d_A: int
    d_B: int

    def __post_init__(self) -> None:
        if self.m_A < 0 or self.m_B < 0:
            raise ValueError("Number of settings must be non-negative.")
        if self.d_A < 1:
            raise ValueError("d_A must be >= 1.")
        if self.d_B < 1:
            raise ValueError("d_B must be >= 1.")

    @staticmethod
    def symmetric(m: int, d: int) -> BellScenario:
        """Create a symmetric scenario: m settings each for Alice and Bob, d outcomes each."""
        return BellScenario(m_A=m, m_B=m, d_A=d, d_B=d)

    @staticmethod
    def asymmetric(m_A: int, m_B: int, d_A: int, d_B: int) -> BellScenario:
        """Create a scenario with different settings/outcomes per party."""
        return BellScenario(m_A=m_A, m_B=m_B, d_A=d_A, d_B=d_B)

    def num_alice_projectors(self) -> int:
        return self.m_A * self.d_A

    def num_bob_projectors(self) -> int:
        return self.m_B * self.d_B

    def num_generators(self) -> int:
        """Number of reduced generators: (d-1) outcomes per setting + identity."""
        return self.m_A * (self.d_A - 1) + self.m_B * (self.d_B - 1) + 1

    # --- Projector constructors ---

    @lru_cache(maxsize=None)
    def alice_projector(self, x: int, a: int) -> BellWord:
        """Create an Alice projector word for setting *x*, outcome *a*."""
        if not (0 <= x < self.m_A):
            raise ValueError(f"Setting x={x} out of range [0, {self.m_A})")
        if not (0 <= a < self.d_A):
            raise ValueError(
                f"Outcome a={a} out of range [0, {self.d_A}) for setting x={x}"
            )
        return BellWord(alice_seq=((x, a),), bob_seq=_EMPTY_SEQ)

    @lru_cache(maxsize=None)
    def bob_projector(self, y: int, b: int) -> BellWord:
        """Create a Bob projector word for setting *y*, outcome *b*."""
        if not (0 <= y < self.m_B):
            raise ValueError(f"Setting y={y} out of range [0, {self.m_B})")
        if not (0 <= b < self.d_B):
            raise ValueError(
                f"Outcome b={b} out of range [0, {self.d_B}) for setting y={y}"
            )
        return BellWord(alice_seq=_EMPTY_SEQ, bob_seq=((y, b),))

    def generators(self) -> List[BellWord]:
        """Reduced generator set: (d-1) outcomes per setting, plus identity."""
        gens: List[BellWord] = [IDENTITY]
        for x in range(self.m_A):
            for a in range(self.d_A - 1):
                gens.append(self.alice_projector(x, a))
        for y in range(self.m_B):
            for b in range(self.d_B - 1):
                gens.append(self.bob_projector(y, b))
        return gens

    def npa_basis(self, k: Union[int, float]) -> BellBasis:
        """Generate the NPA basis at level *k*.

        Supports fractional level ``k=1.5`` (the *local* basis): NPA level 1
        augmented with all Alice × Bob cross-party products.
        """
        if k == 1.5:
            return self._npa_basis_local()

        k = int(k)
        if k < 0:
            raise ValueError("k must be >= 0.")

        I = IDENTITY
        gens = self.generators()

        levels: List[List[BellWord]] = [[] for _ in range(k + 1)]
        min_len: Dict[BellWord, int] = {I: 0}
        frontier: List[BellWord] = [I]
        levels[0] = [I]

        for length in range(1, k + 1):
            next_set: set[BellWord] = set()
            for w in frontier:
                for g in gens:
                    w_new = multiply_words(w, g)
                    if w_new is not None and w_new not in min_len:
                        min_len[w_new] = length
                        next_set.add(w_new)

            next_level = sorted(next_set, key=_word_sort_key)
            levels[length] = next_level
            frontier = next_level

            if not frontier:
                levels = levels[:length + 1]
                break

        all_words = list(min_len.keys())
        all_words_sorted = sorted(
            all_words,
            key=lambda w: (min_len[w], _word_sort_key(w)),
        )
        index = {w: i for i, w in enumerate(all_words_sorted)}

        return BellBasis(
            scenario=self,
            k=k,
            words=all_words_sorted,
            levels=levels,
            min_len=min_len,
            index=index,
        )

    # --- Display helpers ---

    def format_word(self, w: BellWord) -> str:
        """Human-friendly string for *w* in this scenario's context.

        For binary outcomes (d=2) the outcome label is dropped:
        ``A0|0`` → ``A0``, ``B1|0`` → ``B1``.
        For non-binary outcomes the full form is kept.
        """
        if w.is_identity():
            return "I"
        parts: list[str] = []
        drop_a = self.d_A == 2
        drop_b = self.d_B == 2
        for x, a in w.alice_seq:
            parts.append(f"A{x}" if drop_a else f"A{x}|{a}")
        for y, b in w.bob_seq:
            parts.append(f"B{y}" if drop_b else f"B{y}|{b}")
        return " ".join(parts)

    def format_basis(self, basis: BellBasis) -> List[str]:
        """Format every word in *basis* using :meth:`format_word`."""
        return [self.format_word(w) for w in basis.words]

    def format_operator(self, op: Dict[BellWord, float]) -> str:
        """Human-friendly string for a BellOperator."""
        if not op:
            return "0"
        parts: list[str] = []
        for word, coeff in sorted(op.items(), key=lambda t: str(t[0])):
            if abs(coeff) < 1e-12:
                continue
            parts.append(f"{coeff:+.1f} · {self.format_word(word)}")
        return "\n".join(parts) if parts else "0"

    def _npa_basis_local(self) -> BellBasis:
        """NPA 1 + Alice×Bob cross products ('local' / level-1.5 basis)."""
        base = self.npa_basis(1)

        alice_words = [w for w in base.words
                       if w.alice_len() == 1 and w.bob_len() == 0]
        bob_words = [w for w in base.words
                     if w.alice_len() == 0 and w.bob_len() == 1]

        cross: List[BellWord] = []
        seen = set(base.index.keys())
        for aw in alice_words:
            for bw in bob_words:
                w = multiply_words(aw, bw)
                if w is not None and w not in seen:
                    cross.append(w)
                    seen.add(w)

        cross_sorted = sorted(cross, key=_word_sort_key)

        all_words = base.words + cross_sorted
        index = {w: i for i, w in enumerate(all_words)}
        min_len = dict(base.min_len)
        for w in cross_sorted:
            min_len[w] = 2

        levels = list(base.levels)
        if cross_sorted:
            levels.append(cross_sorted)

        return BellBasis(
            scenario=self,
            k=1.5,
            words=all_words,
            levels=levels,
            min_len=min_len,
            index=index,
        )


# ---------------------------------------------------------------------------
# BellWord: ordered sequence representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BellWord:
    """
    A reduced monomial (word) in the Bell projector algebra.

    Storage: two tuples of (setting, outcome) pairs, one per party.
      alice_seq: ((x1, a1), (x2, a2), ...) -- ordered, no adjacent same-setting
      bob_seq:   ((y1, b1), (y2, b2), ...) -- ordered, no adjacent same-setting

    The full operator is:
      A{x1}|{a1} A{x2}|{a2} ... · B{y1}|{b1} B{y2}|{b2} ...

    A word is "reduced" if no two adjacent elements in either sequence share
    the same setting. (If they did, they'd reduce by idempotence or be zero
    by orthogonality.)

    The identity word has alice_seq = bob_seq = ().
    """
    alice_seq: Tuple[Tuple[int, int], ...]
    bob_seq: Tuple[Tuple[int, int], ...]

    def is_identity(self) -> bool:
        return len(self.alice_seq) == 0 and len(self.bob_seq) == 0

    def alice_len(self) -> int:
        return len(self.alice_seq)

    def bob_len(self) -> int:
        return len(self.bob_seq)

    def total_len(self) -> int:
        return len(self.alice_seq) + len(self.bob_seq)

    def dagger(self) -> BellWord:
        """
        Hermitian adjoint: reverse each party's sequence.
        (A{x1}|{a1} A{x2}|{a2} ... B{y1}|{b1} ...)^dag
        = ... B{y1}|{b1} ... A{x2}|{a2} A{x1}|{a1}
        After [A,B]=0 commutation -> reverse(Alice) * reverse(Bob)
        """
        return BellWord(
            alice_seq=self.alice_seq[::-1],
            bob_seq=self.bob_seq[::-1],
        )

    def __repr__(self) -> str:
        parts = []
        for x, a in self.alice_seq:
            parts.append(f"A{x}|{a}")
        for y, b in self.bob_seq:
            parts.append(f"B{y}|{b}")
        if not parts:
            return "I"
        return " ".join(parts)


ZeroWord = None

_EMPTY_SEQ: Tuple[Tuple[int, int], ...] = ()
IDENTITY = BellWord(alice_seq=_EMPTY_SEQ, bob_seq=_EMPTY_SEQ)


@lru_cache(maxsize=None)
def identity_word(scenario: BellScenario) -> BellWord:
    return IDENTITY





# ---------------------------------------------------------------------------
# Reduction and multiplication
# ---------------------------------------------------------------------------

def _concat_and_reduce(
    seq1: Tuple[Tuple[int, int], ...],
    seq2: Tuple[Tuple[int, int], ...],
) -> Optional[Tuple[Tuple[int, int], ...]]:
    """
    Concatenate two already-reduced sequences and reduce the junction.

    Both inputs must be individually reduced (no adjacent same-setting).
    The only reductions that can occur are at the junction: seq1's tail
    meeting seq2's head. Idempotence can cascade *into* seq1 (peeling
    back elements), but once the junction is resolved the rest of seq2
    appends directly — because seq2 is internally reduced, no further
    adjacency conflicts are possible.

    Returns:
        Reduced tuple, or None if the product is zero.
    """
    if not seq1:
        return seq2
    if not seq2:
        return seq1

    # Junction elements have different settings -> no reduction
    if seq1[-1][0] != seq2[0][0]:
        return seq1 + seq2

    # Resolve the junction: seq1[-1] vs seq2[0]
    x_last, a_last = seq1[-1]
    x_first, a_first = seq2[0]

    if a_last == a_first:
        # Idempotence: (Ax|a)^2 = Ax|a — drop one copy
        return seq1[:-1] + seq2
    else:
        # Orthogonality: Ax|a · Ax|a' = 0
        return None


def multiply_words(w1: BellWord, w2: BellWord) -> Optional[BellWord]:
    """
    Multiply two BellWords and return the reduced result.
    (A1*B1)(A2*B2) = A1*A2 * B1*B2  (by [A,B]=0)
    Within each party, concatenate and reduce.
    Returns None if the product is zero (orthogonality).
    """
    alice = _concat_and_reduce(w1.alice_seq, w2.alice_seq)
    if alice is None:
        return None

    bob = _concat_and_reduce(w1.bob_seq, w2.bob_seq)
    if bob is None:
        return None

    return BellWord(alice_seq=alice, bob_seq=bob)


# ---------------------------------------------------------------------------
# NPA Basis Generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BellBasis:
    scenario: BellScenario
    k: Union[int, float]
    words: List[BellWord]
    levels: List[List[BellWord]]
    min_len: Dict[BellWord, int]
    index: Dict[BellWord, int]


def _word_sort_key(w: BellWord) -> Tuple:
    return (w.total_len(), w.alice_seq, w.bob_seq)


# ---------------------------------------------------------------------------
# Moment Matrix Compilation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BellMomentMatrixRep:
    """
    Compiled representation of the moment matrix M_ij = <S_i^dag S_j>.

    Hermiticity (real entries): <S_i^dag S_j> = <S_j^dag S_i>
    so nf(S_i^dag S_j) and nf(S_j^dag S_i) = nf(S_i^dag S_j)^dag
    must share the same moment variable. We canonicalize by picking
    min(w, w^dag) as the representative for each {w, w^dag} pair.
    """
    scenario: BellScenario
    basis: List[BellWord]
    labels: List[BellWord]
    label_index: Dict[BellWord, int]
    label_idx: np.ndarray   # (n, n) int32; -1 means zero entry
    idx_I: int


def _canonicalize(w: BellWord) -> BellWord:
    """Pick canonical representative from {w, w^dag}: the one that sorts first."""
    wd = w.dagger()
    if wd == w:
        return w
    return min(w, wd, key=_word_sort_key)


def compile_moment_matrix_rep(basis: BellBasis) -> BellMomentMatrixRep:
    """
    Compile the moment matrix representation.
    Entry (i,j) corresponds to <S_i^dag S_j>.
    """
    n = len(basis.words)
    if n == 0:
        raise ValueError("Basis must be non-empty.")

    scenario = basis.scenario
    I = identity_word(scenario)

    # Compute upper-triangle products S_i^dag S_j and canonicalize.
    # min(w, w^dag) so both map to the same label.
    raw_upper: dict = {}
    canonical_set: set = {I}

    for i in range(n):
        si_dag = basis.words[i].dagger()
        for j in range(i, n):
            u = multiply_words(si_dag, basis.words[j])
            raw_upper[(i, j)] = u
            if u is not None:
                canonical_set.add(_canonicalize(u))

    labels_rest = sorted(
        (u for u in canonical_set if u != I),
        key=_word_sort_key,
    )
    labels = [I] + labels_rest
    label_index = {u: k for k, u in enumerate(labels)}
    idx_I = 0

    label_idx = np.full((n, n), -1, dtype=np.int32)
    for i in range(n):
        for j in range(i, n):
            u = raw_upper[(i, j)]
            if u is not None:
                k = label_index[_canonicalize(u)]
                label_idx[i, j] = k
                label_idx[j, i] = k

    return BellMomentMatrixRep(
        scenario=scenario,
        basis=basis.words,
        labels=labels,
        label_index=label_index,
        label_idx=label_idx,
        idx_I=idx_I,
    )


# ---------------------------------------------------------------------------
# Type aliases (canonical home — re-exported by bell_sdp too)
# ---------------------------------------------------------------------------

BellOperator = Dict[BellWord, float]
"""Mapping from Bell words to real coefficients."""

Sense = Literal["min", "max"]


def expand_bell_operator(
    op: BellOperator,
    scenario: BellScenario,
) -> BellOperator:
    """
    Expand a BellOperator so it only references outcomes 0..d-2 per setting.

    For each word containing a last-outcome projector (A{x}|{d-1} or B{y}|{d-1}),
    substitute via completeness: A{x}|{d-1} = I - sum_{a<d-1} A{x}|{a}.

    This must be applied to the objective operator before passing it to the SDP
    when the generator set excludes last outcomes.
    """
    expanded: BellOperator = {}

    for w, coef in op.items():
        if coef == 0:
            continue
        _expand_word(w, coef, scenario, expanded)

    return expanded


def _expand_word(
    w: BellWord,
    coef: float,
    scenario: BellScenario,
    out: BellOperator,
) -> None:
    """
    Recursively expand a single word. If it contains a last-outcome projector,
    substitute completeness and recurse. Otherwise add to out.
    """
    # Check Alice sequence for any last-outcome projector
    for i, (x, a) in enumerate(w.alice_seq):
        if a == scenario.d_A - 1:
            # A{x}|{d-1} = I - sum_{a'=0}^{d-2} A{x}|{a'}
            # Build prefix and suffix words
            prefix = BellWord(alice_seq=w.alice_seq[:i], bob_seq=_EMPTY_SEQ)
            suffix = BellWord(alice_seq=w.alice_seq[i+1:], bob_seq=w.bob_seq)

            # Identity term: prefix * suffix (skip E_{d-1|x})
            w_id = multiply_words(prefix, suffix)
            if w_id is not None:
                _expand_word(w_id, coef, scenario, out)

            # Subtracted terms: -E_{a'|x} for a' = 0..d-2
            for a_prime in range(scenario.d_A - 1):
                mid = BellWord(alice_seq=((x, a_prime),), bob_seq=_EMPTY_SEQ)
                w_sub = multiply_words(prefix, multiply_words(mid, suffix))
                if w_sub is not None:
                    _expand_word(w_sub, -coef, scenario, out)
            return

    # Check Bob sequence for any last-outcome projector
    for j, (y, b) in enumerate(w.bob_seq):
        if b == scenario.d_B - 1:
            prefix = BellWord(alice_seq=w.alice_seq, bob_seq=w.bob_seq[:j])
            suffix = BellWord(alice_seq=_EMPTY_SEQ, bob_seq=w.bob_seq[j+1:])

            w_id = multiply_words(prefix, suffix)
            if w_id is not None:
                _expand_word(w_id, coef, scenario, out)

            for b_prime in range(scenario.d_B - 1):
                mid = BellWord(alice_seq=_EMPTY_SEQ, bob_seq=((y, b_prime),))
                w_sub = multiply_words(prefix, multiply_words(mid, suffix))
                if w_sub is not None:
                    _expand_word(w_sub, -coef, scenario, out)
            return

    # No last-outcome projectors — word is already in the reduced basis
    out[w] = out.get(w, 0.0) + coef


# ---------------------------------------------------------------------------
# Convenience: Common Bell Inequalities
# ---------------------------------------------------------------------------

def chsh_operator(scenario: BellScenario) -> BellOperator:
    """Build the CHSH operator and expand to the reduced generator basis."""
    if scenario.m_A < 2 or scenario.m_B < 2:
        raise ValueError("CHSH requires at least 2 settings for each party.")
    if scenario.d_A < 2 or scenario.d_B < 2:
        raise ValueError("CHSH requires at least 2 outcomes for each party.")

    op: BellOperator = {}
    chsh_coefs = {(0, 0): +1, (0, 1): +1, (1, 0): +1, (1, 1): -1}

    for (x, y_setting), c_xy in chsh_coefs.items():
        for a in range(2):
            for b in range(2):
                sign_ab = (-1) ** (a + b)
                coef = c_xy * sign_ab
                w = multiply_words(
                    scenario.alice_projector(x, a),
                    scenario.bob_projector(y_setting, b),
                )
                if w is not None:
                    op[w] = op.get(w, 0) + coef

    return expand_bell_operator(op, scenario)


def probability_word(scenario: BellScenario, x: int, a: int, y: int, b: int) -> Optional[BellWord]:
    return multiply_words(
        scenario.alice_projector(x, a),
        scenario.bob_projector(y, b),
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def basis_summary(basis: BellBasis) -> str:
    lines = [
        f"Bell NPA Basis for scenario ({basis.scenario.m_A}, {basis.scenario.m_B}) "
        f"with d_A={basis.scenario.d_A}, d_B={basis.scenario.d_B}",
        f"Level k = {basis.k}",
        f"Total words: {len(basis.words)}",
    ]
    for ell, level in enumerate(basis.levels):
        lines.append(f"  Level {ell}: {len(level)} words")
    return "\n".join(lines)


def rep_summary(rep: BellMomentMatrixRep) -> str:
    n = len(rep.basis)
    m = len(rep.labels)
    num_zero = np.sum(rep.label_idx == -1)
    lines = [
        f"Moment matrix: {n} x {n}",
        f"Number of moment labels: {m}",
        f"Zero entries: {num_zero} / {n*n}",
    ]
    return "\n".join(lines)
