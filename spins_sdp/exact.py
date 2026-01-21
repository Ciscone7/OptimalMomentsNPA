import numpy as np
import qutip as qt
from typing import List, Tuple, Union, Literal

BoundaryType = Literal["open", "periodic"]
AxisType = Literal["x", "y", "z"]

# Type aliases for QuTiP objects
Qobj = qt.Qobj

def _single_site_op(N: int, i: int, op: Qobj) -> Qobj:
    """Helper to create an operator acting on site i."""
    op_list = [qt.qeye(2)] * N
    op_list[i] = op
    return qt.tensor(op_list)

def ising_hamiltonian(N: int, J: float = 1.0, h: float = 0.0, k: float = 0.0, boundary: BoundaryType = 'periodic') -> qt.Qobj:
    """
    Construct the Hamiltonian for the transverse field Ising model:
    H = -J sum Z_i Z_{i+1} - h sum X_i - k sum Z_i
    """
    # Initialize Hamiltonian as a zero operator
    H = qt.qzero([2] * N)

    sz = qt.sigmaz()
    sx = qt.sigmax()

    # Define Pauli operators for each site
    for i in range(N):
        # Nearest-neighbor interaction: sigma_z * sigma_z
        if i < N - 1 or boundary == 'periodic':
            j = (i + 1) % N
            if N > 1:
                # Optimization: Construct Z_i Z_j directly via tensor product
                op_list = [qt.qeye(2)] * N
                op_list[i] = sz
                op_list[j] = sz
                H += -J * qt.tensor(op_list)
            else:
                # Edge case N=1: Z_0 * Z_0 = I
                H += -J * qt.tensor([qt.qeye(2)] * N)
        
        # Transverse field: sigma_x
        H += -h * _single_site_op(N, i, sx)

        # Parallel field: sigma_z
        H += -k * _single_site_op(N, i, sz)

    return H

def heisenberg_hamiltonian(N: int, boundary: BoundaryType = 'periodic') -> qt.Qobj:
    """
    Construct the Hamiltonian for the Heisenberg chain (case B from paper):
    H = (1/4) sum_i sum_a∈{x,y,z} σ_i^a σ_{i+1}^a
    """
    # Initialize Hamiltonian as a zero operator
    H = qt.qzero([2] * N)

    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    # Sum over all sites
    for i in range(N):
        if i < N - 1 or boundary == 'periodic':
            j = (i + 1) % N
            if N > 1:
                # X_i X_{i+1}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sx
                op_list[j] = sx
                H += 0.25 * qt.tensor(op_list)
                
                # Y_i Y_{i+1}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sy
                op_list[j] = sy
                H += 0.25 * qt.tensor(op_list)
                
                # Z_i Z_{i+1}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sz
                op_list[j] = sz
                H += 0.25 * qt.tensor(op_list)
            else:
                # Edge case N=1
                H += 0.75 * qt.tensor([qt.qeye(2)] * N)

    return H

def heisenberg_j2_hamiltonian(N: int, J2: float, boundary: BoundaryType = 'periodic') -> qt.Qobj:
    """
    Construct the Hamiltonian for the Heisenberg chain with second-neighbor couplings (case C):
    H = (1/4) sum_i sum_a∈{x,y,z} [σ_i^a σ_{i+1}^a + J2 σ_i^a σ_{i+2}^a]
    
    Args:
        N: Number of spins
        J2: Coupling strength for second-neighbor terms
        boundary: 'periodic' or 'open'
        
    Returns:
        QuTiP Hamiltonian operator
    """
    # Initialize Hamiltonian as a zero operator
    H = qt.qzero([2] * N)

    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    # First-neighbor terms: (1/4) sum_i sum_a σ_i^a σ_{i+1}^a
    for i in range(N):
        if i < N - 1 or boundary == 'periodic':
            j = (i + 1) % N
            if N > 1:
                # X_i X_{i+1}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sx
                op_list[j] = sx
                H += 0.25 * qt.tensor(op_list)
                
                # Y_i Y_{i+1}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sy
                op_list[j] = sy
                H += 0.25 * qt.tensor(op_list)
                
                # Z_i Z_{i+1}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sz
                op_list[j] = sz
                H += 0.25 * qt.tensor(op_list)
    
    # Second-neighbor terms: (J2/4) sum_i sum_a σ_i^a σ_{i+2}^a
    for i in range(N):
        if i < N - 2 or boundary == 'periodic':
            j = (i + 2) % N
            if N > 2 or (N == 2 and boundary == 'periodic'):
                # X_i X_{i+2}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sx
                op_list[j] = sx
                H += (J2 * 0.25) * qt.tensor(op_list)
                
                # Y_i Y_{i+2}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sy
                op_list[j] = sy
                H += (J2 * 0.25) * qt.tensor(op_list)
                
                # Z_i Z_{i+2}
                op_list = [qt.qeye(2)] * N
                op_list[i] = sz
                op_list[j] = sz
                H += (J2 * 0.25) * qt.tensor(op_list)

    return H

def magnetization_qutip(N: int, axis: AxisType = 'z', average: bool = True) -> qt.Qobj:
    """
    Construct the total or average magnetization operator along a given axis.
    """
    if axis not in ('x', 'y', 'z'):
        raise ValueError("axis must be one of {'x', 'y', 'z'}")

    # Choose the single-site Pauli operator
    if axis == 'x':
        pauli = qt.sigmax()
    elif axis == 'y':
        pauli = qt.sigmay()
    else:  # axis == 'z'
        pauli = qt.sigmaz()

    # Build the operator sum_i sigma_i^axis
    M = 0
    for i in range(N):
        M += _single_site_op(N, i, pauli)

    if average:
        M = M / N

    return M

def gibbs_state_from_spectrum(energies: Union[List[float], np.ndarray], eigenstates: List[Qobj], beta: float) -> Qobj:
    """
    Construct the Gibbs state rho = exp(-beta H) / Z from the spectrum.

    Parameters
    ----------
    energies : array-like of float
        Eigenvalues E_n of the Hamiltonian.
    eigenstates : list of qt.Qobj
        Corresponding eigenstates |n>, typically kets, such that H|n> = E_n|n>.
    beta : float
        Inverse temperature beta = 1/(k_B T).
        (Assumes k_B = 1 in chosen units.)

    Returns
    -------
    rho : qt.Qobj
        Density matrix of the Gibbs state.
    """
    energies = np.asarray(energies, dtype=float)
    if len(energies) != len(eigenstates):
        raise ValueError("energies and eigenstates must have the same length")

    
    weights = np.exp(-beta * energies)
    Z = np.sum(weights)
    if Z == 0:
        raise ValueError("Partition function Z is zero; check beta and energies.")

    probabilities = weights / Z

    # Build rho = sum_n p_n |n><n|
    rho = sum(p * qt.ket2dm(psi) for p, psi in zip(probabilities, eigenstates))
    return rho

def exact_ground_state_eigenpair(H: qt.Qobj) -> Tuple[float, qt.Qobj]:
    """Compute the exact ground state energy of a Hamiltonian by diagonalization.

    Args:
        H (np.ndarray): Hamiltonian matrix.

    Returns:
        float: Ground state energy.
    """
    evals, evecs = H.eigenstates(eigvals=1)  # just the lowest one
    return (evals[0], evecs[0])