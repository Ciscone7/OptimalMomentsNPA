import numpy as np
import qutip as qt
from typing import List, Tuple, Union, Literal

BoundaryType = Literal["open", "periodic"]
AxisType = Literal["x", "y", "z"]

# Type aliases for QuTiP objects
Qobj = qt.Qobj

def ising_hamiltonian(N: int, J: float = 1.0, h: float = 0.0, k: float = 0.0, boundary: BoundaryType = 'periodic') -> qt.Qobj:
    """
    Construct the Hamiltonian for the transverse field Ising model:
    H = -J sum Z_i Z_{i+1} - h sum X_i - k sum Z_i
    """
    def sigmax_site(N, i):
        op_list = [qt.qeye(2)] * N
        op_list[i] = qt.sigmax()
        return qt.tensor(op_list)

    def sigmaz_site(N, i):
        op_list = [qt.qeye(2)] * N
        op_list[i] = qt.sigmaz()
        return qt.tensor(op_list)

    # Initialize Hamiltonian as a zero operator
    H = qt.qzero([2] * N)

    # Define Pauli operators for each site
    for i in range(N):
        # Nearest-neighbor interaction: sigma_z * sigma_z
        if i < N - 1 or boundary == 'periodic':
            H += -J * sigmaz_site(N, i) * sigmaz_site(N, (i + 1) % N)
        
        # Transverse field: sigma_x
        H += -h * sigmax_site(N, i)

        # Parallel field: sigma_z
        H += -k * sigmaz_site(N, i)

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
        ops = [qt.qeye(2)] * N
        ops[i] = pauli
        M += qt.tensor(ops)

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
