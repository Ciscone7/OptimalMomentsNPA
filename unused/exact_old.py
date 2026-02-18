import numpy as np
import qutip as qt
from typing import List, Tuple, Union, Literal

from spins.models import Qobj

BoundaryType = Literal["open", "periodic"]
AxisType = Literal["x", "y", "z"]


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
