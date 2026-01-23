"""Pytest fixtures and configuration for spiral-time memory tests."""

import pytest
import numpy as np
from typing import Callable, Tuple
import scipy.linalg as la


@pytest.fixture
def hilbert_dim():
    """Default Hilbert space dimension for tests."""
    return 2


@pytest.fixture
def extended_dim():
    """Extended Hilbert space dimension (system ⊗ memory)."""
    return 4


@pytest.fixture
def random_state(hilbert_dim):
    """Generate a random pure quantum state."""
    psi = np.random.randn(hilbert_dim) + 1j * np.random.randn(hilbert_dim)
    psi /= np.linalg.norm(psi)
    return psi


@pytest.fixture
def random_density_matrix(hilbert_dim):
    """Generate a random density matrix."""
    rho = np.random.randn(hilbert_dim, hilbert_dim) + \
          1j * np.random.randn(hilbert_dim, hilbert_dim)
    rho = rho @ rho.conj().T
    rho /= np.trace(rho)
    return rho


@pytest.fixture
def time_grid():
    """Standard time grid for dynamics tests."""
    return np.linspace(0, 10, 100)


@pytest.fixture
def memory_kernel():
    """Exponential memory kernel K(t-τ)."""
    def kernel(t, tau, gamma=0.1):
        return gamma * np.exp(-gamma * (t - tau)) if t >= tau else 0.0
    return kernel


@pytest.fixture
def spiral_time_state():
    """Generate a triadic spiral-time state Ψ(t) = (t, φ(t), χ(t))."""
    def state(t):
        phi_t = 0.1 * np.sin(2 * np.pi * t)
        chi_t = 0.05 * np.exp(-0.1 * t)
        return t, phi_t, chi_t
    return state


@pytest.fixture
def cp_map():
    """Generate a simple CP map (amplitude damping)."""
    def map_fn(rho, gamma=0.1):
        dim = rho.shape[0]
        E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        result = E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
        return result
    return map_fn


@pytest.fixture
def tolerance():
    """Numerical tolerance for tests."""
    return 1e-10


@pytest.fixture
def pauli_matrices():
    """Standard Pauli matrices."""
    return {
        'I': np.eye(2),
        'X': np.array([[0, 1], [1, 0]]),
        'Y': np.array([[0, -1j], [1j, 0]]),
        'Z': np.array([[1, 0], [0, -1]])
    }


@pytest.fixture
def bell_state():
    """Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2."""
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1/np.sqrt(2)  # |00⟩
    psi[3] = 1/np.sqrt(2)  # |11⟩
    return psi
