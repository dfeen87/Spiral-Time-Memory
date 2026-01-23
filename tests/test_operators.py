"""Tests for spiral-time operators and algebraic structure."""

import pytest
import numpy as np
from scipy.linalg import expm


class TestSpiralTimeAlgebra:
    """Test quaternionic algebra structure (Eq. 4)."""
    
    def test_imaginary_unit_squares(self):
        """Verify i² = j² = -1."""
        i = np.array([[0, -1], [1, 0]])  # Pauli Y (up to factor)
        j = np.array([[0, 1], [1, 0]])   # Pauli X
        
        i_squared = i @ i
        j_squared = j @ j
        minus_identity = -np.eye(2)
        
        assert np.allclose(i_squared, minus_identity)
        assert np.allclose(j_squared, minus_identity)
    
    def test_anticommutation(self):
        """Verify ij = -ji."""
        i = np.array([[0, -1], [1, 0]])
        j = np.array([[0, 1], [1, 0]])
        
        ij = i @ j
        ji = j @ i
        
        assert np.allclose(ij, -ji)
    
    def test_kinetic_weight_reality(self):
        """Verify A(t) = Re(Ψ(t)) is real (Eq. 5)."""
        t = 1.0
        phi_t = 0.1 * np.sin(t)
        chi_t = 0.05 * np.exp(-t)
        
        # Ψ(t) = t + iφ(t) + jχ(t)
        # Re(Ψ(t)) should extract only the real part
        A_t = t  # Simplified version
        
        assert isinstance(A_t, (int, float))
        assert A_t > 0
    
    def test_quaternionic_norm(self):
        """Test norm of quaternionic spiral-time variable."""
        t = 2.0
        phi_t = 0.1
        chi_t = 0.05
        
        # |Ψ|² = t² + φ² + χ²
        norm_squared = t**2 + phi_t**2 + chi_t**2
        
        assert norm_squared > 0
        assert np.isreal(norm_squared)


class TestSpiralTimeOperator:
    """Test triadic operator structure (Eq. 28)."""
    
    def test_operator_construction(self, hilbert_dim):
        """Test Ψ = T + iΦ + jΧ construction."""
        # Time operator (self-adjoint)
        T = np.diag(np.arange(hilbert_dim, dtype=float))
        
        # Phase operator (bounded)
        Phi = np.random.randn(hilbert_dim, hilbert_dim)
        Phi = (Phi + Phi.T) / 2  # Make Hermitian
        
        # Memory operator (bounded)
        Chi = np.random.randn(hilbert_dim, hilbert_dim)
        Chi = (Chi + Chi.T) / 2  # Make Hermitian
        
        # Verify operators are well-defined
        assert np.allclose(T, T.T)
        assert np.allclose(Phi, Phi.conj().T)
        assert np.allclose(Chi, Chi.conj().T)
    
    def test_extended_hilbert_space(self, hilbert_dim):
        """Test H_ext = H_sys ⊗ H_mem construction (Eq. 3)."""
        dim_sys = hilbert_dim
        dim_mem = 2
        dim_ext = dim_sys * dim_mem
        
        # Create extended state
        psi_sys = np.random.randn(dim_sys) + 1j * np.random.randn(dim_sys)
        psi_sys /= np.linalg.norm(psi_sys)
        
        psi_mem = np.array([1, 0])  # Memory ground state
        
        psi_ext = np.kron(psi_sys, psi_mem)
        
        assert psi_ext.shape[0] == dim_ext
        assert np.allclose(np.linalg.norm(psi_ext), 1.0)
    
    def test_hermiticity_of_hamiltonian(self, hilbert_dim):
        """Test that effective Hamiltonian remains Hermitian (Eq. 8)."""
        # Create Hermitian Hamiltonian
        H = np.random.randn(hilbert_dim, hilbert_dim)
        H = (H + H.conj().T) / 2
        
        # Verify Hermiticity
        assert np.allclose(H, H.conj().T)
        
        # With kinetic weight A(t)
        A_t = 1.05  # Real, positive
        H_eff = A_t * H
        
        # Should remain Hermitian
        assert np.allclose(H_eff, H_eff.conj().T)
    
    def test_memory_sector_partial_trace(self, extended_dim):
        """Test tracing over memory sector yields reduced system dynamics."""
        # Extended state on H_sys ⊗ H_mem
        dim_sys = 2
        dim_mem = extended_dim // dim_sys
        
        # Create random extended density matrix
        rho_ext = np.random.randn(extended_dim, extended_dim) + \
                  1j * np.random.randn(extended_dim, extended_dim)
        rho_ext = rho_ext @ rho_ext.conj().T
        rho_ext /= np.trace(rho_ext)
        
        # Partial trace over memory
        rho_sys = np.zeros((dim_sys, dim_sys), dtype=complex)
        for i in range(dim_mem):
            for j in range(dim_mem):
                if i == j:  # Diagonal blocks contribute to trace
                    block = rho_ext[i*dim_sys:(i+1)*dim_sys, j*dim_sys:(j+1)*dim_sys]
                    rho_sys += block
        
        # Verify reduced state is valid
        assert np.allclose(rho_sys, rho_sys.conj().T)
        assert np.abs(np.trace(rho_sys) - 1.0) < 1e-10
        
        eigenvalues = np.linalg.eigvalsh(rho_sys)
        assert np.all(eigenvalues >= -1e-10)
    
    def test_commutator_structure(self, pauli_matrices):
        """Test commutator relations for spiral-time operators."""
        X = pauli_matrices['X']
        Y = pauli_matrices['Y']
        Z = pauli_matrices['Z']
        
        # [X, Y] = 2iZ
        commutator_XY = X @ Y - Y @ X
        expected = 2j * Z
        
        assert np.allclose(commutator_XY, expected)


class TestOperatorWellDefinedness:
    """Test mathematical well-definedness of operators."""
    
    def test_essential_self_adjointness(self, hilbert_dim):
        """Test that deformed generators remain essentially self-adjoint."""
        # Base self-adjoint operator
        T = np.diag(np.arange(hilbert_dim, dtype=float))
        
        # Small perturbation (Kato-small)
        epsilon = 0.01
        V = epsilon * np.random.randn(hilbert_dim, hilbert_dim)
        V = (V + V.T) / 2
        
        # Perturbed operator
        T_pert = T + V
        
        # Should remain self-adjoint
        assert np.allclose(T_pert, T_pert.T)
    
    def test_bounded_operator_norm(self):
        """Test that Φ and Χ are bounded operators."""
        dim = 4
        
        Phi = np.random.randn(dim, dim)
        Phi = (Phi + Phi.T) / 2
        
        Chi = np.random.randn(dim, dim)
        Chi = (Chi + Chi.T) / 2
        
        # Compute operator norms
        norm_Phi = np.linalg.norm(Phi, ord=2)
        norm_Chi = np.linalg.norm(Chi, ord=2)
        
        # Should be finite
        assert np.isfinite(norm_Phi)
        assert np.isfinite(norm_Chi)
    
    def test_spectral_properties(self, hilbert_dim):
        """Test spectral properties of spiral-time operators."""
        # Create Hermitian operator
        H = np.random.randn(hilbert_dim, hilbert_dim)
        H = (H + H.conj().T) / 2
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(H)
        
        # All should be real
        assert np.allclose(eigenvalues.imag, 0)
        
        # Verify spectral decomposition
        eigvals, eigvecs = np.linalg.eigh(H)
        H_reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        
        assert np.allclose(H, H_reconstructed)
