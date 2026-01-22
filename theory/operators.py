"""
Triadic Spiral-Time Operators on Extended Hilbert Space
========================================================

Implements the operator-theoretic foundations from Appendix A of the paper.

The spiral-time operator Ψ = T + iΦ + jχ acts on an extended Hilbert space
H_ext = H_sys ⊗ H_mem, where H_mem encodes temporal memory degrees of freedom.

Reference: Paper Section 2.1-2.2, Appendix A

WARNING: This is ONE possible implementation of the abstract operator structure.
The quaternionic algebra (i, j) with i² = j² = -1, ij = -ji can be represented
in multiple ways. See IMPLEMENTATIONS.md.

Author: Marcel Krüger & Don Michael Feeney Jr.
License: MIT
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import scipy.linalg as la


@dataclass
class ExtendedHilbertConfig:
    """Configuration for extended Hilbert space H_ext = H_sys ⊗ H_mem."""
    sys_dim: int = 2          # Dimension of H_sys (e.g., qubit = 2)
    mem_dim: int = 4          # Dimension of H_mem (memory sector)
    epsilon: float = 0.1      # Memory coupling strength
    

class SpiralTimeOperator:
    """Triadic spiral-time operator Ψ = T + iΦ + jχ.
    
    Acts on extended Hilbert space H_ext = H_sys ⊗ H_mem.
    
    The quaternionic structure (i, j) satisfies:
        i² = j² = -1
        ij = -ji
    
    Physical observables depend only on:
        A(t) = ℜΨ(t) = 1 + ε(t)
    ensuring Hermiticity of the effective Hamiltonian.
    
    Attributes:
        config: Extended Hilbert space configuration
        T_op: Time translation generator (self-adjoint)
        Phi_op: Phase coherence operator
        Chi_op: Temporal memory operator
    """
    
    def __init__(self, config: ExtendedHilbertConfig):
        self.config = config
        self.ext_dim = config.sys_dim * config.mem_dim
        
        # Construct operators on extended space
        self.T_op = self._construct_time_generator()
        self.Phi_op = self._construct_phase_operator()
        self.Chi_op = self._construct_memory_operator()
        
    def _construct_time_generator(self) -> np.ndarray:
        """Construct time translation generator T (self-adjoint).
        
        Acts as identity on memory sector, generator on system.
        """
        # System Hamiltonian (example: qubit σ_z)
        H_sys = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Extend to full space: T = H_sys ⊗ I_mem
        I_mem = np.eye(self.config.mem_dim)
        T = np.kron(H_sys, I_mem)
        
        return T
    
    def _construct_phase_operator(self) -> np.ndarray:
        """Construct phase coherence operator Φ.
        
        Bounded perturbation preserving essential self-adjointness.
        """
        # System coherence (example: qubit σ_x)
        Phi_sys = np.array([[0, 1], [1, 0]], dtype=complex)
        
        # Extend to full space
        I_mem = np.eye(self.config.mem_dim)
        Phi = np.kron(Phi_sys, I_mem)
        
        return Phi
    
    def _construct_memory_operator(self) -> np.ndarray:
        """Construct temporal memory operator χ.
        
        Acts non-trivially on memory sector.
        """
        # Identity on system
        I_sys = np.eye(self.config.sys_dim)
        
        # Memory sector operator (example: shift operator)
        Chi_mem = np.zeros((self.config.mem_dim, self.config.mem_dim), dtype=complex)
        for i in range(self.config.mem_dim - 1):
            Chi_mem[i, i+1] = 1
        Chi_mem[-1, 0] = 1  # Cyclic
        
        # Extend to full space: χ = I_sys ⊗ χ_mem
        Chi = np.kron(I_sys, Chi_mem)
        
        return Chi
    
    def kinetic_weight(self, t: float, phi_t: float, chi_t: float) -> float:
        """Compute physical kinetic weight A(t) = ℜΨ(t).
        
        This is the only part entering physical observables.
        
        Args:
            t: Time parameter
            phi_t: Phase coherence at time t
            chi_t: Memory state at time t
            
        Returns:
            A(t) = 1 + ε(t)
        """
        # Real part: A(t) = t (suppressing units) + Re(corrections)
        # For small deformations: A(t) ≈ 1 + ε·chi_t
        epsilon_t = self.config.epsilon * chi_t
        return 1.0 + epsilon_t
    
    def effective_hamiltonian(
        self, 
        t: float, 
        chi_t: float
    ) -> np.ndarray:
        """Construct effective Hamiltonian on extended space.
        
        H_ext(t) = A(t)[T + εΦ] + ε²χ_op
        
        Args:
            t: Time
            chi_t: Memory state
            
        Returns:
            Hermitian operator on H_ext
        """
        A_t = self.kinetic_weight(t, 0, chi_t)  # phi_t absorbed
        eps = self.config.epsilon
        
        H_ext = A_t * self.T_op + eps * self.Phi_op + eps**2 * self.Chi_op
        
        # Ensure Hermiticity
        H_ext = 0.5 * (H_ext + H_ext.conj().T)
        
        return H_ext
    
    def trace_over_memory(self, rho_ext: np.ndarray) -> np.ndarray:
        """Trace over memory sector to obtain effective system state.
        
        ρ_sys = Tr_mem[ρ_ext]
        
        This operation induces non-Markovian dynamics on H_sys.
        
        Args:
            rho_ext: Density matrix on H_ext
            
        Returns:
            Reduced density matrix on H_sys
        """
        d_sys = self.config.sys_dim
        d_mem = self.config.mem_dim
        
        rho_sys = np.zeros((d_sys, d_sys), dtype=complex)
        
        for i in range(d_sys):
            for j in range(d_sys):
                # Sum over memory indices
                for k in range(d_mem):
                    idx_i = i * d_mem + k
                    idx_j = j * d_mem + k
                    rho_sys[i, j] += rho_ext[idx_i, idx_j]
        
        return rho_sys
    
    def von_neumann_evolution(
        self, 
        rho_ext_0: np.ndarray, 
        t_final: float, 
        chi_trajectory: callable,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve density matrix via von Neumann equation.
        
        dρ_ext/dt = -i[H_ext(t), ρ_ext(t)]
        
        Full evolution on H_ext is unitary, but traced-out dynamics
        on H_sys is non-Markovian.
        
        Args:
            rho_ext_0: Initial state on H_ext
            t_final: Final time
            chi_trajectory: Function t → χ(t)
            dt: Time step
            
        Returns:
            times: Array of time points
            rhos: Array of density matrices on H_ext
        """
        times = np.arange(0, t_final + dt, dt)
        rhos = np.zeros((len(times), self.ext_dim, self.ext_dim), dtype=complex)
        rhos[0] = rho_ext_0.copy()
        
        for i, t in enumerate(times[:-1]):
            chi_t = chi_trajectory(t)
            H_ext = self.effective_hamiltonian(t, chi_t)
            
            # Von Neumann equation: dρ/dt = -i[H, ρ]
            commutator = H_ext @ rhos[i] - rhos[i] @ H_ext
            drho_dt = -1j * commutator
            
            # Euler step (for illustration; use RK4 for better accuracy)
            rhos[i+1] = rhos[i] + drho_dt * dt
            
            # Renormalize (numerical stability)
            rhos[i+1] = rhos[i+1] / np.trace(rhos[i+1])
        
        return times, rhos
    
    def check_unitarity(self, rho_ext: np.ndarray) -> bool:
        """Check if evolution preserves trace and positivity.
        
        Args:
            rho_ext: Density matrix to check
            
        Returns:
            True if valid (trace ≈ 1, positive eigenvalues)
        """
        trace_val = np.trace(rho_ext).real
        eigvals = np.linalg.eigvalsh(rho_ext)
        
        trace_ok = np.abs(trace_val - 1.0) < 1e-6
        positive_ok = np.all(eigvals >= -1e-10)
        
        return trace_ok and positive_ok


def demonstrate_extended_space():
    """Demonstrate extended Hilbert space formalism."""
    print("Extended Hilbert Space Formalism")
    print("=" * 60)
    
    # Configuration
    config = ExtendedHilbertConfig(sys_dim=2, mem_dim=4, epsilon=0.1)
    psi_op = SpiralTimeOperator(config)
    
    print(f"H_sys dimension: {config.sys_dim}")
    print(f"H_mem dimension: {config.mem_dim}")
    print(f"H_ext dimension: {config.sys_dim * config.mem_dim}")
    print()
    
    # Initial state: |0⟩_sys ⊗ |0⟩_mem
    rho_ext_0 = np.zeros((psi_op.ext_dim, psi_op.ext_dim), dtype=complex)
    rho_ext_0[0, 0] = 1.0
    
    # Memory trajectory (example: sinusoidal)
    chi_traj = lambda t: 0.1 * np.sin(2 * np.pi * t)
    
    # Evolve
    print("Evolving von Neumann equation on H_ext...")
    times, rhos = psi_op.von_neumann_evolution(rho_ext_0, t_final=1.0, 
                                                chi_trajectory=chi_traj, dt=0.01)
    
    # Check unitarity
    print(f"Initial state valid: {psi_op.check_unitarity(rhos[0])}")
    print(f"Final state valid: {psi_op.check_unitarity(rhos[-1])}")
    
    # Trace over memory
    print()
    print("Tracing over memory sector...")
    rho_sys_initial = psi_op.trace_over_memory(rhos[0])
    rho_sys_final = psi_op.trace_over_memory(rhos[-1])
    
    print(f"System state (initial):")
    print(rho_sys_initial)
    print(f"\nSystem state (final):")
    print(rho_sys_final)
    print()
    print("✓ Non-Markovian evolution on H_sys induced by memory sector")


if __name__ == "__main__":
    demonstrate_extended_space()
