"""
Protocol B: Process Tensor Reconstruction and CP-Divisibility Test
===================================================================

Reconstructs multi-time quantum processes and tests whether they can be
described by CP-divisible (Markovian) dynamics or require non-factorizing
temporal correlations.

Reference: Section 9, Protocol B
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from scipy.linalg import sqrtm, expm
from itertools import product


@dataclass
class ProcessTensorConfig:
    """Configuration for process tensor tomography."""
    n_time_steps: int = 3  # Number of time steps
    n_measurements: int = 1000  # Measurements per configuration
    time_step_duration: float = 1.0  # Duration between measurements
    
    # Memory kernel parameters
    memory_strength: float = 0.01  # ε_χ
    memory_kernel_type: str = "exponential"  # 'exponential' or 'power_law'
    memory_decay_rate: float = 0.5  # For exponential kernel
    
    # Reconstruction parameters
    regularization: float = 1e-6  # For numerical stability
    max_iterations: int = 1000


class PauliOperators:
    """Pauli operator basis for single-qubit systems."""
    
    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    basis = [I, X, Y, Z]
    labels = ['I', 'X', 'Y', 'Z']
    
    @classmethod
    def decompose(cls, rho: np.ndarray) -> np.ndarray:
        """Decompose density matrix in Pauli basis."""
        coeffs = np.zeros(4, dtype=complex)
        for i, P in enumerate(cls.basis):
            coeffs[i] = np.trace(P @ rho) / 2
        return coeffs
    
    @classmethod
    def reconstruct(cls, coeffs: np.ndarray) -> np.ndarray:
        """Reconstruct density matrix from Pauli coefficients."""
        rho = np.zeros((2, 2), dtype=complex)
        for c, P in zip(coeffs, cls.basis):
            rho += c * P
        return rho


class MemoryKernel:
    """Implements various memory kernel types."""
    
    def __init__(self, kernel_type: str = "exponential", 
                 strength: float = 0.01, decay_rate: float = 0.5):
        self.kernel_type = kernel_type
        self.strength = strength
        self.decay_rate = decay_rate
    
    def __call__(self, t: float, tau: float) -> float:
        """
        Evaluate memory kernel K(t - τ).
        
        Args:
            t: Current time
            tau: Past time
        
        Returns:
            Kernel value
        """
        dt = t - tau
        if dt < 0:
            return 0.0
        
        if self.kernel_type == "exponential":
            # K(Δt) = ε * exp(-γ * Δt)
            return self.strength * np.exp(-self.decay_rate * dt)
        
        elif self.kernel_type == "power_law":
            # K(Δt) = ε / (1 + γ * Δt)
            return self.strength / (1 + self.decay_rate * dt)
        
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")


class NonMarkovianEvolution:
    """Simulates non-Markovian qubit evolution with memory."""
    
    def __init__(self, kernel: MemoryKernel, dt: float = 0.01):
        """
        Initialize non-Markovian evolution.
        
        Args:
            kernel: Memory kernel
            dt: Time discretization step
        """
        self.kernel = kernel
        self.dt = dt
        self.history = []  # Stores (time, state) pairs
    
    def evolve(self, rho: np.ndarray, t_final: float) -> np.ndarray:
        """
        Evolve density matrix with memory.
        
        Uses Eq. 18: ρ̇(t) = L[ρ(t)] + ∫₀ᵗ K(t-τ) ρ(τ) dτ
        
        Args:
            rho: Initial density matrix
            t_final: Final time
        
        Returns:
            Evolved density matrix
        """
        # Local Lindblad generator (dephasing)
        def lindblad(rho_t):
            sigma_z = PauliOperators.Z
            dephasing_rate = 0.1
            L_rho = -1j * 0 * rho_t  # No Hamiltonian for simplicity
            # Dephasing: L[ρ] = γ(σ_z ρ σ_z - ρ)
            L_rho += dephasing_rate * (sigma_z @ rho_t @ sigma_z - rho_t)
            return L_rho
        
        # Store initial state
        self.history = [(0.0, rho.copy())]
        
        # Time evolution
        t = 0.0
        rho_current = rho.copy()
        
        while t < t_final:
            # Markovian part
            drho_local = lindblad(rho_current)
            
            # Non-Markovian memory integral
            # ∫₀ᵗ K(t-τ) ρ(τ) dτ
            memory_contribution = np.zeros_like(rho_current)
            
            for t_past, rho_past in self.history:
                if t_past <= t:
                    kernel_val = self.kernel(t, t_past)
                    memory_contribution += kernel_val * rho_past * self.dt
            
            # Total derivative
            drho_total = drho_local + memory_contribution
            
            # Update (Euler step)
            rho_current = rho_current + drho_total * self.dt
            
            # Ensure physicality
            rho_current = (rho_current + rho_current.conj().T) / 2  # Hermiticity
            rho_current = rho_current / np.trace(rho_current)  # Normalization
            
            # Ensure positivity (project to nearest positive semidefinite)
            eigvals, eigvecs = np.linalg.eigh(rho_current)
            eigvals = np.maximum(eigvals, 0)
            rho_current = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
            rho_current = rho_current / np.trace(rho_current)
            
            t += self.dt
            self.history.append((t, rho_current.copy()))
        
        return rho_current


class ProcessTensor:
    """Represents a multi-time quantum process."""
    
    def __init__(self, n_steps: int):
        """
        Initialize process tensor.
        
        Args:
            n_steps: Number of time steps
        """
        self.n_steps = n_steps
        # Process tensor components: maps (operation sequence) -> final state
        self.tensor_data = {}
    
    def add_measurement(self, operations: Tuple[int, ...], 
                       final_state: np.ndarray, weight: float = 1.0):
        """
        Add measurement result to process tensor.
        
        Args:
            operations: Sequence of operation indices
            final_state: Resulting density matrix
            weight: Statistical weight
        """
        if operations not in self.tensor_data:
            self.tensor_data[operations] = []
        self.tensor_data[operations].append((final_state, weight))
    
    def predict_outcome(self, operations: Tuple[int, ...]) -> Optional[np.ndarray]:
        """
        Predict final state for operation sequence.
        
        Args:
            operations: Sequence of operation indices
        
        Returns:
            Predicted density matrix or None if not measured
        """
        if operations not in self.tensor_data:
            return None
        
        # Average over measurements
        states, weights = zip(*self.tensor_data[operations])
        total_weight = sum(weights)
        avg_state = sum(w * s for w, s in zip(weights, states)) / total_weight
        
        return avg_state


class CPDivisibilityTest:
    """Tests whether a process is CP-divisible (Markovian)."""
    
    @staticmethod
    def compute_dynamical_map(rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        """
        Compute Choi matrix of dynamical map: ρ_in -> ρ_out.
        
        For single qubit, Choi matrix is 4×4.
        """
        # Vectorize density matrices
        rho_in_vec = rho_in.flatten()
        rho_out_vec = rho_out.flatten()
        
        # Compute Choi matrix (simplified for pedagogical purposes)
        # In full implementation, would use process tomography
        choi = np.outer(rho_out_vec, rho_in_vec.conj())
        
        return choi
    
    @staticmethod
    def is_completely_positive(choi: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Test if Choi matrix represents a completely positive map.
        
        Args:
            choi: Choi matrix
            tol: Tolerance for negative eigenvalues
        
        Returns:
            True if CP, False otherwise
        """
        eigvals = np.linalg.eigvalsh(choi)
        return np.all(eigvals > -tol)
    
    @staticmethod
    def test_divisibility(
        process_tensor: ProcessTensor,
        time_steps: List[int]
    ) -> Dict[str, any]:
        """
        Test CP-divisibility: E_{t:0} = E_{t:s} ∘ E_{s:0} for all 0 ≤ s ≤ t.
        
        Args:
            process_tensor: Reconstructed process tensor
            time_steps: List of time indices to test
        
        Returns:
            Test results including CP violations
        """
        violations = []
        
        # Test factorization at intermediate times
        for t in time_steps[1:]:
            for s in range(1, t):
                # Check if E_{t:0} = E_{t:s} ∘ E_{s:0}
                # This requires comparing different operation sequences
                
                # For simplicity, we test a specific scenario
                # In full implementation, would test all possible interventions
                
                identity_seq = tuple([0] * t)  # Identity operations
                
                state_0 = process_tensor.predict_outcome(tuple([0] * 0))  # Initial
                state_s = process_tensor.predict_outcome(tuple([0] * s))
                state_t = process_tensor.predict_outcome(identity_seq)
                
                if state_0 is not None and state_s is not None and state_t is not None:
                    # Check if evolution is divisible
                    # Simplified test: compare eigenvalue evolution
                    
                    eig_0 = np.linalg.eigvalsh(state_0)
                    eig_s = np.linalg.eigvalsh(state_s)
                    eig_t = np.linalg.eigvalsh(state_t)
                    
                    # Non-monotonic eigenvalue evolution indicates non-CP-divisibility
                    if not (np.all(eig_s <= eig_0 + 1e-6) and np.all(eig_t <= eig_s + 1e-6)):
                        violations.append({
                            'time': t,
                            'intermediate': s,
                            'type': 'eigenvalue_non_monotonicity'
                        })
        
        return {
            'is_cp_divisible': len(violations) == 0,
            'violations': violations,
            'n_violations': len(violations)
        }


def reconstruct_process_tensor(
    config: ProcessTensorConfig,
    use_memory: bool = True
) -> ProcessTensor:
    """
    Reconstruct process tensor from multi-time measurements.
    
    Args:
        config: Configuration
        use_memory: Whether to include temporal memory
    
    Returns:
        Reconstructed ProcessTensor
    """
    print(f"Reconstructing process tensor: {config.n_time_steps} steps")
    print(f"Memory: {'Enabled' if use_memory else 'Disabled'}")
    
    # Initialize
    if use_memory:
        kernel = MemoryKernel(
            kernel_type=config.memory_kernel_type,
            strength=config.memory_strength,
            decay_rate=config.memory_decay_rate
        )
        evolver = NonMarkovianEvolution(kernel, dt=0.01)
    
    process_tensor = ProcessTensor(config.n_time_steps)
    
    # Initial state: |0⟩
    rho_init = np.array([[1, 0], [0, 0]], dtype=complex)
    
    # Generate all possible operation sequences
    # Operations: 0=Identity, 1=X, 2=Y, 3=Z
    n_operations = 4
    
    for ops_sequence in product(range(n_operations), repeat=config.n_time_steps):
        rho = rho_init.copy()
        
        # Apply operations at each time step
        for step, op_idx in enumerate(ops_sequence):
            # Apply operation
            op = PauliOperators.basis[op_idx]
            rho = op @ rho @ op.conj().T
            rho = rho / np.trace(rho)
            
            # Evolve with/without memory
            if use_memory:
                rho = evolver.evolve(rho, config.time_step_duration)
            else:
                # Standard Markovian evolution (just dephasing)
                dephasing = 0.1 * config.time_step_duration
                rho = (1 - dephasing) * rho + dephasing * np.diag(np.diag(rho))
        
        # Store result
        process_tensor.add_measurement(ops_sequence, rho)
    
    print(f"Tensor reconstructed: {len(process_tensor.tensor_data)} configurations")
    return process_tensor


def run_protocol_b(config: ProcessTensorConfig) -> Dict:
    """
    Run Protocol B: Process tensor tomography and CP-divisibility test.
    
    Args:
        config: Test configuration
    
    Returns:
        Test results
    """
    print("="*70)
    print("PROTOCOL B: PROCESS TENSOR TOMOGRAPHY")
    print("="*70)
    
    # Reconstruct with memory (spiral-time)
    print("\n[1/2] Spiral-time dynamics (with memory)...")
    tensor_memory = reconstruct_process_tensor(config, use_memory=True)
    
    # Reconstruct without memory (Markovian)
    print("\n[2/2] Markovian dynamics (no memory)...")
    tensor_markov = reconstruct_process_tensor(config, use_memory=False)
    
    # Test CP-divisibility
    print("\n" + "="*70)
    print("CP-DIVISIBILITY ANALYSIS")
    print("="*70)
    
    time_steps = list(range(1, config.n_time_steps + 1))
    
    print("\nTesting spiral-time tensor...")
    cp_test_memory = CPDivisibilityTest.test_divisibility(tensor_memory, time_steps)
    
    print("\nTesting Markovian tensor...")
    cp_test_markov = CPDivisibilityTest.test_divisibility(tensor_markov, time_steps)
    
    # Results
    results = {
        'config': config,
        'tensor_memory': tensor_memory,
        'tensor_markov': tensor_markov,
        'cp_test_memory': cp_test_memory,
        'cp_test_markov': cp_test_markov
    }
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Spiral-time CP-divisible: {cp_test_memory['is_cp_divisible']}")
    print(f"  Violations detected: {cp_test_memory['n_violations']}")
    print(f"\nMarkovian CP-divisible: {cp_test_markov['is_cp_divisible']}")
    print(f"  Violations detected: {cp_test_markov['n_violations']}")
    
    if not cp_test_memory['is_cp_divisible'] and cp_test_markov['is_cp_divisible']:
        print("\n✓ SPIRAL-TIME PREDICTION CONFIRMED")
        print("  Non-Markovian process tensor detected with memory")
    elif cp_test_memory['is_cp_divisible']:
        print("\n✗ SPIRAL-TIME PREDICTION NOT CONFIRMED")
        print("  Process tensor remains CP-divisible")
    
    return results


if __name__ == "__main__":
    config = ProcessTensorConfig(
        n_time_steps=3,
        memory_strength=0.05,
        memory_kernel_type="exponential",
        memory_decay_rate=0.3
    )
    
    results = run_protocol_b(config)
    
    print("\n" + "="*70)
    print("Protocol B complete.")
    print("="*70)
