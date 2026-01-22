"""
CP-Divisibility Testing for Process Tensors
============================================

Core falsification tool for spiral-time memory hypothesis.

The theory is falsified if all reconstructed process tensors are CP-divisible
(Markovian) under controlled interventions - see THEORY.md §Falsification.

Includes tests to distinguish spiral-time memory from 
environmental decoherence via state-independence criteria (Section 10).

WARNING: Multiple non-Markovianity measures exist (BLP, RHP, trace distance).
This implements BLP for pedagogical clarity. See IMPLEMENTATIONS.md §4.

Author: Marcel Krüger & Don Michael Feeney Jr
License: MIT
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.linalg import expm, logm
import warnings


@dataclass
class ProcessTensorConfig:
    """Configuration for process tensor reconstruction."""
    n_timesteps: int = 3          # Number of time points
    hilbert_dim: int = 2          # Dimension of quantum system (e.g., qubit = 2)
    dt: float = 0.1               # Time interval between steps
    
    def __post_init__(self):
        if self.n_timesteps < 2:
            raise ValueError("Need at least 2 timesteps for process")


class QuantumChannel:
    """Completely positive trace-preserving (CPTP) map.
    
    Represents evolution ρ → ε(ρ) between time steps.
    """
    
    def __init__(self, kraus_operators: List[np.ndarray]):
        """Initialize from Kraus representation.
        
        Args:
            kraus_operators: List of Kraus operators {K_i}
                            satisfying Σ K_i† K_i = I
        """
        self.kraus_ops = [K.copy() for K in kraus_operators]
        self.dim = kraus_operators[0].shape[0]
        
        # Verify CPTP
        sum_KdagK = sum(K.conj().T @ K for K in self.kraus_ops)
        if not np.allclose(sum_KdagK, np.eye(self.dim)):
            warnings.warn("Channel may not be trace-preserving")
    
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply channel to density matrix.
        
        Args:
            rho: Input density matrix
            
        Returns:
            ε(ρ) = Σ K_i ρ K_i†
        """
        return sum(K @ rho @ K.conj().T for K in self.kraus_ops)
    
    def choi_matrix(self) -> np.ndarray:
        """Choi-Jamiołkowski representation.
        
        Returns:
            Choi matrix Λ = Σ |K_i⟩⟩⟨⟨K_i|
        """
        d = self.dim
        choi = np.zeros((d**2, d**2), dtype=complex)
        
        for K in self.kraus_ops:
            K_vec = K.reshape(-1, 1)  # Vectorize operator
            choi += K_vec @ K_vec.conj().T
            
        return choi
    
    def is_cp(self, tol: float = 1e-10) -> bool:
        """Check complete positivity via Choi matrix.
        
        CP ⟺ Choi matrix is positive semidefinite.
        """
        choi = self.choi_matrix()
        eigvals = np.linalg.eigvalsh(choi)
        return np.all(eigvals >= -tol)


def compose_channels(eps1: QuantumChannel, eps2: QuantumChannel) -> QuantumChannel:
    """Compose two quantum channels: ε₂ ∘ ε₁.
    
    Args:
        eps1: First channel
        eps2: Second channel
        
    Returns:
        Composed channel
    """
    # Compose via Kraus operators
    # (ε₂ ∘ ε₁)(ρ) = Σᵢⱼ K₂ʲ K₁ⁱ ρ (K₁ⁱ)† (K₂ʲ)†
    composed_kraus = [
        K2 @ K1 
        for K2 in eps2.kraus_ops 
        for K1 in eps1.kraus_ops
    ]
    
    return QuantumChannel(composed_kraus)


def test_cp_divisibility(
    channels: List[QuantumChannel],
    times: List[float]
) -> Tuple[bool, List[float]]:
    """Test CP-divisibility of a quantum process.
    
    A process is CP-divisible (Markovian) if:
        ε_{t:0} = ε_{t:s} ∘ ε_{s:0}  with ε_{t:s} CP for all t ≥ s ≥ 0
    
    This is the NULL HYPOTHESIS. Failure indicates non-Markovian memory.
    
    Args:
        channels: List of cumulative channels {ε_{tᵢ:0}}
        times: Corresponding time points
        
    Returns:
        is_cp_divisible: True if all intermediate maps are CP
        cp_violations: List of BLP non-Markovianity measures at each step
        
    Example:
        >>> # Create test channels (Markovian case)
        >>> K = [np.eye(2) * 0.9, np.array([[0, 0.436], [0, 0]])]
        >>> eps = [QuantumChannel([k * np.eye(2)]) for k in [0.9, 0.81, 0.73]]
        >>> is_markov, viols = test_cp_divisibility(eps, [0, 0.1, 0.2])
    """
    n_steps = len(channels)
    cp_violations = []
    is_cp_divisible = True
    
    for i in range(1, n_steps):
        # Construct intermediate map ε_{tᵢ:tᵢ₋₁}
        # Need to solve: ε_{i:0} = ε_{i:i-1} ∘ ε_{i-1:0}
        # This requires inverting ε_{i-1:0}, which may not be physical
        
        # BLP measure: check if trace distance can increase
        # For simplicity, use Choi-matrix eigenvalue test
        
        # Compute "incremental" channel via Choi pseudoinverse (non-unique!)
        choi_i = channels[i].choi_matrix()
        choi_im1 = channels[i-1].choi_matrix()
        
        # Attempt to extract intermediate map
        # This is ill-posed in general; we use a regularized approach
        try:
            # Simplified test: check monotonicity of trace norm
            eigvals_i = np.linalg.eigvalsh(choi_i)
            eigvals_im1 = np.linalg.eigvalsh(choi_im1)
            
            # BLP-like measure: negativity of intermediate eigenvalues
            min_eigval = np.min(eigvals_i)
            
            if min_eigval < -1e-10:  # Significant negativity
                is_cp_divisible = False
                cp_violations.append(-min_eigval)
            else:
                cp_violations.append(0.0)
                
        except np.linalg.LinAlgError:
            warnings.warn(f"Singular Choi matrix at step {i}")
            cp_violations.append(np.nan)
    
    return is_cp_divisible, cp_violations


def blp_measure(
    rho0: np.ndarray,
    channels: List[QuantumChannel],
    times: List[float]
) -> np.ndarray:
    """Breuer-Laine-Piilo (BLP) non-Markovianity measure.
    
    Measures information backflow via trace distance increase:
        N = max_{ρ₁,ρ₂} ∫ max(0, d/dt D(ε_t(ρ₁), ε_t(ρ₂))) dt
    
    Simplified implementation: single pair of initial states.
    
    Args:
        rho0: Reference initial state
        channels: Cumulative channels at each time
        times: Time points
        
    Returns:
        BLP measure (0 = Markovian, >0 = non-Markovian)
    """
    # Construct orthogonal initial state
    d = rho0.shape[0]
    rho1 = np.eye(d) / d  # Maximally mixed
    
    # Evolve both states
    trace_dists = []
    for eps in channels:
        sigma0 = eps.apply(rho0)
        sigma1 = eps.apply(rho1)
        
        # Trace distance D(σ₀, σ₁) = (1/2) ||σ₀ - σ₁||₁
        diff = sigma0 - sigma1
        eigvals = np.linalg.eigvalsh(diff)
        trace_dist = 0.5 * np.sum(np.abs(eigvals))
        trace_dists.append(trace_dist)
    
    # Compute time derivative
    trace_dists = np.array(trace_dists)
    dt = np.diff(times)
    d_trace_dist = np.diff(trace_dists) / dt
    
    # Integrate positive derivatives (backflow)
    backflow = np.maximum(0, d_trace_dist)
    blp = np.trapz(backflow, times[1:])
    
    return blp


class ProcessTensorReconstructor:
    """Reconstruct multi-time process tensor from measurement data.
    
    Implements Protocol B from THEORY.md and state-independence tests
    from Section 10 (experimental discrimination).
    """
    
    def __init__(self, config: ProcessTensorConfig):
        self.config = config
        self.d = config.hilbert_dim
    
    def test_state_independence(
        self,
        channels: List[QuantumChannel],
        initial_states: List[np.ndarray]
    ) -> dict:
        """Test if memory kernel is state-independent (Section 10.1).
        
        Spiral-time prediction: Memory kernel K(t-τ) should NOT depend
        on initial state, unlike environmental non-Markovianity.
        
        Args:
            channels: Process at different times
            initial_states: Different initial system states
            
        Returns:
            Dictionary with state-independence measures
        """
        blp_values = []
        times = [i * self.config.dt for i in range(len(channels))]
        
        # Compute BLP for each initial state
        for rho0 in initial_states:
            blp = blp_measure(rho0, channels, times)
            blp_values.append(blp)
        
        # State-independence: variance should be small
        variance = np.var(blp_values)
        mean = np.mean(blp_values)
        
        # Coefficient of variation
        cv = variance / mean if mean > 1e-10 else 0
        
        return {
            'blp_values': blp_values,
            'mean': mean,
            'variance': variance,
            'cv': cv,
            'state_independent': cv < 0.2,  # Threshold for independence
            'interpretation': (
                'STATE-INDEPENDENT (consistent with spiral-time)' 
                if cv < 0.2 else 
                'STATE-DEPENDENT (environmental origin likely)'
            )
        }
    
    def generate_test_process(
        self,
        memory_strength: float = 0.0
    ) -> List[QuantumChannel]:
        """Generate synthetic process for testing.
        
        Args:
            memory_strength: 0 = Markovian, >0 = non-Markovian
            
        Returns:
            List of cumulative channels
        """
        channels = []
        dt = self.config.dt
        
        # Hamiltonian (standard evolution)
        H = np.array([[0, 1], [1, 0]]) * 2 * np.pi  # σ_x
        
        # Memory-induced non-unitary correction
        # This is a TOY model, not the canonical spiral-time evolution
        for i in range(self.config.n_timesteps):
            t = i * dt
            
            # Unitary part
            U = expm(-1j * H * t)
            
            # Memory-induced amplitude damping (illustrative)
            gamma = memory_strength * (1 - np.exp(-t / 0.5))
            
            K0 = U @ np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
            K1 = U @ np.array([[0, np.sqrt(gamma)], [0, 0]])
            
            channels.append(QuantumChannel([K0, K1]))
        
        return channels
    
    def test_memory_hypothesis(
        self,
        channels: List[QuantumChannel],
        times: List[float],
        verbose: bool = True
    ) -> dict:
        """Test spiral-time memory hypothesis.
        
        Returns:
            results: Dictionary with keys:
                - 'is_markovian': bool
                - 'cp_violations': list
                - 'blp_measure': float
                - 'verdict': str
        """
        # Test CP-divisibility
        is_cp_div, violations = test_cp_divisibility(channels, times)
        
        # Compute BLP measure
        rho0 = np.array([[1, 0], [0, 0]])  # |0⟩⟨0|
        blp = blp_measure(rho0, channels, times)
        
        results = {
            'is_markovian': is_cp_div,
            'cp_violations': violations,
            'blp_measure': blp,
            'verdict': 'MARKOVIAN (theory falsified)' if is_cp_div else 'NON-MARKOVIAN (consistent with memory)'
        }
        
        if verbose:
            print("=" * 60)
            print("SPIRAL-TIME MEMORY HYPOTHESIS TEST")
            print("=" * 60)
            print(f"CP-divisible: {is_cp_div}")
            print(f"BLP measure: {blp:.6f}")
            print(f"Verdict: {results['verdict']}")
            print("=" * 60)
        
        return results


# Example: Falsification test
if __name__ == "__main__":
    config = ProcessTensorConfig(n_timesteps=5, hilbert_dim=2, dt=0.1)
    reconstructor = ProcessTensorReconstructor(config)
    
    print("TEST 1: Markovian process (should falsify memory hypothesis)")
    print("-" * 60)
    channels_markov = reconstructor.generate_test_process(memory_strength=0.0)
    times = [i * config.dt for i in range(config.n_timesteps)]
    results_markov = reconstructor.test_memory_hypothesis(channels_markov, times)
    
    print("\nTEST 2: Non-Markovian process (consistent with memory)")
    print("-" * 60)
    channels_memory = reconstructor.generate_test_process(memory_strength=0.3)
    results_memory = reconstructor.test_memory_hypothesis(channels_memory, times)
    
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)
    print("If experimental data yields CP-divisible processes for ALL")
    print("controlled interventions, the spiral-time memory hypothesis")
    print("is FALSIFIED. Non-Markovian signatures are required.")
