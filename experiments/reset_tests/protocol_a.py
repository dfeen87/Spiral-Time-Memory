"""
Protocol A: Reset Test for History Dependence
==============================================

Tests whether nominally perfect reset operations can eliminate memory effects.
If spiral-time memory is intrinsic, residual history dependence should persist
beyond experimental systematics.

Reference: Section 10.5, Protocol A
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings


@dataclass
class ResetTestConfig:
    """Configuration for reset test protocol."""
    n_cycles: int = 100  # Number of measure-reset cycles
    n_histories: int = 10  # Number of different measurement histories to test
    history_length: int = 5  # Length of each measurement history
    reset_fidelity: float = 0.999  # Fidelity of reset operation
    measurement_basis: str = "computational"  # 'computational' or 'hadamard'
    
    # Memory parameters
    memory_strength: float = 0.01  # ε_χ parameter
    memory_decay_time: float = 10.0  # Characteristic memory timescale
    
    # Statistical parameters
    confidence_level: float = 0.95
    min_effect_size: float = 0.001  # Minimum detectable history dependence


class QuantumState:
    """Represents a qubit state with optional memory sector."""
    
    def __init__(self, rho: np.ndarray, chi: float = 0.0):
        """
        Initialize quantum state.
        
        Args:
            rho: 2x2 density matrix
            chi: Memory sector value
        """
        self.rho = rho / np.trace(rho)  # Normalize
        self.chi = chi
        self._validate()
    
    def _validate(self):
        """Ensure state is physical."""
        if not np.allclose(self.rho, self.rho.conj().T):
            raise ValueError("Density matrix must be Hermitian")
        if not np.isclose(np.trace(self.rho), 1.0):
            raise ValueError("Density matrix must have unit trace")
        eigenvalues = np.linalg.eigvalsh(self.rho)
        if np.any(eigenvalues < -1e-10):
            raise ValueError("Density matrix must be positive semidefinite")
    
    def copy(self) -> 'QuantumState':
        """Create a copy of the state."""
        return QuantumState(self.rho.copy(), self.chi)


class ResetOperation:
    """Models a reset operation with configurable fidelity."""
    
    def __init__(self, fidelity: float = 1.0, target_state: Optional[np.ndarray] = None):
        """
        Initialize reset operation.
        
        Args:
            fidelity: Reset fidelity (0 to 1)
            target_state: Target state (default: |0⟩⟨0|)
        """
        self.fidelity = fidelity
        if target_state is None:
            # Default: reset to |0⟩
            self.target_state = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        else:
            self.target_state = target_state
    
    def apply(self, state: QuantumState, preserve_memory: bool = False) -> QuantumState:
        """
        Apply reset operation.
        
        Args:
            state: Input quantum state
            preserve_memory: If True, memory sector is preserved (spiral-time prediction)
                           If False, memory is also reset (standard QM)
        
        Returns:
            Reset quantum state
        """
        # Standard reset: rho -> fidelity * |0⟩⟨0| + (1-fidelity) * rho
        reset_rho = (self.fidelity * self.target_state + 
                    (1 - self.fidelity) * state.rho)
        
        # Memory sector behavior
        if preserve_memory:
            # Spiral-time prediction: chi persists through reset
            reset_chi = state.chi
        else:
            # Markovian prediction: chi is also reset
            reset_chi = 0.0
        
        return QuantumState(reset_rho, reset_chi)


class MeasurementOperation:
    """Projective measurement in specified basis."""
    
    def __init__(self, basis: str = "computational"):
        """
        Initialize measurement.
        
        Args:
            basis: 'computational' (Z) or 'hadamard' (X)
        """
        self.basis = basis
        if basis == "computational":
            self.projectors = [
                np.array([[1, 0], [0, 0]], dtype=complex),  # |0⟩⟨0|
                np.array([[0, 0], [0, 1]], dtype=complex),  # |1⟩⟨1|
            ]
        elif basis == "hadamard":
            # X basis: |+⟩, |−⟩
            self.projectors = [
                np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex),
                np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex),
            ]
        else:
            raise ValueError(f"Unknown basis: {basis}")
    
    def measure(self, state: QuantumState, update_memory: bool = True,
                memory_strength: float = 0.01) -> Tuple[int, QuantumState]:
        """
        Perform measurement.
        
        Args:
            state: Input quantum state
            update_memory: Whether to update memory sector
            memory_strength: Strength of memory update (ε_χ)
        
        Returns:
            (outcome, post_measurement_state)
        """
        # Calculate Born probabilities
        probs = [np.real(np.trace(P @ state.rho)) for P in self.projectors]
        probs = np.array(probs) / sum(probs)  # Normalize
        
        # Sample outcome
        outcome = np.random.choice(len(probs), p=probs)
        
        # Project state
        P = self.projectors[outcome]
        post_rho = P @ state.rho @ P
        post_rho = post_rho / np.trace(post_rho)
        
        # Update memory sector (Eq. 11)
        if update_memory:
            delta_chi = memory_strength * (2 * outcome - 1)  # ±ε_χ encoding
            post_chi = state.chi + delta_chi
        else:
            post_chi = state.chi
        
        return outcome, QuantumState(post_rho, post_chi)


class HistoryDependenceAnalyzer:
    """Analyzes measurement outcomes for history dependence."""
    
    def __init__(self, config: ResetTestConfig):
        self.config = config
    
    def compute_conditional_probability(
        self,
        outcomes: List[int],
        histories: List[Tuple[int, ...]],
        target_outcome: int
    ) -> Dict[Tuple[int, ...], Tuple[float, float]]:
        """
        Compute P(outcome|history) for each history.
        
        Args:
            outcomes: List of measurement outcomes
            histories: List of measurement history tuples
            target_outcome: Outcome to compute probability for
        
        Returns:
            Dictionary mapping history -> (probability, standard_error)
        """
        history_stats = {}
        
        for history in set(histories):
            # Find all measurements preceded by this history
            mask = [h == history for h in histories]
            relevant_outcomes = [o for o, m in zip(outcomes, mask) if m]
            
            if len(relevant_outcomes) > 0:
                prob = np.mean([o == target_outcome for o in relevant_outcomes])
                se = np.sqrt(prob * (1 - prob) / len(relevant_outcomes))
                history_stats[history] = (prob, se)
        
        return history_stats
    
    def test_history_independence(
        self,
        outcomes: List[int],
        histories: List[Tuple[int, ...]],
    ) -> Dict[str, float]:
        """
        Test null hypothesis: P(outcome|history) is independent of history.
        
        Uses chi-squared test for independence.
        
        Args:
            outcomes: List of measurement outcomes
            histories: List of measurement history tuples
        
        Returns:
            Dictionary with test statistics
        """
        from scipy.stats import chi2_contingency
        
        # Create contingency table: history × outcome
        unique_histories = sorted(set(histories))
        unique_outcomes = sorted(set(outcomes))
        
        contingency = np.zeros((len(unique_histories), len(unique_outcomes)))
        for i, hist in enumerate(unique_histories):
            for j, outcome in enumerate(unique_outcomes):
                mask = [h == hist and o == outcome 
                       for h, o in zip(histories, outcomes)]
                contingency[i, j] = sum(mask)
        
        # Chi-squared test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Effect size (Cramér's V)
        n = contingency.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        
        return {
            'chi_squared': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'n_samples': n,
            'significant': p_value < (1 - self.config.confidence_level),
            'effect_detectable': cramers_v > self.config.min_effect_size
        }


def run_reset_test(
    config: ResetTestConfig,
    spiral_time_mode: bool = True
) -> Dict:
    """
    Run Protocol A: Reset test for history dependence.
    
    Args:
        config: Test configuration
        spiral_time_mode: If True, simulate spiral-time dynamics with memory persistence
                         If False, simulate standard Markovian dynamics
    
    Returns:
        Dictionary containing test results and statistics
    """
    print(f"Running Protocol A: Reset Test")
    print(f"Mode: {'Spiral-Time with Memory' if spiral_time_mode else 'Standard Markovian'}")
    print(f"Cycles: {config.n_cycles}, Histories: {config.n_histories}")
    
    # Initialize
    reset_op = ResetOperation(fidelity=config.reset_fidelity)
    measure_op = MeasurementOperation(basis=config.measurement_basis)
    analyzer = HistoryDependenceAnalyzer(config)
    
    # Storage
    all_outcomes = []
    all_histories = []
    all_chi_values = []
    
    # Generate different measurement histories
    measurement_sequences = [
        tuple(np.random.randint(0, 2, config.history_length))
        for _ in range(config.n_histories)
    ]
    
    for cycle in range(config.n_cycles):
        # Select a measurement history
        history = measurement_sequences[cycle % config.n_histories]
        
        # Initialize state
        state = QuantumState(np.array([[1, 0], [0, 0]], dtype=complex), chi=0.0)
        
        # Apply measurement history
        for target_outcome in history:
            # Rotate to target outcome (simulate preparation)
            if target_outcome == 1:
                state.rho = np.array([[0, 0], [0, 1]], dtype=complex)
            
            # Measure
            outcome, state = measure_op.measure(
                state,
                update_memory=spiral_time_mode,
                memory_strength=config.memory_strength
            )
        
        # Apply reset
        state = reset_op.apply(state, preserve_memory=spiral_time_mode)
        
        # Final measurement
        outcome, state = measure_op.measure(
            state,
            update_memory=False  # Don't update memory in final measurement
        )
        
        # Record
        all_outcomes.append(outcome)
        all_histories.append(history)
        all_chi_values.append(state.chi)
    
    # Analyze history dependence
    conditional_probs = analyzer.compute_conditional_probability(
        all_outcomes, all_histories, target_outcome=0
    )
    
    independence_test = analyzer.test_history_independence(
        all_outcomes, all_histories
    )
    
    # Compute memory persistence metric
    memory_persistence = np.std(all_chi_values) if spiral_time_mode else 0.0
    
    results = {
        'mode': 'spiral_time' if spiral_time_mode else 'markovian',
        'outcomes': all_outcomes,
        'histories': all_histories,
        'chi_values': all_chi_values,
        'conditional_probabilities': conditional_probs,
        'independence_test': independence_test,
        'memory_persistence': memory_persistence,
        'config': config
    }
    
    # Print summary
    print(f"\nResults:")
    print(f"  Chi-squared statistic: {independence_test['chi_squared']:.4f}")
    print(f"  p-value: {independence_test['p_value']:.4e}")
    print(f"  Cramér's V (effect size): {independence_test['cramers_v']:.4f}")
    print(f"  History dependence detected: {independence_test['significant']}")
    print(f"  Memory persistence (std χ): {memory_persistence:.4f}")
    
    if spiral_time_mode and independence_test['significant']:
        print(f"\n✓ Spiral-time prediction confirmed: History dependence persists after reset")
    elif not spiral_time_mode and not independence_test['significant']:
        print(f"\n✓ Markovian prediction confirmed: No history dependence after reset")
    
    return results


if __name__ == "__main__":
    # Example usage
    config = ResetTestConfig(
        n_cycles=500,
        n_histories=8,
        history_length=4,
        reset_fidelity=0.999,
        memory_strength=0.02
    )
    
    print("="*70)
    print("PROTOCOL A: RESET TEST FOR TEMPORAL MEMORY")
    print("="*70)
    
    # Test 1: Spiral-time dynamics
    results_spiral = run_reset_test(config, spiral_time_mode=True)
    
    print("\n" + "="*70)
    
    # Test 2: Standard Markovian dynamics
    results_markov = run_reset_test(config, spiral_time_mode=False)
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Spiral-time history dependence: p={results_spiral['independence_test']['p_value']:.2e}")
    print(f"Markovian history dependence: p={results_markov['independence_test']['p_value']:.2e}")
    print(f"\nΔp = {abs(results_spiral['independence_test']['p_value'] - results_markov['independence_test']['p_value']):.2e}")
