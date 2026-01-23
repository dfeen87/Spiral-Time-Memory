"""
Protocol C: Leggett-Garg Inequality under Memory Suppression
=============================================================

Tests multi-time correlations using Leggett-Garg inequalities with and without
engineered memory suppression. Spiral-time predicts persistent violations even
with memory suppression protocols.

Reference: Section 10.5, Protocol C
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class LeggettGargConfig:
    """Configuration for Leggett-Garg protocol."""

    n_measurements: int = 10000  # Number of measurement runs
    n_time_points: int = 3  # Number of time points (typically 3 or 4)
    time_intervals: List[float] = None  # Time between measurements

    # Memory parameters
    memory_strength: float = 0.02  # ε_χ
    memory_coherence_time: float = 5.0  # τ_mem

    # Memory suppression parameters
    suppression_enabled: bool = False
    suppression_strength: float = 0.5  # 0 = no suppression, 1 = full suppression
    suppression_method: str = "dynamical_decoupling"  # or "frequent_reset"

    def __post_init__(self):
        if self.time_intervals is None:
            # Default: equal spacing
            self.time_intervals = [1.0] * (self.n_time_points - 1)


class QuantumEvolution:
    """Simulates qubit evolution with optional memory effects."""

    def __init__(self, memory_strength: float, coherence_time: float):
        self.memory_strength = memory_strength
        self.coherence_time = coherence_time
        self.chi = 0.0  # Memory sector

    def evolve_with_memory(
        self, rho: np.ndarray, dt: float, suppression: float = 0.0
    ) -> np.ndarray:
        """
        Evolve density matrix with memory kernel.

        Args:
            rho: Input density matrix
            dt: Time interval
            suppression: Memory suppression strength (0 to 1)

        Returns:
            Evolved density matrix
        """
        # Hamiltonian evolution (simple precession)
        omega = 1.0  # Oscillation frequency
        _ = 0.5 * omega * np.array([[1, 0], [0, -1]], dtype=complex)
        U = np.array(
            [
                [np.cos(omega * dt / 2) - 1j * np.sin(omega * dt / 2), 0],
                [0, np.cos(omega * dt / 2) + 1j * np.sin(omega * dt / 2)],
            ]
        )

        rho_evolved = U @ rho @ U.conj().T

        # Memory-induced decoherence (Eq. 18)
        effective_memory = self.memory_strength * (1 - suppression)
        memory_factor = np.exp(-effective_memory * dt / self.coherence_time)

        # Off-diagonal decay due to memory
        rho_evolved[0, 1] *= memory_factor
        rho_evolved[1, 0] *= memory_factor

        # Update memory sector
        self.chi += effective_memory * np.real(rho_evolved[0, 0] - rho_evolved[1, 1])

        return rho_evolved

    def reset_memory(self, strength: float = 1.0):
        """Reset memory sector."""
        self.chi *= 1 - strength


class LeggettGargMeasurement:
    """Implements projective measurements for Leggett-Garg tests."""

    def __init__(self, observable: str = "Z"):
        """
        Initialize measurement.

        Args:
            observable: 'Z', 'X', or 'Y'
        """
        self.observable = observable

        if observable == "Z":
            self.operator = np.array([[1, 0], [0, -1]], dtype=complex)
            self.projectors = [
                np.array([[1, 0], [0, 0]], dtype=complex),  # |0⟩⟨0|
                np.array([[0, 0], [0, 1]], dtype=complex),  # |1⟩⟨1|
            ]
            self.outcomes = [+1, -1]
        elif observable == "X":
            self.operator = np.array([[0, 1], [1, 0]], dtype=complex)
            self.projectors = [
                0.5 * np.array([[1, 1], [1, 1]], dtype=complex),
                0.5 * np.array([[1, -1], [-1, 1]], dtype=complex),
            ]
            self.outcomes = [+1, -1]
        else:
            raise ValueError(f"Unsupported observable: {observable}")

    def measure(
        self, rho: np.ndarray, collapse: bool = True
    ) -> Tuple[float, np.ndarray]:
        """
        Perform measurement.

        Args:
            rho: Input density matrix
            collapse: If True, collapse state; if False, just compute expectation

        Returns:
            (measurement_outcome, post_measurement_state)
        """
        # Compute expectation value
        expectation = np.real(np.trace(self.operator @ rho))

        if collapse:
            # Perform projective measurement
            probs = [np.real(np.trace(P @ rho)) for P in self.projectors]
            probs = np.array(probs) / sum(probs)

            outcome_idx = np.random.choice(len(probs), p=probs)
            outcome = self.outcomes[outcome_idx]

            # Collapse state
            P = self.projectors[outcome_idx]
            rho_collapsed = P @ rho @ P
            rho_collapsed = rho_collapsed / np.trace(rho_collapsed)

            return outcome, rho_collapsed
        else:
            # Non-invasive measurement (ideal)
            return expectation, rho


class LeggettGargInequality:
    """Computes and tests Leggett-Garg inequalities."""

    @staticmethod
    def compute_K3(correlators: Dict[Tuple[int, int], float]) -> float:
        """
        Compute K3 Leggett-Garg parameter.

        K3 = C(t1,t2) + C(t2,t3) - C(t1,t3)

        Classical bound: -3 ≤ K3 ≤ 1
        Quantum bound: K3 can reach 3/2

        Args:
            correlators: Dictionary {(ti, tj): C(ti,tj)}

        Returns:
            K3 value
        """
        C_12 = correlators.get((0, 1), 0)
        C_23 = correlators.get((1, 2), 0)
        C_13 = correlators.get((0, 2), 0)

        K3 = C_12 + C_23 - C_13
        return K3

    @staticmethod
    def compute_K4(correlators: Dict[Tuple[int, int], float]) -> float:
        """
        Compute K4 Leggett-Garg parameter (4 time points).

        K4 = C(t1,t2) + C(t2,t3) + C(t3,t4) - C(t1,t4)

        Classical bound: -4 ≤ K4 ≤ 2

        Args:
            correlators: Dictionary {(ti, tj): C(ti,tj)}

        Returns:
            K4 value
        """
        C_12 = correlators.get((0, 1), 0)
        C_23 = correlators.get((1, 2), 0)
        C_34 = correlators.get((2, 3), 0)
        C_14 = correlators.get((0, 3), 0)

        K4 = C_12 + C_23 + C_34 - C_14
        return K4

    @staticmethod
    def test_violation(
        K: float, n_times: int = 3, significance_level: float = 0.05
    ) -> Dict:
        """
        Test whether K violates classical bound.

        Args:
            K: Leggett-Garg parameter
            n_times: Number of time points (3 or 4)
            significance_level: Statistical significance threshold

        Returns:
            Test results
        """
        if n_times == 3:
            classical_max = 1.0
            classical_min = -3.0
        elif n_times == 4:
            classical_max = 2.0
            classical_min = -4.0
        else:
            raise ValueError("Only 3 or 4 time points supported")

        violates_upper = K > classical_max
        violates_lower = K < classical_min
        violation_magnitude = max(K - classical_max, classical_min - K, 0)

        return {
            "K": K,
            "classical_max": classical_max,
            "classical_min": classical_min,
            "violates": violates_upper or violates_lower,
            "violation_magnitude": violation_magnitude,
            "violation_direction": "upper"
            if violates_upper
            else "lower"
            if violates_lower
            else "none",
        }


def run_leggett_garg_experiment(
    config: LeggettGargConfig, invasive_measurement: bool = True
) -> Dict:
    """
    Run Leggett-Garg experiment.

    Args:
        config: Configuration
        invasive_measurement: If True, measurements collapse state

    Returns:
        Experimental results
    """
    evolver = QuantumEvolution(config.memory_strength, config.memory_coherence_time)
    measurement = LeggettGargMeasurement(observable="Z")

    # Storage for correlations
    measurement_results = {i: [] for i in range(config.n_time_points)}

    # Run measurements
    for run in range(config.n_measurements):
        # Initialize state: |+⟩ (superposition)
        rho = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
        evolver.reset_memory()

        outcomes = []
        t_current = 0.0

        for t_idx in range(config.n_time_points):
            # Evolve to next measurement time
            if t_idx > 0:
                dt = config.time_intervals[t_idx - 1]

                # Apply memory suppression if enabled
                if config.suppression_enabled:
                    if config.suppression_method == "dynamical_decoupling":
                        # Simulate dynamical decoupling (reduces memory coherence)
                        rho = evolver.evolve_with_memory(
                            rho, dt, suppression=config.suppression_strength
                        )
                    elif config.suppression_method == "frequent_reset":
                        # Periodic reset of memory sector
                        rho = evolver.evolve_with_memory(rho, dt, suppression=0.0)
                        evolver.reset_memory(strength=config.suppression_strength)
                else:
                    # Normal evolution with full memory
                    rho = evolver.evolve_with_memory(rho, dt, suppression=0.0)

                t_current += dt

            # Measure
            outcome, rho = measurement.measure(rho, collapse=invasive_measurement)
            outcomes.append(outcome)
            measurement_results[t_idx].append(outcome)

    # Compute two-time correlators
    correlators = {}
    for i in range(config.n_time_points):
        for j in range(i + 1, config.n_time_points):
            # C(ti, tj) = <Q(ti) Q(tj)>
            correlators[(i, j)] = np.mean(
                [
                    measurement_results[i][k] * measurement_results[j][k]
                    for k in range(config.n_measurements)
                ]
            )

    # Compute Leggett-Garg parameter
    if config.n_time_points == 3:
        K = LeggettGargInequality.compute_K3(correlators)
    elif config.n_time_points == 4:
        K = LeggettGargInequality.compute_K4(correlators)
    else:
        K = 0.0

    # Test violation
    test_result = LeggettGargInequality.test_violation(K, config.n_time_points)

    return {
        "correlators": correlators,
        "K": K,
        "test_result": test_result,
        "measurement_results": measurement_results,
        "config": config,
    }


def run_protocol_c(config: LeggettGargConfig) -> Dict:
    """
    Run Protocol C: Compare Leggett-Garg violations with/without memory suppression.

    Args:
        config: Test configuration

    Returns:
        Comparison results
    """
    print("=" * 70)
    print("PROTOCOL C: LEGGETT-GARG INEQUALITY UNDER MEMORY SUPPRESSION")
    print("=" * 70)

    # Test 1: No memory suppression
    print("\n[1/2] Running without memory suppression...")
    config_no_suppression = config
    config_no_suppression.suppression_enabled = False
    results_no_suppression = run_leggett_garg_experiment(config_no_suppression)

    # Test 2: With memory suppression
    print("[2/2] Running with memory suppression...")
    config_with_suppression = config
    config_with_suppression.suppression_enabled = True
    results_with_suppression = run_leggett_garg_experiment(config_with_suppression)

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    K_no_supp = results_no_suppression["K"]
    K_with_supp = results_with_suppression["K"]

    test_no_supp = results_no_suppression["test_result"]
    test_with_supp = results_with_suppression["test_result"]

    print("\nWithout memory suppression:")
    print(f"  K = {K_no_supp:.4f}")
    print(f"  Classical bound: {test_no_supp['classical_max']}")
    print(f"  Violation: {test_no_supp['violates']}")
    print(f"  Violation magnitude: {test_no_supp['violation_magnitude']:.4f}")

    print(
        f"\nWith memory suppression ({config.suppression_method}, "
        f"strength={config.suppression_strength}):"
    )
    print(f"  K = {K_with_supp:.4f}")
    print(f"  Classical bound: {test_with_supp['classical_max']}")
    print(f"  Violation: {test_with_supp['violates']}")
    print(f"  Violation magnitude: {test_with_supp['violation_magnitude']:.4f}")

    print(f"\nChange in K: ΔK = {K_no_supp - K_with_supp:.4f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if test_with_supp["violates"]:
        print(
            "✓ SPIRAL-TIME PREDICTION: Memory suppression does NOT eliminate violation"
        )
        print("  This suggests intrinsic temporal memory beyond environmental effects")
    else:
        print("✗ Violation eliminated by memory suppression")
        print("  Results consistent with environmental (non-intrinsic) memory")

    if test_no_supp["violates"] and test_with_supp["violates"]:
        reduction = (
            test_no_supp["violation_magnitude"] - test_with_supp["violation_magnitude"]
        )
        print(f"\nViolation reduction: {reduction:.4f}")

        if reduction < 0.1 * test_no_supp["violation_magnitude"]:
            print("→ Minimal reduction suggests intrinsic temporal memory")
        else:
            print("→ Significant reduction suggests partial environmental contribution")

    return {
        "no_suppression": results_no_suppression,
        "with_suppression": results_with_suppression,
        "delta_K": K_no_supp - K_with_supp,
    }


if __name__ == "__main__":
    # Configuration
    config = LeggettGargConfig(
        n_measurements=5000,
        n_time_points=3,
        time_intervals=[1.0, 1.0],
        memory_strength=0.03,
        memory_coherence_time=5.0,
        suppression_strength=0.7,
        suppression_method="dynamical_decoupling",
    )

    results = run_protocol_c(config)

    print("\n" + "=" * 70)
    print("Protocol C complete.")
    print("=" * 70)

    # Optional: Plot correlators
    print("\nTwo-time correlators (no suppression):")
    for (i, j), C in results["no_suppression"]["correlators"].items():
        print(f"  C(t{i}, t{j}) = {C:.4f}")

    print("\nTwo-time correlators (with suppression):")
    for (i, j), C in results["with_suppression"]["correlators"].items():
        print(f"  C(t{i}, t{j}) = {C:.4f}")
