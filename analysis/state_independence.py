"""
State-Independence Tests for Temporal Memory
=============================================

Implements Section 10.1 of the paper: State-Independent Temporal Memory.

Key distinction from environmental decoherence:
- Environmental: Memory kernel depends on system state and system-bath correlations
- Spiral-Time: Memory kernel K(t-τ) depends ONLY on temporal separation

This module provides tools to test whether observed non-Markovianity is
state-independent (consistent with Spiral-Time) or state-dependent (environmental).

Reference: Paper Section 10.1
Author: Marcel Krüger & Don Michael Feeney Jr.
License: MIT
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway


@dataclass
class StateIndependenceConfig:
    """Configuration for state-independence tests."""

    n_initial_states: int = 10  # Number of different initial states to test
    n_measurements: int = 1000  # Shots per state
    significance_level: float = 0.05  # Statistical significance threshold
    cv_threshold: float = 0.2  # Coefficient of variation threshold


class StateIndependenceTester:
    """Test whether memory effects are state-independent.

    Spiral-Time prediction: Memory kernel K(t-τ) should be invariant under
    changes to the initial system state ρ(0).

    Environmental prediction: Memory effects depend on ρ(0) and system-bath
    correlations.
    """

    def __init__(self, config: StateIndependenceConfig):
        self.config = config

    def generate_initial_states(
        self, hilbert_dim: int = 2, state_type: str = "mixed"
    ) -> List[np.ndarray]:
        """Generate diverse initial states for testing.

        Args:
            hilbert_dim: Dimension of Hilbert space
            state_type: "pure", "mixed", or "random"

        Returns:
            List of density matrices
        """
        states = []

        if state_type == "pure":
            # Pure states on Bloch sphere
            for i in range(self.config.n_initial_states):
                theta = np.pi * i / (self.config.n_initial_states - 1)
                phi = 2 * np.pi * i / self.config.n_initial_states

                # Bloch vector
                n = np.array(
                    [
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta),
                    ]
                )

                # Pauli matrices
                sigma_x = np.array([[0, 1], [1, 0]])
                sigma_y = np.array([[0, -1j], [1j, 0]])
                sigma_z = np.array([[1, 0], [0, -1]])

                # Density matrix
                rho = 0.5 * (
                    np.eye(2) + n[0] * sigma_x + n[1] * sigma_y + n[2] * sigma_z
                )
                states.append(rho)

        elif state_type == "mixed":
            # Mixed states with varying purity
            for i in range(self.config.n_initial_states):
                purity = 0.5 + 0.5 * i / (self.config.n_initial_states - 1)

                # Random pure state
                psi = np.random.randn(hilbert_dim) + 1j * np.random.randn(hilbert_dim)
                psi = psi / np.linalg.norm(psi)
                rho_pure = np.outer(psi, psi.conj())

                # Mix with maximally mixed state
                rho_mixed = np.eye(hilbert_dim) / hilbert_dim
                rho = purity * rho_pure + (1 - purity) * rho_mixed

                states.append(rho)

        else:  # random
            for _ in range(self.config.n_initial_states):
                # Random density matrix (Ginibre ensemble)
                A = np.random.randn(hilbert_dim, hilbert_dim) + 1j * np.random.randn(
                    hilbert_dim, hilbert_dim
                )
                rho = A @ A.conj().T
                rho = rho / np.trace(rho)
                states.append(rho)

        return states

    def measure_memory_signature(
        self, evolution_operator: Callable, rho_initial: np.ndarray, times: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Measure memory-dependent quantities for a given initial state.

        Args:
            evolution_operator: Function (ρ, t) → evolved ρ
            rho_initial: Initial density matrix
            times: Time points

        Returns:
            Dictionary with memory signatures
        """
        signatures: Dict[str, List[float]] = {
            "purity": [],
            "entropy": [],
            "coherence": [],
            "trace_distance_from_initial": [],
        }

        rho_t = rho_initial.copy()

        for t in times:
            # Evolve
            rho_t = evolution_operator(rho_t, t)

            # Compute signatures
            # Purity: Tr(ρ²)
            purity = np.trace(rho_t @ rho_t).real
            signatures["purity"].append(purity)

            # Von Neumann entropy: -Tr(ρ log ρ)
            eigvals = np.linalg.eigvalsh(rho_t)
            eigvals = eigvals[eigvals > 1e-12]  # Remove numerical zeros
            entropy = -np.sum(eigvals * np.log2(eigvals))
            signatures["entropy"].append(entropy)

            # Coherence: Sum of off-diagonal elements
            coherence = np.sum(np.abs(rho_t - np.diag(np.diag(rho_t))))
            signatures["coherence"].append(coherence)

            # Trace distance from initial state
            diff = rho_t - rho_initial
            eigvals_diff = np.linalg.eigvalsh(diff)
            trace_dist = 0.5 * np.sum(np.abs(eigvals_diff))
            signatures["trace_distance_from_initial"].append(trace_dist)

        # Convert to arrays
        signatures_np: Dict[str, np.ndarray] = {
            key: np.array(values) for key, values in signatures.items()
        }

        return signatures_np

    def test_state_independence(
        self,
        evolution_operator: Callable,
        initial_states: Optional[List[np.ndarray]] = None,
        times: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Test if memory signatures are independent of initial state.

        Args:
            evolution_operator: Function (ρ, t) → evolved ρ
            initial_states: List of initial states (generated if None)
            times: Time points (default: [0, 0.1, ..., 1.0])

        Returns:
            Test results dictionary
        """
        # Generate initial states if not provided
        if initial_states is None:
            initial_states = self.generate_initial_states(
                hilbert_dim=2, state_type="mixed"
            )

        if times is None:
            times = np.linspace(0, 1.0, 11)

        print("State-Independence Test (Section 10.1)")
        print("=" * 70)
        print(f"Testing {len(initial_states)} initial states")
        print(f"Time points: {len(times)}")
        print()

        # Collect signatures for all initial states
        all_signatures = []
        for i, rho0 in enumerate(initial_states):
            sig = self.measure_memory_signature(evolution_operator, rho0, times)
            all_signatures.append(sig)
            print(
                f"State {i+1}/{len(initial_states)}: "
                f"Final purity = {sig['purity'][-1]:.4f}"
            )

        # Statistical analysis
        results = self._analyze_variance(all_signatures, times)

        # Visualization
        self._plot_results(all_signatures, times, results)

        return results

    def _analyze_variance(
        self, all_signatures: List[Dict[str, np.ndarray]], times: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze variance across initial states.

        State-independence implies low variance across initial states.
        """
        print("\n" + "=" * 70)
        print("STATISTICAL ANALYSIS")
        print("=" * 70)

        results: Dict[str, Dict[str, Any]] = {}

        for quantity in [
            "purity",
            "entropy",
            "coherence",
            "trace_distance_from_initial",
        ]:
            # Collect values at each time point
            values_per_time = []
            for t_idx in range(len(times)):
                values = [sig[quantity][t_idx] for sig in all_signatures]
                values_per_time.append(values)

            # Coefficient of variation at each time
            cvs = []
            for values in values_per_time:
                mean = np.mean(values)
                std = np.std(values)
                cv = std / mean if mean > 1e-10 else 0
                cvs.append(cv)

            # Average CV over time
            avg_cv = np.mean(cvs)
            max_cv = np.max(cvs)

            # ANOVA test: Are means significantly different?
            f_stat, p_value = f_oneway(*values_per_time)

            results[quantity] = {
                "values_per_time": values_per_time,
                "cv_avg": avg_cv,
                "cv_max": max_cv,
                "anova_f": f_stat,
                "anova_p": p_value,
                "state_independent": avg_cv < self.config.cv_threshold,
            }

            print(f"\n{quantity.replace('_', ' ').title()}:")
            print(f"  Average CV: {avg_cv:.4f}")
            print(f"  Max CV: {max_cv:.4f}")
            print(f"  ANOVA p-value: {p_value:.4e}")
            verdict = (
                "STATE-INDEPENDENT"
                if avg_cv < self.config.cv_threshold
                else "STATE-DEPENDENT"
            )
            print(f"  Verdict: {verdict}")

        # Overall verdict
        independent_count = sum(1 for r in results.values() if r["state_independent"])
        total_count = len(results)

        overall_independent = independent_count >= total_count * 0.75  # 75% threshold

        print("\n" + "=" * 70)
        print("OVERALL VERDICT")
        print("=" * 70)
        print(f"State-independent measures: {independent_count}/{total_count}")

        if overall_independent:
            verdict = "CONSISTENT WITH SPIRAL-TIME: Memory is state-independent"
        else:
            verdict = "ENVIRONMENTAL ORIGIN LIKELY: Memory is state-dependent"

        print(verdict)
        print("=" * 70)

        results["overall"] = {
            "verdict": verdict,
            "independent_count": independent_count,
            "total_count": total_count,
            "is_spiral_time": overall_independent,
        }

        return results

    def _plot_results(
        self,
        all_signatures: List[Dict[str, np.ndarray]],
        times: np.ndarray,
        results: Dict[str, Any],
    ):
        """Visualize state-independence test results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        quantities = ["purity", "entropy", "coherence", "trace_distance_from_initial"]

        for idx, quantity in enumerate(quantities):
            ax = axes[idx]

            # Plot all trajectories
            for i, sig in enumerate(all_signatures):
                ax.plot(times, sig[quantity], alpha=0.3, color="blue")

            # Plot mean and std
            values_per_time = results[quantity]["values_per_time"]
            means = [np.mean(v) for v in values_per_time]
            stds = [np.std(v) for v in values_per_time]

            ax.plot(times, means, "r-", linewidth=2, label="Mean")
            ax.fill_between(
                times,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.3,
                color="red",
                label="±1 std",
            )

            # Styling
            title = quantity.replace("_", " ").title()
            cv = results[quantity]["cv_avg"]
            verdict = (
                "State-Independent"
                if results[quantity]["state_independent"]
                else "State-Dependent"
            )

            ax.set_title(f"{title}\nCV = {cv:.3f} ({verdict})")
            ax.set_xlabel("Time")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.savefig("state_independence_test.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("\n✓ Figure saved: state_independence_test.png")


def compare_spiral_time_vs_environmental():
    """Demonstrate difference between Spiral-Time and environmental non-Markovianity."""
    print("\n" + "=" * 70)
    print("COMPARISON: Spiral-Time vs Environmental Non-Markovianity")
    print("=" * 70)

    config = StateIndependenceConfig(n_initial_states=8)
    tester = StateIndependenceTester(config)

    # Generate initial states
    initial_states = tester.generate_initial_states(hilbert_dim=2, state_type="mixed")
    times = np.linspace(0, 1.0, 11)

    # Spiral-Time evolution (state-independent kernel)
    def spiral_time_evolution(rho, t):
        """State-independent memory kernel."""
        # Simple exponential decay (state-independent)
        gamma = 0.2
        decay = np.exp(-gamma * t)

        # Dephasing channel (state-independent)
        rho_evolved = decay * rho + (1 - decay) * np.diag(np.diag(rho))
        return rho_evolved

    # Environmental evolution (state-dependent)
    def environmental_evolution(rho, t):
        """State-dependent environmental decoherence."""
        # Decay rate depends on initial purity
        purity = np.trace(rho @ rho).real
        gamma = 0.2 * purity  # State-dependent!

        decay = np.exp(-gamma * t)
        rho_evolved = decay * rho + (1 - decay) * np.diag(np.diag(rho))
        return rho_evolved

    print("\n--- SPIRAL-TIME EVOLUTION (state-independent) ---")
    results_st = tester.test_state_independence(
        spiral_time_evolution, initial_states, times
    )

    print("\n--- ENVIRONMENTAL EVOLUTION (state-dependent) ---")
    results_env = tester.test_state_independence(
        environmental_evolution, initial_states, times
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Spiral-Time verdict: {results_st['overall']['verdict']}")
    print(f"Environmental verdict: {results_env['overall']['verdict']}")


if __name__ == "__main__":
    compare_spiral_time_vs_environmental()
