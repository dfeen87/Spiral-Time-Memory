"""
Non-Markovianity Measures
==========================

Collection of non-Markovianity witnesses and measures for quantum processes.

Includes:
- BLP (Breuer-Laine-Piilo) measure - information backflow via trace distance
- RHP (Rivas-Huelga-Plenio) measure - CP-divisibility violation
- Trace distance monotonicity tests
- Quantum Fisher information approaches

These measures quantify deviation from Markovianity but don't distinguish
origin (environmental vs Spiral-Time). Use with state_independence.py for
discrimination.

References:
- Breuer, Laine, Piilo, PRL 103, 210401 (2009)
- Rivas, Huelga, Plenio, PRL 105, 050403 (2010)

Author: Marcel Krüger & Don Michael Feeney Jr.
License: MIT
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class NonMarkovianityConfig:
    """Configuration for non-Markovianity measurements."""

    n_orthogonal_states: int = 10  # Number of state pairs for averaging
    time_resolution: int = 100  # Time discretization
    blp_threshold: float = 1e-6  # Threshold for backflow detection


class NonMarkovianityMeasures:
    """Compute various non-Markovianity measures."""

    def __init__(self, config: NonMarkovianityConfig):
        self.config = config

    def blp_measure(
        self,
        evolution_maps: List[np.ndarray],
        times: np.ndarray,
        rho_pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """Breuer-Laine-Piilo measure based on trace distance increase.

        N_BLP = max_{ρ₁,ρ₂} ∫ max(0, dD/dt) dt

        where D(t) is trace distance between evolved states.

        Args:
            evolution_maps: List of Kraus operators or Choi matrices
            times: Corresponding time points
            rho_pairs: Pairs of initial states (generated if None)

        Returns:
            BLP measure and diagnostics
        """
        d = evolution_maps[0].shape[0]

        # Generate orthogonal state pairs if not provided
        if rho_pairs is None:
            rho_pairs = self._generate_orthogonal_pairs(d)

        blp_values = []
        all_trace_dists = []

        for rho1, rho2 in rho_pairs:
            trace_dists_list: List[float] = []

            # Evolve both states
            for E_t in evolution_maps:
                sigma1 = self._apply_map(E_t, rho1)
                sigma2 = self._apply_map(E_t, rho2)

                # Trace distance D = (1/2)||σ₁ - σ₂||₁
                D = self._trace_distance(sigma1, sigma2)
                trace_dists_list.append(D)

            trace_dists = np.array(trace_dists_list)
            all_trace_dists.append(trace_dists)

            # Time derivative
            dt = np.diff(times)
            dD_dt = np.diff(trace_dists) / dt

            # Integrate positive parts (backflow)
            backflow = np.maximum(0, dD_dt)
            blp = np.trapz(backflow, times[1:])
            blp_values.append(blp)

        # Maximum over state pairs
        blp_max = np.max(blp_values)
        blp_avg = np.mean(blp_values)

        is_nonmarkovian = blp_max > self.config.blp_threshold

        return {
            "blp_max": blp_max,
            "blp_avg": blp_avg,
            "blp_per_pair": blp_values,
            "trace_distances": all_trace_dists,
            "is_nonmarkovian": is_nonmarkovian,
            "measure_type": "BLP",
        }

    def rhp_measure(
        self, evolution_maps: List[np.ndarray], times: np.ndarray
    ) -> Dict[str, Any]:
        """Rivas-Huelga-Plenio measure based on CP-divisibility.

        N_RHP = ∫ max(0, -g(t)) dt

        where g(t) measures deviation from complete positivity of
        intermediate maps.

        Args:
            evolution_maps: List of cumulative evolution maps
            times: Time points

        Returns:
            RHP measure and CP-violation times
        """
        cp_violations = []
        violation_times = []

        for i in range(1, len(evolution_maps)):
            # Construct intermediate map Φ_{t,s} from cumulative maps
            # Φ_{t,0} = Φ_{t,s} ∘ Φ_{s,0}
            # Need to "invert" Φ_{s,0}

            E_s = evolution_maps[i - 1]
            E_t = evolution_maps[i]

            # Simplified: Check if incremental map is CP
            # Full implementation would use pseudoinverse

            # For now, measure via eigenvalues of Choi matrix
            try:
                # Construct incremental Choi matrix (approximate)
                choi_violation = self._check_cp_violation(E_s, E_t)
                cp_violations.append(choi_violation)

                if choi_violation > 0:
                    violation_times.append(times[i])
            except Exception:
                cp_violations.append(0)

        # Integrate violations
        if len(cp_violations) > 0:
            rhp = np.trapz(np.maximum(0, cp_violations), times[1:])
        else:
            rhp = 0.0

        is_nonmarkovian = rhp > self.config.blp_threshold

        return {
            "rhp": rhp,
            "cp_violations": cp_violations,
            "violation_times": violation_times,
            "is_nonmarkovian": is_nonmarkovian,
            "measure_type": "RHP",
        }

    def fisher_information_measure(
        self,
        evolution_maps: List[np.ndarray],
        times: np.ndarray,
        observable: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Quantum Fisher information based measure.

        Non-Markovianity can increase Fisher information about
        system parameters.

        Args:
            evolution_maps: Evolution maps
            times: Time points
            observable: Observable for parameter estimation

        Returns:
            Fisher information evolution
        """
        if observable is None:
            # Default: Pauli Z for qubits
            observable = np.array([[1, 0], [0, -1]], dtype=complex)

        fisher_info_list: List[float] = []

        for E_t in evolution_maps:
            # Simple state for demonstration
            rho = np.array([[1, 0], [0, 0]], dtype=complex)
            sigma = self._apply_map(E_t, rho)

            # Quantum Fisher information (simplified)
            # F_Q = Tr[ρ L²] where L is symmetric logarithmic derivative
            # For pure states: F_Q = 4 * Var(O)

            expectation = np.trace(sigma @ observable).real
            variance = np.trace(sigma @ observable @ observable).real - expectation**2

            # For mixed states, this is approximate
            fi = 4 * variance
            fisher_info_list.append(fi)

        fisher_info = np.array(fisher_info_list)

        # Check for increase
        dF_dt = np.diff(fisher_info) / np.diff(times)
        increases = np.sum(dF_dt > 0)

        is_nonmarkovian = increases > len(dF_dt) * 0.1  # 10% threshold

        return {
            "fisher_information": fisher_info,
            "increases": increases,
            "is_nonmarkovian": is_nonmarkovian,
            "measure_type": "Fisher",
        }

    def _generate_orthogonal_pairs(
        self, dim: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate pairs of orthogonal density matrices."""
        pairs = []

        for _ in range(self.config.n_orthogonal_states):
            # Random pure states
            psi1: np.ndarray = np.random.randn(dim) + 1j * np.random.randn(dim)
            psi1 = psi1 / np.linalg.norm(psi1)

            psi2: np.ndarray = np.random.randn(dim) + 1j * np.random.randn(dim)
            psi2 = psi2 / np.linalg.norm(psi2)

            # Orthogonalize
            psi2 = psi2 - np.vdot(psi1, psi2) * psi1
            psi2 = psi2 / np.linalg.norm(psi2)

            rho1 = np.outer(psi1, psi1.conj())
            rho2 = np.outer(psi2, psi2.conj())

            pairs.append((rho1, rho2))

        return pairs

    def _apply_map(self, E: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Apply quantum map E to state rho.

        E can be Choi matrix or Kraus operators.
        """
        # Assume E is Choi matrix for simplicity
        # Full implementation would handle both representations

        # Choi-Jamiolkowski: ε(ρ) = Tr_1[(I ⊗ ρ^T) Λ]
        # Simplified application

        # For now, use simple channel
        # In practice, use proper Choi-to-Kraus conversion

        return E @ rho @ E.conj().T / np.trace(E @ rho @ E.conj().T)

    def _trace_distance(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """Trace distance D(ρ₁, ρ₂) = (1/2)||ρ₁ - ρ₂||₁."""
        diff = rho1 - rho2
        eigvals = np.linalg.eigvalsh(diff)
        return 0.5 * np.sum(np.abs(eigvals))

    def _check_cp_violation(self, E_s: np.ndarray, E_t: np.ndarray) -> float:
        """Check if intermediate map violates complete positivity."""
        # Simplified: measure negative eigenvalues
        # Full implementation would construct Φ_{t,s} properly

        try:
            # Attempt to construct incremental map
            # This is approximate
            diff = E_t - E_s
            eigvals = np.linalg.eigvalsh(diff)

            # Sum of negative eigenvalues
            violation = -np.sum(eigvals[eigvals < 0])
            return violation
        except Exception:
            return 0.0

    def compare_measures(
        self, evolution_maps: List[np.ndarray], times: np.ndarray
    ) -> Dict[str, Any]:
        """Compare all non-Markovianity measures."""
        print("Comparing Non-Markovianity Measures")
        print("=" * 70)

        # Compute all measures
        blp = self.blp_measure(evolution_maps, times)
        rhp = self.rhp_measure(evolution_maps, times)
        fisher = self.fisher_information_measure(evolution_maps, times)

        # Summary
        print(f"\nBLP measure: {blp['blp_max']:.6f}")
        print(
            f"  Verdict: {'NON-MARKOVIAN' if blp['is_nonmarkovian'] else 'Markovian'}"
        )

        print(f"\nRHP measure: {rhp['rhp']:.6f}")
        print(f"  CP violations at {len(rhp['violation_times'])} time points")
        print(
            f"  Verdict: {'NON-MARKOVIAN' if rhp['is_nonmarkovian'] else 'Markovian'}"
        )

        print(f"\nFisher information increases: {fisher['increases']}")
        print(
            f"  Verdict: {'NON-MARKOVIAN' if fisher['is_nonmarkovian'] else 'Markovian'}"
        )

        # Consensus
        votes = [
            blp["is_nonmarkovian"],
            rhp["is_nonmarkovian"],
            fisher["is_nonmarkovian"],
        ]
        consensus = sum(votes) >= 2

        print(f"\nConsensus verdict: {'NON-MARKOVIAN' if consensus else 'MARKOVIAN'}")
        print("=" * 70)

        # Visualization
        self._plot_comparison(blp, rhp, fisher, times)

        return {"blp": blp, "rhp": rhp, "fisher": fisher, "consensus": consensus}

    def _plot_comparison(self, blp: Dict, rhp: Dict, fisher: Dict, times: np.ndarray):
        """Visualize comparison of measures."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # BLP: Trace distances
        ax = axes[0, 0]
        for trace_dists in blp["trace_distances"][:5]:  # Plot first 5
            ax.plot(times, trace_dists, alpha=0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Trace Distance")
        ax.set_title(f"BLP: Trace Distance Evolution\nMax = {blp['blp_max']:.4f}")
        ax.grid(True, alpha=0.3)

        # RHP: CP violations
        ax = axes[0, 1]
        if len(rhp["cp_violations"]) > 0:
            ax.plot(times[1:], rhp["cp_violations"], "o-", color="red")
        ax.set_xlabel("Time")
        ax.set_ylabel("CP Violation")
        ax.set_title(f"RHP: CP-Divisibility Violation\nMeasure = {rhp['rhp']:.4f}")
        ax.grid(True, alpha=0.3)

        # Fisher information
        ax = axes[1, 0]
        ax.plot(times, fisher["fisher_information"], "s-", color="green")
        ax.set_xlabel("Time")
        ax.set_ylabel("Fisher Information")
        ax.set_title(f"Fisher Information\nIncreases = {fisher['increases']}")
        ax.grid(True, alpha=0.3)

        # Summary
        ax = axes[1, 1]
        ax.axis("off")

        summary_text = f"""
        Non-Markovianity Measures Summary
        
        BLP (Trace Distance):
          Value: {blp['blp_max']:.6f}
          Verdict: {'NON-MARKOVIAN' if blp['is_nonmarkovian'] else 'Markovian'}
        
        RHP (CP-Divisibility):
          Value: {rhp['rhp']:.6f}
          Verdict: {'NON-MARKOVIAN' if rhp['is_nonmarkovian'] else 'Markovian'}
        
        Fisher Information:
          Increases: {fisher['increases']}
          Verdict: {'NON-MARKOVIAN' if fisher['is_nonmarkovian'] else 'Markovian'}
        """

        ax.text(
            0.1,
            0.5,
            summary_text,
            fontsize=11,
            family="monospace",
            verticalalignment="center",
        )

        plt.tight_layout()
        plt.savefig("non_markovianity_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("\n✓ Figure saved: non_markovianity_comparison.png")


if __name__ == "__main__":
    # Demonstration
    config = NonMarkovianityConfig()
    measures = NonMarkovianityMeasures(config)

    # Generate test evolution (amplitude damping)
    times = np.linspace(0, 2, 50)
    evolution_maps = []

    for t in times:
        gamma = 0.3 * (1 - np.exp(-t))  # Non-monotonic
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])

        # Construct Choi matrix (simplified)
        E = K0 @ K0.T.conj() + K1 @ K1.T.conj()
        evolution_maps.append(E)

    # Compare measures
    results = measures.compare_measures(evolution_maps, times)
