"""
Born Rule Emergence from Resonance-Time Measure
================================================

Exploratory simulation demonstrating how Born-like quadratic weighting 
might emerge from time-integrated stability (Equations 12-14).

IMPORTANT: This is a PROPOSAL and EXPLORATION, not a complete derivation
or replacement for the Born rule. See Paper Section 6 for full context.

Reference: Paper Section 6 (Born Rule Emergence), Equations (12-14)

Author: Marcel Krüger & Don Michael Feeney Jr.  
License: MIT
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ResonanceConfig:
    """Configuration for resonance-time simulations."""

    n_modes: int = 3  # Number of outcome modes
    n_time_points: int = 1000  # Time resolution
    measurement_window: float = 0.1  # Coarse-graining window T
    stability_threshold: float = 0.9  # Threshold for "stable" resonance


def resonance_time_measure(
    psi_modes: List[np.ndarray],
    times: np.ndarray,
    config: Optional[ResonanceConfig] = None,
) -> np.ndarray:
    """Compute probability weights from resonance-time measure.

    Implements Equation (12) from paper:
        P_n ∝ ∫_{t₀}^{t₁} |ψ_n(t)|² dt

    PEDAGOGICAL: Demonstrates how time-integrated stability could yield
    quadratic weighting, but this is NOT a claim to replace Born rule.

    Args:
        psi_modes: List of mode amplitudes ψ_n(t)
        times: Time array
        config: Configuration settings

    Returns:
        Array of probability weights [P_1, P_2, ..., P_n]

    Example:
        >>> times = np.linspace(0, 1, 1000)
        >>> psi1 = np.exp(-times) * np.sin(10*times)
        >>> psi2 = np.exp(-times) * np.cos(10*times)
        >>> P = resonance_time_measure([psi1, psi2], times)
    """
    if config is None:
        config = ResonanceConfig()

    weights = []

    for psi in psi_modes:
        # Time-integrated squared amplitude
        integrand = np.abs(psi) ** 2
        weight = np.trapz(integrand, times)
        weights.append(weight)

    weights = np.array(weights)

    # Normalize: Σ P_n = 1
    weights = weights / np.sum(weights)

    return weights


def temporal_coarse_graining(
    psi: np.ndarray, times: np.ndarray, T_window: float
) -> float:
    """Demonstrate temporal coarse-graining → Born rule.

    Implements Equation (13):
        P_n = lim_{T→0} (1/T) ∫_{t₀}^{t₀+T} |ψ_n(t)|² dt

    For short windows with continuous ψ_n(t):
        P_n → |ψ_n(t₀)|²  (Equation 14)

    Args:
        psi: Mode amplitude array
        times: Time array
        T_window: Coarse-graining window

    Returns:
        Coarse-grained probability at t₀
    """
    # Find window around t=0 (or first point)
    t0_idx = 0
    window_mask = (times >= times[t0_idx]) & (times <= times[t0_idx] + T_window)

    times_window = times[window_mask]
    psi_window = psi[window_mask]

    if len(times_window) < 2:
        # Window too small, return instantaneous
        return np.abs(psi[t0_idx]) ** 2

    # Average over window
    integrand = np.abs(psi_window) ** 2
    P_coarse = np.trapz(integrand, times_window) / T_window

    return P_coarse


def demonstrate_born_emergence(
    n_modes: int = 3, T_measure: float = 0.1, show_plots: bool = True
) -> Dict[str, np.ndarray]:
    """Demonstrate Born rule emergence from resonance-time measure.

    PEDAGOGICAL DEMONSTRATION:
    1. Create stable resonance modes
    2. Compute time-integrated weights (Eq. 12)
    3. Show convergence to instantaneous |ψ|² (Eq. 13-14)

    Args:
        n_modes: Number of quantum modes
        T_measure: Measurement window duration
        show_plots: Whether to display visualizations

    Returns:
        Dictionary with results
    """
    print("Born Rule Emergence: Resonance-Time Demonstration")
    print("=" * 70)
    print("PEDAGOGICAL EXPLORATION - NOT a complete derivation")
    print()

    # Time array
    times = np.linspace(0, 2, 1000)

    # Create example modes with different "stability"
    # Mode amplitudes chosen to give different time-integrated weights
    modes = []
    mode_names = []

    for n in range(n_modes):
        # Each mode has different decay rate and oscillation
        decay_rate = 0.5 + n * 0.3
        freq = 5 + n * 2

        psi_n = np.exp(-decay_rate * times) * np.sin(freq * times)
        modes.append(psi_n)
        mode_names.append(f"Mode {n+1}")

    # 1. Time-integrated weights (Equation 12)
    P_resonance = resonance_time_measure(modes, times)

    # 2. Instantaneous |ψ|² at t=0 (Born rule)
    P_born = np.array([np.abs(psi[0]) ** 2 for psi in modes])
    P_born = P_born / np.sum(P_born)  # Normalize

    # 3. Coarse-graining convergence (Equation 13-14)
    T_windows = np.logspace(-2, 0, 20)  # 0.01 to 1.0
    convergence = []

    for T in T_windows:
        P_coarse = np.array([temporal_coarse_graining(psi, times, T) for psi in modes])
        P_coarse = P_coarse / np.sum(P_coarse)
        convergence.append(P_coarse)

    convergence = np.array(convergence)

    if show_plots:
        _plot_born_emergence(
            times, modes, mode_names, P_resonance, P_born, T_windows, convergence
        )

    # Summary
    print("\nResults:")
    print("-" * 70)
    print("Resonance-time weights (Eq. 12):")
    for i, P in enumerate(P_resonance):
        print(f"  Mode {i+1}: P = {P:.4f}")

    print("\nInstantaneous |ψ|² (Born rule):")
    for i, P in enumerate(P_born):
        print(f"  Mode {i+1}: P = {P:.4f}")

    print()
    print("Observation:")
    print("  As measurement window T → 0, resonance-time weights")
    print("  approach instantaneous |ψ(t₀)|² (Born rule)")
    print()
    print("IMPORTANT:")
    print("  This is a SCHEMATIC demonstration, not a claim that Eq. 12")
    print("  replaces the Born rule in all regimes. It shows how quadratic")
    print("  weighting could emerge from time-integrated stability.")
    print("=" * 70)

    return {
        "times": times,
        "modes": modes,
        "P_resonance": P_resonance,
        "P_born": P_born,
        "T_windows": T_windows,
        "convergence": convergence,
    }


def _plot_born_emergence(
    times: np.ndarray,
    modes: List[np.ndarray],
    mode_names: List[str],
    P_resonance: np.ndarray,
    P_born: np.ndarray,
    T_windows: np.ndarray,
    convergence: np.ndarray,
):
    """Create visualization of Born rule emergence."""
    plt.figure(figsize=(15, 10))

    # 1. Mode amplitudes over time
    ax1 = plt.subplot(2, 3, 1)
    for psi, name in zip(modes, mode_names):
        ax1.plot(times, psi, linewidth=2, label=name, alpha=0.7)
    ax1.set_xlabel("Time t", fontsize=11)
    ax1.set_ylabel("ψ_n(t)", fontsize=11)
    ax1.set_title("Resonance Modes", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="k", linewidth=0.5, alpha=0.5)

    # 2. Squared amplitudes
    ax2 = plt.subplot(2, 3, 2)
    for psi, name in zip(modes, mode_names):
        ax2.plot(times, np.abs(psi) ** 2, linewidth=2, label=name, alpha=0.7)
    ax2.set_xlabel("Time t", fontsize=11)
    ax2.set_ylabel("|ψ_n(t)|²", fontsize=11)
    ax2.set_title("Squared Amplitudes", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Probability comparison
    ax3 = plt.subplot(2, 3, 3)
    x = np.arange(len(modes))
    width = 0.35
    ax3.bar(
        x - width / 2,
        P_resonance,
        width,
        label="Resonance-time (Eq. 12)",
        alpha=0.8,
        color="steelblue",
    )
    ax3.bar(x + width / 2, P_born, width, label="Born |ψ|²", alpha=0.8, color="coral")
    ax3.set_xlabel("Mode", fontsize=11)
    ax3.set_ylabel("Probability", fontsize=11)
    ax3.set_title("Probability Comparison", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"n={i+1}" for i in range(len(modes))])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Convergence with window size
    ax4 = plt.subplot(2, 3, 4)
    for i, name in enumerate(mode_names):
        ax4.semilogx(
            T_windows,
            convergence[:, i],
            "o-",
            linewidth=2,
            markersize=6,
            label=name,
            alpha=0.7,
        )
        ax4.axhline(P_born[i], color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax4.set_xlabel("Measurement Window T", fontsize=11)
    ax4.set_ylabel("Coarse-grained P_n", fontsize=11)
    ax4.set_title(
        "Convergence to Born Rule (Eq. 13-14)", fontsize=12, fontweight="bold"
    )
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. Relative error vs window
    ax5 = plt.subplot(2, 3, 5)
    errors = np.abs(convergence - P_born) / P_born
    for i, name in enumerate(mode_names):
        ax5.loglog(
            T_windows,
            errors[:, i],
            "o-",
            linewidth=2,
            markersize=6,
            label=name,
            alpha=0.7,
        )
    ax5.set_xlabel("Measurement Window T", fontsize=11)
    ax5.set_ylabel("Relative Error |P_coarse - P_Born|/P_Born", fontsize=11)
    ax5.set_title("Error Scaling", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, which="both")

    # 6. Summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    summary_text = """
    Born Rule Emergence Summary
    ═══════════════════════════
    
    Equation (12): Resonance-Time
        Pₙ ∝ ∫|ψₙ(t)|² dt
        
    Equation (13): Temporal Coarse-Graining
        Pₙ = lim(T→0) (1/T) ∫|ψₙ(t)|² dt
        
    Equation (14): Convergence
        Pₙ → |ψₙ(t₀)|²  as T → 0
    
    Key Observation:
    • Time-integrated stability yields
      quadratic weighting
    • Compatible with Gleason's theorem
    • NOT claimed as complete replacement
      for Born rule in all regimes
    
    Status: PROPOSAL for how Born-like
    statistics could emerge from temporal
    structure rather than axiom
    """

    ax6.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
        transform=ax6.transAxes,
    )

    plt.tight_layout()
    plt.savefig("born_rule_emergence.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("✓ Figure saved: born_rule_emergence.png")


# Example usage
if __name__ == "__main__":
    print("PEDAGOGICAL SIMULATION: Born Rule Emergence")
    print("=" * 70)
    print("Exploratory demonstration - not a complete derivation")
    print()

    # Demonstrate with 3 modes
    results = demonstrate_born_emergence(n_modes=3, T_measure=0.1)

    print("\n" + "=" * 70)
    print("PEDAGOGICAL NOTE:")
    print("This demonstrates Paper Section 6 conceptually.")
    print("It is NOT a claim to replace the Born rule.")
    print("See Paper Section 6.1 for full theoretical context.")
    print("=" * 70)
