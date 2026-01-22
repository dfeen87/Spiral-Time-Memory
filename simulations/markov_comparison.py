"""
Markovian vs Memory Dynamics Visualization
===========================================

Illustrative comparison of Markovian (memoryless) evolution with 
spiral-time memory dynamics.

PEDAGOGICAL PURPOSE ONLY - NOT QUANTITATIVE PREDICTIONS

This module demonstrates how temporal memory affects dynamical trajectories,
energy dissipation, and phase space structure. The visualizations are meant
to build intuition for the concepts in Paper Section 3.

Reference: Paper Section 3 (Non-Markovian Dynamics), Equation (7)

Author: Marcel Krüger & Don Michael Feeney Jr.
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, Dict
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from theory.dynamics import (
    MemoryKernelConfig,
    NonMarkovianEvolver,
    compare_markov_vs_memory
)


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    figsize: Tuple[int, int] = (15, 5)
    dpi: int = 150
    style: str = 'seaborn-v0_8-darkgrid'
    save_figures: bool = True


def visualize_memory_effect(
    F: Callable,
    g: Callable,
    x0: np.ndarray,
    t_span: Tuple[float, float] = (0, 10),
    kernel_config: Optional[MemoryKernelConfig] = None,
    config: Optional[VisualizationConfig] = None
) -> Dict[str, np.ndarray]:
    """Visualize the effect of temporal memory on dynamics.
    
    Creates side-by-side comparison showing:
    1. Position trajectories (Markovian vs Memory)
    2. Phase space (position-velocity)
    3. Energy evolution
    
    PEDAGOGICAL: Demonstrates memory-induced damping without environment.
    
    Args:
        F: Markovian dynamics x → F(x)
        g: Memory coupling memory_int → g(memory_int)
        x0: Initial state
        t_span: Time interval
        kernel_config: Memory kernel configuration
        config: Visualization settings
        
    Returns:
        Dictionary with times, states_markov, states_memory
        
    Example:
        >>> # Harmonic oscillator
        >>> def F(x):
        ...     return np.array([x[1], -x[0]])
        >>> 
        >>> def g(mem):
        ...     return np.array([0, -0.1 * mem[1]])
        >>> 
        >>> visualize_memory_effect(F, g, np.array([1.0, 0.0]))
    """
    if kernel_config is None:
        kernel_config = MemoryKernelConfig(kernel_type="exponential", tau_mem=0.5)
    
    if config is None:
        config = VisualizationConfig()
    
    print("Visualizing Memory Effect on Dynamics")
    print("=" * 70)
    print("PEDAGOGICAL SIMULATION - Illustrative only")
    print()
    
    # Compare evolutions
    times, states_m, states_mem = compare_markov_vs_memory(
        x0, F, g, kernel_config, t_span, dt=0.01
    )
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=config.figsize)
    
    # 1. Position trajectories
    ax = axes[0]
    ax.plot(times, states_m[:, 0], 'b-', linewidth=2, label='Markovian', alpha=0.8)
    ax.plot(times, states_mem[:, 0], 'r--', linewidth=2, label='With Memory', alpha=0.8)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Position x(t)', fontsize=11)
    ax.set_title('Position Trajectories', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
    
    # 2. Phase space
    ax = axes[1]
    ax.plot(states_m[:, 0], states_m[:, 1], 'b-', linewidth=2, 
           label='Markovian', alpha=0.7)
    ax.plot(states_mem[:, 0], states_mem[:, 1], 'r--', linewidth=2, 
           label='With Memory', alpha=0.7)
    ax.plot(x0[0], x0[1], 'go', markersize=10, label='Initial', zorder=5)
    ax.set_xlabel('Position x', fontsize=11)
    ax.set_ylabel('Velocity v', fontsize=11)
    ax.set_title('Phase Space', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
    ax.set_aspect('equal')
    
    # 3. Energy evolution
    ax = axes[2]
    # Compute energy (assuming harmonic oscillator-like)
    E_m = 0.5 * (states_m[:, 0]**2 + states_m[:, 1]**2)
    E_mem = 0.5 * (states_mem[:, 0]**2 + states_mem[:, 1]**2)
    
    ax.plot(times, E_m, 'b-', linewidth=2, label='Markovian', alpha=0.8)
    ax.plot(times, E_mem, 'r--', linewidth=2, label='With Memory', alpha=0.8)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Energy E(t)', fontsize=11)
    ax.set_title('Energy Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if config.save_figures:
        plt.savefig('memory_effect_comparison.png', dpi=config.dpi, bbox_inches='tight')
        print(f"✓ Figure saved: memory_effect_comparison.png")
    
    plt.show()
    
    # Summary statistics
    print()
    print("Summary Statistics:")
    print("-" * 70)
    print(f"Initial energy: {E_m[0]:.4f}")
    print(f"Final energy (Markovian): {E_m[-1]:.4f} (conserved: {np.abs(E_m[-1] - E_m[0]) < 0.01})")
    print(f"Final energy (Memory): {E_mem[-1]:.4f} (dissipated: {E_m[0] - E_mem[-1]:.4f})")
    print()
    print("Interpretation:")
    print("  Memory of past states induces effective energy dissipation")
    print("  No explicit environment needed - memory alone causes damping")
    print("=" * 70)
    
    return {
        'times': times,
        'states_markov': states_m,
        'states_memory': states_mem,
        'energy_markov': E_m,
        'energy_memory': E_mem
    }


def compare_energy_dissipation(
    memory_strengths: np.ndarray,
    F: Callable,
    x0: np.ndarray,
    t_final: float = 10.0
) -> Dict[str, np.ndarray]:
    """Compare energy dissipation for different memory strengths.
    
    PEDAGOGICAL: Shows how memory coupling strength affects dissipation rate.
    
    Args:
        memory_strengths: Array of gamma values to test
        F: Markovian dynamics
        x0: Initial state
        t_final: Final time
        
    Returns:
        Dictionary with results
    """
    print("\nEnergy Dissipation vs Memory Strength")
    print("=" * 70)
    
    kernel_config = MemoryKernelConfig(kernel_type="exponential", tau_mem=0.5)
    
    final_energies = []
    
    for gamma in memory_strengths:
        def g(mem):
            return np.array([0, -gamma * mem[1]])
        
        _, _, states = compare_markov_vs_memory(
            x0, F, g, kernel_config, (0, t_final), dt=0.01
        )
        
        E_final = 0.5 * (states[-1, 0]**2 + states[-1, 1]**2)
        final_energies.append(E_final)
    
    final_energies = np.array(final_energies)
    E_initial = 0.5 * (x0[0]**2 + x0[1]**2)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(memory_strengths, final_energies / E_initial, 'o-', 
             linewidth=2, markersize=8, color='darkred')
    plt.axhline(1.0, color='blue', linestyle='--', linewidth=2, 
               label='Markovian (no dissipation)', alpha=0.7)
    plt.xlabel('Memory Coupling Strength γ', fontsize=12)
    plt.ylabel('Normalized Final Energy E(t_final)/E(0)', fontsize=12)
    plt.title('Energy Dissipation vs Memory Strength', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('energy_dissipation_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Figure saved: energy_dissipation_scaling.png")
    print()
    print("Observation: Stronger memory coupling → greater energy dissipation")
    print("=" * 70)
    
    return {
        'memory_strengths': memory_strengths,
        'final_energies': final_energies,
        'normalized': final_energies / E_initial
    }


def plot_phase_space_comparison(
    kernel_types: list,
    F: Callable,
    x0: np.ndarray,
    t_span: Tuple[float, float] = (0, 10)
):
    """Compare phase space trajectories for different memory kernels.
    
    PEDAGOGICAL: Shows how kernel choice affects dynamics.
    
    Args:
        kernel_types: List of kernel types to compare
        F: Markovian dynamics
        x0: Initial state
        t_span: Time interval
    """
    print("\nPhase Space Comparison: Different Memory Kernels")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, len(kernel_types) + 1, 
                            figsize=(5 * (len(kernel_types) + 1), 5))
    
    # Memory coupling
    def g(mem):
        return np.array([0, -0.2 * mem[1]])
    
    # Markovian reference
    ax = axes[0]
    config = MemoryKernelConfig(kernel_type="exponential", tau_mem=0.5)
    _, states_m, _ = compare_markov_vs_memory(
        x0, F, g, config, t_span, dt=0.01
    )
    ax.plot(states_m[:, 0], states_m[:, 1], 'b-', linewidth=2, alpha=0.8)
    ax.plot(x0[0], x0[1], 'go', markersize=10, zorder=5)
    ax.set_title('Markovian (Reference)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Velocity v')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Different kernels
    colors = ['red', 'purple', 'orange', 'green']
    
    for idx, kernel_type in enumerate(kernel_types):
        ax = axes[idx + 1]
        
        config = MemoryKernelConfig(kernel_type=kernel_type, tau_mem=0.5, alpha=0.5)
        _, _, states = compare_markov_vs_memory(
            x0, F, g, config, t_span, dt=0.01
        )
        
        ax.plot(states[:, 0], states[:, 1], color=colors[idx % len(colors)], 
               linewidth=2, alpha=0.8)
        ax.plot(x0[0], x0[1], 'go', markersize=10, zorder=5)
        ax.set_title(f'{kernel_type.replace("_", "-").title()}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Position x')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('phase_space_kernel_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Figure saved: phase_space_kernel_comparison.png")
    print()
    print("IMPORTANT: Kernel choice is NON-UNIQUE (see IMPLEMENTATIONS.md)")
    print("Different kernels → different phenomenology (all valid)")
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    print("PEDAGOGICAL SIMULATION: Markovian vs Memory Dynamics")
    print("=" * 70)
    print("Illustrative demonstration only - not quantitative predictions")
    print()
    
    # Simple harmonic oscillator
    def F_oscillator(x):
        """Undamped harmonic oscillator"""
        omega = 2 * np.pi  # 1 Hz
        return np.array([x[1], -omega**2 * x[0]])
    
    def g_memory(mem):
        """Memory-induced damping"""
        gamma = 0.5
        return np.array([0, -gamma * mem[1]])
    
    # Initial condition
    x0 = np.array([1.0, 0.0])
    
    # Main visualization
    results = visualize_memory_effect(F_oscillator, g_memory, x0, (0, 10))
    
    # Energy dissipation scaling
    gammas = np.linspace(0, 1.0, 10)
    dissipation_results = compare_energy_dissipation(gammas, F_oscillator, x0)
    
    # Kernel comparison
    kernels = ['exponential', 'power_law', 'gaussian']
    plot_phase_space_comparison(kernels, F_oscillator, x0)
    
    print("\n" + "=" * 70)
    print("PEDAGOGICAL NOTE:")
    print("These visualizations demonstrate concepts from Paper Section 3.")
    print("They are NOT predictions - see experiments/ for falsifiable tests.")
    print("=" * 70)
