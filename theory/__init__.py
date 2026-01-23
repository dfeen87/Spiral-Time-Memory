"""
Theory Module: Mathematical Foundations of Spiral-Time with Memory
===================================================================

This module implements the core mathematical structures of the spiral-time
framework as described in the paper.

Core Components
---------------

operators.py
    Extended Hilbert space formalism H_ext = H_sys ⊗ H_mem
    - Triadic spiral-time operator: Ψ = T + iΦ + jχ
    - Von Neumann evolution on extended space
    - Memory sector tracing
    - Quaternionic algebra (i² = j² = -1, ij = -ji)

dynamics.py
    Non-Markovian temporal evolution (Equation 3)
    - Memory kernel implementations: K(t-τ)
    - History-dependent evolution: ẋ(t) = F(x(t), ∫K(t-τ)x(τ)dτ)
    - Markovian limit verification (χ → 0)

Key Concepts
------------

Triadic Time Structure (Axiom I):
    Ψ(t) = t + iφ(t) + jχ(t)
    - t: Linear time ordering
    - φ(t): Phase coherence
    - χ(t): Temporal memory

Extended Hilbert Space (Axiom II):
    H_ext = H_sys ⊗ H_mem
    Full evolution is unitary on H_ext
    Traced evolution on H_sys is non-Markovian

Memory Kernel:
    K(t-τ): How past influences present
    - State-independent (distinguishes from environmental)
    - Multiple forms: exponential, power-law, Gaussian
    - Non-unique (see IMPLEMENTATIONS.md)

Reduction (Axiom III):
    χ → 0, vanishing memory coupling → Standard Markovian QM

Usage Examples
--------------

Basic non-Markovian evolution:
    >>> from theory import MemoryKernelConfig, NonMarkovianEvolver
    >>> 
    >>> # Define system
    >>> def F(x):
    ...     return np.array([x[1], -x[0]])  # Harmonic oscillator
    >>> 
    >>> def g(mem_int):
    ...     return np.array([0, -0.1 * mem_int[1]])  # Memory damping
    >>> 
    >>> # Configure memory
    >>> config = MemoryKernelConfig(kernel_type="exponential", tau_mem=1.0)
    >>> 
    >>> # Evolve
    >>> evolver = NonMarkovianEvolver(F, g, config, dt=0.01)
    >>> times, states = evolver.evolve(x0, (0, 10))

Extended Hilbert space:
    >>> from theory import SpiralTimeOperator, ExtendedHilbertConfig
    >>> 
    >>> config = ExtendedHilbertConfig(sys_dim=2, mem_dim=4, epsilon=0.1)
    >>> psi_op = SpiralTimeOperator(config)
    >>> 
    >>> # Evolve on extended space
    >>> times, rhos = psi_op.von_neumann_evolution(rho_ext_0, t_final=1.0, 
    ...                                             chi_trajectory=chi_traj)
    >>> 
    >>> # Trace over memory
    >>> rho_sys = psi_op.trace_over_memory(rhos[-1])

Compare Markovian vs memory:
    >>> from theory import compare_markov_vs_memory
    >>> 
    >>> times, x_markov, x_memory = compare_markov_vs_memory(
    ...     x0, F, g, config, (0, 10)
    ... )

References
----------
Paper Sections:
    - Section 2: Axioms of Spiral-Time
    - Section 3: Non-Markovian Dynamics
    - Appendix A: Operator Foundations

Important Disclaimers
---------------------
1. All implementations are ILLUSTRATIVE, not canonical
2. Memory kernel forms are NON-UNIQUE theoretical choices
3. Discretization schemes affect numerical accuracy but not theory
4. See IMPLEMENTATIONS.md for detailed discussion

Author: Marcel Krüger
License: MIT
"""

from .dynamics import (
    MemoryKernelConfig,
    NonMarkovianEvolver,
    compare_markov_vs_memory,
    memory_kernel,
)
from .operators import ExtendedHilbertConfig, SpiralTimeOperator

__all__ = [
    # Extended Hilbert space
    "SpiralTimeOperator",
    "ExtendedHilbertConfig",
    # Non-Markovian dynamics
    "MemoryKernelConfig",
    "memory_kernel",
    "NonMarkovianEvolver",
    "compare_markov_vs_memory",
]

__version__ = "0.1.0"
__author__ = "Marcel Krüger"

# Module-level documentation
__doc_sections__ = {
    "operators": "Extended Hilbert space H_ext = H_sys ⊗ H_mem",
    "dynamics": "Non-Markovian evolution with memory kernels",
}
