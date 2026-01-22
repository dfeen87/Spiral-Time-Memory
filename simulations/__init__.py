"""
Simulations Module: Illustrative Comparisons for Spiral-Time Framework
========================================================================

This module provides pedagogical visualizations and exploratory simulations
to illustrate key concepts of the spiral-time framework.

IMPORTANT DISCLAIMER:
--------------------
These simulations are ILLUSTRATIVE and PEDAGOGICAL only. They are NOT:
- Predictions from first principles
- Definitive numerical results
- Claims about specific parameter values
- Canonical implementations

See IMPLEMENTATIONS.md for detailed discussion of non-uniqueness.

Modules
-------

markov_comparison.py
    Side-by-side visualization of Markovian vs non-Markovian dynamics
    - Demonstrates memory effects on trajectories
    - Energy dissipation through temporal memory
    - Phase space evolution comparison
    - Visual illustrations only (not quantitative predictions)

resonance_statistics.py
    Exploration of Born rule emergence from resonance-time measure (Eq. 12-14)
    - Time-integrated stability → probability weights
    - Schematic demonstration of quadratic weighting emergence
    - Compatibility with Gleason's theorem
    - PROPOSAL only - not replacement for Born rule

Key Concepts Illustrated
-------------------------

1. Memory Effects:
   - How history dependence changes dynamics
   - Difference from Markovian evolution
   - Effective damping without environment

2. Born Rule Emergence:
   - Resonance-time measure P_n ∝ ∫|ψ_n(t)|² dt
   - Temporal coarse-graining → |ψ_n(t₀)|²
   - Schematic only - not complete derivation

3. Visualization Tools:
   - Phase space trajectories
   - Energy evolution
   - Probability distributions

Usage Examples
--------------

Compare Markovian vs memory dynamics:
    >>> from simulations import visualize_memory_effect
    >>> 
    >>> # Simple oscillator
    >>> def F(x):
    ...     return np.array([x[1], -x[0]])
    >>> 
    >>> def g(mem):
    ...     return np.array([0, -0.1 * mem[1]])
    >>> 
    >>> visualize_memory_effect(F, g, x0, t_span=(0, 10))

Explore Born rule emergence:
    >>> from simulations import demonstrate_born_emergence
    >>> 
    >>> demonstrate_born_emergence(n_modes=3, T_measure=0.1)

Pedagogical Notes
-----------------

These simulations help build intuition but should be interpreted carefully:

1. They use simplified toy models
2. Parameter choices are for illustration
3. They demonstrate concepts, not make predictions
4. Real experimental tests require Protocols A-C (see experiments/)

For rigorous theoretical foundations, see:
- Paper Section 6 (Born rule emergence)
- Paper Section 3 (Non-Markovian dynamics)
- THEORY.md (Complete framework)

For falsification criteria, see:
- analysis/cp_divisibility.py (Core tests)
- experiments/ (Protocols A-C)

References
----------
Paper Sections:
    - Section 3: Non-Markovian Dynamics
    - Section 6: Born Rule Emergence
    - Section 12: Discussion

Author: Marcel Krüger
License: MIT
"""

from .markov_comparison import (
    visualize_memory_effect,
    compare_energy_dissipation,
    plot_phase_space_comparison,
)

from .resonance_statistics import (
    demonstrate_born_emergence,
    resonance_time_measure,
    temporal_coarse_graining,
)

__all__ = [
    # Markovian comparison
    'visualize_memory_effect',
    'compare_energy_dissipation',
    'plot_phase_space_comparison',
    
    # Born rule emergence
    'demonstrate_born_emergence',
    'resonance_time_measure',
    'temporal_coarse_graining',
]

__version__ = '0.1.0'
__author__ = 'Marcel Krüger'

# Module-level pedagogical note
__pedagogical_note__ = """
IMPORTANT: All simulations in this module are illustrative demonstrations 
of theoretical concepts. They are NOT quantitative predictions or claims 
about specific physical systems. 

For experimental validation, see experiments/protocol_*.py
For theoretical rigor, see paper/spiral_time.md and THEORY.md
"""
