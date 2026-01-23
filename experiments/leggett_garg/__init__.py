"""
Protocol C: Leggett-Garg Inequality under Memory Suppression
=============================================================

Tests multi-time correlations using Leggett-Garg inequalities with and without
engineered memory suppression. Spiral-time predicts persistent violations even
under aggressive suppression.

Key Classes
-----------
LeggettGargConfig : Configuration for LGI tests
QuantumEvolution : Simulates evolution with memory effects
LeggettGargMeasurement : Projective measurements for LGI
LeggettGargInequality : Computes and tests LGI violations

Main Function
-------------
run_protocol_c(config) : Execute Protocol C with suppression comparison
run_leggett_garg_experiment(config, invasive=True) : Single LGI run

Example
-------
>>> from experiments.leggett_garg import LeggettGargConfig, run_protocol_c
>>> config = LeggettGargConfig(
...     n_measurements=5000,
...     memory_strength=0.03,
...     suppression_strength=0.7
... )
>>> results = run_protocol_c(config)
>>> K_no_supp = results['no_suppression']['K']
>>> K_with_supp = results['with_suppression']['K']
>>> print(f"K without suppression: {K_no_supp:.3f}")
>>> print(f"K with suppression: {K_with_supp:.3f}")
>>> print(f"Violation persists: {results['with_suppression']['test_result']['violates']}")

See Also
--------
experiments.leggett_garg.protocol_c : Full implementation
README.md : Detailed protocol documentation
Paper Section 10.5 : Protocol C specification
"""

from .protocol_c import (LeggettGargConfig, LeggettGargInequality,
                         LeggettGargMeasurement, QuantumEvolution,
                         run_leggett_garg_experiment, run_protocol_c)

__all__ = [
    "LeggettGargConfig",
    "run_protocol_c",
    "run_leggett_garg_experiment",
    "QuantumEvolution",
    "LeggettGargMeasurement",
    "LeggettGargInequality",
]
