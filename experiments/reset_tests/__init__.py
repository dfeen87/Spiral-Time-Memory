"""
Protocol A: Reset Test for History Dependence
==============================================

Tests whether nominally perfect reset operations can eliminate temporal memory
effects. Spiral-time predicts residual history dependence after reset.

Key Classes
-----------
ResetTestConfig : Configuration for reset tests
QuantumState : Qubit state with memory sector
ResetOperation : Models reset with configurable fidelity
MeasurementOperation : Projective measurements
HistoryDependenceAnalyzer : Statistical analysis tools

Main Function
-------------
run_reset_test(config, spiral_time_mode=True) : Execute Protocol A

Example
-------
>>> from experiments.reset_tests import ResetTestConfig, run_reset_test
>>> config = ResetTestConfig(n_cycles=500, memory_strength=0.02)
>>> results = run_reset_test(config, spiral_time_mode=True)
>>> print(f"History dependent: {results['independence_test']['significant']}")

See Also
--------
experiments.reset_tests.protocol_a : Full implementation
README.md : Detailed protocol documentation
"""

from .protocol_a import (
    HistoryDependenceAnalyzer,
    MeasurementOperation,
    QuantumState,
    ResetOperation,
    ResetTestConfig,
    run_reset_test,
)

__all__ = [
    "ResetTestConfig",
    "run_reset_test",
    "QuantumState",
    "ResetOperation",
    "MeasurementOperation",
    "HistoryDependenceAnalyzer",
]
