"""
Protocol B: Process Tensor Tomography and CP-Divisibility Test
===============================================================

Reconstructs multi-time quantum processes and tests whether they can be
described by CP-divisible (Markovian) dynamics. This is the decisive test
for intrinsic temporal memory.

Key Classes
-----------
ProcessTensorConfig : Configuration for tomography
ProcessTensor : Stores reconstructed process tensor
MemoryKernel : Implements intrinsic memory kernels K(t-τ)
NonMarkovianEvolution : Simulates evolution with memory (Eq. 18)
CPDivisibilityTest : Tests Markovianity via CP-divisibility
PauliOperators : Pauli basis for single-qubit systems

Main Function
-------------
run_protocol_b(config) : Execute Protocol B

Example
-------
>>> from experiments.process_tensor import ProcessTensorConfig, run_protocol_b
>>> config = ProcessTensorConfig(n_time_steps=3, memory_strength=0.02)
>>> results = run_protocol_b(config)
>>> print(f"CP-divisible: {results['cp_test_memory']['is_cp_divisible']}")
>>> print(f"Violations: {results['cp_test_memory']['n_violations']}")

See Also
--------
experiments.process_tensor.protocol_b : Full implementation
README.md : Detailed protocol documentation
Paper Section 9 : Process tensor formalism
"""

from .protocol_b import (
    ProcessTensorConfig,
    run_protocol_b,
    reconstruct_process_tensor,
    ProcessTensor,
    MemoryKernel,
    NonMarkovianEvolution,
    CPDivisibilityTest,
    PauliOperators
)

__all__ = [
    'ProcessTensorConfig',
    'run_protocol_b',
    'reconstruct_process_tensor',
    'ProcessTensor',
    'MemoryKernel',
    'NonMarkovianEvolution',
    'CPDivisibilityTest',
    'PauliOperators'
]
