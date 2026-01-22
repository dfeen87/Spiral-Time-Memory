# Protocol B: Process Tensor Tomography

## Overview

Protocol B reconstructs the complete multi-time quantum process using process tensor tomography and tests whether the dynamics can be described by a CP-divisible (Markovian) model. This is the **decisive test** for intrinsic temporal memory according to the paper.

## Theoretical Foundation

### Process Tensor Formalism
A quantum process over n time steps is characterized by a process tensor T that maps initial states and intermediate operations to final statistics:
```
P(outcomes) = Tr[T × (ρ₀ ⊗ O₁ ⊗ O₂ ⊗ ... ⊗ Oₙ)]
```

### CP-Divisibility (Markovianity)
**Definition**: A process is CP-divisible if it can be factorized into completely positive maps:
```
E_{t:0} = E_{t:s} ∘ E_{s:0}  for all t ≥ s ≥ 0
```
where each E_{t:s} is a completely positive (CP) map.

**Physical Meaning**: The future depends only on the present state, not on the past history.

### Spiral-Time Prediction

From paper Section 9.3:
> "If time carries intrinsic memory, then even after controlling environment and reset protocols, there exist multi-time interventions for which the reconstructed process tensor is not compatible with a factorized CP-divisible structure."

The memory kernel (Eq. 18) introduces explicit history dependence:
```
ρ̇(t) = L[ρ(t)] + ∫₀ᵗ K(t-τ) ρ(τ) dτ
```

**Key Distinction** (Section 10.2): This violation occurs even in dynamically isolated systems, unlike environmental non-Markovianity which requires system-bath coupling.

## Experimental Protocol

### Step 1: Process Tensor Reconstruction

#### Time Steps
Perform tomography over n = 3 or 4 time steps (t₀, t₁, t₂, ..., tₙ).

#### Operations at Each Step
Apply a complete tomographic set of operations:
- Identity (I)
- Pauli rotations (X, Y, Z)
- Or: Full single-qubit gate set

This generates 4ⁿ different operation sequences.

#### Measurement
For each operation sequence:
1. Prepare initial state ρ₀ = |0⟩⟨0|
2. Apply operation sequence O₁, O₂, ..., Oₙ at times t₁, t₂, ..., tₙ
3. Perform final measurement
4. Repeat N times (typically N = 1000 per sequence)

#### Reconstruction
Use maximum likelihood estimation or linear inversion to reconstruct process tensor T from measurement statistics.

### Step 2: CP-Divisibility Test

#### Method 1: Eigenvalue Non-Monotonicity
Test if the eigenvalues of reduced density matrices evolve monotonically:
```
λ_min(ρ(t)) should be non-increasing in time for CP-divisible processes
```

**Violation**: Non-monotonic eigenvalue evolution indicates non-CP-divisibility.

#### Method 2: Choi Matrix Positivity
For each intermediate time s, compute the Choi matrix of E_{t:s}:
```
Λ_{t:s} = (I ⊗ E_{t:s})[|Φ⁺⟩⟨Φ⁺|]
```

**Test**: Check if Λ_{t:s} is positive semidefinite for all t > s.

**Violation**: Negative eigenvalues indicate non-CP-divisibility.

#### Method 3: Divisibility Relation
Verify the factorization:
```
E_{t:0} ?= E_{t:s} ∘ E_{s:0}
```

**Test**: Compute both sides independently and compare (e.g., via trace distance).

**Violation**: Significant discrepancy indicates non-CP-divisibility.

### Step 3: Environmental Control

**Critical**: Must demonstrate that violations persist under maximal system isolation:
1. Cryogenic cooling (< 100 mK for superconducting qubits)
2. Magnetic field stabilization
3. Electromagnetic shielding
4. Controlled decoherence characterization

## Implementation

### Quick Start
```python
from experiments.process_tensor.protocol_b import (
    ProcessTensorConfig, run_protocol_b
)

# Configure tomography
config = ProcessTensorConfig(
    n_time_steps=3,
    n_measurements=1000,
    memory_strength=0.02,
    memory_kernel_type="exponential"
)

# Run protocol
results = run_protocol_b(config)

# Check CP-divisibility
print(f"CP-divisible: {results['cp_test_memory']['is_cp_divisible']}")
print(f"Violations: {results['cp_test_memory']['n_violations']}")
```

### Key Classes

#### `ProcessTensorConfig`
Configuration for tomography:
- `n_time_steps`: Number of measurement times (typically 3-4)
- `n_measurements`: Statistics per operation sequence
- `memory_kernel_type`: "exponential" or "power_law"
- `memory_strength`: ε_χ parameter

#### `MemoryKernel`
Implements intrinsic memory kernel K(t-τ):
- **Exponential**: K(Δt) = ε exp(-γΔt)
- **Power-law**: K(Δt) = ε/(1 + γΔt)

#### `NonMarkovianEvolution`
Simulates dynamics with memory (Eq. 18):
```python
evolver = NonMarkovianEvolution(kernel, dt=0.01)
rho_final = evolver.evolve(rho_init, t_final)
```

#### `ProcessTensor`
Stores reconstructed process tensor:
- Maps operation sequences → final states
- Computes predicted outcomes
- Provides access to intermediate states

#### `CPDivisibilityTest`
Tests CP-divisibility:
- Eigenvalue monotonicity check
- Choi matrix positivity test
- Factorization verification

## Expected Results

### Spiral-Time Dynamics (with memory)
- **CP-divisible**: NO
- **Violations**: Multiple (typically 2-5 for 3 time steps)
- **Type**: Eigenvalue non-monotonicity and/or negative Choi eigenvalues
- **Persistence**: Violations remain under environmental isolation

### Markovian Dynamics (no memory)
- **CP-divisible**: YES
- **Violations**: None (or only numerical artifacts < 10⁻⁶)
- **Type**: N/A
- **Environmental dependence**: Violations only appear with explicit bath coupling

## Falsification Criteria

From paper Section 9.2 (Criterion 1):

**The spiral-time memory hypothesis is falsified if:**
```
Under controlled intervention and reset protocols:
  1. Process tensor reconstruction yields CP-divisible description
     for ALL tested multi-time settings
  2. Experimental uncertainty is within bounds
  3. NO history-dependent deviations beyond systematics
```

**Quantitative thresholds**:
- CP-divisibility tested to precision δ_CP < 10⁻³
- Memory strength parameter: ε_χ ∈ [10⁻³, 10⁻²]
- Process tensor fidelity: F > 0.95

## Experimental Platforms

### Optimal Systems
1. **Trapped ions**: Excellent coherence, high-fidelity operations
2. **Superconducting qubits**: Fast operations, scalable
3. **NV centers**: Room temperature, long coherence

### Required Specifications
- Single-qubit gate fidelity: > 99.5%
- Readout fidelity: > 98%
- Coherence time: T₂ > 10 × (total process duration)
- Operation parallelism: Simultaneous control + readout

### Current State-of-the-Art
- Process tensor tomography demonstrated (Pollock et al., 2018)
- CP-divisibility tests achieved 10⁻³ precision (White et al., 2020)
- 3-4 time step reconstruction routine on multiple platforms

## Comparison with Environmental Non-Markovianity

| Feature | Environmental | Spiral-Time |
|---------|--------------|-------------|
| Origin | System-bath coupling | Intrinsic temporal structure |
| Memory kernel | State-dependent | **State-independent** |
| CP-divisibility | Violated with bath | **Violated in isolation** |
| Process tensor rank | Finite | **Unbounded** |
| Controllability | Engineered via bath | Intrinsic, not suppressible |

**Key diagnostic** (Section 10): Spiral-time violations persist even under:
- Perfect system isolation
- Maximal decoherence suppression  
- Environmental reset protocols

## Data Analysis Pipeline

1. **Raw data**: Measurement outcomes for each operation sequence
2. **Process tensor reconstruction**: Maximum likelihood or linear inversion
3. **Intermediate state extraction**: Tomography at each time step
4. **Choi matrix computation**: For each time interval
5. **CP tests**: Eigenvalue checks, factorization tests
6. **Statistical analysis**: Confidence intervals, hypothesis tests

### Statistical Validation
- Bootstrap resampling for uncertainty quantification
- Monte Carlo error propagation
- Hypothesis testing: H₀ = "process is CP-divisible"
- Multiple comparison correction (Bonferroni or FDR)

## References

- **Paper Section 9**: Process tensor as direct test
- **Paper Section 10**: Experimental discrimination protocol
- **Paper Equation 18**: Memory kernel dynamics
- Pollock et al., Phys. Rev. Lett. **120**, 040405 (2018): Process tensor formalism
- White et al., Nature Commun. **11**, 6301 (2020): Experimental process tomography

## Advanced Topics

### Multi-Qubit Extension
Extend to n-qubit systems:
- Process tensor dimension: 4^(2n × t_steps)
- Compressed sensing for efficient reconstruction
- Subsystem CP-divisibility tests

### Continuous-Variable Systems
Apply to bosonic systems (cavity QED, optomechanics):
- Wigner function process tomography
- Gaussian CP-divisibility tests
- Phase-space memory kernels

## Contact

For experimental implementation questions, see main repository or contact maintainers.
