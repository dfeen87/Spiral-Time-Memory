# Protocol A: Reset Test for History Dependence

## Overview

Protocol A tests whether nominally perfect reset operations can eliminate temporal memory effects. According to the spiral-time hypothesis, intrinsic temporal memory should persist even after ideal reset operations, leading to residual history dependence in measurement outcomes.

## Theoretical Background

### Standard Quantum Mechanics Prediction
In standard QM, a perfect reset operation should eliminate all correlations with past measurements:
```
ρ_reset = |0⟩⟨0| (with fidelity F ≈ 1)
P(outcome|history) → P(outcome) [history-independent]
```

### Spiral-Time Prediction
With intrinsic temporal memory (Eq. 11 from paper):
```
χ(t⁺) = χ(t⁻) + Δχ_meas
```
Even after system reset, the memory sector χ persists, causing:
```
P(outcome|history) ≠ P(outcome) [history-dependent]
```

## Experimental Protocol

### Setup
1. **Platform**: Single qubit system (NV center, trapped ion, or superconducting qubit)
2. **Operations**:
   - Preparation in |0⟩
   - Single-qubit measurements (computational or Hadamard basis)
   - High-fidelity reset operation (F > 99.9%)

### Procedure
For each measurement cycle:
1. Apply a specific **measurement history** (sequence of measurements/outcomes)
2. Perform **reset operation** (ideally returns system to |0⟩)
3. Perform **final measurement**
4. Record outcome and history

Repeat for multiple different histories and cycles (typically 100-1000 cycles, 5-20 different histories).

### Data Analysis

#### 1. Conditional Probability Computation
For each history sequence h, compute:
```
P(outcome=0|history=h) = (# of outcome=0 after history h) / (# of trials with history h)
```

#### 2. Statistical Test for History Independence
Use χ² test for independence:
- **Null hypothesis (H₀)**: P(outcome|history) is independent of history (Markovian)
- **Alternative (H₁)**: P(outcome|history) depends on history (non-Markovian)

Compute:
- Chi-squared statistic
- p-value
- Effect size (Cramér's V)

#### 3. Memory Persistence Metric
If memory sector χ is modeled, compute:
```
σ_χ = std(χ_values)
```
Higher σ_χ indicates stronger memory persistence.

### Falsification Criteria

**Spiral-time hypothesis is falsified if:**
```
p-value > 0.05 (no significant history dependence)
AND
Cramér's V < 0.001 (negligible effect size)
AND
σ_χ ≈ 0 (no memory persistence)
```
under controlled experimental conditions with systematic errors below statistical threshold.

## Implementation

### Quick Start
```python
from experiments.reset_tests.protocol_a import (
    ResetTestConfig, run_reset_test
)

# Configure experiment
config = ResetTestConfig(
    n_cycles=500,
    n_histories=10,
    history_length=5,
    reset_fidelity=0.999,
    memory_strength=0.02
)

# Run with spiral-time dynamics
results = run_reset_test(config, spiral_time_mode=True)

# Analyze results
print(f"p-value: {results['independence_test']['p_value']}")
print(f"History dependent: {results['independence_test']['significant']}")
```

### Key Classes

#### `ResetTestConfig`
Configuration parameters for the test:
- `n_cycles`: Number of measure-reset cycles
- `n_histories`: Number of different measurement histories
- `history_length`: Length of each history sequence
- `reset_fidelity`: Fidelity of reset operation
- `memory_strength`: ε_χ parameter (typically 0.001-0.05)

#### `QuantumState`
Represents a qubit with memory sector:
- `rho`: 2×2 density matrix
- `chi`: Memory sector value χ(t)

#### `ResetOperation`
Models reset with configurable fidelity:
- Standard mode: Resets both ρ and χ
- Spiral-time mode: Resets ρ but preserves χ

#### `HistoryDependenceAnalyzer`
Statistical analysis:
- Conditional probability computation
- χ² independence test
- Effect size estimation

## Expected Results

### Spiral-Time Mode
- Significant history dependence (p < 0.05)
- Non-zero Cramér's V (typically 0.01-0.10)
- Non-zero memory persistence σ_χ > 0

### Markovian Mode
- No significant history dependence (p > 0.05)
- Cramér's V ≈ 0
- Memory persistence σ_χ = 0

## Experimental Considerations

### Systematic Effects
Control for:
1. **Reset imperfections**: Characterize F_reset independently
2. **Measurement crosstalk**: Ensure measurements don't affect subsequent preparations
3. **Classical memory**: Verify no information stored in classical control system
4. **Environmental correlations**: Test with various decoherence rates

### Parameter Ranges
- Memory strength ε_χ: 10⁻³ to 10⁻²
- Reset fidelity: > 0.995
- Minimum cycles: 200 per history
- Minimum histories: 8

### Statistical Power
Required sample size for 80% power at α=0.05:
```
N ≈ (Z_α + Z_β)² × (1/p₁ + 1/p₂) / (p₁ - p₂)²
```
For detecting Δp ≈ 0.05: N ≈ 400 measurements per history

## References

- **Paper Section 10.5**: Protocol A specification
- **Paper Section 10**: Experimental discrimination criteria
- **Paper Equation 11**: Memory sector update rule

## Contact

For questions about implementation or experimental design, see main repository README.
