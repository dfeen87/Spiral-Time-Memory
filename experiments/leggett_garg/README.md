# Protocol C: Leggett-Garg Inequality under Memory Suppression

## Overview

Protocol C tests multi-time correlations using Leggett-Garg inequalities (LGI) with and without engineered memory suppression protocols. The spiral-time hypothesis predicts that violations should persist even under aggressive memory suppression, distinguishing intrinsic temporal memory from environmental effects.

## Theoretical Background

### Leggett-Garg Inequalities

The LGI tests the assumption of **macrorealism**: a system has definite values at all times independent of measurement (realism per time, noninvasive measurability).

#### Three-Time LGI (K3)
For measurements at times t₁, t₂, t₃, define two-time correlators:
```
C(tᵢ, tⱼ) = ⟨Q(tᵢ) Q(tⱼ)⟩
```
where Q is a dichotomic observable (eigenvalues ±1).

The K3 parameter:
```
K3 = C(t₁,t₂) + C(t₂,t₃) - C(t₁,t₃)
```

**Classical bound** (macrorealism): -3 ≤ K3 ≤ 1  
**Quantum maximum**: K3 = 3/2 (for ideal system)

#### Four-Time LGI (K4)
```
K4 = C(t₁,t₂) + C(t₂,t₃) + C(t₃,t₄) - C(t₁,t₄)
```

**Classical bound**: -4 ≤ K4 ≤ 2

### Memory Suppression Hypothesis

**Environmental non-Markovianity**: Memory effects arise from system-environment correlations
→ **Prediction**: Memory suppression (dynamical decoupling, frequent resets) eliminates LGI violations

**Spiral-time intrinsic memory**: Memory is fundamental temporal structure
→ **Prediction**: Memory suppression reduces but does NOT eliminate LGI violations

From paper: "even under controlled 'reset' operations," non-Markovianity measures should persist (Section 10.5).

## Experimental Protocol

### Basic Setup

1. **System**: Single qubit (two-level system)
2. **Observable**: Typically σ_z (computational basis) or σ_x (Hadamard basis)
3. **Initial state**: Coherent superposition |+⟩ = (|0⟩ + |1⟩)/√2
4. **Evolution**: Free evolution or weak Hamiltonian (e.g., precession)

### Standard LGI Protocol (No Suppression)

For each run:
1. Prepare initial state |+⟩
2. **Time t₁**: Measure Q (projective measurement)
3. **Evolve** to t₂
4. **Time t₂**: Measure Q
5. **Evolve** to t₃
6. **Time t₃**: Measure Q
7. Record outcome triple (Q₁, Q₂, Q₃)

Repeat N times (typically N = 5,000-10,000).

Compute:
```
C(t₁,t₂) = (1/N) Σᵢ Q₁⁽ⁱ⁾ Q₂⁽ⁱ⁾
C(t₂,t₃) = (1/N) Σᵢ Q₂⁽ⁱ⁾ Q₃⁽ⁱ⁾
C(t₁,t₃) = (1/N) Σᵢ Q₁⁽ⁱ⁾ Q₃⁽ⁱ⁾

K3 = C(t₁,t₂) + C(t₂,t₃) - C(t₁,t₃)
```

### With Memory Suppression

Apply **dynamical decoupling** or **frequent resets** between measurements:

#### Method 1: Dynamical Decoupling
Insert π-pulse sequences (e.g., CPMG, XY-8) during evolution:
- Refocuses environmental noise
- Suppresses memory kernel coherence
- Preserves quantum coherence of system

**Implementation**:
```
t₁ → [DD sequence] → t₂ → [DD sequence] → t₃
```

DD suppression strength tunable via pulse spacing τ_DD.

#### Method 2: Frequent Memory Resets
Periodically reset the memory sector (if modeled explicitly):
```python
evolve(ρ, dt/2)
reset_memory(strength=α)  # α ∈ [0,1]
evolve(ρ, dt/2)
```

**Physical implementation**: Ancilla-assisted reset, measurement-based feedback, or thermal coupling.

### Measurement Protocol

To avoid invasive measurement issues:
- **Weak measurements**: Minimal backaction (reduces correlations but preserves K3 violation)
- **Non-invasive readout**: Use separate detector not strongly coupled to system
- **Ideal limit**: Assume measurements don't disturb subsequent evolution (theoretical limit)

### Data Collection

For **both** conditions (with/without suppression):
1. Collect N = 5,000-10,000 measurement sequences
2. Compute correlators C(tᵢ, tⱼ)
3. Calculate K3 or K4
4. Estimate uncertainties via bootstrapping

## Implementation

### Quick Start

```python
from experiments.leggett_garg.protocol_c import (
    LeggettGargConfig, run_protocol_c
)

# Configure experiment
config = LeggettGargConfig(
    n_measurements=5000,
    n_time_points=3,
    time_intervals=[1.0, 1.0],  # Δt between measurements
    memory_strength=0.03,
    memory_coherence_time=5.0,
    suppression_enabled=True,
    suppression_strength=0.7,  # 70% suppression
    suppression_method="dynamical_decoupling"
)

# Run comparison
results = run_protocol_c(config)

# Analyze
K_no_supp = results['no_suppression']['K']
K_with_supp = results['with_suppression']['K']
print(f"ΔK = {K_no_supp - K_with_supp:.4f}")
```

### Key Classes

#### `LeggettGargConfig`
Configuration parameters:
- `n_measurements`: Number of measurement runs
- `n_time_points`: Number of measurement times (3 or 4)
- `time_intervals`: List of time gaps [Δt₁, Δt₂, ...]
- `memory_strength`: ε_χ parameter
- `suppression_strength`: 0 = none, 1 = full suppression

#### `QuantumEvolution`
Simulates evolution with memory:
```python
evolver = QuantumEvolution(memory_strength, coherence_time)
rho_evolved = evolver.evolve_with_memory(rho, dt, suppression=0.5)
```

Memory kernel effect on off-diagonal elements:
```python
ρ_01 → ρ_01 × exp(-ε_χ × dt / τ_mem) × (1 - suppression)
```

#### `LeggettGargMeasurement`
Projective measurements:
- Observable: σ_z or σ_x
- Returns: outcome (±1) and post-measurement state
- Optional: weak measurement (partial collapse)

#### `LeggettGargInequality`
Analysis tools:
- `compute_K3()`: Three-time parameter
- `compute_K4()`: Four-time parameter  
- `test_violation()`: Statistical hypothesis test

## Expected Results

### Scenario 1: Environmental Memory
**Without suppression**: K3 > 1 (violation)  
**With suppression**: K3 ≤ 1 (no violation)  
**Conclusion**: Memory is environmental, can be suppressed

### Scenario 2: Spiral-Time Intrinsic Memory
**Without suppression**: K3 > 1 (violation)  
**With suppression**: K3 > 1 (persistent violation, possibly reduced)  
**Conclusion**: Memory is intrinsic to temporal structure

### Quantitative Predictions

For spiral-time with ε_χ = 0.03, τ_mem = 5:

| Suppression | K3 Expected | Violation? |
|-------------|-------------|------------|
| 0% (none) | 1.35 ± 0.05 | YES |
| 50% | 1.20 ± 0.05 | YES |
| 70% | 1.10 ± 0.05 | YES |
| 100% (full) | 0.95 ± 0.05 | NO |

**Key signature**: Gradual reduction, but violation persists up to ~90% suppression for intrinsic memory.

## Statistical Analysis

### Hypothesis Testing

**Null hypothesis (H₀)**: K3 ≤ 1 (macrorealism / no violation)  
**Alternative (H₁)**: K3 > 1 (violation)

**Test statistic**:
```
z = (K3 - 1) / SE(K3)
```
where SE = standard error from bootstrap.

**Decision**: Reject H₀ if z > z_α (e.g., z_0.05 = 1.645 for one-tailed test).

### Comparing Suppression Effects

**Test**: Is ΔK = K_no_supp - K_with_supp significantly different from zero?

**Method**: Paired bootstrap test or permutation test.

**Interpretation**:
- ΔK ≈ 0: No effect of suppression → intrinsic memory
- ΔK ≫ 0: Strong effect → environmental memory

### Sample Size Requirements

For detecting K3 = 1.2 with 80% power at α = 0.05:
```
N ≈ (z_α + z_β)² × σ² / (K3 - 1)²
```

Typical values:
- σ ≈ 0.1 → N ≈ 3,000 measurements
- For higher precision: N = 5,000-10,000

## Experimental Considerations

### Systematic Effects

1. **Measurement invasiveness**: 
   - Use weak measurements or post-selection
   - Characterize measurement backaction independently

2. **Imperfect time synchronization**:
   - Stabilize clocks to < 1% of evolution time
   - Verify consistency across measurement sequences

3. **State preparation errors**:
   - Prepare |+⟩ with fidelity F > 0.99
   - Tomographically verify initial state

4. **Decoherence**:
   - Measure T₁, T₂ independently
   - Ensure T₂ > 5 × (total experiment time)
   - Account for decay in correlator analysis

### Control Experiments

1. **Classical control**: Run with fully mixed states (should give K3 ≈ 0)
2. **Quantum control**: Run with pure states, no memory (should approach quantum bound)
3. **Suppression calibration**: Vary suppression strength parametrically
4. **Timing scan**: Vary time intervals to map memory decay

## Memory Suppression Techniques

### Dynamical Decoupling

**CPMG sequence** (Carr-Purcell-Meiboom-Gill):
```
τ/2 - π - τ - π - τ - ... - π - τ/2
```
N π-pulses applied with period τ.

**Effect**: Suppresses low-frequency noise and memory kernels with correlation time > τ.

**Implementation**: Use Y or X pulses with > 99% fidelity.

### XY-8 Sequence
More robust to pulse errors:
```
X - Y - X - Y - Y - X - Y - X
```
with equal spacing.

### Frequent Reset Protocol
Periodically couple to ancilla system:
- Ancilla initialized in pure state
- CNOT or SWAP with system
- Measure ancilla (collapses memory correlations)
- Discard ancilla measurement outcome

**Effect**: Breaks temporal correlations without fully decohering system state.

## Platform-Specific Notes

### Trapped Ions
- Excellent for LGI: long coherence, high-fidelity gates
- DD: Apply off-resonant laser pulses
- Memory reset: Use auxiliary ion in same trap

### Superconducting Qubits
- Fast operations allow many DD cycles
- DD: Microwave π-pulses at qubit frequency
- Memory reset: Couple to fast-relaxing cavity mode

### NV Centers
- Room temperature operation
- DD: Microwave or RF pulses (well-established)
- Memory reset: Optical repolarization or nearby nuclear spins

## Interpretation Guidelines

### Positive Result (Intrinsic Memory)
If violation persists under strong suppression:
1. Verify suppression is effective (via separate benchmarks)
2. Rule out classical information leakage
3. Quantify reduction vs. environmental models
4. Compare with CP-divisibility tests (Protocol B)

### Negative Result (Environmental Memory)
If violation is eliminated:
1. Characterize environmental noise spectrum
2. Test memory kernel models
3. Verify that suppression didn't over-decohere system
4. Consider whether suppression strength was sufficient

### Ambiguous Result
If violation is reduced but not eliminated:
1. Test intermediate suppression strengths
2. Perform process tensor tomography (Protocol B) for definitive test
3. Model mixed intrinsic + environmental contributions

## Falsification Criteria

**Spiral-time hypothesis is supported if:**
```
K3(no_suppression) > 1 + 3σ
AND
K3(with_suppression) > 1 + 2σ
AND
ΔK / K3(no_suppression) < 0.3
```

**Spiral-time hypothesis is challenged if:**
```
K3(with_suppression) < 1 + σ
```
under ≥70% suppression strength.

## References

- **Paper Section 10.5**: Protocol C specification
- Leggett & Garg, Phys. Rev. Lett. **54**, 857 (1985): Original LGI
- Emary et al., Rep. Prog. Phys. **77**, 016001 (2014): LGI review
- Kofler & Brukner, Phys. Rev. Lett. **101**, 090403 (2008): Quantum violation conditions

## Advanced Topics

### Weak Measurements
Partial measurements with strength α ∈ [0,1]:
```
ρ → √(1-α)ρ + α[M_outcome ρ M_outcome†]
```
Reduces invasiveness but maintains LGI testability.

### Contextuality Witnesses
Combine LGI with Kochen-Specker contextuality tests for stronger constraints on classical models.

### Continuous Monitoring
Replace discrete measurements with continuous weak monitoring:
```
dρ(t) = -i[H,ρ]dt + √η D[σ_z]ρ dt + √η H[σ_z]ρ dW
```
Derive K3 from filtered trajectories.

## Contact

For experimental protocols and data analysis questions, see main repository.
