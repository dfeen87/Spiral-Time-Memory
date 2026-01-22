# Implementation Disclaimer

## Non-Uniqueness and Non-Authority of Code

**Critical Notice**: The code in this repository is **illustrative, not definitive**.

---

## Why Implementations Are Non-Unique

### 1. Memory Kernel Freedom
The theory specifies that dynamics should include a memory kernel K(t-τ), but does **not** uniquely determine its form.

**Valid choices include**:
```python
# Exponential decay
K(s) = exp(-s/τ_mem)

# Power-law
K(s) = s^(-α)

# Gaussian
K(s) = exp(-s²/2σ²)

# Mittag-Leffler (fractional)
K(s) = E_α(-s^α)
```

**Our implementations** use exponential kernels for computational simplicity. This is **one choice among many**, not "the" spiral-time kernel.

### 2. Discretization Schemes
Continuous integral evolution:
```
ẋ(t) = F(x(t), ∫_{-∞}^t K(t-τ)x(τ)dτ)
```

requires discretization. Choices include:
- Trapezoidal rule
- Simpson's rule  
- Adaptive quadrature
- FFT-based convolution

Each affects numerical stability and accuracy differently. **Our choice** (trapezoidal) prioritizes transparency over performance.

### 3. Resonance-Time Measure
Equation (7) proposes:
```
P_n ∝ ∫|ψ_n(t)|² dt
```

**Implementation questions**:
- Integration window [t₀, t₁]?
- Threshold for "stable" resonance?
- Normalization convention?
- Multi-mode interference handling?

**Our approach**: Fixed window, ad-hoc threshold. **Not canonical**.

### 4. CP-Divisibility Tests
Multiple non-Markovianity witnesses exist:
- BLP measure (Breuer-Laine-Piilo)
- RHP measure (Rivas-Huelga-Plenio)
- Trace distance monotonicity
- Choi-matrix positivity checks

**Our analysis/** uses BLP for pedagogical clarity. Other measures may yield different sensitivity to memory effects.

---

## What Code Does and Does Not Represent

### The code DOES:
✓ Demonstrate that non-Markovian dynamics can be simulated  
✓ Provide pedagogical examples of memory-kernel integration  
✓ Offer templates for CP-divisibility testing  
✓ Illustrate conceptual differences from Markovian evolution  

### The code DOES NOT:
✗ Constitute "the" spiral-time implementation  
✗ Claim numerical results as predictions without error analysis  
✗ Assert uniqueness of algorithmic choices  
✗ Replace rigorous mathematical proofs (see [`paper/appendix.tex`](paper/appendix.tex))  

---

## Reproducibility vs. Canonicality

### Reproducibility ✓
All code includes:
- Fixed random seeds
- Explicit parameter values
- Version-pinned dependencies (`requirements.txt`)
- Unit tests for numerical stability

**You can reproduce our plots**. That does not mean they are unique outcomes of the theory.

### Non-Canonicality ✓
The theory constrains structure (non-Markovian, CP-indivisible) but not implementation details.

**Analogy**: General relativity specifies Einstein's equations. Numerical relativity codes (BSSN, CCZ4, generalized harmonic) all solve them but make different gauge and discretization choices. None is "the" GR implementation.

---

## How to Use This Code Responsibly

### ✓ Appropriate Uses
- Educational exploration of non-Markovian dynamics
- Testing CP-divisibility diagnostics on toy models
- Prototyping experimental data analysis pipelines
- Benchmarking alternative memory kernels

### ✗ Inappropriate Uses
- Claiming numerical outputs as "spiral-time predictions" without sensitivity analysis
- Treating kernel choices as empirically validated
- Citing simulation results as theoretical necessity
- Ignoring systematic uncertainties in discretization

---

## Reporting Results

If you extend or modify this code for publications, **clearly state**:

1. **Which kernel form** you chose and why
2. **Discretization scheme** and convergence tests
3. **Parameter ranges** explored and sensitivity analysis
4. **Systematic uncertainties** from numerical choices

**Example acceptable language**:
> "We implement one possible memory kernel K(s) = exp(-s/τ) as an illustrative case. Alternative kernel forms (power-law, Mittag-Leffler) remain to be explored."

**Example unacceptable language**:
> "Spiral-time predicts oscillations at 2.3 kHz." *(without acknowledging kernel/parameter dependence)*

---

## Alternative Implementations Welcome

We encourage community development of:
- Different memory kernel families
- Optimized numerical integrators
- GPU-accelerated process-tensor tomography
- Experimental data parsers for Protocols A-C

**Please**:
- Fork with acknowledgment
- Document your choices explicitly
- Submit PRs with clear change descriptions
- Open issues for conceptual questions before large refactors

---

## Philosophical Note

The theory is a **structural hypothesis** (non-Markovian temporal dynamics) with **falsification criteria** (process-tensor tests). 

Code is a **tool to explore consequences**, not "the theory itself."

**Mathematical rigor**: See [`paper/spiral_time.md`](paper/spiral_time.md)  
**Experimental grounding**: Protocols A-C in [`experiments/`](experiments/)  
**Numerical exploration**: This directory (with above caveats)

---

## Summary

| Aspect | Status |
|--------|--------|
| Memory kernel form | **Non-unique** (exp, power-law, etc.) |
| Discretization | **One choice among many** |
| CP-divisibility tests | **BLP used; others valid** |
| Numerical parameters | **Illustrative, not canonical** |
| Theory predictions | **Structure (non-Markovian), not numbers** |
| Code authority | **Educational tool, not definitive** |

**Bottom line**: Treat all code as "existence proof that this can be computed," not "the answer."

For theoretical substance, see [`THEORY.md`](THEORY.md) and [`paper/`](paper/).
