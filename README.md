# Spiral-Time Memory
## A Falsifiable Framework for Non-Markovian Temporal Structure in Quantum Mechanics

[![ResearchGate](https://img.shields.io/badge/ResearchGate-Spiral%20Time%20with%20Memory%20Paper-green?logo=researchgate)](https://www.researchgate.net/publication/399958489_Spiral-Time_with_Memory_as_a_Fundamental_Principle_From_Non-Markovian_Dynamics_to_Measurement_without_Collapse?channel=doi&linkId=69714718e806a472e6a50958&showFulltext=true)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Overview

This repository explores a minimal theoretical framework where **time itself carries intrinsic memory**, encoded in a triadic structure Ψ(t) = (t, φ(t), χ(t)). The proposal offers:

- **Dynamical alternative to the projection postulate** - Measurement emerges from memory stabilization
- **Explicit falsification criteria** - Process-tensor tomography with CP-divisibility tests
- **Experimental discrimination** - Three operational criteria distinguish intrinsic temporal memory from environmental decoherence
- **Reduction to standard QM** - Markovian limit recovered when χ → 0

> This repository is intended as a research and validation framework, not a production physics engine.

**Status**: Theoretical framework with outlined experimental tests. No empirical confirmation yet.

---

## Core Idea

Standard quantum mechanics treats time as a memoryless parameter. This work asks: *What if temporal dynamics are fundamentally non-Markovian?*

### The Framework

**Standard QM**: Time is an external parameter
```
ẋ(t) = F(x(t))                    # Present determines future
```

**Spiral-Time**: Time is a dynamical state with memory
```
Ψ(t) = t + iφ(t) + jχ(t)         # Triadic structure
ẋ(t) = F(x(t), ∫K(t-τ)x(τ)dτ)    # History matters
         └─ memory kernel
```

**Extended Hilbert Space**:
```
H_ext = H_sys ⊗ H_mem             # Memory sector encodes temporal history
```

Full evolution on H_ext is unitary; traced dynamics on H_sys is non-Markovian.

---

## Key Features

### Measurement Without Collapse

**Problem**: Projection postulate is ad hoc, not derivable from unitary evolution

**Solution**: Measurement = dynamical stabilization of temporal memory
```
χ(t⁺) = χ(t⁻) + Δχ_meas           # Memory sector stabilizes
```

Competing modes lose coherence relative to stabilized memory → effective "collapse" in coarse-grained descriptions while remaining continuous in (|ψ⟩, χ) space.

### Born Rule Emergence

**Standard approach**: |⟨n|ψ⟩|² is axiomatic

**Spiral-Time proposal**: Emerges from time-integrated stability
```
P_n = lim_{T→0} (1/T) ∫_{t₀}^{t₀+T} |ψ_n(t)|² dt → |ψ_n(t₀)|²
```

Compatible with Gleason's theorem via temporal coarse-graining measure.

### Experimental Discrimination from Environmental Decoherence

**Critical question**: How to distinguish from known environmental non-Markovianity?

**Three Operational Criteria**:

| Criterion | Environmental | Spiral-Time |
|-----------|---------------|-------------|
| **1. Memory Kernel** | State-dependent, coupled to bath | State-independent: K(t-τ) only |
| **2. CP-Divisibility** | Violated due to info backflow | Violated structurally in *isolated* systems |
| **3. Process Tensor** | Finite-rank (finite bath) | No finite-rank representation |

**Operational test**: Must satisfy ALL THREE criteria to claim Spiral-Time memory vs environmental effects.

**Sensitivity**: Predicted deviations O(εχ) ∼ 10⁻³–10⁻², within reach of current trapped-ion and superconducting platforms (resolution ~10⁻³).

---

## Falsification Criteria

### Primary Test: Process Tensor Tomography

**Null Hypothesis (Markovian dynamics)**:
```
ℰ_{t:0} = ℰ_{t:s} ∘ ℰ_{s:0}  with ℰ_{t:s} CP for all t ≥ s ≥ 0
```
*(Evolution factorizes into completely positive maps)*

**Spiral-Time Prediction**:
Even after controlling environment and reset protocols, reconstructed process tensors should exhibit:
- Statistically significant deviations from CP-divisible fits
- Outcome probabilities depending on intervention history
- Persistent non-Markovianity under controlled "reset" operations
- State-independent memory kernel K(t-τ)

### Experimental Protocols

#### Protocol A: Reset Test
- **Platform**: NV centers, trapped ions, superconducting qubits
- **Test**: Repeated measure–reset cycles with varied history sequences
- **Expected**: Residual history dependence beyond systematics (state-independent)
- **Falsification**: No history dependence → theory fails

#### Protocol B: Process Tensor Tomography
- **Method**: Reconstruct process over ≥3 time steps with intermediate operations
- **Test**: Does best-fit model require non-factorizing correlations?
- **Critical**: Test with multiple initial states → verify state-independence
- **Falsification**: All processes CP-divisible → theory fails

#### Protocol C: Leggett-Garg Under Interventions
- **Test**: Multi-time correlations with/without engineered memory suppression
- **Expected**: Violations persist under maximal isolation
- **Falsification**: Violations disappear with environment → environmental origin

### Formal Falsification Statement

**The Spiral-Time memory hypothesis is falsified if**:

Under controlled intervention and reset protocols, process-tensor reconstruction yields CP-divisible (Markovian) descriptions for all tested multi-time settings within experimental uncertainty, **OR** if observed non-Markovianity can be reproduced by finite environmental baths (fails discrimination criteria), **AND** no state-independent history-dependent deviations remain beyond systematics.

---

## Repository Structure

```
spiral-time-memory/
├── README.md                  # Project overview, scope, and navigation entry point
├── THEORY.md                  # Formal axioms, assumptions, protocols, falsification criteria
├── IMPLEMENTATIONS.md         # Rationale for non-unique implementations and equivalence classes
├── QUICKSTART.md              # 5-minute setup and minimal working example
├── requirements.txt           # Pinned runtime and analysis dependencies
│
├── theory/                    # Mathematical foundations
│   ├── operators.py           # Extended Hilbert space: H_ext = H_sys ⊗ H_mem
│   └── dynamics.py            # Non-Markovian evolution with explicit memory kernels
│
├── analysis/                  # Falsification and diagnostic tools
│   └── cp_divisibility.py     # Process-tensor CP-divisibility tests
│                               # + state-independence validation (NEW)
│
├── experiments/               # Protocol implementations
│   ├── reset_tests/
│   │   └── protocol_a.py      # Reset-based memory tests
│   ├── process_tensor/
│   │   └── protocol_b.py      # Process tensor reconstruction protocol
│   └── leggett_garg/
│       └── protocol_c.py      # Leggett–Garg inequality evaluation
│
├── tests/                     # Verification and unit tests
│   ├── test_operators.py
│   ├── test_dynamics.py
│   └── test_cp_divisibility.py
│
└── examples/                  # Interactive tutorials and demonstrations
    └── spiral_time_intro.ipynb # Conceptual walkthrough and usage example

```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/dfeen87/spiral-time-memory.git
cd spiral-time-memory

# Install dependencies
pip install -r requirements.txt

# Or install as editable package with dev tools
pip install -e ".[dev,notebooks]"

# Verify installation
pytest tests/ -v
```

### Run Your First Experiment

```python
import numpy as np
from theory.dynamics import MemoryKernelConfig, NonMarkovianEvolver

# Define system with memory
def F(x):
    """Harmonic oscillator"""
    return np.array([x[1], -x[0]])

def g(mem_int):
    """Memory coupling"""
    return np.array([0, -0.1 * mem_int[1]])

# Configure memory kernel
config = MemoryKernelConfig(kernel_type="exponential", tau_mem=1.0)

# Evolve with memory
evolver = NonMarkovianEvolver(F, g, config, dt=0.01)
x0 = np.array([1.0, 0.0])
times, states = evolver.evolve(x0, (0, 10))

print(f"Memory effect: amplitude decays from {states[0,0]:.3f} to {states[-1,0]:.3f}")
```

### Test Falsification Criterion

```python
from analysis.cp_divisibility import ProcessTensorReconstructor, ProcessTensorConfig

# Configure process tensor test
config = ProcessTensorConfig(n_timesteps=5, hilbert_dim=2, dt=0.1)
reconstructor = ProcessTensorReconstructor(config)

# Generate test process
channels = reconstructor.generate_test_process(memory_strength=0.3)
times = [i * config.dt for i in range(config.n_timesteps)]

# Test hypothesis
results = reconstructor.test_memory_hypothesis(channels, times)
print(results['verdict'])
# Output: "NON-MARKOVIAN (consistent with memory)" or "MARKOVIAN (theory falsified)"
```

### Run Experimental Protocol

```bash
# Protocol A: Reset test for history dependence
python experiments/reset_tests/protocol_a.py

# Or via make
make run-protocol-a
```

---

## Comparison with Standard QM

| Aspect | Standard QM | Spiral-Time |
|--------|-------------|-------------|
| **Time** | External parameter | Dynamical state Ψ = t + iφ + jχ |
| **State Space** | H_sys | H_ext = H_sys ⊗ H_mem |
| **Dynamics** | Markovian (state at t sufficient) | Non-Markovian (history dependence) |
| **Measurement** | Projection postulate (axiom) | Memory stabilization (dynamical) |
| **Born Rule** | Postulate | Temporal coarse-graining (proposal) |
| **Collapse** | Fundamental | Effective (continuous in extended space) |
| **Nonlocality** | Entanglement | Temporal coherence + memory |

**Reduction**: χ → 0 with vanishing memory coupling → standard Markovian QM recovered

---

## Scope and Limitations

### What This Proposes

- Falsifiable extension of quantum mechanics with specific experimental signatures
- Dynamical mechanism for measurement without projection postulate
- Reduction to standard physics in appropriate limits
- Three-part operational test distinguishing from environmental effects

### What This Does NOT Claim

- To replace quantum field theory (it's an embedding principle)
- That implementations here are unique or canonical (see [`IMPLEMENTATIONS.md`](IMPLEMENTATIONS.md))
- Complete derivation of GR or Standard Model masses (Eqs. 9-10 are effective relations)
- Empirical validation (experimental tests remain to be performed)

### Critical Disclaimers

**All code is illustrative, not definitive**:
- Memory kernel forms (exponential, power-law, Gaussian) are non-unique choices
- Discretization schemes are implementation-dependent
- Parameter values are for demonstration
- See [`IMPLEMENTATIONS.md`](IMPLEMENTATIONS.md) for detailed disclaimers

---

## Documentation

- **[`THEORY.md`](THEORY.md)** — Complete axioms, experimental protocols, falsification criteria
- **[`IMPLEMENTATIONS.md`](IMPLEMENTATIONS.md)** — Why code choices are non-unique (READ THIS)
- **[`QUICKSTART.md`](QUICKSTART.md)** — 5-minute getting started guide
- **[`examples/`](examples/)** — Interactive Jupyter notebooks

---

## Citation

If you use this work, please cite:

```
@article{kruger2025spiral,
  title = {Spiral-Time with Memory as a Fundamental Principle: 
           From Non-Markovian Dynamics to Measurement without Collapse},
  author = {Kr{\"u}ger, Marcel},
  year   = {2025},
  note   = {ResearchGate preprint, DOI: 10.13140/RG.2.2.27393.93280}
}
```

See [`CITATION.cff`](CITATION.cff) for structured citation metadata for this repository.

---

## Contributing

Contributions are welcome, especially:

### High Priority
- Experimental implementations of Protocols A-C on real hardware
- Alternative memory kernel formulations with theoretical justification
- CP-divisibility test refinements and validation
- State-independence test protocols (Section 10 criteria)

### Medium Priority
- Additional example notebooks and tutorials
- Performance optimizations for process tensor tomography
- Extended Hilbert space visualization tools
- Cross-platform experimental adapters

### Before Contributing
1. Read [`THEORY.md`](THEORY.md) for theoretical background
2. Read [`IMPLEMENTATIONS.md`](IMPLEMENTATIONS.md) for code philosophy
3. Run `make check` to verify code quality
4. Open an issue for discussion before major PRs

**Code style**: Run `make format` before committing (uses Black, isort)

---

## Development

### Common Commands

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Check code quality (format + lint + tests)
make check

# Run experimental protocol
make run-protocol-a

# Start Jupyter notebook
make notebook-server

# Clean build artifacts
make clean
```

### CI/CD

Automated testing on:
- Python 3.9, 3.10, 3.11, 3.12
- Linux, macOS, Windows
- Formatting (Black), linting (flake8), type checking (mypy)
- Coverage reporting

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

---

## Experimental Status

### Current State
- Theoretical framework complete
- Falsification criteria explicit
- Reference implementations provided
- Discrimination from environmental effects formalized
- Experimental validation pending

### Platforms of Interest
- **Trapped ions**: High-fidelity multi-time measurements
- **Superconducting qubits**: Fast reset protocols
- **NV centers**: Long coherence for memory tests
- **Photonic systems**: Process tensor tomography

### Expected Timeline
- **Near term (6-12 months)**: Protocol A on existing platforms
- **Medium term (1-2 years)**: Full process tensor tomography (Protocol B)
- **Long term (2-5 years)**: Comprehensive discrimination tests

---

## Frequently Asked Questions

### Is this a new interpretation of quantum mechanics?

No. This is a **structural hypothesis** with **falsifiable predictions**. If process tensors remain CP-divisible under controlled interventions, the theory fails. It's not about interpretation—it's about measurable signatures.

### How does this differ from environmental decoherence?

Three decisive criteria (Section 10 of paper):
1. Memory kernel is **state-independent** (not coupled to ρ(t))
2. CP-divisibility violated in **isolated systems** (no environment needed)
3. Process tensor has **no finite-rank representation** (unbounded temporal correlations)

Standard environmental effects fail at least one of these tests.

### What if all experiments show Markovian dynamics?

Then the theory is **falsified**. This is the point—the framework makes testable predictions that could be wrong.

### Are the code implementations canonical?

**No.** See [`IMPLEMENTATIONS.md`](IMPLEMENTATIONS.md). Memory kernel forms, discretization schemes, and numerical parameters are illustrative choices. Alternative implementations are equally valid.

### Can this replace quantum field theory?

No. It's an **embedding principle** that reduces to standard QFT in the χ → 0 limit. Think of it as a framework that could contain QFT as a special case, not a replacement.

### What about relativity?

Spacetime metric emerges as effective structure (Eq. 9: g_μν ∼ ∂_μΨ · ∂_νΨ). Full GR derivation not claimed—this is a consistency-level relation showing how metric could arise from triadic time-state variations.

---

## Support and Community

- **GitHub Issues**: Bug reports, feature requests, theoretical questions
- **Discussions**: Community forum for extended conversations
- **Email**: [dfeen87@gmail.com] for collaboration inquiries

### Getting Help

1. Check [`QUICKSTART.md`](QUICKSTART.md) for setup issues
2. Read [`THEORY.md`](THEORY.md) for theoretical questions
3. See [`IMPLEMENTATIONS.md`](IMPLEMENTATIONS.md) for code questions
4. Search existing GitHub Issues
5. Open new issue with minimal reproducible example

---

## Acknowledgments

This work builds on foundations in:
- Non-Markovian quantum dynamics (Breuer-Laine-Piilo, Rivas-Huelga-Plenio)
- Process tensor formalism (Pollock et al.)
- Quantum foundations (Gleason, Zurek, Leggett)

---

## License

MIT License - see [`LICENSE`](LICENSE) for details.

**In brief**: You can use, modify, and distribute this code freely, with attribution. No warranty is provided.

---

## Version History

- **v0.1.0** (January 2025): Initial public research release
  - Formal theoretical framework with explicit falsification criteria
  - Reference implementations for theory, analysis, and experimental protocols
  - Comprehensive unit test suite
  - Experimental Protocols A–C defined and documented
  - Operational criteria for discrimination from environmental effects

---

## Final Notes

### The Core Claim

**If time carries intrinsic memory**, then:
1. Measurement collapse may be replaced by dynamical memory stabilization
2. Born rule emerges from temporal coarse-graining
3. Multi-time quantum processes will exhibit **state-independent, structurally non-CP-divisible signatures** with no finite-rank representation

### The Test

Perform process tensor tomography under controlled interventions:
- If **all processes are CP-divisible** → theory falsified
- If **non-Markovianity is state-dependent** → environmental origin
- If **reproducible by finite bath** → not Spiral-Time intrinsic memory
- If **all three discrimination criteria pass** → consistent with Spiral-Time

### The Stakes

Either:
- Nature exhibits intrinsic temporal memory → measurement problem resolved dynamically
- Nature is fundamentally Markovian → back to projection postulate

**Let's find out.** 

---

**Repository Version**: 0.1.0  
**Paper Status**: DOI: 10.13140/RG.2.2.27393.93280

**License**: MIT  
**Last Updated**: January 2025

**Ready for collaboration, experimentation, and potential falsification.**
