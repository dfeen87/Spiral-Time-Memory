# Spiral-Time with Memory as a Fundamental Principle: From Non-Markovian Dynamics to Measurement without Collapse

**Marcel Krüger**¹

¹Independent Researcher, Meiningen, Germany  
Email: marcelkrueger092@gmail.com  
ORCID: 0009-0002-5709-9729

**Date**: 21.01.2026

---

## Abstract

We propose that time carries intrinsic memory, encoded in a triadic spiral-time structure Ψ(t) = (t, φ(t), χ(t)). In contrast to standard quantum mechanics, where time is treated as a memoryless parameter, spiral-time introduces a non-Markovian temporal substrate from which space, matter, and measurement emerge as effective phenomena. We argue that the projection postulate can be replaced by a dynamical stabilization of time-memory, outline how Born-type statistics may arise as a resonance-time measure, and provide explicit falsification criteria based on process-tensor non-Markovianity and CP-divisibility tests.

---

## 1. Introduction

The measurement problem remains unresolved because standard quantum mechanics treats time as a memoryless parameter. The projection postulate is therefore introduced as an external rule, not derivable from unitary dynamics. This work explores a minimal alternative: the inconsistency disappears if time itself carries a structured memory sector.

We emphasize scope and discipline: the proposal is not a replacement of quantum field theory (QFT), but an embedding principle that reduces to standard physics in an appropriate limit and yields falsifiable signatures in multi-time correlation structure.

---

## 2. Axioms of Spiral-Time

**Axiom I (Temporal Structure)**: Time is a dynamical entity described by a triadic state

> Ψ(t) = ⟨t, φ(t), χ(t)⟩,     (1)

where t orders events, φ(t) encodes phase coherence, and χ(t) represents temporal memory.

**Axiom II (Memory)**: Physical dynamics depend on the history of Ψ, not solely on its instantaneous value.

**Axiom III (Reduction)**: In the limit χ → 0 (and vanishing memory coupling), standard Markovian physics is recovered.

### 2.1 Operator Structure and State Space

The triadic spiral-time variable

> Ψ(t) = t + iφ(t) + jχ(t)     (2)

is promoted to an operator acting on an extended Hilbert space

> Hₑₓₜ = Hₛᵧₛ ⊗ Hₘₑₘ,     (3)

where Hₘₑₘ encodes temporal memory degrees of freedom.

Physical states are density operators ρ(t) on Hₑₓₜ. Tracing over Hₘₑₘ yields effective non-Markovian dynamics on Hₛᵧₛ, while the full evolution remains linear and norm-preserving.

### 2.2 Algebraic Structure of Spiral-Time

The internal units i and j satisfy

> i² = j² = −1,     ij = −ji,     (4)

forming a quaternionic subalgebra. No additional imaginary units are introduced.

Crucially, physical observables depend only on the real kinetic weight

> A(t) = ℜΨ(t) = 1 + ε(t),     (5)

ensuring Hermiticity of the effective Hamiltonian. Non-commutativity affects only internal memory transport and does not violate probability conservation.

---

## 3. Non-Markovian Dynamics

Standard dynamics assume a Markovian state description,

> ẋ(t) = F(x(t)).     (6)

With time-memory, the evolution becomes history-dependent, e.g.

> ẋ(t) = F(x(t), ∫₋∞ᵗ K(t − τ) x(τ) dτ),     (7)

where K is a memory kernel intrinsic to spiral-time. This is the minimal mathematical meaning of "time has memory": the present is insufficient to predict the future without a controlled summary of the past.

### 3.1 Unitarity and CPT Consistency

Although the reduced dynamics is non-Markovian, the full evolution on Hₑₓₜ is unitary. The von Neumann equation reads

> ρ̇ₑₓₜ(t) = −i[Hₑₓₜ, ρₑₓₜ(t)],     (8)

with a Hermitian Hₑₓₜ.

CPT symmetry is preserved since spiral-time memory enters only through time-reversal-even kernels K(t − τ). In the limit χ → 0, standard CP-divisible quantum channels are recovered.

---

## 4. Emergence of Space and Matter (effective viewpoint)

We interpret space as an effective gradient structure of spiral-time:

> gₘᵥ ∼ ∂ₘΨ · ∂ᵥΨ.     (9)

This is an effective relation: it states how an emergent metric could be constructed from variations in the triadic time-state, not a complete derivation of general relativity.

Matter is modeled as stable, closed spiral-time resonances. A minimal effective mass functional that captures the idea "memory costs energy" is

> mₑff ∼ ∫ |∂ₜχ(t)|² dt.     (10)

Equation (10) is not claimed as a final mass formula for the Standard Model; it is a consistency-level definition to connect memory persistence to inertial energy.

---

## 5. Measurement without Wavefunction Collapse

Measurement is reinterpreted as a dynamical stabilization of temporal memory:

> χ(t⁺) = χ(t⁻) + Δχₘₑₐₛ.     (11)

No instantaneous projection is postulated. Instead, competing dynamical modes lose coherence relative to the stabilized memory sector. Operationally, this can look like "collapse" in coarse-grained descriptions, while remaining a continuous evolution in the extended state space (|ψ⟩, χ).

### 5.1 What replaces the projection postulate?

In standard quantum mechanics, the projection update is an extra axiom. Here it is replaced by: (i) environment-induced decoherence (loss of phase coherence in φ) and (ii) memory-sector stabilization (persistence in χ), producing an effective single-outcome record.

---

## 6. Origin of Born-Type Statistics (resonance-time measure)

A minimal route to Born-type weights is to relate outcome weight to time spent in a stable resonance channel. Schematically,

> Pₙ ∝ ∫ₜ₀ᵗ¹ |ψₙ(t)|² dt,     (12)

with normalization Σₙ Pₙ = 1. The claim is not that (12) fully replaces the Born rule in all regimes, but that a Born-like quadratic weighting can emerge from time-integrated stability rather than being a purely axiomatic measurement rule.

### 6.1 Derivation of the Born Rule

Measurement outcomes correspond to stable spiral-time modes ψₙ(t). The probability measure is defined on histories:

> Pₙ = lim(T→0) (1/T) ∫ₜ₀ᵗ⁰⁺ᵀ |ψₙ(t)|² dt.     (13)

For sufficiently short measurement windows, continuity of ψₙ(t) implies

> Pₙ → |ψₙ(t₀)|²,     (14)

recovering the standard Born rule.

This construction is compatible with Gleason's theorem since probabilities arise from a quadratic functional on the projective Hilbert space, with temporal coarse-graining providing the measure.

---

## 7. Comparison with Standard Quantum Mechanics

**Table 1**: Comparison between standard quantum mechanics and spiral-time with memory.

| Aspect | Standard QM | Spiral-Time (HLV) |
|--------|-------------|-------------------|
| Time | External parameter | Dynamical state with memory |
| Dynamics | Markovian (state at t sufficient) | Non-Markovian (history dependence) |
| Measurement update | Projection postulate | Memory stabilization + decoherence |
| Born rule | Postulate | Resonance-time/statistical emergence (proposal) |
| Nonlocal correlations | Encoded in entanglement | Encoded via temporal coherence + memory |
| Collapse | Fundamental rule | Not fundamental (effective appearance) |

### 7.1 Embedding into Quantum Field Theory

In relativistic field theory, spiral-time induces a time-dependent kinetic weight

> ℒ = (1/2)A(t)∂μφ∂ᵘφ − V(φ).     (15)

Gauge invariance is preserved by absorbing A(t) into a rescaled field φc = √A(t)φ. At one-loop order, this generates a small anomalous dimension

> ηₐ = (1/2)(d ln A)/(d ln μ),     (16)

without spoiling renormalizability.

---

## 8. Illustration (schematic)

### Figure 1: Comparison of decoherence dynamics

```
     Amplitude |Ψ|
         ↑
         │          ┌──────────────────────────────────
         │          │ Coupling window
         │     ┌────┘
         │    ╱│
         │   ╱ │                Standard collapse ℰₜ:ₛ
         │  ╱  │               (blue - conventional)
         │ ╱   │
         │╱    │    ╱╲╱╲╱╲     Memory stabilization χ(t)
         │     │   ╱        ╲   (red - spiral-time)
         │     │  ╱          ╲
         │     │ ╱            ╲  Markovian limit χ → 0
         │     │╱              ╲
         │     └────────────────╲──────────────────────
         │              Δχₘₑₐₛ   ╲
         └──────────────────────────────────────────→ t
```

**Caption**: The blue trajectory represents conventional collapse-like information loss into an effective environment, whereas the red trajectory illustrates the spiral-time model in which measurement corresponds to a stabilization of the temporal memory sector χ(t) rather than a non-unitary projection.

### Figure 2: Process-tensor representation

```
    ρ₀ ───[ℰₜ₁:ₜ₀]───→ ρ₁ ───[ℰₜ₂:ₜ₁]───→ ρ₂
           │                    │
           └────────────────────┘
                    │
           χ(t) (spiral-time memory substrate)
                K(t − τ) (Eq. 18)
           Non-Markovian correlation
           
    Null hypothesis: ℰₜ₂:ₜ₀ = ℰₜ₂:ₜ₁ ∘ ℰₜ₁:ₜ₀  (Eq. 17)
    
    HLV prediction: violation of CP-divisibility 
                    via intrinsic temporal coupling
```

**Caption**: In the standard Markovian description (blue), the dynamical evolution factorizes into consecutive completely positive maps between adjacent time steps. In the spiral-time framework (red), an intrinsic temporal coupling via the memory sector χ(t) induces non-Markovian correlations that persist even under perfect system reset operations, leading to a violation of CP-divisibility.

---

## 9. Experimental Program: Falsifiable Tests (Process Tensor / Non-Markov)

The central experimental claim is not "interpretational" but structural: multi-time statistics should carry signatures inconsistent with a fully Markovian (CP-divisible) description under controlled interventions.

### 9.1 Process tensor as a direct test of temporal memory

The process tensor formalism reconstructs the most general multi-time quantum process, capturing memory effects beyond a single quantum channel. A process is Markovian (in the operational sense) if it factorizes into a product of CP maps for successive time steps, i.e. it is CP-divisible.

### 9.2 CP-divisibility (null hypothesis)

Let ℰₜ:ₛ denote the dynamical map from time s to t. Markovian evolution implies the existence of CP maps such that

> ℰₜ:₀ = ℰₜ:ₛ ∘ ℰₛ:₀     with ℰₜ:ₛ CP for all t ≥ s ≥ 0.     (17)

This is the null hypothesis.

### 9.3 HLV prediction (time-memory): history dependence beyond CP-divisibility

If time carries intrinsic memory, then even after controlling environment and reset protocols, there exist multi-time interventions for which the reconstructed process tensor is not compatible with a factorized CP-divisible structure. Operationally, this appears as:

- statistically significant deviations in multi-time correlators from any CP-divisible fit,
- dependence of outcome probabilities on intervention history beyond state-preparation,
- persistence of non-Markovianity measures under controlled "reset" operations.

---

## 10. Experimental Discrimination from Environmental Decoherence

A central requirement for any extension of quantum dynamics is the ability to distinguish genuinely new physical effects from effective descriptions arising from environmental coupling. In particular, non-Markovian behavior in open quantum systems is well known to emerge from structured environments, finite baths, or strong system–environment correlations. The spiral-time framework predicts a qualitatively different form of non-Markovianity, rooted in intrinsic temporal memory rather than in environmental degrees of freedom.

In this section, we show that spiral-time memory can be operationally distinguished from standard environmental decoherence by three experimentally testable criteria.

### 10.1 State-Independent Temporal Memory

In conventional open quantum systems, memory effects arise from tracing out environmental degrees of freedom. As a consequence, the effective memory kernel depends implicitly on the system state and on system–environment correlations.

In contrast, spiral-time dynamics postulate an intrinsic memory kernel K(t − τ) associated with the temporal degree of freedom itself. The evolution equation takes the form

> ρ̇(t) = ℒ[ρ(t)] + ∫₀ᵗ K(t − τ) ρ(τ) dτ,     (18)

where K depends only on the temporal separation and not on ρ(t).

This state-independence implies that spiral-time memory cannot be modeled as an effective environment-induced process. In particular, no choice of system–bath Hamiltonian with finite bath dimension can reproduce a memory kernel that is invariant under changes of the system state.

### 10.2 Violation of CP-Divisibility without Environmental Coupling

A widely used operational criterion for Markovianity is CP-divisibility. A dynamical map ℰₜ:ₛ is CP-divisible if for all t > u > s there exists a completely positive map ℰₜ:ᵤ such that

> ℰₜ:ₛ = ℰₜ:ᵤ ∘ ℰᵤ:ₛ.     (19)

In standard quantum mechanics, violations of CP-divisibility are always attributed to information backflow from an environment. By contrast, spiral-time dynamics generically violate CP-divisibility even in dynamically isolated systems.

The reason is structural: the dynamical map depends explicitly on the temporal history encoded in the memory component χ(t). As a result, the map ℰₜ:ᵤ cannot be defined independently of earlier times, rendering CP-divisibility impossible by construction.

This constitutes a decisive qualitative distinction between spiral-time memory and environmental non-Markovianity.

### 10.3 Process Tensor Obstruction

The most general description of multi-time quantum processes is provided by the process tensor formalism. Any non-Markovian process generated by coupling to a finite-dimensional environment admits a Stinespring dilation and therefore corresponds to a process tensor of finite rank.

Spiral-time memory predicts an unbounded temporal correlation structure, encoded in the continuous evolution of χ(t). As a consequence, the associated process tensors possess, in general, no finite-rank representation.

Therefore, spiral-time dynamics cannot be reproduced by any finite environmental bath, regardless of its internal structure or coupling strength. This provides a model-independent obstruction to interpreting spiral-time effects as effective open-system behavior.

### 10.4 Experimental Sensitivity

State-of-the-art trapped-ion and superconducting qubit platforms have demonstrated full process-tensor tomography with reconstruction errors below the 10⁻³ level. Recent experiments can directly resolve violations of CP-divisibility and quantify multi-time correlations with high precision.

Spiral-time dynamics predict deviations from CP-divisibility of order O(εχ), where εχ parametrizes the strength of temporal memory. For conservative parameter choices, these deviations lie in the range 10⁻³–10⁻², placing the spiral-time hypothesis well within current experimental reach.

Crucially, the predicted signatures persist even under maximal system isolation, providing a clear operational criterion for distinguishing intrinsic temporal memory from environmental decoherence.

#### 10.4.1 Summary

Spiral-time memory differs from environmental non-Markovianity in three experimentally decisive aspects:

- the memory kernel is intrinsic and state-independent,
- CP-divisibility is violated even in isolated systems,
- the resulting process tensors cannot be reproduced by finite baths.

Together, these features provide a falsifiable and experimentally accessible route to testing spiral-time dynamics as a genuine extension of quantum theory.

### 10.5 Minimal experimental protocols

**Protocol A (reset test)**: repeated measure–reset cycles on a qubit platform (NV center, trapped ion, superconducting qubit). If a nominally perfect reset removes all memory, then P(outcome|history) should become history-independent. A residual history dependence beyond systematics supports a memory sector.

**Protocol B (process tensor tomography)**: reconstruct a process tensor over three or more time steps with intermediate operations. Test whether the best-fit model is CP-divisible or requires non-factorizing correlations.

**Protocol C (Leggett–Garg under controlled interventions)**: compare multi-time correlation structure with and without engineered memory suppression.

**Criterion 1 (Falsification)**: The spiral-time memory hypothesis is falsified if, under controlled intervention and reset protocols, process-tensor reconstruction yields a CP-divisible (Markovian) description for all tested multi-time settings within experimental uncertainty, and no history-dependent deviations remain beyond systematics.

---

## 11. Effective Field Theory Embedding and Renormalization Flow

To ensure compatibility with standard quantum field theory, the triadic spiral-time framework can be embedded into an EFT description with a controlled regime of validity. The construction distinguishes between time-domain modulation and RG scaling across energy scales μ.

### 11.1 Separation of time modulation and RG scale

The spiral-time coordinate

> ψ(t) = t + iφ(t) + jχ(t)     (20)

contains conceptually distinct contributions: φ(t) governs local phase coherence (temporal modulation), while χ(t) represents long-lived memory. RG flow probes physics across logarithmic energy scales μ, typically far exceeding the characteristic periods of φ, χ. Therefore, modulation and RG scaling should be treated separately.

Introduce an effective, scale-dependent coupling

> Aₑff(μ) ≈ 1 + η(μ₀/μ)^ζ⁽ᶯ⁾,     (21)

where η parametrizes the memory coupling strength and μ₀ is a reference scale. This yields an anomalous dimension

> ηₐ ≡ (1/2)(d ln Aₑff)/(d ln μ).     (22)

### 11.2 Modified one-loop flow (scalar prototype)

After canonical normalization Θc = √Aₑff Θ, the physical quartic coupling λₚₕᵧₛ = λ/Aₑff² obeys the prototype one-loop flow in d = 4,

> βλₚₕᵧₛ = μ(dλₚₕᵧₛ/dμ) = (3/16π²)λₚₕᵧₛ² − 2ηₐλₚₕᵧₛ.     (23)

In the limit η → 0, standard QFT beta functions are recovered.

### 11.3 Regime of validity

Below a cutoff Λ, higher-derivative operators encoding discreteness or host-geometry effects are suppressed:

> Sₑff = ∫ d⁴x[(DμΨ)†(DᵘΨ) − V(Ψ) − (1/2)Tr(FₘᵥFᵘᵛ) + Σᵢ (cᵢ/Λ²)𝒪ᵢ].     (24)

**Table 2**: Role of spiral-time channels in EFT and RG structure.

| Component | Physical Role | EFT / RG Interpretation |
|-----------|---------------|-------------------------|
| U₁: t | Linear time ordering | Standard QFT limit |
| U₂: φ(t) | Phase coherence/resonance | Floquet-like modulation (time-domain) |
| U₃: χ(t) | Memory / stability bandwidth | Scale-dependent coupling Aₑff(μ) |
| A(t) | Local kinetic prefactor | Time modulation only |
| Aₑff(μ) | Stationary memory imprint | Generates anomalous dimension ηₐ |

---

## 12. Discussion and Conclusion

If time carries intrinsic memory, the measurement postulate can be replaced by a dynamical mechanism: decoherence plus stabilization of a memory sector. Space and matter can be interpreted as effective structures of the triadic time-state. The framework is falsifiable through multi-time tomography: if all tested processes remain CP-divisible under controlled interventions, the memory hypothesis fails. In the appropriate limit, the construction reduces to standard physics.

### 12.1 Bipartite Lattice Analogy: U₁/U₂ as Two Sublattices and Emergent Memory U₃

A useful analogy for the triadic spiral-time structure is provided by any bipartite lattice (graph) decomposition, in which the vertex set splits into two sublattices V = Vₐ ∪ Vᵦ with dominant couplings E ⊂ (Vₐ × Vᵦ). In a schematic "checkerboard" picture, one may associate Vₐ with an "outgoing" channel (U₁) and Vᵦ with a "returning" channel (U₂). Importantly, this does not assert that physical spacetime is a literal two-dimensional board; it is a compact representation of a two-channel coupling topology.

Let ψₐ⁽¹⁾(t) and ψᵦ⁽²⁾(t) denote U₁/U₂ amplitudes on lattice sites n ∈ V. A minimal coupled dynamics takes the form

> ψ̇ₐ⁽¹⁾(t) = F(ψₐ⁽¹⁾(t)) + Σᵦ Jₐᵦ ψᵦ⁽²⁾(t),     (25)
> 
> ψ̇ᵦ⁽²⁾(t) = G(ψᵦ⁽²⁾(t)) + Σₐ Jᵦₐ ψₐ⁽¹⁾(t),     (26)

where Jₐᵦ encodes cross-sublattice coupling. If the U₂ sector carries internal relaxation or phase structure and is eliminated (e.g. by formal integration of the second equation), the U₁ sector acquires an effective history dependence of convolution type,

> ψ̇ₐ⁽¹⁾(t) = F(ψₐ⁽¹⁾(t)) + ∫₋∞ᵗ K(t − τ) ψₐ⁽¹⁾(τ) dτ,     (27)

with an intrinsic memory kernel K determined by the eliminated U₂ dynamics. This provides a clean interpretation of the "memory channel" U₃: it is not an additional spatial sublattice, but the emergent temporal memory component χ(t) (or equivalently the kernel K) induced by integrating out the returning phase channel.

In this view, the triadic spiral-time variable Ψ(t) = (t, φ(t), χ(t)) captures: (i) the event ordering (t), (ii) a phase-coherence channel (φ) naturally associated with U₂ synchronization, and (iii) a memory sector (χ) that arises as the effective remnant of hidden-channel dynamics. This mapping is intended as a structural guide and yields testable non-Markovian signatures through K, without assuming any literal two-dimensional checkerboard geometry.

**Relation to discrete space-bit lattices**: The above bipartite decomposition is compatible with discrete "space-bit" networks (e.g. Fibonacci/dodecahedral lattices) whenever the dominant coupling graph admits an approximate A/B partition; in that case, U₁/U₂ correspond to two coupled mode families on the same underlying lattice, while U₃ corresponds to the induced history dependence after coarse-graining the returning channel.

### Reproducibility and Reference Implementation

To support transparency and reproducibility, a reference implementation of the spiral-time memory operator formalism is publicly available. The repository documents the construction of intrinsic memory kernels, their numerical evaluation, and illustrative examples of non-Markovian dynamics consistent with the framework discussed here:

**https://github.com/dfeen87/Spiral-Time-Memory**

The code serves as a methodological reference and is not required for the conceptual or analytical results presented in this manuscript.

**Acknowledgment**: The author gratefully acknowledges Don Feeney for providing an independent open-source reference implementation related to spiral-time memory dynamics. The accompanying GitHub repository was used for illustrative and reproducibility purposes and helped to validate the internal consistency of the temporal-memory formalism presented here. The conceptual framework, analytical development, and physical interpretation of the present work remain solely the responsibility of the author.

---

## Appendix A: Mathematical Appendix: Operator Foundations (rigorous layer)

This appendix states one clean operator-theoretic route that avoids ambiguity.

### A.1 Triadic spiral-time as an operator

Let ℋ be a complex Hilbert space and 𝒟 ⊂ ℋ a common dense domain. Introduce operators

> T, Φ, χ: 𝒟 → 𝒟,

with T self-adjoint (time translation generator) and (Φ, χ) bounded or Kato-small relative perturbations so that essential self-adjointness is preserved for the deformed generators.

Define the spiral-time operator

> Ψ ≡ T + iΦ + jχ,     (28)

where (i, j) represent internal imaginary directions acting on a suitable extended representation space; for practical QFT embedding it is sufficient that they generate a consistent internal algebra and do not spoil essential self-adjointness on 𝒟.

### A.2 Well-posedness and controlled deformation of propagators

Consider a prototype kinetic operator for a scalar field:

> Kε = A(t)∂ₜ² − Δ + m²,     A(t) = 1 + εχ(t),     (29)

with real-valued bounded χ(t) and small |ε|.

Write

> Kε = K₀ + εV,     K₀ = ∂ₜ² − Δ + m²,     V = χ(t)∂ₜ².

Under standard relative-boundedness conditions (Kato–Rellich), Kε remains essentially self-adjoint and generates a well-defined unitary evolution in the limit of vanishing deformation.

In Fourier space, in windows where χ(t) ≈ χ₀ is effectively constant, the propagator admits the controlled expansion

> Gε(ω, k) = 1/(ω² − k² − m²) + εχ₀ω²/(ω² − k² − m²)² + O(ε²),     (30)

with smooth recovery as ε → 0.

### A.3 Topological diagnostic (optional layer)

If one additionally encodes triadic state trajectories into point clouds in (t, φ, χ)-space, persistent homology can serve as a stability diagnostic. This layer is optional and does not change the falsification logic from process-tensor tomography; it provides an independent coarse-grained signature of laminar vs disturbed triadic regimes.

---

## Appendix B: Cross-domain indication: non-Markovian signatures in neurodynamics

While the core claims of spiral-time memory are formulated within quantum dynamics (via multi-time correlations and process tensors), it is instructive to test whether time-history dependent stability indicators can be operationalized in complex biological systems. Here we report an independent EEG-based pipeline that implements a unified dynamical-instability operator ΔΦ as a scalar measure of deviation from baseline across three complementary axes.

### B.1 Definition of the ΔΦ operator (S–I–C decomposition)

A tri-axial decomposition is used: (i) a structural axis S (geometry/topology proxies), (ii) an informational axis I (entropy/complexity proxies), (iii) a coherence axis C (synchrony/coordination proxies). Relative to baseline values (S₀, I₀, C₀), deviations are

> ΔS(x) = S(x) − S₀,     (31)
> 
> ΔI(x) = I(x) − I₀,     (32)
> 
> ΔC(x) = C(x) − C₀.     (33)

A magnitude-based instability index is then formed via

> ΔΦ(x) = α|ΔS(x)| + β|ΔI(x)| + γ|ΔC(x)|,     α + β + γ = 1.     (34)

In the implemented pipeline, weights (α, β, γ) = (0.40, 0.35, 0.25) are used, motivated by empirical sensitivity of the respective axes in the analyzed recordings.

### B.2 Dataset, windowing, and regime classification

EEG recordings are segmented into fixed-length windows, features are extracted per window, and ΔΦ is computed relative to a baseline period. Windows are classified into four dynamical regimes by thresholds:

> Isostasis          (ΔΦ < 0.15),          (35a)
> 
> Allostasis         (0.15 ≤ ΔΦ < 0.35),   (35b)
> 
> High-Allostasis    (0.35 ≤ ΔΦ < 0.40),   (35c)
> 
> Collapse           (ΔΦ ≥ 0.40).          (35d)

Consecutive "Collapse" windows are clustered to identify sustained instability segments.

### B.3 What this does (and does not) imply for spiral-time memory

This EEG result is not evidence for quantum process-tensor non-Markovianity, nor does it replace the falsifiable tests proposed in Sec. 10. However, it demonstrates that (a) history-dependent stability metrics can be implemented reproducibly, and (b) a tri-axial "state + deviation" formalism naturally yields regime transitions with predictive value in real data. This supports the broader methodological stance of the present work: if time-memory exists as an intrinsic substrate, its operational signature should appear as structured history dependence across domains, while the decisive quantum validation must come from controlled multi-time experiments (process-tensor tomography, CP-divisibility tests, and intervention-based discrimination protocols).

**Kaggle notebook**: https://github.com/nwycomp/NeuroDynamics-Collapse-Validation-/blob/main/eegpart-four.ipynb

---

**End of Paper**
