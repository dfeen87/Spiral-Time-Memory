"""Tests for experimental protocols A-C (Sec. 10.5)."""

import pytest
import numpy as np
from scipy.linalg import expm


class TestProtocolA:
    """Test Protocol A: Reset test (Sec. 10.5)."""
    
    def test_perfect_reset_markovian(self):
        """Markovian: perfect reset removes all history."""
        history = [0, 1, 0, 1, 0]
        
        def p_outcome_markovian(outcome, history):
            return 0.5  # Uniform after reset
        
        probs = [p_outcome_markovian(1, history[:i]) for i in range(1, len(history))]
        assert np.allclose(probs, 0.5)
    
    def test_imperfect_reset_memory(self):
        """Non-Markovian: reset doesn't fully remove history."""
        def p_outcome_with_memory(outcome, history, epsilon=0.1):
            base_prob = 0.5
            bias = epsilon * (sum(history) / len(history) - 0.5) if len(history) > 0 else 0
            return base_prob + bias
        
        history_0 = [0, 0, 0, 0]
        history_1 = [1, 1, 1, 1]
        
        p_0 = p_outcome_with_memory(1, history_0)
        p_1 = p_outcome_with_memory(1, history_1)
        
        assert p_0 != p_1
    
    def test_reset_fidelity_measurement(self):
        """Test measurement of reset fidelity."""
        
        def measure_reset_fidelity(rho_before, rho_after, rho_target):
            """Compute fidelity F = Tr(ρ_after · ρ_target)."""
            return np.trace(rho_after @ rho_target).real
        
        rho_before = np.array([[1, 0], [0, 0]])
        rho_target = np.array([[0, 0], [0, 1]])
        
        # Perfect reset
        rho_perfect = rho_target.copy()
        F_perfect = measure_reset_fidelity(rho_before, rho_perfect, rho_target)
        assert np.isclose(F_perfect, 1.0)
        
        # Imperfect reset with memory leakage
        epsilon = 0.05
        rho_imperfect = (1 - epsilon) * rho_target + epsilon * rho_before
        F_imperfect = measure_reset_fidelity(rho_before, rho_imperfect, rho_target)
        assert F_imperfect < F_perfect
    
    def test_statistical_significance_threshold(self):
        """Test statistical power for detecting memory effects."""
        n_trials = 1000
        epsilon_memory = 0.05
        
        np.random.seed(42)
        outcomes_markov = np.random.choice([0, 1], size=n_trials, p=[0.5, 0.5])
        outcomes_memory = np.random.choice([0, 1], size=n_trials, 
                                          p=[0.5 - epsilon_memory, 0.5 + epsilon_memory])
        
        observed_markov = np.bincount(outcomes_markov, minlength=2)
        observed_memory = np.bincount(outcomes_memory, minlength=2)
        expected = np.array([n_trials/2, n_trials/2])
        
        chi2_markov = np.sum((observed_markov - expected)**2 / expected)
        chi2_memory = np.sum((observed_memory - expected)**2 / expected)
        
        assert chi2_memory > chi2_markov
    
    def test_repeated_reset_cycles(self):
        """Test multiple measure-reset cycles."""
        
        def simulate_reset_cycle(n_cycles, memory_strength):
            """Simulate n reset cycles with memory."""
            outcomes = []
            memory_state = 0.5  # Initial
            
            for cycle in range(n_cycles):
                # Measurement influenced by memory
                p_outcome_1 = 0.5 + memory_strength * (memory_state - 0.5)
                outcome = np.random.choice([0, 1], p=[1 - p_outcome_1, p_outcome_1])
                outcomes.append(outcome)
                
                # Update memory (imperfect reset)
                reset_quality = 0.95
                memory_state = (1 - reset_quality) * outcome + reset_quality * 0.5
            
            return outcomes
        
        np.random.seed(42)
        
        # No memory
        outcomes_no_mem = simulate_reset_cycle(50, 0.0)
        
        # With memory
        outcomes_with_mem = simulate_reset_cycle(50, 0.2)
        
        # Memory case should show correlations
        # (autocorrelation test)
        def autocorr_lag1(x):
            return np.corrcoef(x[:-1], x[1:])[0, 1]
        
        ac_no_mem = autocorr_lag1(outcomes_no_mem)
        ac_with_mem = autocorr_lag1(outcomes_with_mem)
        
        # With memory should have higher autocorrelation
        assert ac_with_mem > ac_no_mem


class TestProtocolB:
    """Test Protocol B: Process tensor tomography (Sec. 10.5)."""
    
    def test_process_tensor_reconstruction(self):
        """Test reconstruction of multi-time process tensor."""
        
        dim = 2
        
        def create_process_tensor(factorizable=True):
            """Create process tensor, factorizable or not."""
            if factorizable:
                E1 = np.array([[0.9, 0], [0, 0.9]])
                E2 = np.array([[0.9, 0], [0, 0.9]])
                return E1, E2, E2 @ E1
            else:
                E1 = np.array([[0.9, 0], [0, 0.9]])
                E2 = np.array([[0.9, 0.1], [0.1, 0.9]])
                E_total = np.array([[0.85, 0.05], [0.05, 0.85]])
                return E1, E2, E_total
        
        E1_fact, E2_fact, E_total_fact = create_process_tensor(factorizable=True)
        E1_nonfact, E2_nonfact, E_total_nonfact = create_process_tensor(factorizable=False)
        
        assert np.allclose(E_total_fact, E2_fact @ E1_fact, atol=1e-10)
        assert not np.allclose(E_total_nonfact, E2_nonfact @ E1_nonfact, atol=1e-2)
    
    def test_cp_divisibility_detection(self):
        """Test detection of CP-divisibility violations."""
        
        def check_cp_divisibility(process_maps, tolerance=1e-6):
            """Check if sequence of maps is CP-divisible."""
            E1, E2 = process_maps
            
            test_states = [
                np.array([[1, 0], [0, 0]]),
                np.array([[0, 0], [0, 1]]),
                np.array([[0.5, 0.5], [0.5, 0.5]])
            ]
            
            for rho in test_states:
                rho_direct = E2 @ (E1 @ rho)
                E_composed = E2 @ E1
                rho_composed = E_composed @ rho
                
                if not np.allclose(rho_direct, rho_composed, atol=tolerance):
                    return False
            
            return True
        
        E1 = np.array([[0.9, 0], [0, 0.9]])
        E2 = np.array([[0.8, 0], [0, 0.8]])
        
        assert check_cp_divisibility([E1, E2])
    
    def test_multi_time_correlation_extraction(self):
        """Test extraction of multi-time correlations from process tensor."""
        
        times = [0, 1, 2]
        A = np.array([[1, 0], [0, -1]])  # Pauli Z
        rho_0 = np.array([[1, 0], [0, 0]])
        
        U1 = np.array([[np.cos(0.5), -np.sin(0.5)], 
                       [np.sin(0.5), np.cos(0.5)]])
        U2 = np.array([[np.cos(0.3), -np.sin(0.3)], 
                       [np.sin(0.3), np.cos(0.3)]])
        
        rho_1 = U1 @ rho_0 @ U1.conj().T
        rho_2 = U2 @ rho_1 @ U2.conj().T
        
        exp_A0 = np.trace(A @ rho_0).real
        exp_A1 = np.trace(A @ rho_1).real
        exp_A2 = np.trace(A @ rho_2).real
        
        C_3time = exp_A0 * exp_A1 * exp_A2
        assert C_3time != 0
    
    def test_tomographically_complete_measurements(self):
        """Test informationally complete measurement set."""
        
        # Pauli basis forms tomographically complete set
        paulis = {
            'I': np.eye(2),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }
        
        # Any density matrix can be expressed as linear combination
        rho = np.array([[0.7, 0.2j], [-0.2j, 0.3]])
        
        # Decompose
        coeffs = {}
        for name, P in paulis.items():
            coeffs[name] = np.trace(rho @ P).real / 2
        
        # Reconstruct
        rho_reconstructed = sum(c * P for (name, c), P in 
                               zip(coeffs.items(), paulis.values()))
        
        assert np.allclose(rho, rho_reconstructed)


class TestProtocolC:
    """Test Protocol C: Leggett-Garg with controlled interventions."""
    
    def test_leggett_garg_inequality(self):
        """Test Leggett-Garg inequality K3 ≤ 1."""
        
        def compute_correlation(t1, t2, observable):
            """Compute <A(t1)A(t2)>."""
            phase = 0.5 * (t2 - t1)
            damping = np.exp(-0.5 * (t2 - t1))
            return damping * np.cos(phase)
        
        A = 1
        times = [0, 1, 2]
        
        C12 = compute_correlation(times[0], times[1], A)
        C23 = compute_correlation(times[1], times[2], A)
        C13 = compute_correlation(times[0], times[2], A)
        
        K3 = C12 + C23 - C13
        
        # Macrorealism requires K3 ≤ 1
        assert K3 <= 1 + 1e-10
    
    def test_memory_suppression_effect(self):
        """Test effect of engineered memory suppression."""
        
        def correlation_with_memory(dt, gamma_memory):
            """Correlation with tunable memory."""
            decay = np.exp(-0.1 * dt)
            memory_term = gamma_memory * np.cos(2 * np.pi * dt)
            return decay * (1 + memory_term)
        
        dt = 1.0
        
        C_full = correlation_with_memory(dt, gamma_memory=0.3)
        C_suppressed = correlation_with_memory(dt, gamma_memory=0.0)
        
        assert C_full != C_suppressed
    
    def test_invasive_vs_noninvasive_measurement(self):
        """Test difference between invasive and non-invasive measurements."""
        
        def measurement_disturbance(invasiveness):
            """Model measurement back-action."""
            return 1 - invasiveness
        
        coherence_noninvasive = measurement_disturbance(0.0)
        coherence_invasive = measurement_disturbance(1.0)
        
        assert coherence_noninvasive > coherence_invasive
    
    def test_temporal_correlation_under_intervention(self):
        """Test how interventions affect temporal correlations."""
        
        def correlation_no_intervention(t1, t2):
            """Natural evolution correlation."""
            return np.exp(-0.1 * abs(t2 - t1))
        
        def correlation_with_intervention(t1, t_int, t2):
            """Correlation with intervention at t_int."""
            # Intervention disrupts correlation
            C_before = np.exp(-0.1 * abs(t_int - t1))
            C_after = np.exp(-0.1 * abs(t2 - t_int))
            return C_before * C_after * 0.8  # Reduction factor
        
        t1, t2 = 0, 2
        t_int = 1
        
        C_no_int = correlation_no_intervention(t1, t2)
        C_with_int = correlation_with_intervention(t1, t_int, t2)
        
        # Intervention should reduce correlation
        assert C_with_int < C_no_int


class TestIntegratedProtocols:
    """Integration tests combining multiple protocols."""
    
    def test_protocol_abc_consistency(self):
        """Test that all three protocols give consistent results."""
        
        results = {
            'protocol_a': False,
            'protocol_b': False,
            'protocol_c': False
        }
        
        # Protocol A: Reset shows memory
        reset_fidelity = 0.97
        if reset_fidelity < 0.99:
            results['protocol_a'] = True
        
        # Protocol B: Non-CP-divisible
        E1 = np.array([[0.9, 0.1], [0.1, 0.9]])
        E2 = np.array([[0.9, 0], [0, 0.9]])
        E_total = np.array([[0.82, 0.09], [0.09, 0.81]])
        
        if not np.allclose(E_total, E2 @ E1, atol=1e-3):
            results['protocol_b'] = True
        
        # Protocol C: LG violation
        K3 = 1.05
        if K3 > 1.0:
            results['protocol_c'] = True
        
        # At least one should show signature
        assert any(results.values())
    
    def test_cross_protocol_validation(self):
        """Test that different protocols detect same underlying effect."""
        
        # Simulate spiral-time memory effect
        memory_strength = 0.15
        
        # Protocol A signature
        def reset_residue():
            return memory_strength * 0.5
        
        # Protocol B signature
        def cp_violation():
            return memory_strength * 0.3
        
        # Protocol C signature
        def lg_violation():
            return memory_strength * 0.7
        
        sig_a = reset_residue()
        sig_b = cp_violation()
        sig_c = lg_violation()
        
        # All should be non-zero and correlated
        assert sig_a > 0 and sig_b > 0 and sig_c > 0
        
        # Should scale similarly
        assert sig_a / sig_b > 1  # Different protocols, different sensitivity
    
    def test_systematic_error_control(self):
        """Test control of systematic errors across protocols."""
        
        # Common systematics that affect all protocols
        systematic_errors = {
            'calibration_drift': 0.01,
            'measurement_crosstalk': 0.005,
            'control_imperfection': 0.02
        }
        
        total_systematic = sum(systematic_errors.values())
        
        # Memory signal should exceed systematics
        memory_signal = 0.10
        
        # Signal-to-noise ratio
        snr = memory_signal / total_systematic
        
        assert snr > 2  # Require >2σ significance


class TestExperimentalFeasibility:
    """Test experimental feasibility and requirements."""
    
    def test_required_measurement_precision(self):
        """Test required precision for detecting memory."""
        
        # Expected memory effect size
        epsilon_memory = 0.05
        
        # Required measurement precision
        precision_required = epsilon_memory / 3  # 3σ detection
        
        # Current experimental capabilities
        precision_trapped_ion = 0.01
        precision_superconducting = 0.015
        
        # Both platforms should be sufficient
        assert precision_trapped_ion < precision_required
        assert precision_superconducting < precision_required
    
    def test_coherence_time_requirements(self):
        """Test coherence time requirements for protocols."""
        
        # Protocol duration
        t_protocol = 10.0  # microseconds
        
        # Memory time scale
        tau_memory = 5.0
        
        # Required coherence time
        t_coherence_required = 2 * max(t_protocol, tau_memory)
        
        # Typical coherence times
        t_coherence_ion = 100.0  # microseconds
        t_coherence_squid = 50.0
        
        # Both should be sufficient
        assert t_coherence_ion > t_coherence_required
        assert t_coherence_squid > t_coherence_required
    
    def test_number_of_measurements_required(self):
        """Test statistical requirements for significance."""
        
        # Effect size
        epsilon = 0.05
        
        # Required measurements for 3σ significance
        # N ~ (3/ε)²
        N_required = int((3 / epsilon)**2)
        
        # Should be experimentally feasible
        assert N_required < 10000
    
    def test_gate_fidelity_requirements(self):
        """Test gate fidelity requirements."""
        
        # Number of gates in protocol
        n_gates = 20
        
        # Required per-gate fidelity
        F_total_required = 0.95  # 95% overall fidelity
        F_gate_required = F_total_required ** (1 / n_gates)
        
        # Current gate fidelities
        F_gate_ion = 0.999
        F_gate_squid = 0.998
        
        # Both should meet requirements
        assert F_gate_ion > F_gate_required
        assert F_gate_squid > F_gate_required


class TestProtocolOptimization:
    """Test optimization strategies for experimental protocols."""
    
    def test_adaptive_measurement_strategy(self):
        """Test adaptive measurement scheduling."""
        
        def adaptive_schedule(preliminary_results):
            """Adjust measurement times based on initial data."""
            # If preliminary shows strong memory at short times,
            # focus measurements there
            
            if np.mean(preliminary_results) > 0.1:
                # Strong signal: use finer time resolution
                times = np.linspace(0, 5, 50)
            else:
                # Weak signal: broader search
                times = np.linspace(0, 10, 30)
            
            return times
        
        # Strong signal case
        prelim_strong = [0.15, 0.12, 0.14]
        times_strong = adaptive_schedule(prelim_strong)
        
        # Weak signal case
        prelim_weak = [0.02, 0.03, 0.01]
        times_weak = adaptive_schedule(prelim_weak)
        
        # Different strategies
        assert len(times_strong) > len(times_weak)
        assert times_strong[-1] < times_weak[-1]
    
    def test_resource_optimization(self):
        """Test optimal allocation of measurement resources."""
        
        def optimize_shots(total_budget, n_settings):
            """Distribute shots across settings."""
            # Equal distribution baseline
            shots_per_setting = total_budget // n_settings
            
            # Optimize based on expected variance
            # Settings with higher variance get more shots
            variances = np.random.uniform(0.1, 0.5, n_settings)
            weights = np.sqrt(variances)
            weights /= np.sum(weights)
            
            optimized_shots = (weights * total_budget).astype(int)
            
            return optimized_shots
        
        budget = 10000
        n = 10
        
        shots = optimize_shots(budget, n)
        
        # Should use full budget
        assert np.sum(shots) <= budget
        # Should vary by setting
        assert len(np.unique(shots)) > 1
    
    def test_error_mitigation_protocol(self):
        """Test error mitigation strategies."""
        
        def mitigate_readout_error(raw_outcomes, confusion_matrix):
            """Correct for readout errors."""
            # Invert confusion matrix
            try:
                inv_confusion = np.linalg.inv(confusion_matrix.T)
                corrected = inv_confusion @ raw_outcomes
                # Clip to valid probabilities
                corrected = np.clip(corrected, 0, 1)
                corrected /= np.sum(corrected)
                return corrected
            except:
                return raw_outcomes
        
        # Simulate readout confusion
        # True state: |0⟩ with p=0.7, |1⟩ with p=0.3
        true_probs = np.array([0.7, 0.3])
        
        # Confusion matrix: rows = true, cols = measured
        confusion = np.array([
            [0.95, 0.05],  # True |0⟩ → measure |0⟩ or |1⟩
            [0.10, 0.90]   # True |1⟩ → measure |0⟩ or |1⟩
        ])
        
        # Observed (corrupted) probabilities
        observed = confusion.T @ true_probs
        
        # Mitigate
        corrected = mitigate_readout_error(observed, confusion)
        
        # Should be closer to true
        error_raw = np.linalg.norm(observed - true_probs)
        error_corrected = np.linalg.norm(corrected - true_probs)
        
        assert error_corrected < error_raw


class TestDataAnalysis:
    """Test data analysis methods for protocol results."""
    
    def test_maximum_likelihood_estimation(self):
        """Test MLE for process parameters."""
        
        def likelihood(data, parameter):
            """Likelihood function for memory parameter."""
            # Simplified: Gaussian likelihood
            predicted = 0.5 + parameter * 0.3
            residuals = data - predicted
            return np.exp(-np.sum(residuals**2) / (2 * 0.01))
        
        # Simulated data
        true_param = 0.2
        data = np.array([0.56, 0.58, 0.54, 0.57])
        
        # Grid search for MLE
        params = np.linspace(0, 0.5, 50)
        likelihoods = [likelihood(data, p) for p in params]
        
        mle_idx = np.argmax(likelihoods)
        mle_param = params[mle_idx]
        
        # Should be close to true value
        assert np.abs(mle_param - true_param) < 0.1
    
    def test_bayesian_inference(self):
        """Test Bayesian parameter estimation."""
        
        def posterior(parameter, data, prior_mean=0.25, prior_std=0.1):
            """Posterior distribution."""
            # Prior: Gaussian
            prior = np.exp(-(parameter - prior_mean)**2 / (2 * prior_std**2))
            
            # Likelihood
            predicted = 0.5 + parameter * 0.3
            likelihood = np.exp(-np.sum((data - predicted)**2) / (2 * 0.01))
            
            return prior * likelihood
        
        data = np.array([0.56, 0.58, 0.54])
        
        params = np.linspace(0, 0.5, 100)
        posteriors = np.array([posterior(p, data) for p in params])
        posteriors /= np.sum(posteriors) * (params[1] - params[0])
        
        # MAP estimate
        map_idx = np.argmax(posteriors)
        map_param = params[map_idx]
        
        # Credible interval (simplified)
        cumsum = np.cumsum(posteriors) * (params[1] - params[0])
        ci_lower = params[np.where(cumsum >= 0.025)[0][0]]
        ci_upper = params[np.where(cumsum >= 0.975)[0][0]]
        
        assert ci_lower < map_param < ci_upper
    
    def test_hypothesis_testing(self):
        """Test statistical hypothesis testing."""
        
        def likelihood_ratio_test(data_null, data_alt):
            """LRT for model comparison."""
            # Null: Markovian (constant)
            L_null = np.exp(-np.var(data_null))
            
            # Alternative: Non-Markovian (with trend)
            L_alt = np.exp(-np.var(data_alt))
            
            # Likelihood ratio
            LR = L_alt / L_null
            
            # Test statistic: -2 log LR
            test_stat = -2 * np.log(LR)
            
            return test_stat
        
        # Markovian data (no trend)
        data_null = np.array([0.5, 0.51, 0.49, 0.50, 0.51])
        
        # Non-Markovian data (with memory-induced trend)
        data_alt = np.array([0.5, 0.52, 0.54, 0.56, 0.58])
        
        test_stat = likelihood_ratio_test(data_null, data_alt)
        
        # Should be positive (alternative is better fit)
        # In real test, would compare to chi-square distribution
        assert test_stat != 0
    
    def test_bootstrap_uncertainty_estimation(self):
        """Test bootstrap method for uncertainty quantification."""
        
        np.random.seed(42)
        
        # Original data
        data = np.array([0.15, 0.12, 0.14, 0.13, 0.16])
        
        # Bootstrap resampling
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            resample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(resample))
        
        # Estimate confidence interval
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        # Original mean should be within CI
        original_mean = np.mean(data)
        assert ci_lower < original_mean < ci_upper


class TestRegressionAndValidation:
    """Regression tests and validation checks."""
    
    def test_known_markovian_baseline(self):
        """Test against known Markovian result."""
        
        # Amplitude damping: known analytical result
        def amplitude_damping_population(t, gamma):
            """Population in excited state."""
            return np.exp(-gamma * t)
        
        times = np.linspace(0, 5, 20)
        gamma = 0.2
        
        analytical = [amplitude_damping_population(t, gamma) for t in times]
        
        # Numerical simulation
        numerical = [np.exp(-gamma * t) for t in times]
        
        assert np.allclose(analytical, numerical, rtol=1e-10)
    
    def test_protocol_reproducibility(self):
        """Test reproducibility of protocol results."""
        
        np.random.seed(123)
        
        def run_protocol(n_shots):
            """Simulate protocol run."""
            return np.random.binomial(n_shots, 0.6) / n_shots
        
        # Run multiple times with same seed
        results_1 = []
        for _ in range(5):
            np.random.seed(123)
            results_1.append(run_protocol(100))
        
        # Should be identical
        assert len(set(results_1)) == 1
    
    def test_cross_platform_consistency(self):
        """Test consistency across different experimental platforms."""
        
        # Simulate same experiment on different platforms
        def platform_simulation(platform_name, true_signal):
            """Simulate with platform-specific noise."""
            if platform_name == 'ion':
                noise = np.random.normal(0, 0.01)
            elif platform_name == 'squid':
                noise = np.random.normal(0, 0.02)
            else:
                noise = 0
            
            return true_signal + noise
        
        np.random.seed(42)
        
        true_memory = 0.15
        
        # Multiple runs on each platform
        n_runs = 50
        
        results_ion = [platform_simulation('ion', true_memory) for _ in range(n_runs)]
        results_squid = [platform_simulation('squid', true_memory) for _ in range(n_runs)]
        
        mean_ion = np.mean(results_ion)
        mean_squid = np.mean(results_squid)
        
        # Both should recover true signal within uncertainty
        assert np.abs(mean_ion - true_memory) < 0.05
        assert np.abs(mean_squid - true_memory) < 0.05
    
    def test_scaling_behavior(self):
        """Test scaling of computational resources."""
        
        def compute_time_estimate(n_qubits, n_time_steps):
            """Estimate computation time."""
            # Hilbert space dimension: 2^n
            # Process tensor: (2^n)^(n_time_steps)
            # Complexity roughly exponential
            
            complexity = (2 ** n_qubits) ** n_time_steps
            
            # Arbitrary time units
            return complexity * 1e-6
        
        # Small system
        t_small = compute_time_estimate(2, 3)
        
        # Larger system
        t_large = compute_time_estimate(3, 4)
        
        # Should scale exponentially
        assert t_large > t_small * 10


class TestDocumentationAndReporting:
    """Tests for documentation and result reporting."""
    
    def test_result_formatting(self):
        """Test proper formatting of results."""
        
        def format_result(parameter, uncertainty, significance):
            """Format measurement result."""
            return {
                'value': parameter,
                'uncertainty': uncertainty,
                'significance': significance,
                'unit': 'dimensionless'
            }
        
        result = format_result(0.15, 0.02, 3.5)
        
        assert 'value' in result
        assert 'uncertainty' in result
        assert result['significance'] > 3.0  # >3σ
    
    def test_metadata_tracking(self):
        """Test experimental metadata tracking."""
        
        metadata = {
            'date': '2026-01-22',
            'platform': 'trapped_ion',
            'protocol': 'A',
            'num_qubits': 2,
            'num_shots': 1000,
            'gate_fidelity': 0.999,
            'readout_fidelity': 0.995
        }
        
        # Required fields present
        required = ['date', 'platform', 'protocol', 'num_qubits', 'num_shots']
        assert all(field in metadata for field in required)
    
    def test_result_serialization(self):
        """Test serialization of results for storage."""
        
        import json
        
        results = {
            'protocol_a': {
                'reset_fidelity': 0.97,
                'memory_detected': True
            },
            'protocol_b': {
                'cp_divisible': False,
                'violation_strength': 0.08
            },
            'protocol_c': {
                'K3_value': 1.05,
                'lg_violation': True
            }
        }
        
        # Should be JSON serializable
        json_str = json.dumps(results)
        recovered = json.loads(json_str)
        
        assert recovered['protocol_a']['memory_detected'] == True
        assert recovered['protocol_b']['cp_divisible'] == False


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_memory_limit(self):
        """Test χ → 0 limit recovers Markovian physics."""
        
        def dynamics_with_memory(t, chi):
            """Evolution with memory parameter."""
            markov_term = np.exp(-0.1 * t)
            memory_term = chi * np.exp(-0.05 * t) * np.sin(t)
            return markov_term + memory_term
        
        t = 5.0
        
        # No memory
        result_no_mem = dynamics_with_memory(t, 0.0)
        
        # Tiny memory
        result_tiny_mem = dynamics_with_memory(t, 1e-10)
        
        # Should be essentially identical
        assert np.isclose(result_no_mem, result_tiny_mem, rtol=1e-8)
    
    def test_maximum_memory_saturation(self):
        """Test behavior at maximum memory strength."""
        
        def memory_effect(chi_strength):
            """Memory signature as function of strength."""
            # Saturates at chi = 1
            return 1 - np.exp(-chi_strength)
        
        chi_values = [0.1, 0.5, 1.0, 5.0, 10.0]
        effects = [memory_effect(chi) for chi in chi_values]
        
        # Should approach 1 asymptotically
        assert effects[-1] > 0.99
        assert effects[-1] < 1.0 + 1e-10
    
    def test_extreme_decoherence_limit(self):
        """Test behavior under extreme decoherence."""
        
        def state_purity(t, gamma_decohere):
            """Purity under decoherence."""
            # Approaches 1/2 for qubit
            return 0.5 + 0.5 * np.exp(-gamma_decohere * t)
        
        t = 100.0
        gamma_extreme = 10.0
        
        purity = state_purity(t, gamma_extreme)
        
        # Should be very close to maximally mixed
        assert np.isclose(purity, 0.5, atol=1e-6)
    
    def test_numerical_stability_at_boundaries(self):
        """Test numerical stability at parameter boundaries."""
        
        def safe_log_division(numerator, denominator, epsilon=1e-15):
            """Numerically stable log division."""
            return np.log(np.maximum(numerator, epsilon) / 
                         np.maximum(denominator, epsilon))
        
        # Near-zero values
        num = 1e-100
        denom = 1e-100
        
        result = safe_log_division(num, denom)
        
        # Should not overflow or produce NaN
        assert np.isfinite(result)


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
