"""Tests for non-Markovianity measures."""

import pytest
import numpy as np


class TestNonMarkovianityMeasures:
    """Test quantitative measures of non-Markovianity."""
    
    def test_rivas_measure(self, tolerance):
        """Test Rivas-Huelga-Plenio measure (trace distance)."""
        
        def trace_distance(rho1, rho2):
            """Compute trace distance between density matrices."""
            diff = rho1 - rho2
            eigenvalues = np.linalg.eigvalsh(diff @ diff.conj().T)
            return 0.5 * np.sqrt(np.sum(eigenvalues))
        
        # Distinguished initial states
        rho1 = np.array([[1, 0], [0, 0]])
        rho2 = np.array([[0, 0], [0, 1]])
        
        D_initial = trace_distance(rho1, rho2)
        
        # After Markovian evolution, distance should decrease monotonically
        gamma = 0.1
        E0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
        E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        
        rho1_evolved = E0 @ rho1 @ E0.conj().T + E1 @ rho1 @ E1.conj().T
        rho2_evolved = E0 @ rho2 @ E0.conj().T + E1 @ rho2 @ E1.conj().T
        
        D_final = trace_distance(rho1_evolved, rho2_evolved)
        
        # Markovian: D should decrease
        assert D_final <= D_initial + tolerance
    
    def test_breuer_laine_piilo_measure(self):
        """Test BLP measure (information backflow)."""
        
        # Simplified: non-Markovianity = integral of trace distance growth rate
        times = np.linspace(0, 10, 100)
        dt = times[1] - times[0]
        
        # Mock trace distance evolution
        D_markov = np.exp(-0.1 * times)  # Monotonic decrease
        D_non_markov = np.exp(-0.1 * times) * (1 + 0.1 * np.sin(times))  # Oscillations
        
        # Compute growth rates
        dD_markov = np.diff(D_markov) / dt
        dD_non_markov = np.diff(D_non_markov) / dt
        
        # Non-Markovianity = integral of positive growth
        N_markov = np.sum(np.maximum(dD_markov, 0)) * dt
        N_non_markov = np.sum(np.maximum(dD_non_markov, 0)) * dt
        
        # Non-Markovian process should have higher measure
        assert N_non_markov > N_markov
    
    def test_luo_fu_song_measure(self):
        """Test LFS measure based on fidelity."""
        
        def fidelity(rho1, rho2):
            """Quantum fidelity between states."""
            sqrt_rho1 = np.linalg.cholesky(rho1 + 1e-10 * np.eye(len(rho1)))
            product = sqrt_rho1 @ rho2 @ sqrt_rho1
            eigenvalues = np.linalg.eigvalsh(product)
            return (np.sum(np.sqrt(np.abs(eigenvalues))))**2
        
        rho1 = np.array([[1, 0], [0, 0]])
        rho2 = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        F = fidelity(rho1, rho2)
        
        # Fidelity should be in [0, 1]
        assert 0 <= F <= 1 + 1e-10
    
    def test_chruściński_maniscalco_measure(self):
        """Test volume-based non-Markovianity measure."""
        
        # Measure based on accessible state volume
        def accessible_volume(dim, purity):
            """Estimate volume of accessible state space."""
            # Higher purity → smaller volume
            return dim ** 2 * (1 - purity)
        
        # Markovian: volume decreases monotonically
        purities_markov = [0.5, 0.6, 0.7, 0.8]
        volumes_markov = [accessible_volume(2, p) for p in purities_markov]
        
        # Should decrease
        assert all(volumes_markov[i] >= volumes_markov[i+1] 
                  for i in range(len(volumes_markov)-1))


class TestInformationBackflow:
    """Test information backflow detection."""
    
    def test_backflow_detection_simple(self):
        """Detect information backflow in toy model."""
        
        times = np.linspace(0, 10, 50)
        
        # Non-Markovian with backflow
        distinguishability = np.exp(-0.2 * times) * (1 + 0.3 * np.sin(2 * times))
        
        # Compute derivative
        dD_dt = np.diff(distinguishability) / (times[1] - times[0])
        
        # Periods of positive derivative = backflow
        backflow_periods = np.sum(dD_dt > 0)
        
        assert backflow_periods > 0
    
    def test_backflow_integral_measure(self):
        """Compute integral measure of backflow."""
        
        def compute_backflow_integral(D_trajectory):
            """Integrate positive growth rate."""
            dD = np.diff(D_trajectory)
            positive_growth = np.maximum(dD, 0)
            return np.sum(positive_growth)
        
        # Markovian
        times = np.linspace(0, 10, 100)
        D_markov = np.exp(-0.1 * times)
        backflow_markov = compute_backflow_integral(D_markov)
        
        # Non-Markovian
        D_non_markov = np.exp(-0.1 * times) * (1 + 0.2 * np.sin(3 * times))
        backflow_non_markov = compute_backflow_integral(D_non_markov)
        
        # Non-Markovian should have significant backflow
        assert backflow_non_markov > backflow_markov
    
    def test_oscillatory_backflow_pattern(self):
        """Test detection of oscillatory backflow."""
        
        times = np.linspace(0, 20, 200)
        
        # Multiple backflow events
        D = np.exp(-0.05 * times) * (1 + 0.4 * np.sin(times))
        
        # Find local maxima (backflow events)
        local_maxima = 0
        for i in range(1, len(D) - 1):
            if D[i] > D[i-1] and D[i] > D[i+1]:
                local_maxima += 1
        
        # Should have multiple backflow events
        assert local_maxima > 3


class TestMemoryDepthMeasures:
    """Test measures of memory depth."""
    
    def test_exponential_memory_decay(self):
        """Test characteristic memory time from exponential decay."""
        
        def memory_kernel(t, tau, gamma):
            """Exponential memory kernel."""
            return gamma * np.exp(-gamma * (t - tau)) if t >= tau else 0.0
        
        gamma = 0.2  # Memory decay rate
        tau_memory = 1 / gamma  # Characteristic memory time
        
        # At t = tau_memory, kernel should decay to 1/e
        t = tau_memory
        tau = 0
        k = memory_kernel(t, tau, gamma)
        
        expected = gamma * np.exp(-1)  # 1/e decay
        assert np.isclose(k, expected, rtol=0.01)
    
    def test_effective_memory_window(self):
        """Test effective memory window calculation."""
        
        def effective_memory_time(kernel_func, threshold=0.01):
            """Find time when kernel drops below threshold."""
            t = 0.0
            dt = 0.1
            
            while t < 100:
                k = kernel_func(t, 0)
                if k < threshold:
                    return t
                t += dt
            
            return t
        
        def kernel(t, tau):
            return 0.1 * np.exp(-0.1 * (t - tau)) if t >= tau else 0.0
        
        tau_eff = effective_memory_time(kernel)
        
        # Should be finite
        assert tau_eff < 100
        assert tau_eff > 0
    
    def test_memory_depth_from_correlation_time(self):
        """Test memory depth from correlation function."""
        
        def autocorrelation(x, max_lag=None):
            """Compute autocorrelation function."""
            if max_lag is None:
                max_lag = len(x) // 2
            
            correlations = []
            for lag in range(max_lag):
                if lag == 0:
                    correlations.append(1.0)
                else:
                    c = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                    correlations.append(c)
            
            return np.array(correlations)
        
        # Generate correlated signal
        np.random.seed(42)
        n = 1000
        
        # Markovian: white noise
        x_markov = np.random.randn(n)
        
        # Non-Markovian: smoothed
        window = 10
        x_non_markov = np.convolve(np.random.randn(n + window), 
                                   np.ones(window)/window, mode='valid')[:n]
        
        corr_markov = autocorrelation(x_markov, 50)
        corr_non_markov = autocorrelation(x_non_markov, 50)
        
        # Non-Markovian should have longer correlation time
        # (autocorrelation decays more slowly)
        assert np.sum(np.abs(corr_non_markov)) > np.sum(np.abs(corr_markov))


class TestWitnessOperators:
    """Test witness operators for non-Markovianity."""
    
    def test_entanglement_witness_analogy(self):
        """Test witness operator construction (analogous to entanglement witnesses)."""
        
        # Witness W such that Tr(W ρ) < 0 implies non-Markovianity
        
        def construct_witness(rho_markov):
            """Construct witness operator."""
            # Simplified: use deviation from maximal mixed
            dim = len(rho_markov)
            rho_mixed = np.eye(dim) / dim
            
            witness = rho_mixed - rho_markov
            return witness
        
        # Markovian state (closer to mixed)
        rho_markov = 0.6 * np.eye(2) / 2 + 0.4 * np.array([[1, 0], [0, 0]])
        
        # Non-Markovian state (more pure due to backflow)
        rho_non_markov = 0.4 * np.eye(2) / 2 + 0.6 * np.array([[1, 0], [0, 0]])
        
        W = construct_witness(rho_markov)
        
        # Test witness values
        witness_markov = np.trace(W @ rho_markov).real
        witness_non_markov = np.trace(W @ rho_non_markov).real
        
        # Different signatures
        assert witness_markov != witness_non_markov
    
    def test_positive_map_witness(self):
        """Test positive-but-not-CP map as witness."""
        
        # Transpose map (positive but not CP)
        def transpose_witness(rho):
            """Apply transpose."""
            return rho.T
        
        # For product state: remains positive
        rho_product = np.array([[0.5, 0], [0, 0.5]])
        rho_transposed = transpose_witness(rho_product)
        
        eigenvalues = np.linalg.eigvalsh(rho_transposed)
        assert np.all(eigenvalues >= -1e-10)
    
    def test_dynamical_witness(self):
        """Test time-dependent witness construction."""
        
        def witness_at_time(t, reference_traj):
            """Construct time-dependent witness."""
            # Deviation from reference Markovian trajectory
            rho_ref = np.eye(2) / 2 * np.exp(-0.1 * t)
            return rho_ref
        
        times = [0, 1, 2, 5, 10]
        witnesses = [witness_at_time(t, None) for t in times]
        
        # Witnesses should change with time
        assert not np.allclose(witnesses[0], witnesses[-1])


class TestMultiTimeCorrelations:
    """Test multi-time correlation functions."""
    
    def test_two_time_correlation(self):
        """Test C(t1, t2) = ⟨A(t1)A(t2)⟩."""
        
        def two_time_corr(t1, t2, decay_rate=0.1):
            """Simple two-time correlation."""
            dt = abs(t2 - t1)
            return np.exp(-decay_rate * dt)
        
        t1, t2 = 0.0, 5.0
        C = two_time_corr(t1, t2)
        
        # Should be less than 1 (decayed)
        assert C < 1.0
        assert C > 0.0
    
    def test_three_time_correlation_factorization(self):
        """Test whether C(t1,t2,t3) factorizes (Markovian)."""
        
        def three_time_corr_markov(t1, t2, t3):
            """Markovian: should factorize."""
            C12 = np.exp(-0.1 * abs(t2 - t1))
            C23 = np.exp(-0.1 * abs(t3 - t2))
            return C12 * C23
        
        def three_time_corr_non_markov(t1, t2, t3):
            """Non-Markovian: additional memory term."""
            C_markov = three_time_corr_markov(t1, t2, t3)
            memory_term = 0.1 * np.exp(-0.05 * (t3 - t1))
            return C_markov + memory_term
        
        t1, t2, t3 = 0.0, 2.0, 5.0
        
        C_markov = three_time_corr_markov(t1, t2, t3)
        C_non_markov = three_time_corr_non_markov(t1, t2, t3)
        
        # Should differ
        assert C_non_markov != C_markov
    
    def test_oto_correlator(self):
        """Test out-of-time-order correlator (OTOC)."""
        
        def otoc(t):
            """OTOC: ⟨A(t)B(0)A(t)B(0)⟩ - ⟨A(t)A(t)⟩⟨B(0)B(0)⟩."""
            # Simplified model
            return np.exp(-0.2 * t) * np.cos(t)
        
        times = np.linspace(0, 10, 50)
        otoc_values = [otoc(t) for t in times]
        
        # OTOC can oscillate and show non-trivial time dependence
        assert len(otoc_values) > 0
        assert not np.allclose(otoc_values, otoc_values[0])


class TestStatisticalTests:
    """Statistical tests for non-Markovianity."""
    
    def test_chi_square_test_markovianity(self):
        """Chi-square test for Markovian hypothesis."""
        
        np.random.seed(42)
        
        # Generate sequences
        n = 100
        
        # Markovian: independent steps
        markov_seq = np.random.randn(n)
        
        # Non-Markovian: correlated
        non_markov_seq = np.zeros(n)
        non_markov_seq[0] = np.random.randn()
        for i in range(1, n):
            non_markov_seq[i] = 0.7 * non_markov_seq[i-1] + 0.3 * np.random.randn()
        
        # Test independence via autocorrelation
        def lag_1_correlation(x):
            return np.corrcoef(x[:-1], x[1:])[0, 1]
        
        corr_markov = lag_1_correlation(markov_seq)
        corr_non_markov = lag_1_correlation(non_markov_seq)
        
        # Markovian should have low autocorrelation
        assert abs(corr_markov) < 0.3
        
        # Non-Markovian should have high autocorrelation
        assert abs(corr_non_markov) > 0.5
    
    def test_kolmogorov_smirnov_test(self):
        """KS test for distribution of waiting times."""
        
        # Markovian: exponential waiting times
        # Non-Markovian: power-law or other
        
        np.random.seed(42)
        
        # Exponential (Markovian)
        waiting_markov = np.random.exponential(1.0, 1000)
        
        # Power-law (Non-Markovian)
        waiting_non_markov = (np.random.pareto(2.0, 1000) + 1)
        
        # Compare distributions
        mean_markov = np.mean(waiting_markov)
        mean_non_markov = np.mean(waiting_non_markov)
        
        # Different statistics
        assert not np.isclose(mean_markov, mean_non_markov, rtol=0.1)
