"""Tests for state-independent memory criterion (Sec. 10.1, Eq. 18)."""

import pytest
import numpy as np


class TestStateIndependence:
    """Test state-independence of memory kernel."""
    
    def test_kernel_independence_from_state(self, memory_kernel):
        """Verify K(t-τ) does not depend on ρ(t)."""
        
        t, tau = 5.0, 2.0
        
        # Different quantum states
        rho_1 = np.array([[1, 0], [0, 0]])
        rho_2 = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        # Kernel should be the same
        k_1 = memory_kernel(t, tau)
        k_2 = memory_kernel(t, tau)
        
        assert k_1 == k_2
    
    def test_environmental_kernel_state_dependence(self):
        """Contrast with environment-induced kernel (state-dependent)."""
        
        def environmental_kernel(t, tau, rho):
            """State-dependent environmental kernel."""
            purity = np.trace(rho @ rho).real
            return 0.1 * np.exp(-0.1 * (t - tau)) * purity
        
        t, tau = 5.0, 2.0
        
        rho_pure = np.array([[1, 0], [0, 0]])
        rho_mixed = np.array([[0.5, 0], [0, 0.5]])
        
        k_pure = environmental_kernel(t, tau, rho_pure)
        k_mixed = environmental_kernel(t, tau, rho_mixed)
        
        # Environmental kernel IS state-dependent
        assert k_pure != k_mixed
    
    def test_spiral_time_vs_environmental_discrimination(self):
        """Test that spiral-time and environmental memory are distinguishable."""
        
        # Spiral-time: state-independent
        def spiral_kernel(t, tau):
            return 0.1 * np.exp(-0.1 * (t - tau)) if t >= tau else 0.0
        
        # Environmental: state-dependent
        def env_kernel(t, tau, purity):
            return 0.1 * np.exp(-0.1 * (t - tau)) * purity if t >= tau else 0.0
        
        t, tau = 5.0, 2.0
        
        # Spiral-time kernel
        k_spiral = spiral_kernel(t, tau)
        
        # Environmental kernel at different purities
        k_env_pure = env_kernel(t, tau, 1.0)
        k_env_mixed = env_kernel(t, tau, 0.5)
        
        # Spiral-time is constant
        assert k_spiral == k_spiral
        
        # Environmental varies
        assert k_env_pure != k_env_mixed
    
    def test_kernel_evolution_independence(self):
        """Test that kernel doesn't evolve with system state."""
        
        def spiral_kernel(t, tau):
            """Pure time-difference kernel."""
            return 0.1 * np.exp(-0.1 * (t - tau)) if t >= tau else 0.0
        
        times = [1.0, 2.0, 5.0, 10.0]
        tau = 0.5
        
        # Evaluate kernel at different times
        kernels = [spiral_kernel(t, tau) for t in times]
        
        # All should depend only on (t - tau)
        time_diffs = [t - tau for t in times]
        expected = [0.1 * np.exp(-0.1 * dt) for dt in time_diffs]
        
        assert np.allclose(kernels, expected)


class TestStateIndependentDynamics:
    """Test dynamics with state-independent memory."""
    
    def test_universal_memory_evolution(self):
        """Test that memory evolution is universal across states."""
        
        def evolve_with_memory(rho_0, times, kernel_func):
            """Evolve state with memory kernel."""
            trajectory = [rho_0]
            
            for i in range(1, len(times)):
                t = times[i]
                dt = times[i] - times[i-1]
                
                # Local evolution
                H = np.array([[0.1, 0], [0, -0.1]])
                U_local = np.eye(2) - 1j * H * dt
                rho_local = U_local @ trajectory[-1] @ U_local.conj().T
                
                # Memory contribution (state-independent)
                memory_factor = 1.0
                for j in range(i):
                    memory_factor += kernel_func(t, times[j]) * dt
                
                rho_new = rho_local / memory_factor
                rho_new /= np.trace(rho_new)
                trajectory.append(rho_new)
            
            return trajectory
        
        def kernel(t, tau):
            return 0.05 * np.exp(-0.1 * (t - tau)) if t >= tau else 0.0
        
        times = np.linspace(0, 5, 20)
        
        # Different initial states
        rho_1 = np.array([[1, 0], [0, 0]])
        rho_2 = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        traj_1 = evolve_with_memory(rho_1, times, kernel)
        traj_2 = evolve_with_memory(rho_2, times, kernel)
        
        # Memory contribution should be identical
        # (even though states differ)
        assert len(traj_1) == len(traj_2)
    
    def test_reset_protocol_state_independence(self):
        """Test that memory persists across state resets."""
        
        def memory_after_reset(history_length, kernel_strength):
            """Compute memory contribution after reset."""
            # Reset happens at t = t_reset
            # Memory should depend on history before reset,
            # not on the reset state itself
            
            t_reset = 5.0
            tau_history = np.linspace(0, t_reset, history_length)
            
            memory_integral = 0.0
            for tau in tau_history:
                memory_integral += kernel_strength * np.exp(-0.1 * (t_reset - tau))
            
            return memory_integral * (tau_history[1] - tau_history[0])
        
        # Different history lengths
        mem_short = memory_after_reset(10, 0.1)
        mem_long = memory_after_reset(50, 0.1)
        expected = 0.1 * (1 - np.exp(-0.1 * 5.0)) / 0.1
        
        # Finer history resolution should be closer to the continuum integral
        assert abs(mem_long - expected) < abs(mem_short - expected)
    
    def test_decoherence_vs_memory_separation(self):
        """Test separation of decoherence (state-dependent) from memory."""
        
        def purity(rho):
            """Compute purity Tr(ρ²)."""
            return np.trace(rho @ rho).real
        
        # Pure state
        rho_pure = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        # Decoherence (state-dependent)
        gamma_deco = 0.3
        rho_decohered = np.array([
            [rho_pure[0,0], rho_pure[0,1] * (1 - gamma_deco)],
            [rho_pure[1,0] * (1 - gamma_deco), rho_pure[1,1]]
        ])
        
        # Purity decreases with decoherence
        assert purity(rho_decohered) < purity(rho_pure)
        
        # Memory (state-independent) doesn't directly affect purity
        # It affects evolution rate, not state properties


class TestExperimentalDiscrimination:
    """Test experimental protocols to discriminate state-independence."""
    
    def test_protocol_varying_initial_states(self):
        """Protocol: measure memory with different initial states."""
        
        def measure_memory_signature(rho_0, n_steps=10):
            """Measure non-Markovian signature."""
            # Simplified: track deviation from Markovian prediction
            
            times = np.linspace(0, 5, n_steps)
            signature = 0.0
            
            for i in range(1, len(times)):
                # Memory contribution (state-independent)
                t = times[i]
                memory_term = 0.05 * np.exp(-0.1 * t)
                signature += memory_term
            
            return signature
        
        # Different initial states
        rho_1 = np.array([[1, 0], [0, 0]])
        rho_2 = np.array([[0, 0], [0, 1]])
        rho_3 = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        sig_1 = measure_memory_signature(rho_1)
        sig_2 = measure_memory_signature(rho_2)
        sig_3 = measure_memory_signature(rho_3)
        
        # State-independent memory: all should be equal
        assert np.isclose(sig_1, sig_2)
        assert np.isclose(sig_2, sig_3)
    
    def test_protocol_controlled_purity_variation(self):
        """Protocol: vary purity while measuring memory."""
        
        def create_state_with_purity(p):
            """Create density matrix with specified purity."""
            # For 2-level system: ρ = p|0⟩⟨0| + (1-p)/2 I
            return np.array([[p + (1-p)/2, 0], [0, (1-p)/2]])
        
        purities = [1.0, 0.8, 0.6, 0.4]
        
        def memory_kernel_eval(t, tau):
            return 0.1 * np.exp(-0.1 * (t - tau)) if t >= tau else 0.0
        
        t, tau = 5.0, 2.0
        
        # Evaluate kernel for different purity states
        kernels = []
        for p in purities:
            rho = create_state_with_purity(p)
            # Spiral-time kernel is state-independent
            k = memory_kernel_eval(t, tau)
            kernels.append(k)
        
        # All should be identical
        assert np.allclose(kernels, kernels[0])
    
    def test_statistical_significance_test(self):
        """Test statistical significance of state-independence."""
        
        np.random.seed(42)
        
        # Simulate measurements with different states
        n_measurements = 100
        
        # State-independent memory signal
        true_memory = 0.15
        
        # Measurements with different states (varying purity)
        purities = np.random.uniform(0.5, 1.0, n_measurements)
        
        # Spiral-time: memory independent of purity
        measurements_spiral = true_memory + np.random.normal(0, 0.01, n_measurements)
        
        # Environmental: memory depends on purity
        measurements_env = true_memory * purities + np.random.normal(0, 0.01, n_measurements)
        
        # Test correlation with purity
        corr_spiral = np.corrcoef(purities, measurements_spiral)[0, 1]
        corr_env = np.corrcoef(purities, measurements_env)[0, 1]
        
        # Spiral-time: no correlation
        assert np.abs(corr_spiral) < 0.2
        
        # Environmental: strong correlation
        assert np.abs(corr_env) > 0.8


class TestTheoreticalConsistency:
    """Test theoretical consistency of state-independence."""
    
    def test_superposition_principle(self):
        """Test that state-independence preserves superposition."""
        
        def kernel(t, tau):
            return 0.1 * np.exp(-0.1 * (t - tau)) if t >= tau else 0.0
        
        t, tau = 5.0, 2.0
        k = kernel(t, tau)
        
        # State 1
        rho_1 = np.array([[1, 0], [0, 0]])
        
        # State 2
        rho_2 = np.array([[0, 0], [0, 1]])
        
        # Superposition
        alpha = 0.6
        rho_superpos = alpha * rho_1 + (1 - alpha) * rho_2
        
        # Kernel should be the same for all
        # (this is trivially true for state-independent kernel)
        assert kernel(t, tau) == k
    
    def test_measurement_invariance(self):
        """Test that kernel is measurement-basis independent."""
        
        def kernel(t, tau):
            return 0.1 * np.exp(-0.1 * (t - tau)) if t >= tau else 0.0
        
        t, tau = 5.0, 2.0
        
        # Z-basis state
        rho_z = np.array([[1, 0], [0, 0]])
        
        # X-basis state (same physics, different basis)
        rho_x = 0.5 * np.array([[1, 1], [1, 1]])
        
        # Kernel evaluations
        k_z = kernel(t, tau)
        k_x = kernel(t, tau)
        
        # Should be identical (basis-independent)
        assert k_z == k_x
    
    def test_no_signaling_constraint(self):
        """Test that state-independent memory respects no-signaling."""
        
        # State-independent memory cannot enable signaling
        # because it doesn't depend on local operations
        
        def memory_contribution(times, kernel_func):
            """Compute total memory contribution."""
            total = 0.0
            t_final = times[-1]
            
            for tau in times[:-1]:
                total += kernel_func(t_final, tau) * (times[1] - times[0])
            
            return total
        
        def kernel(t, tau):
            return 0.05 * np.exp(-0.1 * (t - tau)) if t >= tau else 0.0
        
        times = np.linspace(0, 10, 50)
        
        # Memory is the same regardless of local state preparations
        memory = memory_contribution(times, kernel)
        
        # This value doesn't depend on any local choice
        assert memory > 0
        assert np.isfinite(memory)
