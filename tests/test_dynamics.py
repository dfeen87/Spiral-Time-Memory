"""Tests for non-Markovian dynamics."""

import pytest
import numpy as np
from scipy.integrate import odeint
from scipy.linalg import expm


class TestMemoryKernel:
    """Test memory kernel properties (Eq. 7)."""
    
    def test_kernel_causality(self, memory_kernel):
        """Verify K(t-τ) = 0 for τ > t."""
        t = 5.0
        tau_future = 6.0
        
        k_val = memory_kernel(t, tau_future)
        assert k_val == 0.0
    
    def test_kernel_decay(self, memory_kernel):
        """Verify exponential decay of kernel."""
        t = 10.0
        taus = np.linspace(0, t, 100)
        
        k_vals = [memory_kernel(t, tau) for tau in taus]
        
        # As tau approaches t, (t - tau) shrinks and kernel increases
        assert k_vals[0] <= k_vals[-1]
        assert k_vals[-1] > 0
    
    def test_kernel_normalization(self, memory_kernel):
        """Test integral normalization of kernel."""
        t = 10.0
        taus = np.linspace(0, t, 1000)
        dt = taus[1] - taus[0]
        
        k_vals = [memory_kernel(t, tau) for tau in taus]
        integral = np.sum(k_vals) * dt
        
        # For exponential kernel with γ=0.1:
        # ∫₀ᵗ γe^(-γ(t-τ))dτ = 1 - e^(-γt)
        expected = 1 - np.exp(-0.1 * t)
        
        assert np.abs(integral - expected) < 0.01
    
    def test_kernel_time_translation_invariance(self, memory_kernel):
        """Test that kernel depends only on time difference."""
        t1, tau1 = 10.0, 5.0
        t2, tau2 = 15.0, 10.0
        
        # Same time difference
        assert (t1 - tau1) == (t2 - tau2)
        
        k1 = memory_kernel(t1, tau1)
        k2 = memory_kernel(t2, tau2)
        
        # Should give same value
        assert np.isclose(k1, k2)


class TestNonMarkovianDynamics:
    """Test history-dependent evolution (Eq. 7)."""
    
    def test_history_dependence(self, time_grid, memory_kernel):
        """Verify dynamics depend on history, not just current state."""
        
        def markovian_evolution(x, t):
            """Standard Markovian: dx/dt = F(x)."""
            return -0.1 * x
        
        def non_markovian_evolution(x, t, history, times):
            """Non-Markovian with memory kernel."""
            markov_term = -0.1 * x
            
            # Memory integral
            memory_term = 0.0
            for i, t_past in enumerate(times):
                if t_past < t:
                    dt = times[1] - times[0] if len(times) > 1 else 0.1
                    memory_term += memory_kernel(t, t_past) * history[i] * dt
            
            return markov_term + memory_term
        
        # Initial condition
        x0 = 1.0
        
        # Markovian evolution
        x_markov = odeint(markovian_evolution, x0, time_grid).flatten()
        
        # Non-Markovian evolution (simplified)
        x_non_markov = np.zeros_like(time_grid)
        x_non_markov[0] = x0
        
        for i in range(1, len(time_grid)):
            t = time_grid[i]
            dt = time_grid[i] - time_grid[i-1]
            dx = non_markovian_evolution(
                x_non_markov[i-1], t, x_non_markov[:i], time_grid[:i]
            )
            x_non_markov[i] = x_non_markov[i-1] + dx * dt
        
        # Solutions should differ due to memory
        assert not np.allclose(x_markov, x_non_markov)
    
    def test_markovian_limit(self, time_grid):
        """Verify recovery of Markovian dynamics when χ → 0."""
        
        def dynamics_with_memory(x, t, chi):
            """Dynamics with tunable memory strength."""
            return -0.1 * x * (1 + chi * np.exp(-0.1 * t))
        
        x0 = 1.0
        
        # Evolution with memory
        chi_values = [0.0, 0.01, 0.1]
        solutions = []
        
        for chi in chi_values:
            sol = odeint(lambda x, t: dynamics_with_memory(x, t, chi), 
                        x0, time_grid)
            solutions.append(sol.flatten())
        
        # χ=0 should give standard exponential decay
        assert np.allclose(solutions[0], x0 * np.exp(-0.1 * time_grid))
    
    def test_memory_depth_effect(self, time_grid):
        """Test effect of memory depth on dynamics."""
        
        def evolution_with_depth(t, tau_memory):
            """Evolution with finite memory depth."""
            # Only remember back to t - tau_memory
            if t < tau_memory:
                return np.exp(-0.1 * t)
            else:
                # Memory saturates
                return np.exp(-0.1 * tau_memory) * np.exp(-0.05 * (t - tau_memory))
        
        # Different memory depths
        tau_short = 2.0
        tau_long = 5.0
        
        trajectory_short = [evolution_with_depth(t, tau_short) for t in time_grid]
        trajectory_long = [evolution_with_depth(t, tau_long) for t in time_grid]
        
        # Should differ after memory depth is exceeded
        idx_late = len(time_grid) // 2
        assert not np.isclose(trajectory_short[idx_late], trajectory_long[idx_late])


class TestUnitarity:
    """Test unitarity of extended evolution (Eq. 8)."""
    
    def test_trace_preservation(self, random_density_matrix, tolerance):
        """Verify Tr(ρ) = 1 is preserved."""
        rho = random_density_matrix
        
        # Simple unitary evolution
        H = np.array([[0, 1], [1, 0]])  # Pauli X
        dt = 0.1
        U = expm(-1j * H * dt)
        
        rho_evolved = U @ rho @ U.conj().T
        
        assert np.abs(np.trace(rho_evolved) - 1.0) < tolerance
    
    def test_positivity_preservation(self, random_density_matrix, tolerance):
        """Verify ρ remains positive semi-definite."""
        rho = random_density_matrix
        
        H = np.array([[1, 0], [0, -1]])  # Pauli Z
        dt = 0.1
        U = expm(-1j * H * dt)
        
        rho_evolved = U @ rho @ U.conj().T
        
        eigenvalues = np.linalg.eigvalsh(rho_evolved)
        assert np.all(eigenvalues >= -tolerance)
    
    def test_unitary_group_property(self):
        """Test that evolution operators form a group."""
        H = np.array([[1, 0.5], [0.5, -1]])
        H = (H + H.conj().T) / 2  # Ensure Hermitian
        
        dt = 0.1
        U1 = expm(-1j * H * dt)
        U2 = expm(-1j * H * dt)
        
        # Composition
        U_composed = U2 @ U1
        U_direct = expm(-1j * H * 2 * dt)
        
        assert np.allclose(U_composed, U_direct, atol=1e-10)
        
        # Inverse
        U_inv = U1.conj().T
        assert np.allclose(U1 @ U_inv, np.eye(2), atol=1e-10)
    
    def test_von_neumann_equation(self, random_density_matrix):
        """Test von Neumann equation dρ/dt = -i[H,ρ]."""
        rho_0 = random_density_matrix
        H = np.array([[1, 0], [0, -1]])
        
        dt = 0.01
        
        # Numerical evolution
        U = expm(-1j * H * dt)
        rho_numerical = U @ rho_0 @ U.conj().T
        
        # Analytical (first order)
        commutator = H @ rho_0 - rho_0 @ H
        rho_analytical = rho_0 - 1j * dt * commutator
        
        # Should agree to first order
        assert np.allclose(rho_numerical, rho_analytical, atol=1e-3)


class TestMemoryDynamicsIntegration:
    """Test integration of memory into quantum dynamics."""
    
    def test_volterra_equation_solution(self, time_grid, memory_kernel):
        """Test numerical solution of Volterra integral equation."""
        
        def volterra_integro_differential(x, t, history, times):
            """dx/dt = f(x) + ∫K(t-τ)x(τ)dτ."""
            local_term = -0.1 * x
            
            integral_term = 0.0
            for i, tau in enumerate(times):
                if tau < t:
                    dt_step = times[1] - times[0] if len(times) > 1 else 0.1
                    integral_term += memory_kernel(t, tau) * history[i] * dt_step
            
            return local_term + integral_term
        
        # Solve
        x = np.zeros_like(time_grid)
        x[0] = 1.0
        
        for i in range(1, len(time_grid)):
            t = time_grid[i]
            dt = time_grid[i] - time_grid[i-1]
            dx = volterra_integro_differential(x[i-1], t, x[:i], time_grid[:i])
            x[i] = x[i-1] + dx * dt
        
        # Solution should remain positive and bounded
        assert x[-1] > 0
        assert x[-1] < x[0] + 5
    
    def test_memory_induced_recurrence(self, time_grid):
        """Test that memory can induce recurrence in dynamics."""
        
        def dynamics_with_recurrence(t, omega_memory=2.0):
            """Dynamics with oscillatory memory."""
            decay = np.exp(-0.05 * t)
            oscillation = np.cos(omega_memory * t)
            return decay * (1 + 0.3 * oscillation)
        
        trajectory = [dynamics_with_recurrence(t) for t in time_grid]
        
        # Should have local maxima (recurrences)
        # Find peaks
        peaks = 0
        for i in range(1, len(trajectory) - 1):
            if trajectory[i] > trajectory[i-1] and trajectory[i] > trajectory[i+1]:
                peaks += 1
        
        assert peaks > 0  # Should have recurrent behavior
    
    def test_non_markovian_witness_function(self, time_grid):
        """Test witness function for non-Markovianity."""
        
        def compute_witness(trajectory):
            """Compute d/dt of trajectory variance."""
            variance = np.var(trajectory)
            
            # Witness: if variance increases, non-Markovian
            diffs = np.diff(trajectory)
            increasing_variance_periods = np.sum(diffs > 0)
            
            return increasing_variance_periods / len(diffs)
        
        # Markovian: monotonic decay
        markov_traj = np.exp(-0.1 * time_grid)
        witness_markov = compute_witness(markov_traj)
        
        # Non-Markovian: oscillations
        non_markov_traj = np.exp(-0.1 * time_grid) * (1 + 0.3 * np.sin(2 * time_grid))
        witness_non_markov = compute_witness(non_markov_traj)
        
        # Non-Markovian should have higher witness value
        assert witness_non_markov > witness_markov
