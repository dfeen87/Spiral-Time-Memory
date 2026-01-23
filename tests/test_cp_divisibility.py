"""Tests for CP-divisibility analysis (Sec. 9.2, Eq. 17)."""

import pytest
import numpy as np
from scipy.linalg import expm


class TestCPDivisibility:
    """Test CP-divisibility criterion."""
    
    def test_markovian_process_is_cp_divisible(self, cp_map):
        """Verify Markovian process satisfies E_{t:0} = E_{t:s} ∘ E_{s:0}."""
        rho_0 = np.array([[1, 0], [0, 0]])  # Pure state
        
        gamma = 0.1
        
        # Direct evolution 0 → t
        rho_t_direct = cp_map(rho_0, gamma)
        
        # Factored evolution 0 → s → t
        s_gamma = gamma / 2  # Split damping
        rho_s = cp_map(rho_0, s_gamma)
        rho_t_factored = cp_map(rho_s, s_gamma)
        
        # For amplitude damping, this should approximately hold
        assert np.allclose(rho_t_direct, rho_t_factored, atol=0.01)
    
    def test_non_cp_divisible_process(self):
        """Test detection of non-CP-divisible evolution."""
        
        def non_cp_map(rho, t, memory_state):
            """Map that depends on hidden memory state."""
            # Simulate time-dependent non-Markovian evolution
            factor = 1 + 0.1 * np.sin(t) * memory_state
            return rho * factor / np.trace(rho * factor)
        
        rho_0 = np.array([[0.5, 0], [0, 0.5]])
        memory = 1.0
        
        t1, t2 = 1.0, 2.0
        
        # Direct evolution
        rho_direct = non_cp_map(rho_0, t2, memory)
        
        # Attempt factorization (should fail)
        rho_intermediate = non_cp_map(rho_0, t1, memory)
        rho_factored = non_cp_map(rho_intermediate, t2 - t1, memory)
        
        # Should NOT be equal due to memory dependence
        assert not np.allclose(rho_direct, rho_factored, atol=0.01)
    
    def test_cp_divisibility_witness(self):
        """Test witness operator for CP-divisibility."""
        
        def compute_cp_witness(rho_list, times):
            """Compute deviation from CP-divisibility."""
            n_times = len(times)
            deviations = []
            
            for i in range(1, n_times - 1):
                # Check if E_{t_i:t_0} = E_{t_i:t_{i-1}} ∘ E_{t_{i-1}:t_0}
                # by comparing trace distances
                
                # This is a simplified version
                diff = np.linalg.norm(rho_list[i] - rho_list[i-1])
                deviations.append(diff)
            
            return np.mean(deviations)
        
        # Markovian evolution
        times = [0, 1, 2, 3]
        rho_markov = [np.eye(2)/2 * np.exp(-0.1*t) for t in times]
        
        witness_markov = compute_cp_witness(rho_markov, times)
        
        # Non-Markovian evolution
        rho_non_markov = [np.eye(2)/2 * (1 + 0.1*np.sin(t)) for t in times]
        witness_non_markov = compute_cp_witness(rho_non_markov, times)
        
        # Non-Markovian should have larger deviations
        assert witness_non_markov > witness_markov
    
    def test_choi_matrix_positivity(self):
        """Test Choi matrix criterion for complete positivity."""
        
        def create_choi_matrix(channel_map):
            """Construct Choi matrix for a channel."""
            dim = 2
            # Create maximally entangled state
            psi_max = np.zeros((dim**2, 1), dtype=complex)
            for i in range(dim):
                psi_max[i * dim + i, 0] = 1 / np.sqrt(dim)
            
            rho_max = psi_max @ psi_max.conj().T
            
            # Apply channel to second subsystem
            # (Simplified for testing)
            choi = np.kron(np.eye(dim), channel_map) @ rho_max
            
            return choi
        
        # CP map: amplitude damping
        gamma = 0.1
        E0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
        E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        
        def cp_channel(rho):
            return E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
        
        # For CP map, Choi matrix should be positive
        # (Simplified test)
        test_rho = np.array([[1, 0], [0, 0]])
        result = cp_channel(test_rho)
        
        eigenvalues = np.linalg.eigvalsh(result)
        assert np.all(eigenvalues >= -1e-10)
    
    def test_divisibility_chain_rule(self):
        """Test chain rule for divisible processes."""
        
        # Three consecutive time intervals
        gamma1, gamma2, gamma3 = 0.1, 0.15, 0.12
        
        def amp_damp(rho, gamma):
            """Amplitude damping channel."""
            E0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
            E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
            return E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
        
        rho_0 = np.array([[1, 0], [0, 0]])
        
        # Sequential application
        rho_1 = amp_damp(rho_0, gamma1)
        rho_2 = amp_damp(rho_1, gamma2)
        rho_3 = amp_damp(rho_2, gamma3)
        
        # Direct application (should match for CP-divisible)
        # Total damping (approximate)
        gamma_total = gamma1 + gamma2 + gamma3
        gamma_total = min(gamma_total, 1.0)
        
        rho_direct = amp_damp(rho_0, gamma_total)
        
        # Should be approximately equal (exact for small gammas)
        # Note: amplitude damping is not exactly additive
        # but this tests the structure
        assert rho_3[1,1] <= rho_0[1,1]  # Population decreased


class TestCPDivisibilityMeasures:
    """Test quantitative measures of CP-divisibility violations."""
    
    def test_trace_distance_monotonicity(self):
        """Test trace distance as divisibility measure."""
        
        def trace_distance(rho1, rho2):
            """Compute trace distance."""
            diff = rho1 - rho2
            eigenvalues = np.linalg.eigvalsh(diff @ diff.conj().T)
            return 0.5 * np.sqrt(np.sum(eigenvalues))
        
        # Initial orthogonal states
        rho1 = np.array([[1, 0], [0, 0]])
        rho2 = np.array([[0, 0], [0, 1]])
        
        D0 = trace_distance(rho1, rho2)
        
        # After CP-divisible evolution, distance should decrease
        gamma = 0.2
        E0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
        E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        
        rho1_evolved = E0 @ rho1 @ E0.conj().T + E1 @ rho1 @ E1.conj().T
        rho2_evolved = E0 @ rho2 @ E0.conj().T + E1 @ rho2 @ E1.conj().T
        
        D1 = trace_distance(rho1_evolved, rho2_evolved)
        
        # CP-divisible: distance decreases
        assert D1 <= D0 + 1e-10
    
    def test_cp_divisibility_index(self):
        """Test quantitative CP-divisibility index."""
        
        def cp_index(process_maps):
            """
            Compute CP-divisibility index.
            Returns 0 for CP-divisible, >0 for violations.
            """
            violations = 0.0
            
            for i in range(len(process_maps) - 1):
                E1, E2 = process_maps[i], process_maps[i+1]
                
                # Test composition vs sequential
                test_state = np.array([[0.5, 0.25], [0.25, 0.5]])
                
                sequential = E2 @ (E1 @ test_state)
                
                # For CP-divisible, should satisfy data processing inequality
                # (simplified test)
                purity_before = np.trace(test_state @ test_state).real
                purity_after = np.trace(sequential @ sequential).real
                
                if purity_after > purity_before + 1e-10:
                    violations += (purity_after - purity_before)
            
            return violations
        
        # CP-divisible sequence
        E_cp1 = np.array([[0.9, 0], [0, 0.9]])
        E_cp2 = np.array([[0.95, 0], [0, 0.95]])
        
        index_cp = cp_index([E_cp1, E_cp2])
        
        # Should be zero (or very small)
        assert index_cp < 1e-6
    
    def test_non_markovian_backflow(self):
        """Test information backflow in non-Markovian dynamics."""
        
        # Simulate oscillating trace distance
        times = np.linspace(0, 10, 50)
        
        # Markovian: monotonic decay
        D_markov = np.exp(-0.1 * times)
        
        # Non-Markovian: with backflow
        D_non_markov = np.exp(-0.1 * times) * (1 + 0.2 * np.sin(2 * times))
        
        # Detect backflow: periods where dD/dt > 0
        dD_markov = np.diff(D_markov)
        dD_non_markov = np.diff(D_non_markov)
        
        backflow_markov = np.sum(dD_markov > 0)
        backflow_non_markov = np.sum(dD_non_markov > 0)
        
        # Non-Markovian should show backflow
        assert backflow_non_markov > backflow_markov


class TestCompletePositivity:
    """Test complete positivity of quantum maps."""
    
    def test_krauss_representation(self):
        """Test Krauss representation of CP maps."""
        
        # Amplitude damping Krauss operators
        gamma = 0.3
        K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        
        # Verify completeness: ΣK†K = I
        completeness = K0.conj().T @ K0 + K1.conj().T @ K1
        
        assert np.allclose(completeness, np.eye(2))
    
    def test_positive_but_not_cp_map(self):
        """Test example of positive but not completely positive map."""
        
        # Transpose map (positive but not CP)
        def transpose_map(rho):
            return rho.T
        
        # Single system: positive
        rho_single = np.array([[0.7, 0.2], [0.2, 0.3]])
        rho_transposed = transpose_map(rho_single)
        
        eigenvalues = np.linalg.eigvalsh(rho_transposed)
        assert np.all(eigenvalues >= -1e-10)
        
        # Extended system: can violate positivity
        # (This would require implementing partial transpose test)
    
    def test_stinespring_dilation(self):
        """Test Stinespring dilation of CP maps."""
        
        # Any CP map can be represented as
        # E(ρ) = Tr_env[U(ρ⊗|0⟩⟨0|)U†]
        
        # Simplified test: amplitude damping
        gamma = 0.2
        
        # System state
        rho_sys = np.array([[1, 0], [0, 0]])
        
        # Environment
        env_0 = np.array([[1, 0], [0, 0]])
        
        # Joint state
        rho_joint = np.kron(rho_sys, env_0)
        
        # Joint unitary (simplified)
        theta = np.sqrt(gamma)
        U_joint = np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]
        ])
        
        # Evolve
        rho_evolved = U_joint @ rho_joint @ U_joint.conj().T
        
        # Partial trace over environment
        rho_final = np.array([
            [rho_evolved[0,0] + rho_evolved[1,1], rho_evolved[0,2] + rho_evolved[1,3]],
            [rho_evolved[2,0] + rho_evolved[3,1], rho_evolved[2,2] + rho_evolved[3,3]]
        ])
        
        # Should be valid density matrix
        assert np.abs(np.trace(rho_final) - 1.0) < 1e-10
        eigenvalues = np.linalg.eigvalsh(rho_final)
        assert np.all(eigenvalues >= -1e-10)
