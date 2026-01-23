"""Tests for environmental baseline models (Sec. 10)."""

import pytest
import numpy as np
from scipy.linalg import expm


class TestEnvironmentalBaselines:
    """Test standard environmental decoherence models."""
    
    def test_pure_dephasing(self):
        """Test pure dephasing channel."""
        
        def pure_dephasing(rho, gamma):
            """Z-basis dephasing."""
            return np.array([
                [rho[0, 0], rho[0, 1] * np.exp(-gamma)],
                [rho[1, 0] * np.exp(-gamma), rho[1, 1]]
            ])
        
        rho_0 = np.array([[0.5, 0.5], [0.5, 0.5]])
        gamma = 0.5
        
        rho_final = pure_dephasing(rho_0, gamma)
        
        # Diagonal elements unchanged
        assert np.allclose(rho_final[0, 0], rho_0[0, 0])
        assert np.allclose(rho_final[1, 1], rho_0[1, 1])
        
        # Off-diagonal decayed
        assert np.abs(rho_final[0, 1]) < np.abs(rho_0[0, 1])
    
    def test_amplitude_damping(self):
        """Test amplitude damping channel."""
        
        def amplitude_damping(rho, gamma):
            """Energy relaxation."""
            E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
            E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
            return E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
        
        rho_0 = np.array([[1, 0], [0, 0]])  # Excited state
        gamma = 0.3
        
        rho_final = amplitude_damping(rho_0, gamma)
        
        # Population should transfer from |1⟩ to |0⟩
        assert rho_final[1, 1] < rho_0[1, 1]
        assert rho_final[0, 0] > rho_0[0, 0]
    
    def test_depolarizing_channel(self):
        """Test depolarizing channel."""
        
        def depolarize(rho, p):
            """Depolarizing channel."""
            return (1 - p) * rho + p * np.eye(2) / 2
        
        rho_0 = np.array([[1, 0], [0, 0]])
        p = 0.5
        
        rho_final = depolarize(rho_0, p)
        
        # Should approach maximally mixed state
        assert np.allclose(rho_final, 0.5 * rho_0 + 0.25 * np.eye(2))
    
    def test_phase_damping(self):
        """Test phase damping (random phase kick)."""
        
        def phase_damping(rho, lambda_param):
            """Phase damping channel."""
            K0 = np.array([[1, 0], [0, np.sqrt(1 - lambda_param)]])
            K1 = np.array([[0, 0], [0, np.sqrt(lambda_param)]])
            return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T
        
        rho_0 = np.array([[0.5, 0.5], [0.5, 0.5]])
        lambda_param = 0.4
        
        rho_final = phase_damping(rho_0, lambda_param)
        
        # Coherences decay
        assert np.abs(rho_final[0, 1]) <= np.abs(rho_0[0, 1])


class TestSpinBosonModel:
    """Test spin-boson model as environmental baseline."""
    
    def test_ohmic_spectral_density(self):
        """Test Ohmic bath spectral density."""
        
        def spectral_density_ohmic(omega, alpha, omega_c):
            """J(ω) = παω e^(-ω/ω_c)."""
            return np.pi * alpha * omega * np.exp(-omega / omega_c)
        
        omega = 1.0
        alpha = 0.1
        omega_c = 5.0
        
        J = spectral_density_ohmic(omega, alpha, omega_c)
        
        assert J > 0
        assert np.isfinite(J)
    
    def test_thermal_bath_correlation(self):
        """Test thermal bath correlation function."""
        
        def thermal_correlation(t, beta, omega_c):
            """C(t) for thermal bath at inverse temperature β."""
            # Simplified: classical limit
            return (1 / beta) * np.exp(-omega_c * abs(t))
        
        t = 1.0
        beta = 1.0  # Temperature
        omega_c = 2.0
        
        C = thermal_correlation(t, beta, omega_c)
        
        assert C > 0
        assert C < 1 / beta  # Should decay
    
    def test_bloch_redfield_master_equation(self):
        """Test Bloch-Redfield master equation."""
        
        def bloch_redfield_evolution(rho, H_sys, gamma_list, dt):
            """Single step of Bloch-Redfield evolution."""
            # Unitary part
            U = expm(-1j * H_sys * dt)
            rho_unitary = U @ rho @ U.conj().T
            
            # Dissipative part (simplified)
            dim = len(rho)
            dissipator = np.zeros_like(rho, dtype=complex)
            
            for gamma in gamma_list:
                # Simple damping
                dissipator += gamma * (np.eye(dim) / dim - rho)
            
            rho_final = rho_unitary + dt * dissipator
            rho_final /= np.trace(rho_final)
            
            return rho_final
        
        rho_0 = np.array([[1, 0], [0, 0]])
        H_sys = np.array([[1, 0], [0, -1]])
        gamma_list = [0.1]
        dt = 0.01
        
        rho_evolved = bloch_redfield_evolution(rho_0, H_sys, gamma_list, dt)
        
        # Should remain valid density matrix
        assert np.abs(np.trace(rho_evolved) - 1.0) < 1e-6
        eigenvalues = np.linalg.eigvalsh(rho_evolved)
        assert np.all(eigenvalues >= -1e-10)


class TestCollisionalModels:
    """Test collisional (repeated interactions) models."""
    
    def test_single_collision(self):
        """Test single collision with environment particle."""
        
        def collision(rho_sys, rho_env, U_interaction):
            """Single collision."""
            # Joint state
            rho_joint = np.kron(rho_sys, rho_env)
            
            # Interaction
            rho_after = U_interaction @ rho_joint @ U_interaction.conj().T
            
            # Partial trace over environment
            dim_sys = len(rho_sys)
            dim_env = len(rho_env)
            
            rho_sys_final = np.zeros((dim_sys, dim_sys), dtype=complex)
            for i in range(dim_env):
                block_start = i * dim_sys
                block_end = (i + 1) * dim_sys
                rho_sys_final += rho_after[block_start:block_end, block_start:block_end]
            
            return rho_sys_final
        
        rho_sys = np.array([[1, 0], [0, 0]])
        rho_env = np.array([[1, 0], [0, 0]])
        
        # CNOT-like interaction
        U_int = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        
        rho_final = collision(rho_sys, rho_env, U_int)
        
        assert np.abs(np.trace(rho_final) - 1.0) < 1e-10
    
    def test_repeated_collisions(self):
        """Test repeated collision model."""
        
        def repeated_collisions(rho_sys, n_collisions, coupling_strength):
            """Multiple collision sequence."""
            rho = rho_sys.copy()
            
            for _ in range(n_collisions):
                # Fresh environment particle
                rho_env = np.array([[1, 0], [0, 0]])
                
                # Weak interaction
                theta = coupling_strength
                U = np.array([
                    [1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]
                ])
                
                # Joint evolution
                rho_joint = np.kron(rho, rho_env)
                rho_joint = U @ rho_joint @ U.conj().T
                
                # Trace out environment
                rho = np.array([
                    [rho_joint[0,0] + rho_joint[1,1], rho_joint[0,2] + rho_joint[1,3]],
                    [rho_joint[2,0] + rho_joint[3,1], rho_joint[2,2] + rho_joint[3,3]]
                ])
                
                rho /= np.trace(rho)
            
            return rho
        
        rho_0 = np.array([[0.5, 0.5], [0.5, 0.5]])
        rho_final = repeated_collisions(rho_0, 10, 0.1)
        
        # Coherences should decay
        assert np.abs(rho_final[0, 1]) < np.abs(rho_0[0, 1])


class TestStructuredEnvironments:
    """Test structured (non-Markovian) environmental models."""
    
    def test_finite_temperature_bath(self):
        """Test finite-temperature bath effects."""
        
        def finite_temp_damping(rho, gamma, n_thermal):
            """Damping with thermal excitations."""
            # Generalized amplitude damping
            E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
            E1 = np.array([[0, np.sqrt(gamma * (1 + n_thermal))], [0, 0]])
            E2 = np.array([[np.sqrt(gamma * n_thermal), 0], [0, 0]])
            
            result = E0 @ rho @ E0.conj().T
            result += E1 @ rho @ E1.conj().T
            result += E2 @ rho @ E2.conj().T
            
            return result
        
        rho_0 = np.array([[1, 0], [0, 0]])
        gamma = 0.2
        n_thermal = 0.3  # Average thermal photons
        
        rho_final = finite_temp_damping(rho_0, gamma, n_thermal)
        
        # Valid density matrix
        assert np.abs(np.trace(rho_final) - 1.0) < 1e-10
        eigenvalues = np.linalg.eigvalsh(rho_final)
        assert np.all(eigenvalues >= -1e-10)
    
    def test_cavity_qed_environment(self):
        """Test cavity QED environmental model."""
        
        def cavity_decay(rho, kappa, dt):
            """Cavity photon decay."""
            # Lindblad form: dρ/dt = κ(aρa† - ½{a†a,ρ})
            
            # Annihilation operator (truncated)
            a = np.array([[0, 1], [0, 0]])
            a_dag = a.T
            
            # Lindblad superoperator
            lindblad_term = kappa * (a @ rho @ a_dag - 
                                    0.5 * (a_dag @ a @ rho + rho @ a_dag @ a))
            
            rho_new = rho + dt * lindblad_term
            rho_new /= np.trace(rho_new)
            
            return rho_new
        
        rho_0 = np.array([[0, 0], [0, 1]])  # One photon
        kappa = 0.5  # Decay rate
        dt = 0.1
        
        rho_final = cavity_decay(rho_0, kappa, dt)
        
        # Population should decrease
        assert rho_final[1, 1] < rho_0[1, 1]
    
    def test_non_markovian_spectral_density(self):
        """Test spectral density leading to non-Markovian dynamics."""
        
        def lorentzian_spectral_density(omega, gamma, omega_0, width):
            """Lorentzian J(ω) with sharp cutoff."""
            return gamma * width / ((omega - omega_0)**2 + width**2)
        
        # Sharp spectral feature → memory effects
        omega_values = np.linspace(0, 10, 100)
        omega_0 = 5.0
        width = 0.5  # Narrow → non-Markovian
        
        J = [lorentzian_spectral_density(omega, 1.0, omega_0, width) 
             for omega in omega_values]
        
        # Peak at resonance
        peak_idx = np.argmax(J)
        assert np.isclose(omega_values[peak_idx], omega_0, atol=0.1)


class TestEnvironmentalCharacterization:
    """Test characterization of environmental properties."""
    
    def test_bath_correlation_time(self):
        """Test extraction of bath correlation time."""
        
        def bath_correlation(t, tau_c):
            """Bath correlation function with time scale τ_c."""
            return np.exp(-abs(t) / tau_c)
        
        tau_c = 2.0
        times = np.linspace(0, 10, 50)
        
        corr = [bath_correlation(t, tau_c) for t in times]
        
        # At t = τ_c, should decay to 1/e
        idx_tau = np.argmin(np.abs(times - tau_c))
        assert np.isclose(corr[idx_tau], np.exp(-1), atol=0.1)
    
    def test_bath_temperature_estimation(self):
        """Test temperature estimation from thermal statistics."""
        
        def thermal_occupation(omega, beta):
            """Bose-Einstein distribution."""
            return 1 / (np.exp(beta * omega) - 1)
        
        omega = 1.0
        T = 1.0  # Temperature
        beta = 1 / T
        
        n_thermal = thermal_occupation(omega, beta)
        
        # Should be finite and positive
        assert n_thermal > 0
        assert np.isfinite(n_thermal)
    
    def test_decoherence_rate_scaling(self):
        """Test scaling of decoherence rate with bath properties."""
        
        def decoherence_rate(coupling, spectral_density):
            """Γ ∝ g² J(ω)."""
            return coupling**2 * spectral_density
        
        couplings = [0.1, 0.2, 0.3]
        J = 1.0
        
        rates = [decoherence_rate(g, J) for g in couplings]
        
        # Should scale as g²
        assert np.isclose(rates[1] / rates[0], 4.0, rtol=0.01)
        assert np.isclose(rates[2] / rates[0], 9.0, rtol=0.01)
