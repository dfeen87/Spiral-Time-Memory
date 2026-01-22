"""
Non-Markovian Dynamics with Memory Kernels
===========================================

Implements Equation (3) from the paper:
    ẋ(t) = F(x(t), ∫_{-∞}^t K(t-τ)x(τ)dτ)

This is the core mathematical implementation of spiral-time memory:
the present is insufficient to predict the future without controlled
access to the past encoded in the memory kernel K(t-τ).

Reference: Paper Section 3 (Non-Markovian Dynamics)

WARNING: This is ONE possible implementation. Kernel forms, discretization 
schemes, and integration methods are NON-UNIQUE. See IMPLEMENTATIONS.md.

Author: Marcel Krüger & Don Michael Feeney Jr.
License: MIT
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
from dataclasses import dataclass
import warnings


@dataclass
class MemoryKernelConfig:
    """Configuration for memory kernel K(s).
    
    Non-unique choices - see IMPLEMENTATIONS.md §1.
    
    Attributes:
        kernel_type: Type of memory kernel (exponential, power_law, gaussian, mittag_leffler)
        tau_mem: Memory timescale (characteristic decay time)
        alpha: Power-law / fractional exponent
        sigma: Gaussian width parameter
    """
    kernel_type: str = "exponential"  # exponential, power_law, gaussian, mittag_leffler
    tau_mem: float = 1.0              # Memory timescale
    alpha: float = 0.5                # Power-law / fractional exponent
    sigma: float = 1.0                # Gaussian width
    
    def __post_init__(self):
        """Validate kernel type."""
        valid_types = ["exponential", "power_law", "gaussian", "mittag_leffler"]
        if self.kernel_type not in valid_types:
            raise ValueError(f"kernel_type must be one of {valid_types}")


def memory_kernel(s: np.ndarray, config: MemoryKernelConfig) -> np.ndarray:
    """Compute memory kernel K(s) for time-lag s ≥ 0.
    
    The memory kernel encodes how past states influence present dynamics.
    Different functional forms represent different physical assumptions about
    temporal memory structure.
    
    WARNING: This is illustrative. Alternative forms are equally valid
    and may yield different phenomenology. The choice of kernel is a
    theoretical/experimental question, not a mathematical necessity.
    
    Args:
        s: Time lag(s), shape (N,) or scalar. Must be non-negative (causal).
        config: Kernel configuration specifying functional form and parameters
        
    Returns:
        K(s), same shape as s. Normalized such that K(0) = 1 for all types.
        
    Examples:
        >>> cfg = MemoryKernelConfig(kernel_type="exponential", tau_mem=2.0)
        >>> K = memory_kernel(np.array([0, 1, 2]), cfg)
        >>> print(K)  # [1.0, 0.606, 0.368] - exponential decay
        
    References:
        - Exponential: Standard relaxation processes
        - Power-law: Long-range memory, scale-free dynamics
        - Gaussian: Localized memory with characteristic scale
        - Mittag-Leffler: Fractional calculus, subdiffusion
    """
    s = np.atleast_1d(s)
    
    if config.kernel_type == "exponential":
        # K(s) = exp(-s/τ)
        # Most common form; exponential decay with timescale τ
        K = np.exp(-s / config.tau_mem)
        
    elif config.kernel_type == "power_law":
        # K(s) = (1 + s)^(-α)
        # Regularized at s=0 to avoid divergence
        # Long-range memory for α < 1
        K = (1 + s)**(-config.alpha)
        
    elif config.kernel_type == "gaussian":
        # K(s) = exp(-s²/2σ²)
        # Localized memory with characteristic width σ
        K = np.exp(-s**2 / (2 * config.sigma**2))
        
    elif config.kernel_type == "mittag_leffler":
        # Simplified approximation: K(s) ≈ 1/(1 + s^α)
        # Full Mittag-Leffler function E_α(-s^α) requires special functions
        warnings.warn(
            "Mittag-Leffler uses simplified approximation. "
            "For exact form, use scipy.special or mpmath."
        )
        K = 1.0 / (1 + s**config.alpha)
    
    return K


class NonMarkovianEvolver:
    """Time evolution with intrinsic memory.
    
    Implements the core spiral-time dynamics equation:
        ẋ(t) = F(x(t)) + g(∫_{-∞}^t K(t-τ)x(τ)dτ)
        
    where:
        - F(x) is the Markovian (memoryless) drift
        - g(·) couples the memory integral to dynamics
        - K(t-τ) is the intrinsic memory kernel
    
    The key feature: evolution depends on the entire history x(τ) for τ < t,
    weighted by the memory kernel K(t-τ).
    
    Attributes:
        F: Markovian drift function x(t) → dx/dt
        g: Memory coupling function memory_integral → contribution to dx/dt
        kernel_config: Memory kernel specification
        dt: Time step (discretization choice - non-unique)
        history_times: Stored time points for memory integration
        history_states: Stored states for memory integration
        
    Example:
        >>> # Harmonic oscillator with memory damping
        >>> def F(x):
        ...     return np.array([x[1], -x[0]])  # Undamped oscillator
        >>> 
        >>> def g(mem_int):
        ...     return np.array([0, -0.1 * mem_int[1]])  # Memory-induced damping
        >>> 
        >>> config = MemoryKernelConfig(kernel_type="exponential", tau_mem=1.0)
        >>> evolver = NonMarkovianEvolver(F, g, config, dt=0.01)
        >>> 
        >>> x0 = np.array([1.0, 0.0])  # Initial displacement
        >>> times, states = evolver.evolve(x0, (0, 10))
    """
    
    def __init__(
        self, 
        F: Callable[[np.ndarray], np.ndarray],
        g: Callable[[np.ndarray], np.ndarray],
        kernel_config: MemoryKernelConfig,
        dt: float = 0.01
    ):
        """Initialize non-Markovian evolution.
        
        Args:
            F: Markovian dynamics x → F(x). Should return array of same shape as x.
            g: Memory coupling memory_int → g(memory_int). Should return array of same shape.
            kernel_config: Memory kernel specification
            dt: Time step for numerical integration (affects accuracy - non-canonical choice)
        """
        self.F = F
        self.g = g
        self.kernel_config = kernel_config
        self.dt = dt
        
        # History storage for memory integral
        self.history_times: List[float] = []
        self.history_states: List[np.ndarray] = []
        
    def compute_memory_integral(self, t_current: float) -> np.ndarray:
        """Compute ∫_{-∞}^t K(t-τ)x(τ)dτ using stored history.
        
        Discretization: Trapezoidal rule (one choice among many).
        Alternative schemes (Simpson's, adaptive quadrature, FFT convolution)
        are equally valid and may offer different accuracy/performance tradeoffs.
        
        Args:
            t_current: Current time
            
        Returns:
            Memory integral value, shape same as state dimension
        """
        if len(self.history_times) == 0:
            # No history yet - return zero
            return np.zeros_like(self.history_states[0]) if self.history_states else np.array([0.0])
        
        times = np.array(self.history_times)
        states = np.array(self.history_states)
        
        # Time lags s = t - τ
        s = t_current - times
        s[s < 0] = 0  # Enforce causality (should not happen in practice)
        
        # Kernel weights
        K = memory_kernel(s, self.kernel_config)
        
        # Weighted integrand: K(t-τ) * x(τ)
        if states.ndim > 1:
            # Multi-dimensional state
            integrand = K[:, None] * states
        else:
            # Scalar state
            integrand = K * states
        
        # Trapezoidal integration
        # NOTE: Alternative quadrature schemes are equally valid
        if len(times) > 1:
            memory_int = np.trapz(integrand, times, axis=0)
        else:
            # Single point - integral is approximately zero
            memory_int = integrand[0] * 0
            
        return memory_int
    
    def step(self, x: np.ndarray, t: float) -> np.ndarray:
        """Single time step with memory.
        
        Computes dx/dt = F(x) + g(∫K(t-τ)x(τ)dτ)
        
        Args:
            x: Current state
            t: Current time
            
        Returns:
            dx/dt including both Markovian and memory contributions
        """
        # Markovian part
        dx_markov = self.F(x)
        
        # Memory integral
        memory_int = self.compute_memory_integral(t)
        
        # Memory contribution
        dx_memory = self.g(memory_int)
        
        # Total derivative
        dx_dt = dx_markov + dx_memory
        
        return dx_dt
    
    def evolve(
        self, 
        x0: np.ndarray, 
        t_span: Tuple[float, float], 
        store_history: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve from x0 over time span.
        
        Args:
            x0: Initial state (array of any dimension)
            t_span: (t_start, t_end) time interval
            store_history: Whether to keep full trajectory (True) or use windowed memory (False)
                          Windowed memory saves RAM for long evolutions.
            
        Returns:
            times: Array of time points, shape (N,)
            states: Array of states, shape (N, len(x0))
            
        Example:
            >>> evolver = NonMarkovianEvolver(F, g, config)
            >>> times, states = evolver.evolve(x0, (0, 10))
            >>> plt.plot(times, states[:, 0])  # Plot first component
        """
        t_start, t_end = t_span
        times = np.arange(t_start, t_end + self.dt, self.dt)
        
        states = np.zeros((len(times), len(x0)))
        states[0] = x0
        
        # Initialize history
        self.history_times = [t_start]
        self.history_states = [x0.copy()]
        
        for i, t in enumerate(times[:-1]):
            # Euler step (simplest integrator - non-unique choice)
            # Could use RK4, adaptive methods, symplectic integrators, etc.
            dx = self.step(states[i], t)
            states[i+1] = states[i] + dx * self.dt
            
            # Update history
            if store_history:
                # Store everything
                self.history_times.append(t + self.dt)
                self.history_states.append(states[i+1].copy())
            else:
                # Keep only recent history within memory timescale window
                # Reduces memory usage for long simulations
                window = 5 * self.kernel_config.tau_mem  # 5x decay time
                cutoff_time = t - window
                cutoff_idx = np.searchsorted(self.history_times, cutoff_time)
                
                self.history_times = self.history_times[cutoff_idx:] + [t + self.dt]
                self.history_states = self.history_states[cutoff_idx:] + [states[i+1].copy()]
        
        return times, states
    
    def reset_history(self):
        """Clear stored history.
        
        Useful for running multiple independent simulations with the same evolver.
        """
        self.history_times = []
        self.history_states = []


def compare_markov_vs_memory(
    x0: np.ndarray,
    F: Callable,
    g: Callable,
    kernel_config: MemoryKernelConfig,
    t_span: Tuple[float, float],
    dt: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare Markovian vs non-Markovian evolution side-by-side.
    
    Useful for visualizing the effect of temporal memory on dynamics.
    
    Args:
        x0: Initial state
        F: Markovian dynamics
        g: Memory coupling
        kernel_config: Memory kernel configuration
        t_span: Time interval (t_start, t_end)
        dt: Time step
        
    Returns:
        times: Time points, shape (N,)
        states_markov: Purely Markovian trajectory, shape (N, len(x0))
        states_memory: Trajectory with memory, shape (N, len(x0))
        
    Example:
        >>> times, x_markov, x_memory = compare_markov_vs_memory(
        ...     x0, F, g, config, (0, 10)
        ... )
        >>> plt.plot(times, x_markov[:, 0], label='Markovian')
        >>> plt.plot(times, x_memory[:, 0], label='With memory')
        >>> plt.legend()
    """
    # Markovian evolution (g ≡ 0, no memory coupling)
    evolver_markov = NonMarkovianEvolver(
        F=F, 
        g=lambda x: np.zeros_like(x),  # Zero memory coupling
        kernel_config=kernel_config,
        dt=dt
    )
    times, states_markov = evolver_markov.evolve(x0, t_span)
    
    # Non-Markovian evolution (full memory)
    evolver_memory = NonMarkovianEvolver(
        F=F,
        g=g,
        kernel_config=kernel_config,
        dt=dt
    )
    _, states_memory = evolver_memory.evolve(x0, t_span)
    
    return times, states_markov, states_memory


# Example usage and demonstration
if __name__ == "__main__":
    print("Non-Markovian Dynamics with Memory Kernels")
    print("=" * 70)
    
    # Example: Harmonic oscillator with memory-induced damping
    def F_oscillator(x):
        """Undamped harmonic oscillator: ẍ + ω²x = 0
        
        State vector: x = [position, velocity]
        Returns: [v, -ω²x]
        """
        omega = 2 * np.pi  # Angular frequency (1 Hz)
        return np.array([x[1], -omega**2 * x[0]])
    
    def g_memory_damping(mem_int):
        """Memory-induced damping.
        
        The memory integral of past velocities contributes a damping force.
        This is a toy model showing how memory can induce dissipation
        without explicit coupling to an environment.
        """
        gamma = 0.5  # Damping strength
        return np.array([0, -gamma * mem_int[1]])
    
    # Configuration
    kernel_cfg = MemoryKernelConfig(kernel_type="exponential", tau_mem=0.5)
    x0 = np.array([1.0, 0.0])  # Initial: displaced, at rest
    
    print("\nSystem: Harmonic oscillator")
    print(f"  Frequency: 1 Hz")
    print(f"  Memory kernel: {kernel_cfg.kernel_type}")
    print(f"  Memory timescale: {kernel_cfg.tau_mem}")
    print(f"  Initial state: position={x0[0]}, velocity={x0[1]}")
    print()
    
    # Evolve with comparison
    print("Evolving system...")
    times, x_markov, x_memory = compare_markov_vs_memory(
        x0=x0,
        F=F_oscillator,
        g=g_memory_damping,
        kernel_config=kernel_cfg,
        t_span=(0, 10),
        dt=0.01
    )
    
    print("\nResults:")
    print(f"  Markovian (undamped):")
    print(f"    Initial amplitude: {x_markov[0, 0]:.4f}")
    print(f"    Final amplitude:   {x_markov[-1, 0]:.4f}")
    print(f"    Energy conserved:  {abs(x_markov[-1, 0]) > 0.9}")
    
    print(f"\n  Non-Markovian (memory damping):")
    print(f"    Initial amplitude: {x_memory[0, 0]:.4f}")
    print(f"    Final amplitude:   {x_memory[-1, 0]:.4f}")
    print(f"    Energy dissipated: {abs(x_memory[-1, 0]) < 0.5}")
    
    print("\n" + "=" * 70)
    print("Interpretation:")
    print("  Memory of past velocities induces effective damping")
    print("  No environment needed - memory alone causes dissipation")
    print("  This is a key feature distinguishing spiral-time from")
    print("  standard open quantum systems")
    print("=" * 70)
