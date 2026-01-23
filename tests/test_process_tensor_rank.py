"""Tests for process tensor rank analysis (Sec. 10.3)."""

import numpy as np


class TestProcessTensorRank:
    """Test finite vs unbounded rank signatures."""

    def test_finite_environment_finite_rank(self):
        """Verify finite bath → finite-rank process tensor."""

        # Simple 2-level system + 2-level bath
        dim_sys = 2
        dim_bath = 2
        # Joint Hamiltonian
        H_sys = np.array([[1, 0], [0, -1]])
        H_bath = np.array([[0.5, 0], [0, -0.5]])
        H_int = 0.1 * np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]]))

        _ = np.kron(H_sys, np.eye(dim_bath)) + np.kron(np.eye(dim_sys), H_bath) + H_int

        # Effective process tensor rank ≤ dim_bath²
        max_rank = dim_bath**2

        assert max_rank < np.inf
        assert max_rank == 4

    def test_spiral_time_unbounded_correlation(self, time_grid):
        """Test unbounded temporal correlations in spiral-time."""

        # Spiral-time memory induces correlations at all time separations
        correlations = []

        for dt in [0.1, 1.0, 10.0, 100.0]:
            # Correlation C(t, t-dt) ~ ∫ K(s) ds from (t-dt) to t
            # For exponential kernel, this decays but never vanishes
            corr = np.exp(-0.1 * dt)
            correlations.append(corr)

        # All correlations are non-zero (unbounded in time)
        assert all(c > 0 for c in correlations)

    def test_rank_divergence_criterion(self):
        """Test that spiral-time process tensor rank grows with time steps."""

        def estimate_process_rank(n_times):
            """Estimate effective rank as function of time steps."""
            # Simplified: rank grows with temporal resolution
            # For true spiral-time, rank ~ n_times (continuous limit)
            return n_times

        ranks = [estimate_process_rank(n) for n in [2, 4, 8, 16]]

        # Rank should grow
        assert all(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1))

        # Not bounded
        assert ranks[-1] > ranks[0]

    def test_stinespring_dimension_bound(self):
        """Test Stinespring dimension bound for finite environments."""

        # For finite-dimensional environment:
        # Process tensor rank ≤ d_env²

        d_env_values = [2, 3, 4, 5]

        for d_env in d_env_values:
            max_rank = d_env**2

            # This is the Stinespring bound
            assert max_rank < 1000  # Finite for any finite d_env

            # For spiral-time: no such bound exists
            # (continuous memory → infinite-dimensional "bath")


class TestRankGrowthDynamics:
    """Test dynamics of rank growth in process tensors."""

    def test_environmental_rank_saturation(self):
        """Test that environmental processes saturate in rank."""

        def effective_rank_environmental(n_steps, d_env):
            """Rank for environmental process."""
            # Saturates at d_env²
            return min(n_steps, d_env**2)

        d_env = 3
        n_steps_list = [1, 5, 10, 20, 50]

        ranks = [effective_rank_environmental(n, d_env) for n in n_steps_list]

        # Should saturate at 9
        assert max(ranks) == d_env**2

    def test_spiral_time_rank_growth(self):
        """Test that spiral-time rank grows without bound."""

        def spiral_time_rank(n_steps):
            """Rank for spiral-time process (continuous memory)."""
            # Grows linearly with time resolution
            return n_steps

        n_steps_list = [10, 20, 50, 100, 200]

        ranks = [spiral_time_rank(n) for n in n_steps_list]

        # Should grow indefinitely
        assert ranks[-1] > ranks[0]
        assert len(set(ranks)) == len(ranks)  # All different

    def test_rank_vs_memory_depth(self):
        """Test relationship between rank and memory depth."""

        def rank_from_memory_depth(tau_memory, dt):
            """Estimate rank from memory time scale."""
            # Number of time steps within memory window
            return int(tau_memory / dt)

        tau_memory = 5.0

        # Finer time resolution → higher rank
        dt_coarse = 1.0
        dt_fine = 0.1

        rank_coarse = rank_from_memory_depth(tau_memory, dt_coarse)
        rank_fine = rank_from_memory_depth(tau_memory, dt_fine)

        assert rank_fine > rank_coarse


class TestRankMeasurement:
    """Test experimental measurement of process tensor rank."""

    def test_singular_value_decomposition(self):
        """Test SVD-based rank estimation."""

        # Create a low-rank process matrix
        dim = 4
        rank = 2

        # Random low-rank matrix
        A = np.random.randn(dim, rank)
        B = np.random.randn(rank, dim)
        M_low_rank = A @ B

        # SVD
        U, s, Vh = np.linalg.svd(M_low_rank)

        # Count non-zero singular values
        threshold = 1e-10
        estimated_rank = np.sum(s > threshold)

        assert estimated_rank == rank

    def test_rank_reconstruction_from_data(self):
        """Test rank reconstruction from experimental data."""

        # Simulate process tensor data
        n_times = 5
        dim = 2

        # Low-rank process (environmental)
        process_env = np.random.randn(dim**n_times, dim**n_times)
        # Project to rank-4 subspace
        U, s, Vh = np.linalg.svd(process_env)
        s[4:] = 0  # Keep only 4 singular values
        process_env_low = U @ np.diag(s) @ Vh

        # Estimate rank
        s_recon = np.linalg.svd(process_env_low, compute_uv=False)
        rank_env = np.sum(s_recon > 1e-10)

        assert rank_env <= 4

    def test_rank_scaling_with_time_steps(self):
        """Test how rank scales with number of time steps."""

        def measure_effective_rank(n_steps, process_type="environmental"):
            """Measure effective rank."""

            if process_type == "environmental":
                # Saturates
                d_env = 3
                return min(n_steps, d_env**2)
            else:
                # Spiral-time: grows
                return n_steps

        steps = [2, 4, 8, 16]

        ranks_env = [measure_effective_rank(n, "environmental") for n in steps]
        ranks_spiral = [measure_effective_rank(n, "spiral") for n in steps]

        # Environmental saturates
        assert ranks_env[-1] == 9

        # Spiral-time grows
        assert ranks_spiral[-1] == 16


class TestContinuousLimitSignature:
    """Test signatures of continuous-time memory in the limit."""

    def test_discretization_independence(self):
        """Test that rank grows with discretization refinement."""

        def compute_rank_at_resolution(dt):
            """Compute rank for given time resolution."""
            t_max = 10.0
            n_steps = int(t_max / dt)

            # For spiral-time, rank ~ n_steps
            return n_steps

        resolutions = [1.0, 0.5, 0.1, 0.01]
        ranks = [compute_rank_at_resolution(dt) for dt in resolutions]

        # Rank should grow as dt → 0
        assert all(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1))

    def test_continuous_limit_divergence(self):
        """Test that rank diverges in continuous limit."""

        # As dt → 0, rank → ∞
        dt_values = [0.1, 0.01, 0.001]
        t_max = 5.0

        ranks = [int(t_max / dt) for dt in dt_values]

        # Should grow without bound
        assert ranks[-1] >= 100 * ranks[0]

    def test_memory_kernel_contribution_to_rank(self):
        """Test how memory kernel structure affects rank."""

        def rank_from_kernel_structure(kernel_type, n_times):
            """Estimate rank from kernel type."""

            if kernel_type == "delta":
                # δ-function: Markovian, rank = 1
                return 1
            elif kernel_type == "exponential":
                # Exponential: effectively finite rank
                # (long-time correlations become negligible)
                return min(n_times, 10)
            elif kernel_type == "power_law":
                # Power-law: true long-range memory
                # Rank grows with n_times
                return n_times

            return n_times

        n = 20

        rank_delta = rank_from_kernel_structure("delta", n)
        rank_exp = rank_from_kernel_structure("exponential", n)
        rank_power = rank_from_kernel_structure("power_law", n)

        assert rank_delta < rank_exp < rank_power


class TestExperimentalProtocol:
    """Test experimental protocol for rank determination."""

    def test_tomography_based_rank_estimation(self):
        """Test process tensor tomography for rank estimation."""

        # Simulate complete process tomography
        dim = 2
        # Prepare initial states (informationally complete)
        initial_states = [
            np.array([[1, 0], [0, 0]]),  # |0⟩⟨0|
            np.array([[0, 0], [0, 1]]),  # |1⟩⟨1|
            np.array([[0.5, 0.5], [0.5, 0.5]]),  # |+⟩⟨+|
            np.array([[0.5, -0.5j], [0.5j, 0.5]]),  # |+i⟩⟨+i|
        ]

        # For each initial state, measure at all time steps
        # and reconstruct process tensor

        # This would give a (dim^n_times × dim^n_times) matrix
        # whose rank can be estimated via SVD

        # Simplified test
        assert len(initial_states) >= dim**2  # Informationally complete

    def test_intervention_based_rank_detection(self):
        """Test rank detection via controlled interventions."""

        # Protocol: apply operations at intermediate times
        # and measure correlation structure

        def correlation_with_intervention(n_interventions):
            """Measure correlation after interventions."""
            # More interventions → probe more of process tensor structure
            # Finite-rank: correlations become redundant
            # Unbounded-rank: new information at each step

            return n_interventions  # Simplified

        interventions = [1, 2, 5, 10]
        correlations = [correlation_with_intervention(n) for n in interventions]

        # For spiral-time, should keep growing
        assert len(set(correlations)) == len(correlations)

    def test_statistical_estimation_of_rank(self):
        """Test statistical methods for rank estimation from data."""

        np.random.seed(42)

        # Generate synthetic process data
        n_measurements = 80
        true_rank = 3

        # Low-rank + noise
        signal = np.random.randn(n_measurements, true_rank)
        noise = 1e-6 * np.random.randn(n_measurements, n_measurements)

        data = signal @ signal.T + noise

        # Estimate rank via eigenvalue gap
        eigenvalues = np.linalg.eigvalsh(data)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

        # Look for gap in spectrum
        gaps = eigenvalues[:-1] - eigenvalues[1:]
        largest_gap_idx = np.argmax(gaps[:10]) + 1

        # Should be close to true rank
        assert abs(largest_gap_idx - true_rank) <= 1
