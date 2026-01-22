"""
Process Tensor Rank Analysis
=============================

Implements Section 10.3 of the paper: Process Tensor Obstruction.

Key distinction from environmental decoherence:
- Environmental: Finite-dimensional bath → process tensor of finite rank
- Spiral-Time: Unbounded temporal correlations in χ(t) → no finite-rank representation

This module tests whether a reconstructed process tensor can be approximated
by a finite-rank representation (environmental) or requires unbounded rank (Spiral-Time).

Reference: Paper Section 10.3
Author: Marcel Krüger & Don Michael Feeney Jr.
License: MIT
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.linalg import svd, svdvals


@dataclass
class ProcessTensorRankConfig:
    """Configuration for process tensor rank analysis."""
    rank_threshold: float = 0.95    # Cumulative singular value threshold
    min_rank_ratio: float = 0.1     # Minimum ratio for finite-rank claim
    n_timesteps: int = 5            # Number of time steps
    hilbert_dim: int = 2            # System dimension


class ProcessTensorRankAnalyzer:
    """Analyze rank structure of process tensors.
    
    Spiral-Time prediction: Process tensor cannot be well-approximated by
    low-rank decomposition due to unbounded χ(t) correlations.
    
    Environmental prediction: Process tensor admits finite-rank approximation
    corresponding to finite bath dimension.
    """
    
    def __init__(self, config: ProcessTensorRankConfig):
        self.config = config
    
    def construct_process_tensor(
        self,
        channels: List[np.ndarray],
        times: List[float]
    ) -> np.ndarray:
        """Construct process tensor from sequence of channels.
        
        The process tensor encodes all multi-time correlations.
        
        Args:
            channels: List of cumulative channel Choi matrices
            times: Corresponding time points
            
        Returns:
            Process tensor (high-dimensional array)
        """
        n_steps = len(channels)
        d = self.config.hilbert_dim
        
        # Process tensor dimension: d^(2n) for n time steps
        # We use a simplified representation here
        
        # For illustration, construct from channel composition
        # Full process tensor would be rank-4 tensor
        
        # Simplified: Use matrix representation
        pt_dim = d**(2 * n_steps)
        process_tensor = np.zeros((pt_dim, pt_dim), dtype=complex)
        
        # Build from channels (simplified construction)
        # In practice, would use full process matrix formalism
        
        for i in range(n_steps):
            # Add channel contribution
            # This is a placeholder for full process tensor construction
            choi = channels[i]
            
            # Kronecker product expansion
            if i == 0:
                process_tensor[:choi.shape[0], :choi.shape[1]] = choi
            else:
                # Compose with previous
                # Full implementation would use proper tensor composition
                pass
        
        return process_tensor
    
    def analyze_rank_structure(
        self,
        process_tensor: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, any]:
        """Analyze singular value spectrum of process tensor.
        
        Args:
            process_tensor: Process tensor matrix
            verbose: Print analysis
            
        Returns:
            Dictionary with rank analysis results
        """
        if verbose:
            print("Process Tensor Rank Analysis (Section 10.3)")
            print("=" * 70)
        
        # Singular value decomposition
        singular_values = svdvals(process_tensor)
        singular_values = np.sort(singular_values)[::-1]  # Descending order
        
        # Normalize
        sv_normalized = singular_values / singular_values[0]
        
        # Cumulative sum
        sv_cumsum = np.cumsum(singular_values)
        sv_cumsum_normalized = sv_cumsum / sv_cumsum[-1]
        
        # Effective rank (entropy-based)
        sv_prob = singular_values / np.sum(singular_values)
        sv_prob = sv_prob[sv_prob > 1e-12]
        effective_rank_entropy = np.exp(-np.sum(sv_prob * np.log(sv_prob)))
        
        # Participation ratio
        participation_ratio = 1.0 / np.sum(sv_prob**2)
        
        # Rank at threshold
        idx_threshold = np.argmax(sv_cumsum_normalized >= self.config.rank_threshold)
        rank_at_threshold = idx_threshold + 1
        
        # Ratio to full rank
        full_rank = len(singular_values)
        rank_ratio = rank_at_threshold / full_rank
        
        # Verdict
        is_finite_rank = rank_ratio < self.config.min_rank_ratio
        
        if verbose:
            print(f"Full rank: {full_rank}")
            print(f"Effective rank (entropy): {effective_rank_entropy:.2f}")
            print(f"Participation ratio: {participation_ratio:.2f}")
            print(f"Rank at {self.config.rank_threshold*100:.0f}% threshold: {rank_at_threshold}")
            print(f"Rank ratio: {rank_ratio:.4f}")
            print()
            
            if is_finite_rank:
                print("VERDICT: FINITE-RANK (consistent with environmental origin)")
                print(f"  → Can be approximated by {rank_at_threshold}-dimensional bath")
            else:
                print("VERDICT: NO FINITE-RANK REPRESENTATION")
                print("  → Consistent with unbounded Spiral-Time correlations")
            print("=" * 70)
        
        return {
            'singular_values': singular_values,
            'sv_normalized': sv_normalized,
            'sv_cumsum_normalized': sv_cumsum_normalized,
            'effective_rank': effective_rank_entropy,
            'participation_ratio': participation_ratio,
            'rank_at_threshold': rank_at_threshold,
            'full_rank': full_rank,
            'rank_ratio': rank_ratio,
            'is_finite_rank': is_finite_rank,
            'verdict': 'ENVIRONMENTAL' if is_finite_rank else 'SPIRAL-TIME'
        }
    
    def test_rank_growth(
        self,
        process_generator: callable,
        max_timesteps: int = 10
    ) -> Dict[str, any]:
        """Test how rank scales with number of time steps.
        
        Environmental: Rank saturates at bath dimension
        Spiral-Time: Rank grows unboundedly
        
        Args:
            process_generator: Function n_steps → process tensor
            max_timesteps: Maximum time steps to test
            
        Returns:
            Scaling analysis results
        """
        print("\nRank Scaling Analysis")
        print("=" * 70)
        
        timesteps_range = range(2, max_timesteps + 1)
        ranks = []
        effective_ranks = []
        
        for n_steps in timesteps_range:
            # Generate process tensor
            pt = process_generator(n_steps)
            
            # Analyze rank
            result = self.analyze_rank_structure(pt, verbose=False)
            ranks.append(result['rank_at_threshold'])
            effective_ranks.append(result['effective_rank'])
            
            print(f"n={n_steps}: rank={result['rank_at_threshold']}, "
                  f"eff_rank={result['effective_rank']:.2f}")
        
        # Fit scaling
        timesteps_arr = np.array(list(timesteps_range))
        ranks_arr = np.array(ranks)
        
        # Linear fit: rank ~ a * n + b
        coeffs_linear = np.polyfit(timesteps_arr, ranks_arr, 1)
        
        # Exponential fit: rank ~ a * exp(b * n)
        # Use log-linear
        log_ranks = np.log(ranks_arr + 1)
        coeffs_exp = np.polyfit(timesteps_arr, log_ranks, 1)
        
        # Determine scaling
        growth_rate = coeffs_linear[0]
        
        if growth_rate > 0.5:  # Growing significantly
            scaling = "UNBOUNDED (consistent with Spiral-Time)"
        else:
            scaling = "SATURATING (consistent with finite bath)"
        
        print()
        print(f"Linear growth rate: {coeffs_linear[0]:.3f}")
        print(f"Scaling verdict: {scaling}")
        print("=" * 70)
        
        # Visualization
        self._plot_rank_scaling(timesteps_arr, ranks_arr, effective_ranks, 
                               coeffs_linear, scaling)
        
        return {
            'timesteps': timesteps_arr,
            'ranks': ranks_arr,
            'effective_ranks': effective_ranks,
            'growth_rate': growth_rate,
            'scaling_verdict': scaling
        }
    
    def _plot_rank_scaling(
        self,
        timesteps: np.ndarray,
        ranks: np.ndarray,
        effective_ranks: List[float],
        fit_coeffs: np.ndarray,
        verdict: str
    ):
        """Plot rank scaling with time steps."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Singular values
        ax1.semilogy(timesteps, ranks, 'o-', linewidth=2, markersize=8, label='Rank at 95%')
        
        # Linear fit
        fit_line = np.polyval(fit_coeffs, timesteps)
        ax1.plot(timesteps, fit_line, '--', linewidth=2, alpha=0.7, label='Linear fit')
        
        ax1.set_xlabel('Number of Time Steps')
        ax1.set_ylabel('Rank')
        ax1.set_title(f'Rank Scaling\n{verdict}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Effective rank
        ax2.plot(timesteps, effective_ranks, 's-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Number of Time Steps')
        ax2.set_ylabel('Effective Rank (Entropy)')
        ax2.set_title('Effective Rank Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('process_tensor_rank_scaling.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Figure saved: process_tensor_rank_scaling.png")
    
    def visualize_singular_spectrum(
        self,
        singular_values: np.ndarray,
        title: str = "Singular Value Spectrum"
    ):
        """Visualize singular value spectrum."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Normalized singular values
        sv_norm = singular_values / singular_values[0]
        
        ax1.semilogy(sv_norm, 'o-', linewidth=2, markersize=6)
        ax1.axhline(1e-2, color='red', linestyle='--', label='1% threshold')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Normalized Singular Value')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative sum
        sv_cumsum = np.cumsum(singular_values) / np.sum(singular_values)
        
        ax2.plot(sv_cumsum, 'o-', linewidth=2, markersize=6)
        ax2.axhline(0.95, color='red', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Cumulative Sum (Normalized)')
        ax2.set_title('Cumulative Singular Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('singular_spectrum.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Figure saved: singular_spectrum.png")


def demonstrate_finite_vs_unbounded():
    """Demonstrate finite-rank vs unbounded process tensors."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Finite-Rank vs Unbounded Process Tensors")
    print("=" * 70)
    
    config = ProcessTensorRankConfig(n_timesteps=5, hilbert_dim=2)
    analyzer = ProcessTensorRankAnalyzer(config)
    
    # Example 1: Finite-rank (environmental)
    print("\n--- FINITE-RANK PROCESS (Environmental) ---")
    d = config.hilbert_dim
    n = config.n_timesteps
    
    # Construct low-rank process tensor
    rank = 4  # Bath dimension
    pt_finite = np.random.randn(d**(2*n), rank) @ np.random.randn(rank, d**(2*n))
    pt_finite = pt_finite @ pt_finite.T.conj()  # Make Hermitian
    
    result_finite = analyzer.analyze_rank_structure(pt_finite)
    analyzer.visualize_singular_spectrum(
        result_finite['singular_values'], 
        "Finite-Rank (Environmental)"
    )
    
    # Example 2: Full-rank (Spiral-Time)
    print("\n--- FULL-RANK PROCESS (Spiral-Time) ---")
    pt_full = np.random.randn(d**(2*n), d**(2*n)) + \
              1j * np.random.randn(d**(2*n), d**(2*n))
    pt_full = pt_full @ pt_full.T.conj()
    
    result_full = analyzer.analyze_rank_structure(pt_full)
    analyzer.visualize_singular_spectrum(
        result_full['singular_values'],
        "Full-Rank (Spiral-Time)"
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Finite-rank verdict: {result_finite['verdict']}")
    print(f"  Rank ratio: {result_finite['rank_ratio']:.4f}")
    print(f"\nFull-rank verdict: {result_full['verdict']}")
    print(f"  Rank ratio: {result_full['rank_ratio']:.4f}")


if __name__ == "__main__":
    demonstrate_finite_vs_unbounded()
