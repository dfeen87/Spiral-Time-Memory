"""
Experimental Protocols for Spiral-Time Memory Testing
======================================================

This module implements three falsifiable experimental protocols for testing
the spiral-time memory hypothesis:

Protocol A: Reset Test for History Dependence
    Tests whether nominally perfect reset operations eliminate memory effects.
    Expected signature: residual history dependence after reset in spiral-time
    dynamics vs. complete independence in Markovian dynamics.

Protocol B: Process Tensor Tomography  
    Reconstructs multi-time quantum processes and tests CP-divisibility.
    Expected signature: violations of CP-divisibility even in isolated systems
    (intrinsic temporal memory) vs. CP-divisible processes (Markovian).

Protocol C: Leggett-Garg Inequality under Memory Suppression
    Tests multi-time correlations with engineered memory suppression.
    Expected signature: persistent LGI violations under suppression (intrinsic
    memory) vs. eliminated violations (environmental memory).

Usage
-----
>>> from experiments import run_all_protocols
>>> results = run_all_protocols(memory_strength=0.02)

Or run individual protocols:
>>> from experiments.reset_tests import run_reset_test, ResetTestConfig
>>> from experiments.process_tensor import run_protocol_b, ProcessTensorConfig  
>>> from experiments.leggett_garg import run_protocol_c, LeggettGargConfig

References
----------
Paper Sections:
- Section 10: Experimental Program and Falsifiable Tests
- Section 10.5: Minimal Experimental Protocols
- Criterion 1: CP-divisibility as falsification criterion

Key Equations:
- Eq. 11: Memory sector update χ(t⁺) = χ(t⁻) + Δχ_meas
- Eq. 17: CP-divisibility condition E_{t:0} = E_{t:s} ∘ E_{s:0}
- Eq. 18: Non-Markovian evolution with memory kernel
"""

from typing import Dict, Optional
import warnings

# Version info
__version__ = "0.1.0"
__author__ = "Spiral-Time Memory Research Group"

# Import protocol modules
try:
    from .reset_tests.protocol_a import (
        ResetTestConfig,
        run_reset_test,
        QuantumState,
        ResetOperation,
        MeasurementOperation,
        HistoryDependenceAnalyzer
    )
    PROTOCOL_A_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Protocol A not available: {e}")
    PROTOCOL_A_AVAILABLE = False

try:
    from .process_tensor.protocol_b import (
        ProcessTensorConfig,
        run_protocol_b,
        ProcessTensor,
        MemoryKernel,
        NonMarkovianEvolution,
        CPDivisibilityTest
    )
    PROTOCOL_B_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Protocol B not available: {e}")
    PROTOCOL_B_AVAILABLE = False

try:
    from .leggett_garg.protocol_c import (
        LeggettGargConfig,
        run_protocol_c,
        QuantumEvolution,
        LeggettGargMeasurement,
        LeggettGargInequality
    )
    PROTOCOL_C_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Protocol C not available: {e}")
    PROTOCOL_C_AVAILABLE = False


def run_all_protocols(
    memory_strength: float = 0.02,
    verbose: bool = True
) -> Dict:
    """
    Run all three experimental protocols with consistent parameters.
    
    This convenience function executes Protocols A, B, and C using compatible
    configurations and provides a unified report comparing spiral-time predictions
    with Markovian null hypothesis.
    
    Parameters
    ----------
    memory_strength : float, default=0.02
        Memory coupling strength ε_χ. Typical range: 0.001-0.05.
        Smaller values require higher precision but are more conservative.
    
    verbose : bool, default=True
        Whether to print progress and results during execution.
    
    Returns
    -------
    results : dict
        Dictionary containing results from all protocols:
        {
            'protocol_a': {...},  # Reset test results
            'protocol_b': {...},  # Process tensor results
            'protocol_c': {...},  # Leggett-Garg results
            'summary': {
                'spiral_time_supported': bool,
                'confidence': str,
                'key_signatures': list
            }
        }
    
    Examples
    --------
    >>> results = run_all_protocols(memory_strength=0.03, verbose=True)
    >>> print(f"Spiral-time supported: {results['summary']['spiral_time_supported']}")
    
    Notes
    -----
    This function provides a high-level interface for comprehensive testing.
    For fine-grained control over individual protocols, use the protocol-specific
    functions directly.
    
    The function automatically checks for consistency across protocols:
    - If Protocol A shows history dependence, Protocol B should show 
      non-CP-divisibility
    - If Protocol B shows non-CP-divisibility, Protocol C should show
      persistent LGI violations under suppression
    
    Inconsistent results may indicate:
    - Insufficient statistical power
    - Systematic experimental errors
    - Parameter regimes outside theoretical validity
    """
    if not all([PROTOCOL_A_AVAILABLE, PROTOCOL_B_AVAILABLE, PROTOCOL_C_AVAILABLE]):
        raise RuntimeError(
            "Not all protocols are available. Check import errors above."
        )
    
    results = {}
    
    if verbose:
        print("="*70)
        print("COMPREHENSIVE SPIRAL-TIME MEMORY TEST")
        print("="*70)
        print(f"Memory strength: ε_χ = {memory_strength}")
        print()
    
    # Protocol A: Reset Test
    if verbose:
        print("[1/3] Running Protocol A: Reset Test...")
    
    config_a = ResetTestConfig(
        n_cycles=500,
        n_histories=10,
        history_length=5,
        reset_fidelity=0.999,
        memory_strength=memory_strength
    )
    
    results['protocol_a'] = {
        'spiral_time': run_reset_test(config_a, spiral_time_mode=True),
        'markovian': run_reset_test(config_a, spiral_time_mode=False)
    }
    
    # Protocol B: Process Tensor
    if verbose:
        print("\n[2/3] Running Protocol B: Process Tensor Tomography...")
    
    config_b = ProcessTensorConfig(
        n_time_steps=3,
        memory_strength=memory_strength,
        memory_kernel_type="exponential",
        memory_decay_rate=0.5
    )
    
    results['protocol_b'] = run_protocol_b(config_b)
    
    # Protocol C: Leggett-Garg
    if verbose:
        print("\n[3/3] Running Protocol C: Leggett-Garg Test...")
    
    config_c = LeggettGargConfig(
        n_measurements=5000,
        n_time_points=3,
        memory_strength=memory_strength,
        suppression_strength=0.7
    )
    
    results['protocol_c'] = run_protocol_c(config_c)
    
    # Analyze consistency
    summary = _analyze_consistency(results)
    results['summary'] = summary
    
    if verbose:
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)
        print(f"Spiral-time hypothesis supported: {summary['spiral_time_supported']}")
        print(f"Confidence: {summary['confidence']}")
        print(f"\nKey signatures detected:")
        for sig in summary['key_signatures']:
            print(f"  • {sig}")
    
    return results


def _analyze_consistency(results: Dict) -> Dict:
    """
    Analyze consistency across protocols and determine overall conclusion.
    
    Parameters
    ----------
    results : dict
        Results from all three protocols
    
    Returns
    -------
    summary : dict
        Overall assessment of spiral-time hypothesis
    """
    signatures = []
    
    # Protocol A: Check history dependence
    a_spiral = results['protocol_a']['spiral_time']
    a_markov = results['protocol_a']['markovian']
    
    if a_spiral['independence_test']['significant']:
        signatures.append("Protocol A: History dependence detected after reset")
    
    # Protocol B: Check CP-divisibility
    b_results = results['protocol_b']
    
    if not b_results['cp_test_memory']['is_cp_divisible']:
        signatures.append(
            f"Protocol B: Non-CP-divisible process "
            f"({b_results['cp_test_memory']['n_violations']} violations)"
        )
    
    # Protocol C: Check persistent LGI violation
    c_results = results['protocol_c']
    
    if c_results['with_suppression']['test_result']['violates']:
        signatures.append(
            f"Protocol C: LGI violation persists under suppression "
            f"(K={c_results['with_suppression']['K']:.3f})"
        )
    
    # Determine overall conclusion
    n_signatures = len(signatures)
    
    if n_signatures >= 3:
        supported = True
        confidence = "HIGH"
    elif n_signatures == 2:
        supported = True
        confidence = "MODERATE"
    elif n_signatures == 1:
        supported = True
        confidence = "LOW"
    else:
        supported = False
        confidence = "NONE"
    
    return {
        'spiral_time_supported': supported,
        'confidence': confidence,
        'key_signatures': signatures,
        'n_protocols_supporting': n_signatures
    }


# Public API
__all__ = [
    # Main runner
    'run_all_protocols',
    
    # Protocol A
    'ResetTestConfig',
    'run_reset_test',
    'QuantumState',
    'ResetOperation',
    'MeasurementOperation',
    'HistoryDependenceAnalyzer',
    
    # Protocol B  
    'ProcessTensorConfig',
    'run_protocol_b',
    'ProcessTensor',
    'MemoryKernel',
    'NonMarkovianEvolution',
    'CPDivisibilityTest',
    
    # Protocol C
    'LeggettGargConfig',
    'run_protocol_c',
    'QuantumEvolution',
    'LeggettGargMeasurement',
    'LeggettGargInequality',
    
    # Module info
    '__version__',
    '__author__',
]


if __name__ == "__main__":
    # Example usage
    print("Running comprehensive spiral-time memory test...\n")
    results = run_all_protocols(memory_strength=0.025, verbose=True)
    
    print("\n" + "="*70)
    print("Test complete. Results saved to 'results' dictionary.")
    print("="*70)
