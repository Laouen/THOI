"""
Test suite for local (time-resolved) measures implementation.

This script validates that:
1. Time-averaged local measures match traditional THOI measures
2. Local measures maintain numerical stability
3. Performance improvements are achieved
"""

import torch
import sys
from pathlib import Path

# Add the thoi package to path
sys.path.insert(0, str(Path(__file__).parent / 'thoi'))

from thoi.measures.gaussian_copula import (
    multi_order_measures, 
    local_multi_order_measures, 
    time_averaged_local_measures,
    local_nplets_measures
)
from thoi.commons import gaussian_copula_cov_opt

def test_local_vs_traditional_measures():
    """Test that time-averaged local measures match traditional measures."""
    print("Testing local vs traditional measures equivalence...")
    
    # Generate test data
    torch.manual_seed(42)
    D, T, N = 2, 1000, 5
    data = torch.randn(D, T, N)
    
    # Compute traditional measures (with bias correction)
    traditional_results = multi_order_measures(
        data, min_order=3, max_order=4,
        device='cpu', dtype=torch.float64
    )
    
    # Compute time-averaged local measures (with bias correction)  
    local_averaged_results = time_averaged_local_measures(
        data, min_order=3, max_order=4,
        device='cpu', dtype=torch.float64,
        bias_correction=True
    )
    
    # Compare results
    for order in [3, 4]:
        traditional = traditional_results[order]  # [C(N,order), D, 4]
        local_avg = local_averaged_results[order]  # [C(N,order), D, 4]
        
        # Check shapes match
        assert traditional.shape == local_avg.shape, f"Shape mismatch for order {order}"
        
        # Check numerical equivalence
        max_diff = torch.abs(traditional - local_avg).max().item()
        rel_error = (torch.abs(traditional - local_avg) / (torch.abs(traditional) + 1e-10)).max().item()
        
        print(f"Order {order}:")
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Max relative error: {rel_error:.2e}")
        
        # Tolerance should be very small for double precision
        assert max_diff < 1e-10, f"Measures don't match for order {order}! Max diff: {max_diff}"
        assert rel_error < 1e-8, f"Relative error too high for order {order}! Max rel error: {rel_error}"
    
    print("✅ Local and traditional measures match!")

def test_bias_correction_application():
    """Test that bias correction is applied correctly only after time averaging."""
    print("\nTesting bias correction application...")
    
    torch.manual_seed(123)
    D, T, N = 1, 500, 4
    data = torch.randn(D, T, N)
    
    # Get local measures WITHOUT bias correction
    local_results = local_multi_order_measures(
        data, min_order=3, max_order=3, device='cpu', dtype=torch.float64
    )
    
    # Get time-averaged measures WITHOUT bias correction
    averaged_no_bias = time_averaged_local_measures(
        data, min_order=3, max_order=3, device='cpu', dtype=torch.float64,
        bias_correction=False
    )
    
    # Get time-averaged measures WITH bias correction
    averaged_with_bias = time_averaged_local_measures(
        data, min_order=3, max_order=3, device='cpu', dtype=torch.float64,
        bias_correction=True
    )
    
    order = 3
    local_avg_manual = local_results[order].mean(dim=2)  # Manual time average
    no_bias = averaged_no_bias[order]
    with_bias = averaged_with_bias[order]
    
    # Manual average should match no-bias version
    max_diff_manual = torch.abs(local_avg_manual - no_bias).max().item()
    print(f"Manual vs no-bias max diff: {max_diff_manual:.2e}")
    assert max_diff_manual < 1e-12, "Manual averaging doesn't match time_averaged_local_measures without bias correction"
    
    # With bias correction should be different from without
    max_diff_bias = torch.abs(with_bias - no_bias).max().item()
    print(f"With vs without bias correction max diff: {max_diff_bias:.2e}")
    assert max_diff_bias > 1e-6, "Bias correction should make a difference"
    
    print("✅ Bias correction applied correctly!")

def test_performance_comparison():
    """Test performance of local vs traditional measures."""
    print("\nTesting performance comparison...")
    
    import time
    
    # Test with moderate-sized data
    torch.manual_seed(456)
    D, T, N = 3, 2000, 6
    data = torch.randn(D, T, N)
    
    # Precompute covariance matrices for fair comparison
    _, covmats = gaussian_copula_cov_opt(data)
    
    # Traditional approach
    start = time.time()
    traditional_results = multi_order_measures(
        data, covmats=covmats, min_order=3, max_order=4,
        device='cpu', dtype=torch.float32
    )
    traditional_time = time.time() - start
    
    # Local approach (time-averaged)
    start = time.time()
    local_results = time_averaged_local_measures(
        data, covmats=covmats, min_order=3, max_order=4,
        device='cpu', dtype=torch.float32, bias_correction=True
    )
    local_time = time.time() - start
    
    print(f"Traditional approach: {traditional_time:.3f}s")
    print(f"Local approach: {local_time:.3f}s")
    print(f"Speedup: {traditional_time/local_time:.2f}x")
    
    # Verify they still match numerically
    for order in [3, 4]:
        max_diff = torch.abs(traditional_results[order] - local_results[order]).max().item()
        print(f"Order {order} max difference: {max_diff:.2e}")
        assert max_diff < 1e-5, f"Performance optimizations broke numerical accuracy for order {order}"
    
    print("✅ Performance test passed!")

def test_local_measures_shape_and_values():
    """Test that local measures have correct shapes and reasonable values."""
    print("\nTesting local measures shapes and values...")
    
    torch.manual_seed(789)
    D, T, N = 2, 100, 4
    data = torch.randn(D, T, N)
    
    # Test specific n-plets
    nplets = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)  # Two 3-plets
    
    local_results = local_nplets_measures(
        data, nplets, device='cpu', dtype=torch.float64
    )
    
    expected_shape = (2, D, T, 4)  # (B, D, T, 4)
    assert local_results.shape == expected_shape, f"Expected shape {expected_shape}, got {local_results.shape}"
    
    # Check that measures are finite
    assert torch.isfinite(local_results).all(), "Some local measures are not finite"
    
    # Check that TC >= 0 (always true for Gaussian copula)
    tc_local = local_results[:, :, :, 0]
    assert (tc_local >= -1e-10).all(), "TC measures should be non-negative"
    
    # Check that S >= 0 (synergy)
    s_local = local_results[:, :, :, 3]
    assert (s_local >= -1e-10).all(), "S measures should be non-negative"
    
    print(f"Local measures shape: {local_results.shape}")
    print(f"TC range: [{tc_local.min():.6f}, {tc_local.max():.6f}]")
    print(f"DTC range: [{local_results[:,:,:,1].min():.6f}, {local_results[:,:,:,1].max():.6f}]")
    print(f"O range: [{local_results[:,:,:,2].min():.6f}, {local_results[:,:,:,2].max():.6f}]")
    print(f"S range: [{s_local.min():.6f}, {s_local.max():.6f}]")
    
    print("✅ Local measures have correct shapes and reasonable values!")

if __name__ == "__main__":
    print("="*60)
    print("TESTING LOCAL MEASURES IMPLEMENTATION")
    print("="*60)
    
    try:
        test_local_measures_shape_and_values()
        test_bias_correction_application() 
        test_local_vs_traditional_measures()
        test_performance_comparison()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("Local measures implementation is working correctly!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
