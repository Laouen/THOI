"""
Test script to verify the gaussian_copula_cov_opt implementation
matches the original gaussian_copula_covmat functionality.
"""
import numpy as np
import torch
import sys
import os

# Add the thoi directory to the path
sys.path.append(os.path.abspath('.'))

from thoi.commons import gaussian_copula_covmat, gaussian_copula_cov_opt, _normalize_input_data

def test_basic_functionality():
    """Test that the new function produces similar results to the old one."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create test data
    T, N = 100, 5
    data = np.random.randn(T, N)
    
    # Test old function
    old_cov = gaussian_copula_covmat(data)
    
    # Test new function (single dataset)
    data_tensor = torch.from_numpy(data).unsqueeze(0)  # (1, T, N)
    _, new_cov = gaussian_copula_cov_opt(data_tensor)
    new_cov_np = new_cov[0].numpy()
    
    # Compare results
    diff = np.abs(old_cov - new_cov_np)
    max_diff = np.max(diff)
    
    print(f"Max difference between old and new implementation: {max_diff:.10f}")
    print(f"Results are {'SIMILAR' if max_diff < 1e-10 else 'DIFFERENT'}")
    
    return max_diff < 1e-6  # Allow for small numerical differences

def test_normalize_input_data():
    """Test that _normalize_input_data works with the new implementation."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create multiple datasets
    datasets = [
        np.random.randn(100, 5),
        np.random.randn(120, 5),
        np.random.randn(80, 5)
    ]
    
    # Test the normalize function
    try:
        covmats, D, N, T_list = _normalize_input_data(datasets, device=torch.device('cpu'))
        
        print(f"Successfully processed {D} datasets")
        print(f"Number of variables: {N}")
        print(f"Sample sizes: {T_list}")
        print(f"Covariance matrices shape: {covmats.shape}")
        
        # Check that covariance matrices are positive definite
        for i in range(D):
            eigenvals = torch.linalg.eigvals(covmats[i]).real
            is_pos_def = torch.all(eigenvals > -1e-10)
            print(f"Dataset {i}: {'Positive definite' if is_pos_def else 'Not positive definite'}")
        
        return True
    except Exception as e:
        print(f"Error in _normalize_input_data: {e}")
        return False

def test_batch_processing():
    """Test batch processing functionality."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create test data with multiple datasets
    D, T, N = 3, 100, 4
    data_batch = torch.randn(D, T, N)
    
    # Test with different batch sizes
    _, cov1 = gaussian_copula_cov_opt(data_batch, batch_D=1)
    _, cov2 = gaussian_copula_cov_opt(data_batch, batch_D=2)  
    _, cov3 = gaussian_copula_cov_opt(data_batch)  # No batching
    
    # All should give the same result
    diff12 = torch.max(torch.abs(cov1 - cov2))
    diff13 = torch.max(torch.abs(cov1 - cov3))
    
    print(f"Max difference batch_D=1 vs batch_D=2: {diff12:.10f}")
    print(f"Max difference batch_D=1 vs no_batch: {diff13:.10f}")
    
    return diff12 < 1e-10 and diff13 < 1e-10

if __name__ == "__main__":
    print("Testing gaussian_copula_cov_opt implementation...")
    print("=" * 50)
    
    print("\n1. Testing basic functionality...")
    test1_passed = test_basic_functionality()
    
    print("\n2. Testing _normalize_input_data integration...")
    test2_passed = test_normalize_input_data()
    
    print("\n3. Testing batch processing...")
    test3_passed = test_batch_processing()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Basic functionality: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Integration test: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Batch processing: {'PASSED' if test3_passed else 'FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
