"""
Test local measures using THOI's existing test data.
"""

import torch
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add the thoi package to path
sys.path.insert(0, str(Path(__file__).parent / 'thoi'))

from thoi.measures.gaussian_copula import multi_order_measures
from thoi.commons import gaussian_copula_covmat

def test_with_thoi_data():
    """Test using THOI's existing test data."""
    print("Testing with THOI's existing test data...")
    
    # Load THOI test data
    current_dir = Path(__file__).parent
    data_path = current_dir / 'tests' / 'data' / 'X_random.tsv'
    
    if not data_path.exists():
        print(f"❌ Test data not found at {data_path}")
        return False
        
    X = pd.read_csv(data_path, sep='\t', header=None).values
    print(f"Loaded test data shape: {X.shape}")
    
    # Convert to tensor format for our functions
    X_tensor = torch.tensor(X.T, dtype=torch.float64).unsqueeze(0)  # (1, T, N)
    D, T, N = X_tensor.shape
    print(f"Tensor shape: {X_tensor.shape}")
    
    try:
        # Test import
        from thoi.measures.gaussian_copula import (
            local_nplets_measures, 
            time_averaged_local_measures
        )
        print("✅ Successfully imported local measures functions")
        
        # Compute covariance matrix
        covmat = gaussian_copula_covmat(X)
        covmat_tensor = torch.tensor(covmat, dtype=torch.float64).unsqueeze(0)
        print(f"Covariance matrix shape: {covmat_tensor.shape}")
        
        # Test local measures for a simple 3-plet
        nplets = torch.tensor([[0, 1, 2]], dtype=torch.long)
        
        local_result = local_nplets_measures(
            X_tensor, nplets, covmats=covmat_tensor,
            device='cpu', dtype=torch.float64
        )
        print(f"✅ Local measures computed! Shape: {local_result.shape}")
        
        # Test time-averaged measures
        time_avg_result = time_averaged_local_measures(
            X_tensor, covmats=covmat_tensor, min_order=3, max_order=3,
            device='cpu', dtype=torch.float64, bias_correction=True
        )
        print(f"✅ Time-averaged measures computed! Available orders: {list(time_avg_result.keys())}")
        
        # Compare with traditional THOI
        traditional_result = multi_order_measures(
            X_tensor, covmats=covmat_tensor, min_order=3, max_order=3,
            device='cpu', dtype=torch.float64
        )
        print(f"✅ Traditional measures computed!")
        
        # Compare the results
        order = 3
        local_avg = time_avg_result[order][0, 0, :]  # First n-plet, first dataset
        traditional = traditional_result[order][0, 0, :]  # Same
        
        diff = torch.abs(local_avg - traditional)
        max_diff = diff.max().item()
        rel_error = (diff / (torch.abs(traditional) + 1e-10)).max().item()
        
        print(f"\nComparison for order {order}:")
        print(f"  Traditional: {traditional}")
        print(f"  Local avg:   {local_avg}")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Max relative error: {rel_error:.2e}")
        
        if max_diff < 1e-8:
            print("✅ Results match!")
            return True
        else:
            print("❌ Results don't match!")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_thoi_data()
    if not success:
        sys.exit(1)
