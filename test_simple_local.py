"""
Simple test for local measures functionality.
"""

import torch
import sys
from pathlib import Path

# Add the thoi package to path
sys.path.insert(0, str(Path(__file__).parent / 'thoi'))

def simple_test():
    """Quick test to verify local measures work."""
    print("Testing basic functionality...")
    
    # Generate simple test data
    torch.manual_seed(42)
    D, T, N = 1, 100, 4
    data = torch.randn(D, T, N)
    
    try:
        from thoi.measures.gaussian_copula_local import local_nplets_measures
        print("✅ Successfully imported local_nplets_measures")
        
        # Test with a simple 3-plet
        nplets = torch.tensor([[0, 1, 2]], dtype=torch.long)
        
        result = local_nplets_measures(
            data, nplets, device='cpu', dtype=torch.float64
        )
        
        print(f"✅ Local measures computed successfully!")
        print(f"   Shape: {result.shape}")
        print(f"   TC range: [{result[0,0,:,0].min():.6f}, {result[0,0,:,0].max():.6f}]")
        
        # Test multi-order version
        from thoi.measures.gaussian_copula_local import local_multi_order_measures
        print("✅ Successfully imported local_multi_order_measures")
        
        multi_result = local_multi_order_measures(
            data, min_order=3, max_order=3, device='cpu', dtype=torch.float64
        )
        
        print(f"✅ Multi-order local measures computed!")
        print(f"   Orders available: {list(multi_result.keys())}")
        print(f"   Order 3 shape: {multi_result[3].shape}")
        
        # Test time averaging
        from thoi.measures.gaussian_copula_local import time_averaged_local_measures
        print("✅ Successfully imported time_averaged_local_measures")
        
        avg_result = time_averaged_local_measures(
            data, min_order=3, max_order=3, device='cpu', dtype=torch.float64,
            bias_correction=True
        )
        
        print(f"✅ Time-averaged measures computed with bias correction!")
        print(f"   Shape: {avg_result[3].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n🎉 All basic tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
