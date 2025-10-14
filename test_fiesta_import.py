import sys
import os
import numpy as np

# Add FIESTA path
sys.path.append('/work2/lbuc/iara/GitHub/ESSP')

print("Trying to import FIESTA...")
try:
    from FIESTA_II import FIESTA
    print("FIESTA imported successfully!")
    print(f"FIESTA type: {type(FIESTA)}")
    
    # Test with proper 2D dummy data (FIESTA expects CCF.shape[1] to exist)
    print("\nTesting FIESTA with 2D dummy data...")
    V_grid = np.linspace(-10, 10, 100)
    # Create 2D CCF data: (velocity_points, n_spectra)
    n_spectra = 5
    CCF = np.random.randn(100, n_spectra) * 0.1
    # Add a Gaussian-like signal
    for i in range(n_spectra):
        CCF[:, i] += np.exp(-(V_grid**2)/2)
    
    eCCF = np.ones_like(CCF) * 0.1
    
    print(f"V_grid shape: {V_grid.shape}")
    print(f"CCF shape: {CCF.shape}")
    print(f"eCCF shape: {eCCF.shape}")
    
    result = FIESTA(V_grid, CCF, eCCF)
    print(f"✅ FIESTA test successful! Result keys: {list(result.keys())}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Runtime error: {e}")
    import traceback
    traceback.print_exc()