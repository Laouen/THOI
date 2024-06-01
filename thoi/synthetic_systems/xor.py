import numpy as np
import pandas as pd

def generate_continuos_xor(alpha: float=1.0, T: int=10000):
    X1, X2, Z = np.random.normal(0, 1, (3, T))

    X1_XOR_X2 = np.logical_xor(X1 > 0, X2 > 0).astype(int)
    Zxor = alpha*(Z + 4*X1_XOR_X2) + (1-alpha)*Z

    return pd.DataFrame({'X1': X1, 'X2': X2, 'Zxor': Zxor})