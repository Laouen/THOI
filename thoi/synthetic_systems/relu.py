import numpy as np
import pandas as pd

def ReLU(X, cutoff=0):
    return np.maximum(X,cutoff)

def relu(alpha: float=1.0, beta: float=1.0, pow_factor: float=0.5, T: float=10000):

    assert 0 <= alpha <= 1.0, 'alpha must be in range [0,1]'
    assert 0 <= beta <= 1.0, 'beta must be in range [0,1]'

    Z_syn, Z_red = np.random.normal(0, 1, (2,T))

    X1 = alpha*np.power(ReLU(Z_syn), pow_factor)    + beta*Z_red
    X2 = -alpha*np.power(ReLU(-Z_syn), pow_factor)  + beta*Z_red

    return pd.DataFrame({'X1': X1, 'X2': X2, 'Z_syn': Z_syn, 'Z_red': Z_red})