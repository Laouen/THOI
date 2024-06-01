import numpy as np
import pandas as pd

def flat(alpha: float=1.0, beta: float=1.0, gamma: float=1.0, T: int=1000):

    assert 0 <= alpha <= 1.0, 'alpha must be in range [0,1]'
    assert 0 <= beta <= 1.0, 'beta must be in range [0,1]'
    assert 0 <= gamma <= 1.0, 'gamma must be in range [0,1]'

    # Generate random samples
    Z00, Z01, Z1, Z2, Z3, Z4, Z5, Z6 = [np.random.normal(0, 1, T) for _ in range(8)]

    # Define the variables
    X1 = alpha*np.log(np.abs(Z00) + 1) + beta*Z01 + gamma*Z1
    X2 = alpha*Z00                     + beta*Z01 + gamma*Z2
    X3 = alpha*np.power(Z00,2)         + beta*Z01 + gamma*Z3
    X4 = alpha*np.exp(Z00)             + beta*Z01 + gamma*Z4
    X5 = alpha*np.sin(Z00)             + beta*Z01 + gamma*Z5
    X6 = alpha*np.cos(Z00)             + beta*Z01 + gamma*Z6

    return pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'X6': X6,
        'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'Z4': Z4, 'Z5': Z5, 'Z6': Z6,
        'Z00': Z00, 'Z01': Z01
    })