import numpy as np
import scipy as sp
import torch
from thoi.measures.constants import TWOPIE


def _all_min_1_ids(n_variables):
    return [np.setdiff1d(range(n_variables),x) for x in range(n_variables)]


def gaussian_copula(X):
    """
    Transform the data into a Gaussian copula and compute the covariance matrix.

    Parameters:
    - X: A 2D numpy array of shape (T, N) where T is the number of samples and N is the number of variables.

    Returns:
    - X_gaussian: The data transformed into the Gaussian copula (same shape as the parameter input).
    - X_gaussian_covmat: The covariance matrix of the Gaussian copula transformed data.
    """

    assert X.ndim == 2, f'data must be 2D but got {X.ndim}D data input'

    T = X.shape[0]

    # Step 1 & 2: Rank the data and normalize the ranks
    sortid = np.argsort(X, axis=0) # sorting indices
    copdata = np.argsort(sortid, axis=0) # sorting sorting indices
    copdata = (copdata+1)/(T+1) # normalized indices in the [0,1] range 

    # Step 3: Apply the inverse CDF of the standard normal distribution
    X_gaussian = sp.special.ndtri(copdata) #uniform data to gaussian

    # Handle infinite values by setting them to 0 (optional and depends on use case)
    X_gaussian[np.isinf(X_gaussian)] = 0

    # Step 4: Compute the covariance matrix
    X_gaussian_covmat = np.cov(X_gaussian.T)

    return X_gaussian, X_gaussian_covmat


def _gaussian_entropy_bias_correction(N,T):
    """Computes the bias of the entropy estimator of a 
    N-dimensional multivariate gaussian with T sample"""
    psiterms = sp.special.psi((T - np.arange(1,N+1))/2)
    return (N*np.log(2/(T-1)) + np.sum(psiterms))/2


def _gaussian_entropy_estimation(cov_det, n_variables):
    return 0.5 * (n_variables*torch.log(TWOPIE) + torch.log(cov_det))