from typing import List, Union, Optional
import numpy as np
import scipy as sp
import torch


def _get_string_metric(batched_res: np.ndarray, metric:str):
    '''
    Get the metric from the batched results returning the average over the D axis.

    params:
    - batched_res (np.ndarray): The batched results with shape (batch_size, D, 4) where 4 is the number of metrics (tc, dtc, o, s)
    - metric (str): The metric to get. One of tc, dtc, o or s
    '''

    METRICS = ['tc', 'dtc', 'o', 's']
    metric_idx = METRICS.index(metric)

    # |batch_size|
    return batched_res[:,:,metric_idx].mean(axis=1)


def gaussian_copula(X: np.ndarray):
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

def gaussian_copula_covmat(X: np.ndarray):
    return gaussian_copula(X)[1]

def _to_numpy(X):
    if isinstance(X, torch.Tensor):
        # If the tensor is on a GPU/TPU, move it to CPU first, then convert to NumPy
        return X.detach().cpu().numpy()
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise TypeError(f"Unsupported type: {type(X)}")


def _get_device(use_cpu:bool=False):
    """Set the use of GPU if available"""
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')
    return device


def _normalize_input_data(X: Union[np.ndarray, torch.tensor, List[np.ndarray], List[torch.tensor]],
                         covmat_precomputed: bool=False,
                         T: Optional[Union[int, List[int]]]=None,
                         use_cpu: bool=False):
    '''
    brief: Normalize the input data to be a list of covariance matrices with shape (D, N, N) where D is the lenght of the list and N is the number of variables in the system.

    Parameters:
    - X: A list of 2D numpy arrays or tensors of shape: 1. (T, N) where T is the number of samples if X are multivariate series. 2. a list of 2D covariance matrices with shape (N, N).
    - covmat_precomputed: A boolean flag to indicate if the input data is a list of covariance matrices or multivariate series.
    - T (optional): A list of integers indicating the number of samples for each multivariate series.
    - use_cpu: A boolean flag to indicate if the computation should be done on the CPU.
    '''

    # Get device to use
    device = _get_device(use_cpu)

    # Handle different options for X parameter. Accept multivariate data or covariance matrix
    if covmat_precomputed:
        
        if isinstance(X, (np.ndarray, torch.Tensor)):
            assert X.shape[-2] == X.shape[-1], 'Covariance matrix should be square'
            assert len(X.shape) in [2, 3], 'Covariance matrix should have dimensions (N, N) or (D, N, N)'
            covmats = torch.as_tensor(X)
            covmats = covmats.unsqueeze(0) if len(covmats.shape) == 2 else covmats
        else:
            assert all([len(x.shape) == 2 for x in X]), 'All covariance matrices should have dimensions (N, N)'
            assert all([x.shape[0] == x.shape[1] == X[0].shape[0] for x in X]), 'All covariance matrices should have same dimensions (N, N)'
            covmats = torch.stack([torch.as_tensor(x) for x in X])
    else:
        
        if isinstance(X, (np.ndarray, torch.Tensor)):
            X = _to_numpy(X)
            assert len(X.shape) in [2, 3], 'Covariance matrix should have dimensions (T, N) or (D, T, N)'
            if len(X.shape) == 2:
                X = [X]
            else:
                X = [X[i] for i in range(X.shape[0])]
        else:
            assert all([len(x.shape) == 2 for x in X]), 'All multivariate series should have dimensions (T, N) where T my vary and N be constant across all series'
            assert all([x.shape[1] == X[0].shape[1] for x in X]), 'All multivariate series should have dimensions (T, N) where T my vary and N be constant across all series'
            X = [_to_numpy(x) for x in X]
        
        covmats = torch.stack([torch.from_numpy(gaussian_copula_covmat(x)) for x in X])
        T = [x.shape[0] for x in X]

    D, N = covmats.shape[:2]
    
    # Handle different options for T parameter
    if isinstance(T, int):
        T = [T] * D

    return covmats, D, N, T, device
