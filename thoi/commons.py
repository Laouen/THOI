from typing import List, Union, Optional
import numpy as np
import scipy as sp
import torch

from thoi.typing import TensorLikeArray


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
    .. _gaussian_copula:
    
    Gaussian Copula Transformation
    ==============================
    
    Transform the data into a Gaussian copula and compute the covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array of shape (T, N) where T is the number of samples and N is the number of variables.

    Returns
    -------
    X_gaussian : np.ndarray
        The data transformed into the Gaussian copula (same shape as the input).
    X_gaussian_covmat : np.ndarray
        The covariance matrix of the Gaussian copula transformed data.

    Notes
    -----
    - The Gaussian copula transformation involves ranking the data, normalizing the ranks, and applying the inverse CDF of the standard normal distribution.
    - Infinite values resulting from the inverse CDF transformation are set to 0.
    - The covariance matrix is computed from the Gaussian copula transformed data.
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

@torch.no_grad()
def gaussian_copula_cov_opt(
    X: torch.Tensor,
    *,
    correction: int = 1,
    batch_D: int | None = None,
    return_xg: bool = False,
    in_place: bool = False,
    out_dtype: torch.dtype | None = None,
):
    """
    CPU-optimized Gaussian copula covariance computation.
    Equivalent to monolithic version when batch_D >= D.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data with shape (D, T, N) where D is number of datasets,
        T is time points, N is number of variables. Can be CPU or CUDA.
    correction : int, default=1
        Bias correction for covariance (0 or 1).
    batch_D : int or None, default=None
        Batch size for processing D dimension. If None or >= D, processes all at once.
    return_xg : bool, default=False
        Whether to return Gaussian-transformed data.
    in_place : bool, default=False
        Whether to modify input tensor in place (when possible).
    out_dtype : torch.dtype or None, default=None
        Output data type. If None, uses input dtype.

    Returns
    -------
    Xg_out : torch.Tensor or None
        Gaussian-transformed data if return_xg=True, else None.
    cov_out : torch.Tensor
        Covariance matrices with shape (D, N, N).
    """
    assert X.ndim == 3, f"expected 3D, got {X.ndim}D"
    D, T, N = X.shape
    if out_dtype is None:
        out_dtype = X.dtype
    if batch_D is None or batch_D >= D:
        batch_D = D
    assert correction in (0, 1)

    # Output tensors
    cov_out = torch.empty((D, N, N), dtype=out_dtype, device=X.device)
    Xg_out  = torch.empty_like(X, dtype=out_dtype) if return_xg else None

    finfo = torch.finfo(out_dtype)
    lo = float(finfo.tiny)
    hi = float(1.0 - finfo.eps)
    denom = float(T - correction)

    # Reuse single work buffer to avoid allocator churn
    if not in_place:
        work_buf = torch.empty((min(batch_D, D), T, N), dtype=out_dtype, device=X.device)
    else:
        work_buf = None  # work on input views

    # Stable argsort if version supports it; default without stable (faster on CPU)
    def _argsort(t, dim):
        try:
            return torch.argsort(t, dim=dim, stable=False)
        except TypeError:
            return torch.argsort(t, dim=dim)

    # Process in batches of D
    for s in range(0, D, batch_D):
        e  = min(s + batch_D, D)
        Db = e - s
        Xb = X[s:e]  # (Db,T,N)

        # Buffer: copy once per batch if not in_place; if in_place and dtype matches, operate directly
        if in_place:
            buf = Xb if Xb.dtype == out_dtype else Xb.to(out_dtype)
        else:
            buf = work_buf[:Db]
            if Xb.dtype == out_dtype:
                buf.copy_(Xb)
            else:
                # conversion + copy in one step
                buf.copy_(Xb.to(out_dtype))

        # ranks = argsort(argsort) along time dimension
        sortid = _argsort(buf, dim=1)
        ranks  = _argsort(sortid, dim=1).to(out_dtype)  # (Db,T,N)

        # buf := U = (ranks+1)/(T+1), exact clamp, ndtri in-place
        buf.copy_(ranks).add_(1.0).div_(T + 1.0)
        buf.clamp_(min=lo, max=hi)
        torch.special.ndtri(buf, out=buf)  # now buf = Xg

        # Temporal centering and batched covariance
        buf.sub_(buf.mean(dim=1, keepdim=True))
        cov = torch.bmm(buf.transpose(1, 2), buf).div_(denom)  # (Db,N,N)

        cov_out[s:e] = cov
        if return_xg:
            Xg_out[s:e] = buf

        # Free temporaries to reduce GC pressure on CPU
        del sortid, ranks, cov
        # 'buf' is view of reusable buffer or input; not freed

    return Xg_out, cov_out

def gaussian_copula_covmat(X: np.ndarray):
    """
    Compute the covariance matrix of the Gaussian copula transformed data.

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array of shape (T, N) where T is the number of samples and N is the number of variables.

    Returns
    -------
    np.ndarray
        The covariance matrix of the Gaussian copula transformed data.

    Notes
    -----
    - This function is a wrapper around `gaussian_copula` to directly return the covariance matrix. For more details, see :ref:`gaussian_copula`.
    """
    return gaussian_copula(X)[1]

def _to_numpy(X):
    if isinstance(X, torch.Tensor):
        # If the tensor is on a GPU/TPU, move it to CPU first, then convert to NumPy
        return X.detach().cpu().numpy()
    elif isinstance(X, np.ndarray):
        return X
    return np.array(X)

def _get_device(use_cpu:bool=False):
    """Set the use of GPU if available"""
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')
    return device

def _normalize_input_data(X: TensorLikeArray,
                         covmat_precomputed: bool=False,
                         T: Optional[Union[int, List[int]]]=None,
                         device: torch.device=torch.device('cpu')):
    """
    Normalize the input data to be a list of covariance matrices with shape (D, N, N).

    Parameters
    ----------
    X : TensorLikeArray
        A list or a single 2D numpy array or tensor of shape:
        1. (T, N) where T is the number of samples if X are multivariate series.
        2. A list of 2D covariance matrices with shape (N, N).
    covmat_precomputed : bool, optional
        A boolean flag to indicate if the input data is a list of covariance matrices or multivariate series. Default is False.
    T : int or list of int, optional
        A list of integers indicating the number of samples for each multivariate series. Default is None.
    device : torch.device, optional
        The device to use for the computation. Default is 'cpu'.

    Returns
    -------
    covmats : torch.Tensor
        The normalized covariance matrices with shape (D, N, N).
    D : int
        The number of datasets.
    N : int
        The number of variables in the system.
    T : list of int
        The number of samples for each multivariate series.

    Notes
    -----
    - If `covmat_precomputed` is True, the input data is treated as covariance matrices.
    - If `covmat_precomputed` is False, the input data is treated as multivariate series and the covariance matrices are computed using the Gaussian copula transformation.
    - The function ensures that the covariance matrices are sent to the specified device.
    """

    # Handle different options for X parameter. Accept multivariate data or covariance matrix
    if covmat_precomputed:
        covmats = torch.as_tensor(X)
        covmats = covmats.unsqueeze(0) if len(covmats.shape) == 2 else covmats
        assert covmats.shape[-2] == covmats.shape[-1], 'Covariance matrix should be square'
        assert len(covmats.shape) == 3, 'Covariance matrix should have dimensions (N, N) or (D, N, N)'
    else:
        
        try:
            X = _to_numpy(X)
            assert len(X.shape) in [2, 3], 'Covariance matrix should have dimensions (T, N) or (D, T, N)'
            X = [X] if len(X.shape) == 2 else [X[i] for i in range(X.shape[0])]
        except:
            X = [_to_numpy(x) for x in X]
            assert all([len(x.shape) == 2 for x in X]), 'All multivariate series should have dimensions (T, N) where T may vary and N be constant across all series'
            assert all([x.shape[1] == X[0].shape[1] for x in X]), 'All multivariate series should have dimensions (T, N) where T may vary and N be constant across all series'

        # Process each dataset individually if they have different sizes
        # Check if all datasets have the same temporal dimension
        T_sizes = [x.shape[0] for x in X]
        all_same_size = all(t == T_sizes[0] for t in T_sizes)
        
        if all_same_size:
            # All datasets have same size, we can stack and process in batch
            X_tensor = torch.stack([torch.from_numpy(x) for x in X])  # (D, T, N)
            _, covmats = gaussian_copula_cov_opt(X_tensor, return_xg=False)
        else:
            # Different sizes, process each dataset individually
            covmat_list = []
            for x in X:
                x_tensor = torch.from_numpy(x).unsqueeze(0)  # (1, T, N)
                _, cov = gaussian_copula_cov_opt(x_tensor, return_xg=False)
                covmat_list.append(cov[0])  # Extract the single covariance matrix
            covmats = torch.stack(covmat_list)  # (D, N, N)
        
        T = [x.shape[0] for x in X]

    D, N = covmats.shape[:2]
    
    # Handle different options for T parameter
    if isinstance(T, int):
        T = [T] * D
    
    # Send covmat to device  
    covmats = covmats.to(device).contiguous()

    return covmats, D, N, T
