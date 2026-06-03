from typing import List, Union, Optional
import warnings
import numpy as np
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

@torch.no_grad()
def gaussian_copula_covmat(
    X: torch.Tensor,
    *,
    correction: int = 1,
    batch_size_D: Optional[int] = None,
    return_xg: bool = False,
    in_place: bool = False,
    out_dtype: Optional[torch.dtype] = None):
    """
    CPU-optimized Gaussian copula covariance computation.
    Equivalent to monolithic version when batch_size_D >= D.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data with shape (D, T, N) where D is number of datasets,
        T is time points, N is number of variables. Can be CPU or CUDA.
    correction : int, default=1
        Bias correction for covariance (0 or 1).
    batch_size_D : int or None, default=None
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

    assert correction in (0, 1), f"correction must be 0 or 1, got {correction}"
    assert X.ndim == 3, f"expected 3D, got {X.ndim}D"
    
    D, T, N = X.shape
    batch_size_D = batch_size_D if batch_size_D is not None else D
    
    assert 0 < batch_size_D <= D, f"batch_size_D must be in (0, {D}] or None"
    
    if out_dtype is None:
        out_dtype = X.dtype if X.dtype.is_floating_point else torch.get_default_dtype()

    # Output tensors
    cov_out = torch.empty((D, N, N), dtype=out_dtype, device=X.device)
    Xg_out  = torch.empty_like(X, dtype=out_dtype) if return_xg else None

    finfo = torch.finfo(out_dtype)
    lo = float(finfo.tiny)
    hi = float(1.0 - finfo.eps)
    denom = float(T - correction)

    # Reuse single work buffer to avoid allocator churn
    if not in_place:
        work_buf = torch.empty((min(batch_size_D, D), T, N), dtype=out_dtype, device=X.device)
    else:
        work_buf = None  # work on input views

    # Stable argsort if version supports it; default without stable (faster on CPU)
    def _argsort(t, dim):
        try:
            return torch.argsort(t, dim=dim, stable=False)
        except TypeError:
            return torch.argsort(t, dim=dim)

    # Process in batches of D
    for s in range(0, D, batch_size_D):
        e  = min(s + batch_size_D, D)
        Db = e - s
        Xb = X[s:e]  # (Db,T,N)

        # Buffer: copy once per batch if not in_place; if in_place and dtype matches, operate directly
        if in_place:
            if Xb.dtype != out_dtype:
                warnings.warn(
                    f"in_place=True but X.dtype ({Xb.dtype}) != out_dtype ({out_dtype}); "
                    "dtype conversion requires a copy — input will not be modified in place.",
                    UserWarning, stacklevel=2
                )
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

def _get_device(use_cpu:bool=False):
    """Set the use of GPU if available"""
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')
    return device

def _normalize_input_data(X: TensorLikeArray,
                         covmat_precomputed: bool=False,
                         T: Optional[Union[int, List[int]]]=None,
                         device: torch.device=torch.device('cpu'),
                         batch_size_D: Optional[int]=None):
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
    batch_size_D : int or None, optional
        Number of datasets to process per batch during Gaussian copula covariance computation.
        Reduces peak memory when D is large. Default is None (all datasets at once).

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
        covmats = torch.as_tensor(np.array(X) if isinstance(X, list) else X)
        covmats = covmats.unsqueeze(0) if len(covmats.shape) == 2 else covmats
        assert covmats.shape[-2] == covmats.shape[-1], 'Covariance matrix should be square'
        assert len(covmats.shape) == 3, 'Covariance matrix should have dimensions (N, N) or (D, N, N)'
    else:
        try:
            X_tensor = torch.as_tensor(np.array(X) if isinstance(X, (list, tuple)) else X)
            assert X_tensor.ndim in [2, 3], 'Data should have dimensions (T, N) or (D, T, N)'
            X_tensor = X_tensor.unsqueeze(0) if X_tensor.ndim == 2 else X_tensor
            T = [X_tensor.shape[1]] * X_tensor.shape[0]
            _, covmats = gaussian_copula_covmat(X_tensor, return_xg=False, batch_size_D=batch_size_D)
        except Exception:
            X_list = [torch.as_tensor(x) for x in X]
            assert all(x.ndim == 2 for x in X_list), 'All multivariate series should have dimensions (T, N) where T may vary and N be constant across all series'
            assert all(x.shape[1] == X_list[0].shape[1] for x in X_list), 'All multivariate series should have dimensions (T, N) where T may vary and N be constant across all series'
            T = [x.shape[0] for x in X_list]

            if all(x.shape[0] == X_list[0].shape[0] for x in X_list):
                _, covmats = gaussian_copula_covmat(torch.stack(X_list), return_xg=False, batch_size_D=batch_size_D)
            else:
                covmats = torch.stack([
                    gaussian_copula_covmat(x.unsqueeze(0), return_xg=False, batch_size_D=batch_size_D)[1][0]
                    for x in X_list
                ])

    D, N = covmats.shape[:2]
    
    # Handle different options for T parameter
    if isinstance(T, int):
        T = [T] * D
    
    # Send covmat to device  
    covmats = covmats.to(device).contiguous()

    return covmats, D, N, T
