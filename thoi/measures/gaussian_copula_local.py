from typing import Optional

import numpy as np
import torch

from thoi.typing import TensorLikeArray
from thoi.commons import gaussian_copula_covmat

# TODO: 
# 1. check if I don't have to use the same torch Dataset strategy to make it work on GPU as efficiently as in the non local version.
# 2. re check all tests, make sure they pass and there are no new test cases to consider

def gaussian_tc_bias_correction(K: int, T: int, device='cpu', dtype=torch.float64) -> torch.Tensor:
    """Bias for TC = sum H(X_i) - H(X_1..X_K)."""
    from thoi.measures.utils import _gaussian_entropy_bias_correction
    return K * _gaussian_entropy_bias_correction(1, T).to(device=device, dtype=dtype) - \
           _gaussian_entropy_bias_correction(K, T).to(device=device, dtype=dtype)

def gaussian_dtc_bias_correction(K: int, T: int, device='cpu', dtype=torch.float64) -> torch.Tensor:
    """Bias for DTC = sum_i H(X_{-i}) - (K-1) H(X_1..X_K)."""
    from thoi.measures.utils import _gaussian_entropy_bias_correction
    return K * _gaussian_entropy_bias_correction(K - 1, T).to(device=device, dtype=dtype) - \
           (K - 1) * _gaussian_entropy_bias_correction(K, T).to(device=device, dtype=dtype)

def _batched_chol_logdet(S, eps=1e-10):
    """Compute Cholesky decomposition and log determinant for batched matrices."""
    k = S.shape[-1]
    L = torch.linalg.cholesky(S + eps*torch.eye(k, device=S.device, dtype=S.dtype))
    logdet = 2.0*torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    return L, logdet  # L:[B,k,k], logdet:[B]

def _quad_from_chol(L, X):
    """Compute quadratic form using Cholesky decomposition."""
    # L:[B,k,k], X:[B,Lc,k] -> [B,Lc]
    ZT = torch.cholesky_solve(X.transpose(1,2), L)  # [B,k,Lc]
    return (X.transpose(1,2) * ZT).sum(1)           # [B,Lc]

def _gather_nplet_covs(covmats, nplets):
    """Extract covariance submatrices for specified n-plets."""
    B, K = nplets.shape
    D, N, _ = covmats.shape
    device, dtype = covmats.device, covmats.dtype
    d_idx = torch.arange(D, device=device).view(1,D,1,1).expand(B,D,K,K)
    r_idx = nplets[:,None,:,None].expand(B,D,K,K)
    c_idx = nplets[:,None,None,:].expand(B,D,K,K)
    sub = covmats[d_idx, r_idx, c_idx]   # [B,D,K,K]
    return sub.reshape(B*D, K, K)

def _gather_nplet_data(data, nplets, t0, t1):
    """Extract data subsets for specified n-plets and time range."""
    # slice only [t0:t1] to save memory
    B,K = nplets.shape
    D,T,N = data.shape
    Lc = t1-t0
    device = data.device
    d_idx = torch.arange(D, device=device).view(1,D,1,1).expand(B,D,Lc,K)
    t_idx = torch.arange(t0,t1, device=device).view(1,1,Lc,1).expand(B,D,Lc,K)
    n_idx = nplets[:,None,None,:].expand(B,D,Lc,K)
    sub = data[d_idx, t_idx, n_idx]  # [B,D,Lc,K]
    return sub.reshape(B*D, Lc, K)

def _leave_one_out_stats(S: torch.Tensor, Xc: torch.Tensor, Lj: torch.Tensor, logdet_j: torch.Tensor, eps: float = 1e-10):
    """
    Compute leave-one-out statistics for DTC calculation.
    
    This function computes the sum of all leave-one-out negative log-likelihoods
    which is needed for DTC and S calculations. Uses direct computation of each
    leave-one-out subsystem for accuracy.
    
    Parameters
    ----------
    S : torch.Tensor
        Covariance matrices, shape [B, K, K]
    Xc : torch.Tensor  
        Data chunk, shape [B, Lc, K]
    Lj : torch.Tensor
        Cholesky decomposition of S
    logdet_j : torch.Tensor
        Log determinant of S
        
    Returns
    -------
    logdet_mi_sum : torch.Tensor
        Sum of log determinants for all leave-one-out subsystems, shape [B]
    qmi_sum : torch.Tensor
        Sum of quadratic forms for all leave-one-out subsystems, shape [B, Lc]
    """
    B, K, _ = S.shape
    Lc = Xc.shape[1]
    
    # Initialize accumulators
    logdet_mi_sum = torch.zeros(B, device=S.device, dtype=S.dtype)
    qmi_sum = torch.zeros(B, Lc, device=S.device, dtype=S.dtype)
    
    # Compute each leave-one-out subsystem directly
    for i in range(K):
        # Indices for leave-one-out (excluding i-th variable)
        loo_indices = torch.cat([torch.arange(i), torch.arange(i+1, K)])
        
        # Extract leave-one-out covariance matrices and data
        S_loo = S[:, loo_indices][:, :, loo_indices]  # [B, K-1, K-1]
        Xc_loo = Xc[:, :, loo_indices]                # [B, Lc, K-1]
        
        # Use Cholesky decomposition for numerical stability (like traditional THOI)
        try:
            L_loo = torch.linalg.cholesky(S_loo + eps * torch.eye(K-1, device=S.device, dtype=S.dtype))
            logdet_loo = 2 * torch.sum(torch.log(torch.diagonal(L_loo, dim1=-2, dim2=-1)), dim=-1)
        except:
            # Fallback to regular logdet if Cholesky fails
            logdet_loo = torch.logdet(S_loo + eps * torch.eye(K-1, device=S.device, dtype=S.dtype))
            
        logdet_mi_sum += logdet_loo
        
        # Compute quadratic form using Cholesky solve (more stable than inverse)
        # Solve L @ y = x for each time point, then compute ||y||^2
        try:
            # L_loo: [B, K-1, K-1], Xc_loo: [B, Lc, K-1]
            # We need to solve for each time point: L @ y = x^T
            Xc_loo_t = Xc_loo.transpose(-1, -2)  # [B, K-1, Lc]
            y = torch.linalg.solve_triangular(L_loo, Xc_loo_t, upper=False)  # [B, K-1, Lc]
            q_loo = torch.sum(y * y, dim=1)  # [B, Lc]: sum over K-1 dimension for quadratic form
        except:
            # Fallback to inverse method
            S_loo_inv = torch.inverse(S_loo + eps * torch.eye(K-1, device=S.device, dtype=S.dtype))
            temp = torch.bmm(Xc_loo, S_loo_inv)
            q_loo = torch.sum(temp * Xc_loo, dim=2)  # [B, Lc]
            
        qmi_sum += q_loo

    return logdet_mi_sum, qmi_sum

def _local_nplets_from_xg(Xg: torch.Tensor,
                          covmats: torch.Tensor,
                          nplets: torch.Tensor,
                          *,
                          device: torch.device,
                          dtype: torch.dtype,
                          batch_size: int,
                          time_chunk: int,
                          eps: float) -> torch.Tensor:
    """Core local nplets computation from pre-normalized Gaussian data."""
    nplets = torch.as_tensor(nplets, device=device, dtype=torch.long).contiguous()
    B_total, K = nplets.shape
    D, T, _ = Xg.shape

    out = torch.empty(B_total, D, T, 4, device=device, dtype=dtype)

    for start in range(0, B_total, batch_size):
        npl = nplets[start:start + batch_size]
        B = npl.shape[0]

        S = _gather_nplet_covs(covmats, npl)
        Lj, logdet_j = _batched_chol_logdet(S, eps=eps)
        var = torch.diagonal(S, dim1=-2, dim2=-1)

        for t0 in range(0, T, time_chunk):
            t1 = min(T, t0 + time_chunk)
            Xc = _gather_nplet_data(Xg, npl, t0, t1)
            Lc = t1 - t0

            qj = _quad_from_chol(Lj, Xc)
            joint_nll = 0.5 * (K * np.log(2 * np.pi) + logdet_j.unsqueeze(1) + qj)

            q_uni = Xc ** 2 / var.unsqueeze(1)
            logdet_uni_per_var = torch.log(var)
            uni_nll = 0.5 * (np.log(2 * np.pi) + logdet_uni_per_var.unsqueeze(1) + q_uni)
            uni_nll_sum = uni_nll.sum(dim=2)

            tc_loc = uni_nll_sum - joint_nll

            if K >= 3:
                logdet_mi, qmi = _leave_one_out_stats(S, Xc, Lj, logdet_j, eps)
                loo_nll = 0.5 * (K * (K - 1) * np.log(2 * np.pi) + logdet_mi.unsqueeze(1) + qmi)
                dtc_loc = loo_nll - (K - 1) * joint_nll
                o_loc = tc_loc - dtc_loc
                s_loc = tc_loc + dtc_loc
            else:
                dtc_loc = tc_loc
                o_loc = torch.zeros_like(tc_loc)
                s_loc = tc_loc + dtc_loc

            out[start:start + B, :, t0:t1, 0] = tc_loc.view(B, D, Lc)
            out[start:start + B, :, t0:t1, 1] = dtc_loc.view(B, D, Lc)
            out[start:start + B, :, t0:t1, 2] = o_loc.view(B, D, Lc)
            out[start:start + B, :, t0:t1, 3] = s_loc.view(B, D, Lc)

    return out

@torch.no_grad()
def local_nplets_measures(X, nplets=None, *,
                          covmats: Optional[torch.Tensor] = None,
                          batch_size_D: Optional[int] = None,
                          device='cpu', dtype=torch.float32,
                          batch_size=100000, time_chunk=2048, eps=1e-10) -> torch.Tensor:
    """
    Compute local (time-resolved) higher-order information measures.

    This function computes local versions of TC, DTC, O, and S for each time point
    using the corrected negative log-likelihood approach that ensures theoretical
    consistency (TC ≥ 0) and proper convergence to traditional measures.

    Parameters
    ----------
    X : torch.Tensor or array-like
        Input data with shape (T, N) or (D, T, N).
        When covmats=None (default): raw timeseries — Gaussian copula normalization is applied internally.
        When covmats is provided: must be pre-normalized Gaussian data (the Xg returned by
        ``gaussian_copula_covmat(..., return_xg=True)``). The normalization step is skipped entirely;
        X and covmats must come from the same ``gaussian_copula_covmat`` call.
    nplets : torch.Tensor or array-like, optional
        N-plets to analyze, shape (B, K) where B is number of n-plets, K is order.
        If None, uses the single full-system n-plet [[0, 1, ..., N-1]].
    covmats : torch.Tensor, optional
        Pre-computed covariance matrices with shape (N, N) or (D, N, N).
        If None, X is treated as raw timeseries and both Xg and covmats are computed via
        ``gaussian_copula_covmat``. If provided, X must be the matching pre-normalized Gaussian
        data — the normalization step is skipped completely, saving the full ``_gcc`` computation.
        This is useful when calling this function multiple times on the same data with different nplets.
    batch_size_D : int or None, default=None
        Number of datasets to process per batch during Gaussian copula computation.
        Only used when covmats=None. Reduces peak memory when D is large.
    device : str, default='cpu'
        Device for computation.
    dtype : torch.dtype, default=torch.float32
        Data type for computation.
    batch_size : int, default=100000
        Batch size for n-plet processing.
    time_chunk : int, default=2048
        Time chunk size for memory optimization.
    eps : float, default=1e-10
        Numerical stability epsilon.

    Returns
    -------
    torch.Tensor
        Local measures with shape [B, D, T, 4] where last dimension is (TC, DTC, O, S).
    """
    X_tensor = torch.as_tensor(np.array(X) if isinstance(X, (list, tuple)) else X)
    X_tensor = X_tensor.unsqueeze(0) if X_tensor.ndim == 2 else X_tensor

    if covmats is None:
        Xg, covmats_t = gaussian_copula_covmat(X_tensor, return_xg=True, batch_size_D=batch_size_D)
    else:
        # X is already Gaussian-normalized data (Xg); skip the _gcc computation entirely.
        Xg = X_tensor
        covmats_t = torch.as_tensor(np.array(covmats) if isinstance(covmats, list) else covmats)
        covmats_t = covmats_t.unsqueeze(0) if covmats_t.ndim == 2 else covmats_t
        assert Xg.shape[0] == covmats_t.shape[0], (
            f'Xg has D={Xg.shape[0]} datasets but covmats has D={covmats_t.shape[0]}; '
            'they must come from the same gaussian_copula_covmat call.'
        )
        assert Xg.shape[2] == covmats_t.shape[1], (
            f'Xg has N={Xg.shape[2]} variables but covmats has N={covmats_t.shape[1]}; '
            'they must come from the same gaussian_copula_covmat call.'
        )

    Xg = Xg.to(device=device, dtype=dtype)
    covmats = covmats_t.to(device=device, dtype=dtype)
    N = Xg.shape[2]

    if nplets is None:
        nplets = torch.arange(N, device=device).unsqueeze(0)

    return _local_nplets_from_xg(
        Xg, covmats, nplets,
        device=device, dtype=dtype,
        batch_size=batch_size, time_chunk=time_chunk, eps=eps,
    )

@torch.no_grad()
def local_multi_order_measures(X: TensorLikeArray,
                               min_order: int = 3,
                               max_order: Optional[int] = None,
                               *,
                               covmats: Optional[torch.Tensor] = None,
                               batch_size_D: Optional[int] = None,
                               device: str = 'cpu',
                               dtype: torch.dtype = torch.float32,
                               batch_size: int = 100000,
                               time_chunk: int = 4096,
                               eps: float = 1e-10) -> dict:
    """
    Compute local measures for every order in [min_order, max_order] on the full set of variables.

    Parameters
    ----------
    X : TensorLikeArray
        Input data with shape (T, N) or (D, T, N).
        When covmats=None (default): raw timeseries — Gaussian copula normalization is applied internally.
        When covmats is provided: must be pre-normalized Gaussian data (the Xg returned by
        ``gaussian_copula_covmat(..., return_xg=True)``). The normalization step is skipped entirely;
        X and covmats must come from the same ``gaussian_copula_covmat`` call.
    min_order : int, default=3
        Minimum order of interactions to compute.
    max_order : int, optional
        Maximum order of interactions to compute. If None, uses N (number of variables).
    covmats : torch.Tensor, optional
        Pre-computed covariance matrices with shape (N, N) or (D, N, N).
        If None, X is treated as raw timeseries and both Xg and covmats are computed via
        ``gaussian_copula_covmat``. If provided, X must be the matching pre-normalized Gaussian
        data — the normalization step is skipped completely, saving the full ``_gcc`` computation.
    batch_size_D : int or None, default=None
        Number of datasets to process per batch during Gaussian copula computation.
        Only used when covmats=None. Reduces peak memory when D is large.
    device : str, default='cpu'
        Device for computation.
    dtype : torch.dtype, default=torch.float32
        Data type for computation.
    batch_size : int, default=100000
        Batch size for n-plet processing.
    time_chunk : int, default=4096
        Time chunk size for memory optimization.
    eps : float, default=1e-10
        Numerical stability epsilon.

    Returns
    -------
    dict
        Dictionary with keys as orders and values as tensors [C(N,order), D, T, 4]
        where last dimension is (TC, DTC, O, S).
    """

    X_tensor = torch.as_tensor(np.array(X) if isinstance(X, (list, tuple)) else X)
    X_tensor = X_tensor.unsqueeze(0) if X_tensor.ndim == 2 else X_tensor

    if covmats is None:
        Xg, covmats_t = gaussian_copula_covmat(X_tensor, return_xg=True, batch_size_D=batch_size_D)
    else:
        # X is already Gaussian-normalized data (Xg); skip _gcc entirely.
        Xg = X_tensor
        covmats_t = torch.as_tensor(np.array(covmats) if isinstance(covmats, list) else covmats)
        covmats_t = covmats_t.unsqueeze(0) if covmats_t.ndim == 2 else covmats_t
        assert Xg.shape[0] == covmats_t.shape[0], (
            f'Xg has D={Xg.shape[0]} datasets but covmats has D={covmats_t.shape[0]}; '
            'they must come from the same gaussian_copula_covmat call.'
        )
        assert Xg.shape[2] == covmats_t.shape[1], (
            f'Xg has N={Xg.shape[2]} variables but covmats has N={covmats_t.shape[1]}; '
            'they must come from the same gaussian_copula_covmat call.'
        )

    Xg = Xg.to(device=device, dtype=dtype)
    covmats_t = covmats_t.to(device=device, dtype=dtype)
    N = Xg.shape[2]

    if max_order is None:
        max_order = N

    out = {}
    for K in range(min_order, max_order + 1):
        nplets = torch.combinations(torch.arange(N, device=device), r=K, with_replacement=False)
        out[K] = _local_nplets_from_xg(
            Xg, covmats_t, nplets,
            device=device, dtype=dtype,
            batch_size=batch_size, time_chunk=time_chunk, eps=eps,
        )

    return out


def time_averaged_local_measures(
    X: TensorLikeArray,
    min_order: int = 3,
    max_order: Optional[int] = None,
    *,
    covmats: Optional[torch.Tensor] = None,
    batch_size_D: Optional[int] = None,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    batch_size: int = 100000,
    time_chunk: int = 4096,
    eps: float = 1e-10,
    bias_correction: bool = True,
) -> dict:
    """
    Compute time-averaged local measures with optional bias correction.

    This function computes local measures and then averages them over time,
    applying bias correction to the averaged results (not to individual local measures).

    Parameters
    ----------
    X : TensorLikeArray
        Raw timeseries input data with shape (T, N) or (D, T, N).
    min_order : int, default=3
        Minimum order of interactions to compute.
    max_order : int, optional
        Maximum order of interactions to compute. If None, uses N (number of variables).
    covmats : torch.Tensor, optional
        Pre-computed covariance matrices with shape (N, N) or (D, N, N).
        If None, covariance matrices are computed from X via gaussian_copula_covmat.
    batch_size_D : int or None, default=None
        Number of datasets to process per batch during Gaussian copula computation.
        Reduces peak memory when D is large.
    device : str, default='cpu'
        Device for computation.
    dtype : torch.dtype, default=torch.float32
        Data type for computation.
    batch_size : int, default=100000
        Batch size for n-plet processing.
    time_chunk : int, default=4096
        Chunk size for temporal processing.
    eps : float, default=1e-10
        Small value for numerical stability.
    bias_correction : bool, default=True
        Whether to apply bias correction after temporal averaging.

    Returns
    -------
    dict
        Dictionary mapping order -> tensor of averaged measures with shape
        (n_combinations, n_samples, 4) where last dim is [TC, DTC, O, S].
    """

    # First get local measures
    local_results = local_multi_order_measures(
        X,
        min_order=min_order,
        max_order=max_order,
        covmats=covmats,
        batch_size_D=batch_size_D,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        time_chunk=time_chunk,
        eps=eps,
    )

    # Time-average the results and apply bias correction AFTER temporal averaging
    averaged_results = {}

    for K, measures in local_results.items():
        # measures shape: [n_combinations, D, T, 4]
        # Average over time dimension
        time_averaged = measures.mean(dim=2)  # -> [n_combinations, D, 4]

        if bias_correction and K >= 2:
            # Number of samples used to compute bias correction is the temporal length
            T_samples = measures.shape[2]

            # Compute corrections using the helper functions defined above
            tc_correction = gaussian_tc_bias_correction(K, T_samples, device=time_averaged.device, dtype=time_averaged.dtype)
            dtc_correction = gaussian_dtc_bias_correction(K, T_samples, device=time_averaged.device, dtype=time_averaged.dtype)

            # Apply corrections (broadcasting will handle scalar or tensor shapes)
            corrected = time_averaged.clone()
            corrected[..., 0] = time_averaged[..., 0] - tc_correction
            corrected[..., 1] = time_averaged[..., 1] - dtc_correction

            # Recompute O and S from the corrected TC/DTC to keep theoretical relations
            corrected[..., 2] = corrected[..., 0] - corrected[..., 1]  # O = TC - DTC
            corrected[..., 3] = corrected[..., 0] + corrected[..., 1]  # S = TC + DTC

            averaged_results[K] = corrected
        else:
            averaged_results[K] = time_averaged

    return averaged_results