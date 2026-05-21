from typing import Optional, List, Tuple

import numpy as np
import torch

from thoi.typing import TensorLikeArray
from thoi.commons import gaussian_copula_covmat_batched


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
    L = torch.linalg.cholesky(S + eps * torch.eye(k, device=S.device, dtype=S.dtype))
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    return L, logdet  # L:[B,k,k], logdet:[B]


def _quad_from_chol(L, X):
    """Compute quadratic form using Cholesky decomposition."""
    # L:[B,k,k], X:[B,Lc,k] -> [B,Lc]
    ZT = torch.cholesky_solve(X.transpose(1, 2), L)  # [B,k,Lc]
    return (X.transpose(1, 2) * ZT).sum(1)           # [B,Lc]


def _gather_nplet_covs(covmats, nplets):
    """Extract covariance submatrices for specified n-plets."""
    B, K = nplets.shape
    D, N, _ = covmats.shape
    device = covmats.device
    d_idx = torch.arange(D, device=device).view(1, D, 1, 1).expand(B, D, K, K)
    r_idx = nplets[:, None, :, None].expand(B, D, K, K)
    c_idx = nplets[:, None, None, :].expand(B, D, K, K)
    sub = covmats[d_idx, r_idx, c_idx]   # [B,D,K,K]
    return sub.reshape(B * D, K, K)


def _gather_nplet_data(data, nplets, t0, t1):
    """Extract data subsets for specified n-plets and time range."""
    B, K = nplets.shape
    D, T, N = data.shape
    Lc = t1 - t0
    device = data.device
    d_idx = torch.arange(D, device=device).view(1, D, 1, 1).expand(B, D, Lc, K)
    t_idx = torch.arange(t0, t1, device=device).view(1, 1, Lc, 1).expand(B, D, Lc, K)
    n_idx = nplets[:, None, None, :].expand(B, D, Lc, K)
    sub = data[d_idx, t_idx, n_idx]  # [B,D,Lc,K]
    return sub.reshape(B * D, Lc, K)


def _leave_one_out_nll(S, Xc, Lj, logdet_j, eps=1e-10):
    """
    Compute the sum of leave-one-out negative log-likelihoods needed for DTC.

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

    logdet_mi_sum = torch.zeros(B, device=S.device, dtype=S.dtype)
    qmi_sum = torch.zeros(B, Lc, device=S.device, dtype=S.dtype)

    for i in range(K):
        loo_indices = torch.cat([
            torch.arange(i, device=S.device),
            torch.arange(i + 1, K, device=S.device)
        ])

        S_loo = S[:, loo_indices][:, :, loo_indices]  # [B, K-1, K-1]
        Xc_loo = Xc[:, :, loo_indices]                # [B, Lc, K-1]

        try:
            L_loo = torch.linalg.cholesky(S_loo + eps * torch.eye(K - 1, device=S.device, dtype=S.dtype))
            logdet_loo = 2 * torch.sum(torch.log(torch.diagonal(L_loo, dim1=-2, dim2=-1)), dim=-1)
        except torch.linalg.LinAlgError:
            logdet_loo = torch.logdet(S_loo + eps * torch.eye(K - 1, device=S.device, dtype=S.dtype))

        logdet_mi_sum += logdet_loo

        try:
            Xc_loo_t = Xc_loo.transpose(-1, -2)  # [B, K-1, Lc]
            y = torch.linalg.solve_triangular(L_loo, Xc_loo_t, upper=False)  # [B, K-1, Lc]
            q_loo = torch.sum(y * y, dim=1)  # [B, Lc]
        except torch.linalg.LinAlgError:
            S_loo_inv = torch.inverse(S_loo + eps * torch.eye(K - 1, device=S.device, dtype=S.dtype))
            temp = torch.bmm(Xc_loo, S_loo_inv)
            q_loo = torch.sum(temp * Xc_loo, dim=2)  # [B, Lc]

        qmi_sum += q_loo

    return logdet_mi_sum, qmi_sum


@torch.no_grad()
def local_nplets_measures(
    data, nplets, covmats, *,
    device='cpu', dtype=torch.float32,
    batch_size=100000, time_chunk=2048, eps=1e-10
):
    """
    Compute local (time-resolved) TC, DTC, O-information and S-information.

    Parameters
    ----------
    data : torch.Tensor or array-like
        Normalized input data with shape (D, T, N) or (T, N).
        Must be the output of gaussian_copula_covmat_batched.
    nplets : torch.Tensor or array-like
        N-plets to analyze, shape (B, K).
    covmats : torch.Tensor
        Covariance matrices with shape (D, N, N) from gaussian_copula_covmat_batched.
    device : str, default='cpu'
    dtype : torch.dtype, default=torch.float32
    batch_size : int, default=100000
    time_chunk : int, default=2048
    eps : float, default=1e-10

    Returns
    -------
    torch.Tensor
        Local measures with shape [B, D, T, 4] where last dim is (TC, DTC, O, S).
    """
    if hasattr(data, 'dim') and data.dim() == 3:
        D, T, N = data.shape
        X = data.to(device=device, dtype=dtype)
    elif isinstance(data, (list, tuple)):
        D = len(data)
        T, N = data[0].shape
        X = torch.stack([torch.as_tensor(d, device=device, dtype=dtype) for d in data], 0)
    else:
        X = torch.as_tensor(data, device=device, dtype=dtype).unsqueeze(0)
        D, T, N = 1, X.shape[1], X.shape[2]

    covmats = torch.as_tensor(covmats, device=device, dtype=dtype)
    if covmats.dim() == 2:
        covmats = covmats.unsqueeze(0)

    nplets = torch.as_tensor(nplets, device=device, dtype=torch.long).contiguous()
    B_total, K = nplets.shape

    out = torch.empty(B_total, D, T, 4, device=device, dtype=dtype)

    for start in range(0, B_total, batch_size):
        npl = nplets[start:start + batch_size]
        B = npl.shape[0]

        S = _gather_nplet_covs(covmats, npl)          # [B*D, K, K]
        Lj, logdet_j = _batched_chol_logdet(S, eps=eps)
        var = torch.diagonal(S, dim1=-2, dim2=-1)     # [B*D, K]

        for t0 in range(0, T, time_chunk):
            t1 = min(T, t0 + time_chunk)
            Xc = _gather_nplet_data(X, npl, t0, t1)   # [B*D, Lc, K]
            Lc = t1 - t0

            qj = _quad_from_chol(Lj, Xc)                                      # [B*D, Lc]
            joint_nll = 0.5 * (K * np.log(2 * np.pi) + logdet_j.unsqueeze(1) + qj)

            q_uni = (Xc ** 2 / var.unsqueeze(1))                              # [B*D, Lc, K]
            logdet_uni_per_var = torch.log(var)                               # [B*D, K]
            uni_nll = 0.5 * (np.log(2 * np.pi) + logdet_uni_per_var.unsqueeze(1) + q_uni)
            uni_nll_sum = uni_nll.sum(dim=2)                                  # [B*D, Lc]

            tc_loc = uni_nll_sum - joint_nll                                  # [B*D, Lc]

            if K >= 3:
                logdet_mi, qmi = _leave_one_out_nll(S, Xc, Lj, logdet_j, eps)
                loo_nll = 0.5 * (K * (K - 1) * np.log(2 * np.pi) +
                                 logdet_mi.unsqueeze(1) + qmi)
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
def local_multi_order_measures(
    X: TensorLikeArray,
    min_order: int = 3,
    max_order: Optional[int] = None,
    *,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    batch_size: int = 100000,
    time_chunk: int = 4096,
    eps: float = 1e-10,
    covmats: Optional[torch.Tensor] = None,
    precomputed: bool = False
) -> dict:
    """
    Compute local measures for every order in [min_order, max_order].

    Parameters
    ----------
    X : TensorLikeArray
        Input data (T, N) or list of (T, N) arrays.
        When precomputed=True: normalized data with shape (D, T, N).
    min_order : int, default=3
    max_order : int, optional
        If None, uses N (number of variables).
    device : str, default='cpu'
    dtype : torch.dtype, default=torch.float32
    batch_size : int, default=100000
    time_chunk : int, default=4096
    eps : float, default=1e-10
    covmats : torch.Tensor, optional
        Required when precomputed=True.
    precomputed : bool, default=False

    Returns
    -------
    dict
        Keys are orders, values are tensors [C(N, order), D, T, 4] where
        last dim is (TC, DTC, O, S).
    """
    if precomputed:
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')

        normalized_data = torch.as_tensor(X, dtype=dtype, device=device)
        if normalized_data.dim() == 2:
            normalized_data = normalized_data.unsqueeze(0)

        covmats = torch.as_tensor(covmats, dtype=dtype, device=device)
        if covmats.dim() == 2:
            covmats = covmats.unsqueeze(0)

        D, T, N = normalized_data.shape
    else:
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X[np.newaxis] if X.ndim == 2 else X, dtype=dtype)
        elif isinstance(X, torch.Tensor):
            X_tensor = (X.unsqueeze(0) if X.ndim == 2 else X).to(dtype=dtype)
        elif isinstance(X, (list, tuple)):
            X_tensor = torch.stack([torch.tensor(x, dtype=dtype) for x in X])
        else:
            X_tensor = torch.tensor(X, dtype=dtype)
            if X_tensor.ndim == 2:
                X_tensor = X_tensor.unsqueeze(0)

        normalized_data, covmats = gaussian_copula_covmat_batched(X_tensor, return_xg=True)
        D, T, N = normalized_data.shape

    if max_order is None:
        max_order = N

    out = {}
    for K in range(min_order, max_order + 1):
        nplets = torch.combinations(torch.arange(N, device=device), r=K, with_replacement=False)
        out[K] = local_nplets_measures(
            normalized_data, nplets, covmats,
            device=device, batch_size=batch_size,
            time_chunk=time_chunk, eps=eps, dtype=dtype
        )

    return out


def time_averaged_local_measures(
    X: TensorLikeArray,
    min_order: int = 3,
    max_order: Optional[int] = None,
    *,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    batch_size: int = 100000,
    time_chunk: int = 4096,
    eps: float = 1e-10,
    bias_correction: bool = True,
    covmats: Optional[torch.Tensor] = None,
    precomputed: bool = False
) -> Tuple[dict, dict]:
    """
    Master function: compute both local (temporal) and time-averaged measures.

    Computes local measures once, then averages over time and applies optional
    bias correction — avoiding duplicate computation when both outputs are needed.

    Parameters
    ----------
    X : TensorLikeArray
        Input data (T, N) or list of (T, N) arrays.
    min_order : int, default=3
    max_order : int, optional
    device : str, default='cpu'
    dtype : torch.dtype, default=torch.float32
    batch_size : int, default=100000
    time_chunk : int, default=4096
    eps : float, default=1e-10
    bias_correction : bool, default=True
        Apply bias correction after temporal averaging.
    covmats : torch.Tensor, optional
        Required when precomputed=True.
    precomputed : bool, default=False

    Returns
    -------
    local_measures : dict
        Keys are orders, values are tensors [n_combinations, D, T, 4].
    averaged_measures : dict
        Keys are orders, values are tensors [n_combinations, D, 4]
        with optional bias correction applied.
    """
    local_results = local_multi_order_measures(
        X,
        min_order=min_order,
        max_order=max_order,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        time_chunk=time_chunk,
        eps=eps,
        covmats=covmats,
        precomputed=precomputed
    )

    averaged_results = {}
    for K, measures in local_results.items():
        # measures: [n_combinations, D, T, 4]
        time_averaged = measures.mean(dim=2)  # -> [n_combinations, D, 4]

        if bias_correction and K >= 2:
            T_samples = measures.shape[2]
            tc_correction = gaussian_tc_bias_correction(
                K, T_samples, device=time_averaged.device, dtype=time_averaged.dtype
            )
            dtc_correction = gaussian_dtc_bias_correction(
                K, T_samples, device=time_averaged.device, dtype=time_averaged.dtype
            )

            corrected = time_averaged.clone()
            corrected[..., 0] = time_averaged[..., 0] - tc_correction
            corrected[..., 1] = time_averaged[..., 1] - dtc_correction
            corrected[..., 2] = corrected[..., 0] - corrected[..., 1]  # O = TC - DTC
            corrected[..., 3] = corrected[..., 0] + corrected[..., 1]  # S = TC + DTC
            averaged_results[K] = corrected
        else:
            averaged_results[K] = time_averaged

    return local_results, averaged_results
