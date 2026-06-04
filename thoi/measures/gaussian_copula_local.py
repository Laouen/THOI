from typing import Callable, Optional

import numpy as np
import torch

from thoi.typing import TensorLikeArray
from thoi.commons import gaussian_copula_covmat
from thoi.measures.utils import _all_min_1_ids, _get_single_exclusion_covmats
from thoi.measures.gaussian_copula import _batch_processing_multi_order


def gaussian_tc_bias_correction(K: int, T: int, device: torch.device = torch.device('cpu'), dtype=torch.float64) -> torch.Tensor:
    """Bias for TC = sum H(X_i) - H(X_1..X_K)."""
    from thoi.measures.utils import _gaussian_entropy_bias_correction
    return K * _gaussian_entropy_bias_correction(1, T).to(device=device, dtype=dtype) - \
           _gaussian_entropy_bias_correction(K, T).to(device=device, dtype=dtype)


def gaussian_dtc_bias_correction(K: int, T: int, device: torch.device = torch.device('cpu'), dtype=torch.float64) -> torch.Tensor:
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
    device = covmats.device
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


def _leave_one_out_stats(S: torch.Tensor, Xc: torch.Tensor, allmin1: torch.Tensor, eps: float = 1e-10):
    """
    Compute leave-one-out log-determinant sum and quadratic-form sum for DTC.

    Fully vectorized: one batched Cholesky and one batched triangular solve over
    all B×K leave-one-out subsystems, replacing the previous K-iteration Python loop.

    Parameters
    ----------
    S : torch.Tensor
        Covariance matrices, shape (B, K, K).
    Xc : torch.Tensor
        Data chunk, shape (B, Lc, K).
    allmin1 : torch.Tensor
        Leave-one-out index table, shape (K, K-1). Precomputed via _all_min_1_ids(K).
    eps : float, default=1e-10
        Regularisation added to the diagonal before Cholesky.

    Returns
    -------
    logdet_mi_sum : torch.Tensor  shape (B,)
        Sum of log|Σ_{-i}| over all K leave-one-out subsystems.
    qmi_sum : torch.Tensor  shape (B, Lc)
        Sum of x_{-i}ᵀ Σ_{-i}⁻¹ x_{-i} over all K leave-one-out subsystems.
    """
    B, K, _ = S.shape
    Lc = Xc.shape[1]

    # --- covariance submatrices ---
    # (B, K, K-1, K-1) then flatten to (B*K, K-1, K-1)
    S_loo = _get_single_exclusion_covmats(S, allmin1).reshape(B * K, K - 1, K - 1)

    # one batched Cholesky for all B*K subsystems
    eye = eps * torch.eye(K - 1, device=S.device, dtype=S.dtype)
    L_loo = torch.linalg.cholesky(S_loo + eye)                            # (B*K, K-1, K-1)
    logdet_loo = 2 * torch.log(torch.diagonal(L_loo, dim1=-2, dim2=-1)).sum(-1)  # (B*K,)
    logdet_mi_sum = logdet_loo.view(B, K).sum(dim=1)                      # (B,)

    # --- data subsets ---
    # Xc[:, :, allmin1] → (B, Lc, K, K-1) → permute → (B, K, Lc, K-1) → (B*K, Lc, K-1)
    Xc_loo = Xc[:, :, allmin1].permute(0, 2, 1, 3).reshape(B * K, Lc, K - 1)

    # one batched triangular solve for all B*K subsystems
    y = torch.linalg.solve_triangular(
        L_loo, Xc_loo.transpose(-1, -2), upper=False
    )                                                                       # (B*K, K-1, Lc)
    qmi_sum = (y * y).sum(dim=1).view(B, K, Lc).sum(dim=1)               # (B, Lc)

    return logdet_mi_sum, qmi_sum


def _local_single_batch_from_xg(
    Xg: torch.Tensor,
    covmats: torch.Tensor,
    nplets: torch.Tensor,
    *,
    allmin1: Optional[torch.Tensor],
    device: torch.device,
    time_chunk: int,
    eps: float,
) -> torch.Tensor:
    """Compute local measures for a single batch of nplets across all time steps.

    Parameters
    ----------
    Xg : Tensor[D, T, N]
        Pre-normalized Gaussian data.
    covmats : Tensor[D, N, N]
        Covariance matrices.
    nplets : Tensor[B, K]
        One batch of n-plet indices (all same order K).
    allmin1 : Tensor[K, K-1] or None
        Leave-one-out index table for order K. None when K < 3.

    Returns
    -------
    Tensor[B, D, T, 4]
        Local measures (TC, DTC, O, S) per n-plet and time point.
    """
    B, K = nplets.shape
    D, T, _ = Xg.shape

    S = _gather_nplet_covs(covmats, nplets)           # [B*D, K, K]
    Lj, logdet_j = _batched_chol_logdet(S, eps=eps)  # [B*D, K, K], [B*D]
    var = torch.diagonal(S, dim1=-2, dim2=-1)         # [B*D, K]

    batch_out = torch.empty(B, D, T, 4, device=device, dtype=Xg.dtype)

    for t0 in range(0, T, time_chunk):
        t1 = min(T, t0 + time_chunk)
        Xc = _gather_nplet_data(Xg, nplets, t0, t1)  # [B*D, Lc, K]
        Lc = t1 - t0

        qj = _quad_from_chol(Lj, Xc)                  # [B*D, Lc]
        joint_nll = 0.5 * (K * np.log(2 * np.pi) + logdet_j.unsqueeze(1) + qj)  # [B*D, Lc]

        q_uni = Xc ** 2 / var.unsqueeze(1)             # [B*D, Lc, K]
        uni_nll_sum = 0.5 * (
            np.log(2 * np.pi) + torch.log(var).unsqueeze(1) + q_uni
        ).sum(dim=2)                                   # [B*D, Lc]

        tc_loc = uni_nll_sum - joint_nll               # [B*D, Lc]

        if K >= 3:
            logdet_mi, qmi = _leave_one_out_stats(S, Xc, allmin1, eps)
            loo_nll = 0.5 * (K * (K - 1) * np.log(2 * np.pi) + logdet_mi.unsqueeze(1) + qmi)
            dtc_loc = loo_nll - (K - 1) * joint_nll
            o_loc = tc_loc - dtc_loc
            s_loc = tc_loc + dtc_loc
        else:
            dtc_loc = tc_loc
            o_loc = torch.zeros_like(tc_loc)
            s_loc = tc_loc + dtc_loc

        batch_out[:, :, t0:t1, 0] = tc_loc.view(B, D, Lc)
        batch_out[:, :, t0:t1, 1] = dtc_loc.view(B, D, Lc)
        batch_out[:, :, t0:t1, 2] = o_loc.view(B, D, Lc)
        batch_out[:, :, t0:t1, 3] = s_loc.view(B, D, Lc)

    return batch_out  # [B, D, T, 4]


def _local_nplets_from_xg(Xg: torch.Tensor,
                          covmats: torch.Tensor,
                          nplets: torch.Tensor,
                          *,
                          device: torch.device,
                          batch_size: int,
                          time_chunk: int,
                          eps: float) -> torch.Tensor:
    """Core local nplets computation from pre-normalized Gaussian data."""
    K = nplets.shape[1]
    # Precompute leave-one-out index table once — reused across every batch and time chunk.
    allmin1 = _all_min_1_ids(K, device=device) if K >= 3 else None  # (K, K-1)

    results = []
    for start in range(0, nplets.shape[0], batch_size):
        npl = nplets[start:start + batch_size]
        results.append(_local_single_batch_from_xg(
            Xg, covmats, npl,
            allmin1=allmin1, device=device,
            time_chunk=time_chunk, eps=eps,
        ))
    return torch.cat(results, dim=0)

@torch.no_grad()
def local_nplets_measures(X, nplets=None, *,
                          covmats: Optional[torch.Tensor] = None,
                          batch_size_D: Optional[int] = None,
                          device: torch.device = torch.device('cpu'),
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
    device : torch.device, default=torch.device('cpu')
        Device for computation.
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

    Xg = Xg.to(device=device)
    covmats = covmats_t.to(device=device)
    N = Xg.shape[2]

    if nplets is None:
        nplets = torch.arange(N, device=device).unsqueeze(0)
    else:
        nplets = torch.as_tensor(nplets, device=device, dtype=torch.long).contiguous()

    return _local_nplets_from_xg(
        Xg, covmats, nplets,
        device=device,
        batch_size=batch_size, time_chunk=time_chunk, eps=eps,
    )

@torch.no_grad()
def local_multi_order_measures(X: TensorLikeArray,
                               min_order: int = 3,
                               max_order: Optional[int] = None,
                               *,
                               covmats: Optional[torch.Tensor] = None,
                               batch_size_D: Optional[int] = None,
                               batch_size: int = 100000,
                               device: torch.device = torch.device('cpu'),
                               time_chunk: int = 4096,
                               eps: float = 1e-10,
                               offload_to_cpu: bool = True,
                               batch_data_collector: Optional[Callable] = None,
                               batch_aggregation: Optional[Callable] = None) -> dict:
    """
    Compute local measures for every order in [min_order, max_order] on the full set of variables.

    N-plets are generated lazily via CovarianceDataset + DataLoader (no full materialization),
    then processed in batches. The result collection and aggregation are controlled by
    ``batch_data_collector`` and ``batch_aggregation``, following the same architecture as
    ``multi_order_measures``.

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
    device : torch.device, default=torch.device('cpu')
        Device for computation.
    batch_size : int, default=100000
        Batch size for n-plet processing.
    time_chunk : int, default=4096
        Time chunk size for memory optimization.
    eps : float, default=1e-10
        Numerical stability epsilon.
    offload_to_cpu : bool, default=True
        When True, each batch is moved to CPU immediately after computation.
        Set to False if the device has sufficient memory and you want to avoid
        repeated small transfers.  See ``_batch_processing_multi_order`` for details.
    batch_data_collector : callable, optional
        ``(nplets: Tensor[B, K], result: Tensor[B, D, T, 4], bn: int) -> Any``
        Post-processes each batch result. Defaults to the identity (returns the result tensor).
        Can be used to write results to disk batch-by-batch to avoid materializing the full output.
    batch_aggregation : callable, optional
        ``(items: list[Any]) -> Any``
        Aggregates all collected items for one order.
        Defaults to ``torch.cat(items, dim=0)`` → ``Tensor[C(N, order), D, T, 4]``.

    Returns
    -------
    dict
        ``{order: aggregated_result}`` for each order in [min_order, max_order].
        With default callbacks the values are ``Tensor[C(N, order), D, T, 4]``
        where the last dimension is (TC, DTC, O, S).
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

    Xg = Xg.to(device=device)
    covmats_t = covmats_t.to(device=device)
    N = Xg.shape[2]

    if max_order is None:
        max_order = N

    # Cache allmin1 per order so it is computed once, not once per batch.
    _allmin1_cache: dict = {}

    def _batch_fn(nplets, K):
        if K not in _allmin1_cache:
            _allmin1_cache[K] = _all_min_1_ids(K, device=device) if K >= 3 else None
        return _local_single_batch_from_xg(
            Xg, covmats_t, nplets,
            allmin1=_allmin1_cache[K],
            device=device,
            time_chunk=time_chunk, eps=eps,
        )

    return _batch_processing_multi_order(
        N=N, min_order=min_order, max_order=max_order,
        batch_fn=_batch_fn,
        batch_size=batch_size,
        device=device,
        offload_to_cpu=offload_to_cpu,
        batch_data_collector=batch_data_collector,
        batch_aggregation=batch_aggregation,
    )


def time_averaged_local_measures(
    X: TensorLikeArray,
    min_order: int = 3,
    max_order: Optional[int] = None,
    *,
    covmats: Optional[torch.Tensor] = None,
    batch_size_D: Optional[int] = None,
    device: torch.device = torch.device('cpu'),
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
    device : torch.device, default=torch.device('cpu')
        Device for computation.
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

    # Get raw (nplets, result) tuples — bypass the DataFrame default so we can do
    # tensor operations (mean, bias correction) on the time dimension.
    raw_batches = local_multi_order_measures(
        X,
        min_order=min_order,
        max_order=max_order,
        covmats=covmats,
        batch_size_D=batch_size_D,
        device=device,
        batch_size=batch_size,
        time_chunk=time_chunk,
        eps=eps,
        batch_aggregation=lambda items: items,
    )

    # Group by order K (recoverable from nplets.shape[1]) and time-average per group.
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for nplets, measures in raw_batches:
        groups[nplets.shape[1]].append(measures)

    # Time-average the results and apply bias correction AFTER temporal averaging
    averaged_results = {}

    for K, measure_list in groups.items():
        measures = torch.cat(measure_list, dim=0)  # [C(N,K), D, T, 4]
        # Average over time dimension
        time_averaged = measures.mean(dim=2)  # -> [C(N,K), D, 4]

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
