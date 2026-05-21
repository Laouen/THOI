from typing import Optional, List, Tuple, Union

import numpy as np
import scipy as sp
import torch

from thoi.typing import TensorLikeArray
from thoi.commons import gaussian_copula_covmat_batched


def build_lagged_embedding(
    data_list: List[torch.Tensor],
    shifts: torch.Tensor
) -> torch.Tensor:
    """
    Build a time-lagged embedding by stacking shifted copies of multivariate series.

    Parameters
    ----------
    data_list : list of L tensors, each of shape (T, N)
    shifts : tensor of lag values (e.g., torch.tensor([0, 1, 2, 5]))

    Returns
    -------
    torch.Tensor of shape (L, T - max(shifts), len(shifts) * N)
    """
    L = len(data_list)
    T, N = data_list[0].shape
    device = data_list[0].device
    shifts = shifts.to(device)
    max_shift = int(shifts.max().item())

    X_tensor = torch.stack(data_list)  # (L, T, N)

    idx_base = torch.arange(T - max_shift, device=device)
    idx = idx_base.unsqueeze(0) + shifts.view(-1, 1)  # (len(shifts), T - max_shift)

    lagged = X_tensor[:, idx, :]  # (L, len(shifts), T - max_shift, N)
    lagged = lagged.permute(0, 2, 1, 3).reshape(L, T - max_shift, -1)

    return lagged  # (L, T - max_shift, len(shifts) * N)


@torch.no_grad()
def batch_local_ais_torch(
    data_batch: torch.Tensor,   # [L, T, k*N] output of build_lagged_embedding
    cov_batch: torch.Tensor,    # [L, k*N, k*N] covariance per block, same order as shifts
    shifts: torch.Tensor,       # [k] with shifts; shifts[0] must be 0
    eps: float = 1e-10,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Compute local Active Information Storage (AIS) per lag.

    Returns
    -------
    i_local : torch.Tensor, shape [L, T, k-1]
        Local AIS in nats per lag (excluding lag 0).
    """
    L, T, D = data_batch.shape
    k = shifts.numel()
    assert D % k == 0, "D must be a multiple of len(shifts)"
    N = D // k

    data_batch = data_batch.to(device)
    cov_batch = cov_batch.to(device)
    eyeN = torch.eye(N, device=device)

    def block(i, j):
        return cov_batch[:, i * N:(i + 1) * N, j * N:(j + 1) * N]  # [L,N,N]

    Sigma_t = block(0, 0) + eps * eyeN                                              # [L,N,N]
    Sigma_y = torch.stack([block(s, s) for s in range(1, k)], 1) + eps * eyeN      # [L,k-1,N,N]
    Sigma_ty = torch.stack([block(0, s) for s in range(1, k)], 1)                  # [L,k-1,N,N]
    Sigma_yt = Sigma_ty.transpose(-1, -2)                                           # [L,k-1,N,N]

    L_t = torch.linalg.cholesky(Sigma_t)                                            # [L,N,N]
    logdet_t = 2.0 * torch.log(torch.diagonal(L_t, dim1=-2, dim2=-1)).sum(-1)      # [L]

    L_y = torch.linalg.cholesky(Sigma_y)                                            # [L,k-1,N,N]
    M = torch.cholesky_solve(Sigma_yt, L_y)                                         # [L,k-1,N,N]
    Sigma_cond = Sigma_t.unsqueeze(1) - torch.matmul(Sigma_ty, M) + eps * eyeN     # [L,k-1,N,N]
    L_cond = torch.linalg.cholesky(Sigma_cond)                                      # [L,k-1,N,N]
    logdet_cond = 2.0 * torch.log(torch.diagonal(L_cond, dim1=-2, dim2=-1)).sum(-1) # [L,k-1]

    x = data_batch[:, :, :N]                                                        # [L,T,N]
    y = torch.stack([data_batch[:, :, s * N:(s + 1) * N] for s in range(1, k)], 2) # [L,T,k-1,N]

    z_marg = torch.cholesky_solve(x.transpose(1, 2), L_t)                           # [L,N,T]
    quad_marg = (x.transpose(1, 2) * z_marg).sum(1)                                 # [L,T]

    y_rhs = y.permute(0, 2, 3, 1)                                                   # [L,k-1,N,T]
    y_inv = torch.cholesky_solve(y_rhs, L_y)                                        # [L,k-1,N,T]
    mu = torch.matmul(Sigma_ty, y_inv).permute(0, 3, 1, 2)                          # [L,T,k-1,N]

    diff = x.unsqueeze(2) - mu                                                      # [L,T,k-1,N]
    diff_T = diff.permute(0, 2, 3, 1)                                               # [L,k-1,N,T]
    z_cond = torch.cholesky_solve(diff_T, L_cond)                                   # [L,k-1,N,T]
    quad_cond = (diff_T * z_cond).sum(2).permute(0, 2, 1)                           # [L,T,k-1]

    # i_local(t,τ) = 0.5[log|Σ_t| - log|Σ_{t|y_τ}|] + 0.5[quad_marg - quad_cond]
    i_local = (
        0.5 * (logdet_t[:, None, None] - logdet_cond[:, None, :])
        + 0.5 * (quad_marg[:, :, None] - quad_cond)
    )  # [L,T,k-1]

    return i_local


def gaussian_mi_bias_correction(T: int) -> torch.Tensor:
    """Bias correction for Gaussian mutual information."""
    return torch.tensor((sp.special.psi((T - 1) / 2) - sp.special.psi((T - 2) / 2)) / 2)


def gaussian_ais_bias_correction(N: int, T: int, device='cpu') -> torch.Tensor:
    """Bias correction for Gaussian AIS."""
    if T is None:
        return torch.tensor(0.0, device=device)

    def bias_H(n):
        psi_terms = torch.from_numpy(
            sp.special.psi((T - np.arange(1, n + 1)) / 2)
        ).to(device)
        return 0.5 * (n * torch.log(torch.tensor(2.0 / (T - 1), device=device)) + psi_terms.sum())

    return bias_H(N) + bias_H(N) - bias_H(2 * N)


def compute_ais(
    cov_batch: torch.Tensor,
    lags: torch.Tensor,
    bias: torch.Tensor,
    device: str = 'cpu',
) -> torch.Tensor:
    """Compute AIS from lagged covariance matrices."""
    B, D, _ = cov_batch.shape
    k = len(lags)
    N = D // k
    cov_batch = cov_batch + 1e-10 * torch.eye(D, device=device).unsqueeze(0)
    idx_t = torch.arange(N, device=device)
    Sigma_t = cov_batch[:, idx_t][:, :, idx_t]
    Sigma_lag = torch.stack([
        cov_batch[:, tau * N:(tau + 1) * N, tau * N:(tau + 1) * N]
        for tau in range(1, k)
    ], dim=1)
    joint_idx = torch.stack([
        torch.cat([idx_t, torch.arange(tau * N, (tau + 1) * N, device=device)])
        for tau in range(1, k)
    ])
    batch_idx = torch.arange(B, device=device)[:, None, None, None]
    ji = joint_idx.unsqueeze(0).expand(B, -1, -1)
    rows = ji.unsqueeze(3).expand(-1, -1, -1, 2 * N)
    cols = ji.unsqueeze(2).expand(-1, -1, 2 * N, -1)
    Sigma_joint = cov_batch[batch_idx, rows, cols]
    log_det_past = torch.logdet(Sigma_lag)
    log_det_joint = torch.logdet(Sigma_joint)
    log_det_t = torch.logdet(Sigma_t)
    ais = 0.5 * (log_det_past + log_det_t.unsqueeze(1) - log_det_joint) - bias
    ais = torch.clamp(ais, min=0.0)
    return ais


def extract_lagged_subcovariance(
    cov_full: torch.Tensor,
    idxs: torch.Tensor,   # shape: (S, g)
    N_total: int,
) -> torch.Tensor:
    """
    Extract sub-covariances from a full lagged covariance matrix.

    Supports both a single matrix (D, D) and a batch (B, D, D). The function
    accounts for the lag structure: indices in idxs refer to channels in the
    original (non-lagged) space and are replicated across all lag blocks.

    Parameters
    ----------
    cov_full : torch.Tensor
        Shape (N, N) for a single matrix or (B, N, N) for a batch.
        N = N_total * k where k is the number of lags.
    idxs : torch.Tensor, shape (S, g)
        Channel indices in the original (lag-0) space.
    N_total : int
        Number of channels per lag block.

    Returns
    -------
    torch.Tensor
        Shape (S, g*k, g*k) for a single matrix or (B, S, g*k, g*k) for a batch.
    """
    S, g = idxs.shape
    batched = cov_full.dim() == 3
    if batched:
        B, N, _ = cov_full.shape
    else:
        N = cov_full.size(0)
    device = cov_full.device

    # infer number of lags from total dimension
    k = N // N_total

    # block offsets: [0, N_total, 2*N_total, ..., (k-1)*N_total]
    lag_offsets = torch.arange(k, device=device).view(k, 1, 1) * N_total  # (k,1,1)

    # for each subset and each lag, add the offset to get absolute indices
    base = idxs.unsqueeze(0) + lag_offsets  # (k, S, g)

    # flatten to (S, k*g)
    flat = base.permute(1, 0, 2).reshape(S, k * g)

    # construct row/column indices (S, k*g, k*g)
    rows = flat.unsqueeze(2).expand(-1, -1, k * g)
    cols = flat.unsqueeze(1).expand(-1, k * g, -1)

    if not batched:
        return cov_full[rows, cols]  # (S, k*g, k*g)
    else:
        rows_b = rows.unsqueeze(0).expand(B, -1, -1, -1)
        cols_b = cols.unsqueeze(0).expand(B, -1, -1, -1)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(-1, S, k * g, k * g)
        return cov_full[batch_idx, rows_b, cols_b]  # (B, S, k*g, k*g)


@torch.no_grad()
def batch_compute_tdmi_xcorr_torch(
    cov_batch: torch.Tensor,  # (B, D, D)
    lags: torch.Tensor,       # (k,)
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute TDMI and cross-correlation for all variable pairs across specified lags.

    Parameters
    ----------
    cov_batch : torch.Tensor, shape (B, D, D)
        Batch of full lagged covariance matrices. D = N_vars * n_lags.
    lags : torch.Tensor, shape (k,)
        Lag indices (including zero).
    device : str, default 'cpu'

    Returns
    -------
    tdmi : torch.Tensor, shape (B, k, N, N)
        TDMI for each variable pair at each lag.
    xcorr : torch.Tensor, shape (B, k, N, N)
        Time-delayed Pearson cross-correlation for each pair at each lag.
    """
    B, D, _ = cov_batch.shape
    k = lags.numel()
    N = D // k

    cov_batch = cov_batch.to(device) + 1e-10 * torch.eye(D, device=device).unsqueeze(0)

    Sigma_lagged = torch.stack([
        cov_batch[:, tau * N:(tau + 1) * N, :N]
        for tau in range(k)
    ], dim=1)  # (B, k, N, N)

    Sigma_t = cov_batch[:, :N, :N]                                          # (B, N, N)
    var_t = torch.diagonal(Sigma_t, offset=0, dim1=1, dim2=2)               # (B, N)
    std_t = torch.sqrt(var_t)                                                # (B, N)

    denom_corr = std_t.unsqueeze(2) * std_t.unsqueeze(1)                    # (B, N, N)
    denom_corr = denom_corr.unsqueeze(1).expand(-1, k, -1, -1)              # (B, k, N, N)
    xcorr = Sigma_lagged / (denom_corr + 1e-10)
    xcorr = torch.clamp(xcorr, -0.999999, 0.999999)

    # Gaussian MI: I = -½ log(1 - ρ²)
    tdmi = -0.5 * torch.log1p(-xcorr.pow(2))

    return tdmi, xcorr


def build_full_tdmi(measure: torch.Tensor) -> torch.Tensor:
    """
    Construct the full bidirectional TDMI tensor for lags -(k-1) to +(k-1).

    Parameters
    ----------
    measure : torch.Tensor, shape (B, k, N, N)
        Forward-lag TDMI: measure[b, τ, i, j] = I(X_i(t); X_j(t-τ)).

    Returns
    -------
    full_tdmi : torch.Tensor, shape (B, N, N, 2*k - 1)
        Diagonal entries (i==j) are set to NaN.
    """
    _, k, N, _ = measure.shape

    m_fwd = measure
    m_bwd = torch.flip(measure.transpose(2, 3), dims=[1])[:, :-1, :, :]  # (B, k-1, N, N)

    m_full = torch.cat([m_bwd, m_fwd], dim=1)         # (B, 2k-1, N, N)
    full_tdmi = m_full.permute(0, 2, 3, 1)            # (B, N, N, 2k-1)

    eye = torch.eye(N, device=measure.device, dtype=torch.bool)
    mask = eye.unsqueeze(0).unsqueeze(-1)
    full_tdmi = full_tdmi.masked_fill(mask, float('nan'))

    return full_tdmi


@torch.no_grad()
def local_ais(
    X: TensorLikeArray,
    shifts: Union[List[int], torch.Tensor, np.ndarray],
    *,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-10,
    covmats: Optional[torch.Tensor] = None,
    precomputed: bool = False,
) -> torch.Tensor:
    """
    Compute local (time-resolved) Active Information Storage across lags.

    Parameters
    ----------
    X : TensorLikeArray
        Input data (T, N) or list of (T, N) arrays.
    shifts : array-like
        Lag values starting with 0. Example: [0, 1, 2, 5].
    device : str, default='cpu'
    dtype : torch.dtype, default=torch.float32
    eps : float, default=1e-10
    covmats : torch.Tensor, optional
        Required when precomputed=True.
    precomputed : bool, default=False

    Returns
    -------
    torch.Tensor, shape [L, T, k-1]
        Local AIS per lag (excluding lag 0).
    """
    shifts_tensor = torch.as_tensor(shifts, dtype=torch.int32, device=device)
    if shifts_tensor.numel() == 0 or shifts_tensor[0].item() != 0:
        raise ValueError("shifts must start with 0 (present time)")

    def _ensure_batched_cov(cov):
        c = torch.as_tensor(cov, dtype=dtype, device=device)
        return c.unsqueeze(0) if c.dim() == 2 else c

    if precomputed:
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        covmats = _ensure_batched_cov(covmats)
        normalized_data = torch.as_tensor(X, dtype=dtype, device=device)
        if normalized_data.dim() == 2:
            normalized_data = normalized_data.unsqueeze(0)
    else:
        if isinstance(X, (torch.Tensor, np.ndarray)):
            arr = torch.as_tensor(X, dtype=dtype, device=device)
            if arr.dim() != 2:
                raise ValueError("Single dataset must have shape (T, N)")
            data_list = [arr]
        elif isinstance(X, (list, tuple)):
            data_list = [torch.as_tensor(x, dtype=dtype, device=device) for x in X]
        else:
            raise ValueError("X must be array-like or list of array-like")

        lagged_data = build_lagged_embedding(data_list, shifts_tensor)
        normalized_data, covmats = gaussian_copula_covmat_batched(lagged_data, return_xg=True)

    return batch_local_ais_torch(normalized_data, covmats, shifts_tensor, eps=eps, device=device)


@torch.no_grad()
def ais(
    X: TensorLikeArray,
    shifts: Union[List[int], torch.Tensor, np.ndarray],
    *,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    bias_correction: bool = True,
    covmats: Optional[torch.Tensor] = None,
    T: Optional[int] = None,
    precomputed: bool = False,
) -> torch.Tensor:
    """
    Compute Active Information Storage (time-averaged) across lags.

    Parameters
    ----------
    X : TensorLikeArray
        Input data (T, N) or list of (T, N) arrays.
    shifts : array-like
        Lag values starting with 0.
    device : str, default='cpu'
    dtype : torch.dtype, default=torch.float32
    bias_correction : bool, default=True
    covmats : torch.Tensor, optional
        Required when precomputed=True.
    T : int, optional
        Number of samples for bias correction when precomputed=True.
    precomputed : bool, default=False

    Returns
    -------
    torch.Tensor, shape [B, k-1]
        AIS per lag (excluding lag 0).
    """
    shifts_tensor = torch.as_tensor(shifts, dtype=torch.int32, device=device)
    if shifts_tensor.numel() == 0 or shifts_tensor[0].item() != 0:
        raise ValueError("shifts must start with 0 (present time)")
    k = shifts_tensor.numel()

    def _ensure_batched_cov(cov):
        c = torch.as_tensor(cov, dtype=dtype, device=device)
        return c.unsqueeze(0) if c.dim() == 2 else c

    def _build_bias(B, N_vars, T_samples):
        if bias_correction and T_samples is not None:
            bias_corr = gaussian_ais_bias_correction(N_vars, T_samples, device=device)
            return bias_corr.unsqueeze(0).expand(B, k - 1).to(device)
        return torch.zeros(B, k - 1, device=device)

    if precomputed:
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        covmats = _ensure_batched_cov(covmats)
        B, D, _ = covmats.shape
        N_vars = D // k
        T_samples = T
        if X is not None:
            X_tensor = torch.as_tensor(X, dtype=dtype, device=device)
            T_samples = X_tensor.shape[0] if X_tensor.dim() == 2 else X_tensor.shape[1]
        bias_tensor = _build_bias(B, N_vars, T_samples)
    else:
        if isinstance(X, (torch.Tensor, np.ndarray)):
            arr = torch.as_tensor(X, dtype=dtype, device=device)
            if arr.dim() != 2:
                raise ValueError("Single dataset must have shape (T, N)")
            data_list = [arr]
        elif isinstance(X, (list, tuple)):
            data_list = [torch.as_tensor(x, dtype=dtype, device=device) for x in X]
        else:
            raise ValueError("X must be array-like or list of array-like")

        lagged_data = build_lagged_embedding(data_list, shifts_tensor)
        normalized_data, covmats = gaussian_copula_covmat_batched(lagged_data, return_xg=True)
        T_samples = normalized_data.shape[1]
        N_vars = lagged_data.shape[2] // k
        B = covmats.shape[0]
        bias_tensor = _build_bias(B, N_vars, T_samples)

    return compute_ais(covmats, shifts_tensor, bias_tensor, device=device)


@torch.no_grad()
def nplets_ais(
    X: Optional[TensorLikeArray] = None,
    shifts: Union[List[int], torch.Tensor, np.ndarray] = None,
    idxs: Union[List[int], torch.Tensor, np.ndarray] = None,
    *,
    covmats: Optional[torch.Tensor] = None,
    T: Optional[int] = None,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    bias_correction: bool = True,
    precomputed: bool = False,
) -> torch.Tensor:
    """
    Compute AIS for specified subsets (n-plets) of variables.

    Parameters
    ----------
    X : TensorLikeArray, optional
        Input data (T, N) or list of (T, N) arrays.
    shifts : array-like
        Lag values starting with 0.
    idxs : array-like
        Variable indices defining the subset(s), shape (g,) or (S, g).
    covmats : torch.Tensor, optional
        Full lagged covariance matrices, shape (B, D, D) or (D, D).
    T : int, optional
        Number of samples for bias correction when providing covmats directly.
    device : str, default='cpu'
    dtype : torch.dtype, default=torch.float32
    bias_correction : bool, default=True
    precomputed : bool, default=False

    Returns
    -------
    torch.Tensor
        AIS values, shape [B, k-1] or [B, S, k-1] for multiple index sets.
    """
    if shifts is None:
        raise ValueError('shifts must be provided')

    shifts_tensor = torch.as_tensor(shifts, dtype=torch.long, device=device)
    if shifts_tensor.numel() == 0 or shifts_tensor[0].item() != 0:
        raise ValueError('shifts must start with 0 (present time)')
    k = shifts_tensor.numel()

    covmats_full = None
    normalized_data = None

    if precomputed:
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        covmats_full = torch.as_tensor(covmats, dtype=dtype, device=device)
        if covmats_full.dim() == 2:
            covmats_full = covmats_full.unsqueeze(0)
        if X is not None:
            normalized_data = torch.as_tensor(X, dtype=dtype, device=device)
            if normalized_data.dim() == 2:
                normalized_data = normalized_data.unsqueeze(0)
    else:
        if X is not None:
            if isinstance(X, (torch.Tensor, np.ndarray)):
                arr = torch.as_tensor(X, dtype=dtype, device=device)
                if arr.dim() != 2:
                    raise ValueError('Single dataset must have shape (T, N)')
                data_list = [arr]
            elif isinstance(X, (list, tuple)):
                data_list = [torch.as_tensor(x, dtype=dtype, device=device) for x in X]
            else:
                raise ValueError('X must be array-like or list of array-like')

            lagged_data = build_lagged_embedding(data_list, shifts_tensor)
            normalized_data, covmats_full = gaussian_copula_covmat_batched(lagged_data, return_xg=True)

        if covmats is not None:
            covmats_full = torch.as_tensor(covmats, dtype=dtype, device=device)

    if covmats_full is None:
        raise ValueError('Unable to obtain covariance matrices from X or covmats')

    if covmats_full.dim() == 2:
        covmats_full = covmats_full.unsqueeze(0)
    B, D, _ = covmats_full.shape

    idxs_tensor = torch.as_tensor(idxs, dtype=torch.long, device=device)
    if idxs_tensor.dim() == 1:
        idxs_tensor = idxs_tensor.unsqueeze(0)  # (S=1, g)
    S, g = idxs_tensor.shape

    N_total = D // k
    if N_total * k != D:
        raise ValueError('covmats dimensionality is not compatible with provided shifts')

    subcovs = extract_lagged_subcovariance(covmats_full, idxs_tensor, N_total)
    if subcovs.dim() == 3:
        subcovs = subcovs.unsqueeze(0)

    gk = g * k
    cov_sub_batch = subcovs.reshape(B * S, gk, gk)

    T_samples = normalized_data.shape[1] if normalized_data is not None else T

    if bias_correction and T_samples is not None:
        bias_corr = gaussian_ais_bias_correction(g, T_samples, device=device)
        bias_tensor = bias_corr.unsqueeze(0).expand(B * S, k - 1).to(device)
    else:
        bias_tensor = torch.zeros(B * S, k - 1, device=device)

    ais_vals = compute_ais(cov_sub_batch, shifts_tensor, bias_tensor, device=device)
    ais_vals = ais_vals.view(B, S, -1)
    if S == 1:
        ais_vals = ais_vals.squeeze(1)

    return ais_vals


@torch.no_grad()
def nplets_local_ais(
    X: Optional[TensorLikeArray] = None,
    shifts: Union[List[int], torch.Tensor, np.ndarray] = None,
    idxs: Union[List[int], torch.Tensor, np.ndarray] = None,
    *,
    covmats: Optional[torch.Tensor] = None,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-10,
    precomputed: bool = False,
) -> torch.Tensor:
    """
    Compute local (time-resolved) AIS for specified subsets (n-plets) of variables.

    Parameters
    ----------
    X : TensorLikeArray
        Raw data (T, N) or list of (T, N) arrays.
        When precomputed=True: stacked lagged normalized data (B, T_eff, D).
    shifts : array-like
        Lag values starting with 0.
    idxs : array-like
        Variable indices, shape (g,) or (S, g).
    covmats : torch.Tensor, optional
        Overrides covariances computed from X when provided.
    device : str, default='cpu'
    dtype : torch.dtype, default=torch.float32
    eps : float, default=1e-10
    precomputed : bool, default=False

    Returns
    -------
    torch.Tensor, shape [B, S, T, k-1] or [B, T, k-1] when S==1.
    """
    if shifts is None or idxs is None:
        raise ValueError('shifts and idxs must be provided')

    shifts_tensor = torch.as_tensor(shifts, dtype=torch.long, device=device)
    if shifts_tensor.numel() == 0 or shifts_tensor[0].item() != 0:
        raise ValueError('shifts must start with 0')
    k = shifts_tensor.numel()

    covmats_full = None
    normalized_data = None

    if precomputed:
        if covmats is None or X is None:
            raise ValueError('covmats and X must be provided when precomputed=True')
        covmats_full = torch.as_tensor(covmats, dtype=dtype, device=device)
        if covmats_full.dim() == 2:
            covmats_full = covmats_full.unsqueeze(0)
        normalized_data = torch.as_tensor(X, dtype=dtype, device=device)
        if normalized_data.dim() == 2:
            normalized_data = normalized_data.unsqueeze(0)
        B, T_eff, D = normalized_data.shape
    else:
        if X is None:
            raise ValueError('X must be provided')
        if isinstance(X, (torch.Tensor, np.ndarray)):
            arr = torch.as_tensor(X, dtype=dtype, device=device)
            if arr.dim() != 2:
                raise ValueError('Single dataset must have shape (T, N)')
            data_list = [arr]
        elif isinstance(X, (list, tuple)):
            data_list = [torch.as_tensor(x, dtype=dtype, device=device) for x in X]
        else:
            raise ValueError('X must be array-like or list of array-like')

        lagged_data = build_lagged_embedding(data_list, shifts_tensor)
        normalized_data, covmats_full = gaussian_copula_covmat_batched(lagged_data, return_xg=True)

        if covmats is not None:
            covmats_full = torch.as_tensor(covmats, dtype=dtype, device=device)
        if covmats_full.dim() == 2:
            covmats_full = covmats_full.unsqueeze(0)
        B, T_eff, D = normalized_data.shape

    idxs_tensor = torch.as_tensor(idxs, dtype=torch.long, device=device)
    if idxs_tensor.dim() == 1:
        idxs_tensor = idxs_tensor.unsqueeze(0)
    S, g = idxs_tensor.shape

    N_total = D // k

    lag_offsets = torch.arange(k, device=device, dtype=torch.long).view(k, 1) * N_total
    positions = (lag_offsets.unsqueeze(1) + idxs_tensor.unsqueeze(0)).permute(1, 0, 2).reshape(S, k * g)

    Xexp = normalized_data.unsqueeze(1).expand(B, S, T_eff, D)
    idx_gather = positions.unsqueeze(0).unsqueeze(2).expand(B, S, T_eff, k * g)
    data_selected = torch.gather(Xexp, dim=3, index=idx_gather)  # (B, S, T_eff, k*g)
    data_batch = data_selected.reshape(B * S, T_eff, k * g)

    subcovs = extract_lagged_subcovariance(covmats_full, idxs_tensor, N_total)
    if subcovs.dim() == 3:
        subcovs = subcovs.unsqueeze(0)
    cov_sub_batch = subcovs.reshape(B * S, g * k, g * k)

    i_local = batch_local_ais_torch(
        data_batch, cov_sub_batch, shifts_tensor, eps=eps, device=device
    )  # [B*S, T_eff, k-1]

    i_local = i_local.view(B, S, T_eff, -1)
    if S == 1:
        i_local = i_local.squeeze(1)

    return i_local


@torch.no_grad()
def time_delayed_mutual_information(
    X: Optional[TensorLikeArray] = None,
    lags: Union[List[int], torch.Tensor, np.ndarray] = None,
    *,
    covmats: Optional[torch.Tensor] = None,
    T: Optional[int] = None,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    bias_correction: bool = True,
    return_full: bool = True,
    precomputed: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Compute Time-Delayed Mutual Information (TDMI) and cross-correlation.

    Returns the pairwise MI between all variable pairs across specified time lags,
    optionally including the full bidirectional (time-lagged connectivity) matrix.

    Parameters
    ----------
    X : TensorLikeArray, optional
        Input data (T, N) or list of (T, N) arrays.
        Required when precomputed=False.
    lags : array-like
        Lag values starting with 0.
    covmats : torch.Tensor, optional
        Precomputed lagged covariance matrices (B, D, D) where D = N * k.
        Required when precomputed=True.
    T : int, optional
        Number of time samples. Required for bias correction when precomputed=True.
    device : str, default='cpu'
    dtype : torch.dtype, default=torch.float32
    bias_correction : bool, default=True
    return_full : bool, default=True
        If True, also return the full bidirectional TDMI matrix.
    precomputed : bool, default=False

    Returns
    -------
    tdmi_vals : torch.Tensor, shape (B, k, N, N)
    xcorr_vals : torch.Tensor, shape (B, k, N, N)
    full_tdmi : torch.Tensor, shape (B, N, N, 2*k-1)  (only if return_full=True)
    """
    if lags is None:
        raise ValueError('lags must be provided')

    lags_tensor = torch.as_tensor(lags, dtype=torch.int32, device=device)
    if lags_tensor.numel() == 0 or lags_tensor[0].item() != 0:
        raise ValueError('lags must start with 0 (present time)')
    k = lags_tensor.numel()

    if precomputed:
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        if T is None:
            raise ValueError('T must be provided when precomputed=True')
        covmats_final = torch.as_tensor(covmats, dtype=dtype, device=device)
        if covmats_final.dim() == 2:
            covmats_final = covmats_final.unsqueeze(0)
        T_samples = T
    else:
        if X is None:
            raise ValueError('X must be provided when precomputed=False')
        if isinstance(X, (torch.Tensor, np.ndarray)):
            arr = torch.as_tensor(X, dtype=dtype, device=device)
            if arr.dim() != 2:
                raise ValueError("Single dataset must have shape (T, N)")
            data_list = [arr]
        elif isinstance(X, (list, tuple)):
            data_list = [torch.as_tensor(x, dtype=dtype, device=device) for x in X]
        else:
            raise ValueError("X must be array-like or list of array-like")

        lagged_data = build_lagged_embedding(data_list, lags_tensor)
        normalized_data, covmats_final = gaussian_copula_covmat_batched(lagged_data, return_xg=True)
        T_samples = normalized_data.shape[1]

    tdmi_vals, xcorr_vals = batch_compute_tdmi_xcorr_torch(covmats_final, lags_tensor, device=device)

    if bias_correction:
        bias_corr = gaussian_mi_bias_correction(T_samples)
        tdmi_vals = tdmi_vals - bias_corr

    if return_full:
        full_tdmi = build_full_tdmi(tdmi_vals)
        return tdmi_vals, xcorr_vals, full_tdmi

    return tdmi_vals, xcorr_vals
