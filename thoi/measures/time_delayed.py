from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from thoi.commons import gaussian_copula_covmat
from thoi.measures.utils import _gaussian_entropy_bias_correction
from thoi.typing import TensorLikeArray


def _as_lag_tensor(
    lags: Union[List[int], torch.Tensor, np.ndarray],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Validate lag inputs and return them as a tensor on the target device."""
    if lags is None:
        raise ValueError('lags must be provided')

    lags_tensor = torch.as_tensor(lags, dtype=torch.long, device=device)
    if lags_tensor.numel() == 0 or lags_tensor[0].item() != 0:
        raise ValueError('lags must start with 0 (present time)')
    if torch.any(lags_tensor < 0):
        raise ValueError('lags must be non-negative')

    return lags_tensor


def _as_data_list(
    X: TensorLikeArray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Normalize accepted timeseries inputs into a list of (T, N) tensors."""
    if isinstance(X, (torch.Tensor, np.ndarray)):
        X_tensor = torch.as_tensor(X, dtype=dtype, device=device)
        if X_tensor.ndim == 2:
            return [X_tensor]
        if X_tensor.ndim == 3:
            return [x for x in X_tensor]
        raise ValueError('X must have shape (T, N), (D, T, N), or be a list of (T, N) arrays')

    if isinstance(X, (list, tuple)):
        data_list = [torch.as_tensor(x, dtype=dtype, device=device) for x in X]
        if not data_list:
            raise ValueError('X must contain at least one dataset')
        if not all(x.ndim == 2 for x in data_list):
            raise ValueError('All datasets must have shape (T, N)')
        if not all(x.shape[1] == data_list[0].shape[1] for x in data_list):
            raise ValueError('All datasets must have the same number of variables')
        return data_list

    raise ValueError('X must be array-like or a list of array-like datasets')


def _ensure_batched_covmat(
    covmats: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    covmats = torch.as_tensor(covmats, dtype=dtype, device=device)
    if covmats.ndim == 2:
        covmats = covmats.unsqueeze(0)
    if covmats.ndim != 3 or covmats.shape[-1] != covmats.shape[-2]:
        raise ValueError('covmats must have shape (D, N, N) or (N, N)')
    return covmats


def _validate_temporal_covmat(covmats: torch.Tensor, n_lags: int) -> Tuple[int, int, int]:
    B, D, _ = covmats.shape
    if D % n_lags != 0:
        raise ValueError('covmats dimensionality is not compatible with the provided lags')
    return B, D, D // n_lags


def generate_stacked_lagged_batches(
    data_list: List[torch.Tensor],
    lags: Union[List[int], torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """
    Stack lagged versions of multivariate time series.

    Parameters
    ----------
    data_list : list of torch.Tensor
        Datasets with shape ``(T, N)``. All datasets must have the same temporal
        length and number of variables.
    lags : array-like
        Lag values, including zero as the first element.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(D, T - max(lags), len(lags) * N)``.
    """
    if not data_list:
        raise ValueError('data_list must contain at least one dataset')

    lags_tensor = _as_lag_tensor(lags, device=data_list[0].device)
    T, N = data_list[0].shape

    if not all(x.ndim == 2 for x in data_list):
        raise ValueError('All datasets must have shape (T, N)')
    if not all(x.shape == (T, N) for x in data_list):
        raise ValueError('All datasets must have the same shape')

    max_lag = int(lags_tensor.max().item())
    T_eff = T - max_lag
    if T_eff <= 1:
        raise ValueError('Temporal embedding needs at least two effective samples')

    X_tensor = torch.stack(data_list)
    idx_base = torch.arange(max_lag, T, device=data_list[0].device)
    idx = idx_base.unsqueeze(0) - lags_tensor.view(-1, 1)
    lagged = X_tensor[:, idx, :]

    return lagged.permute(0, 2, 1, 3).reshape(len(data_list), T_eff, len(lags_tensor) * N)


@torch.no_grad()
def precompute_temporal_embedding(
    X: TensorLikeArray,
    lags: Union[List[int], torch.Tensor, np.ndarray],
    *,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
    batch_size_D: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
    """
    Precompute the lagged Gaussian-copula embedding used by AIS and TDMI.

    Returns
    -------
    Xg : torch.Tensor
        Gaussian-copula normalized lagged data with shape ``(D, T_eff, k*N)``.
        ``T_eff`` is the effective number of samples after temporal embedding:
        ``T_eff = T - max(lags)``.
    covmats : torch.Tensor
        Covariance matrices with shape ``(D, k*N, k*N)``.
    T_eff : int
        Effective number of temporal samples after lag embedding, defined as
        ``T_eff = T - max(lags)``.
    N : int
        Number of variables in the original datasets.
    lags : torch.Tensor
        Validated lag tensor on ``device``.
    """
    device = torch.device(device)
    lags_tensor = _as_lag_tensor(lags, device=device)
    data_list = _as_data_list(X, device=device, dtype=dtype)
    lagged_data = generate_stacked_lagged_batches(data_list, lags_tensor)
    Xg, covmats = gaussian_copula_covmat(
        lagged_data,
        return_xg=True,
        batch_size_D=batch_size_D,
        out_dtype=dtype,
    )
    T_eff = Xg.shape[1]
    N = Xg.shape[2] // lags_tensor.numel()

    return Xg.to(device=device, dtype=dtype), covmats.to(device=device, dtype=dtype), T_eff, N, lags_tensor


def gaussian_mi_bias_correction(
    T: int,
    *,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Bias for Gaussian mutual information between two scalar variables."""
    bias = (
        2 * _gaussian_entropy_bias_correction(1, T)
        - _gaussian_entropy_bias_correction(2, T)
    )
    return bias.to(device=device, dtype=dtype)


def gaussian_ais_bias_correction(
    N: int,
    T: int,
    *,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Bias for Gaussian AIS between an N-variate present and N-variate lagged state."""
    bias = (
        2 * _gaussian_entropy_bias_correction(N, T)
        - _gaussian_entropy_bias_correction(2 * N, T)
    )
    return bias.to(device=device, dtype=dtype)


def compute_ais(
    covmats: torch.Tensor,
    lags: Union[List[int], torch.Tensor, np.ndarray],
    bias: torch.Tensor,
    *,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute Active Information Storage (AIS) from lagged covariance matrices.

    This low-level kernel assumes that ``covmats`` was computed from a temporal
    embedding whose columns are ordered by lag blocks:
    ``[X(t), X(t-lag_1), ..., X(t-lag_k)]``. The zero-lag block is treated as
    the present state and each later block is compared with it independently.

    Parameters
    ----------
    covmats : torch.Tensor
        Lagged covariance matrices with shape ``(D, k*N, k*N)`` or
        ``(k*N, k*N)``. ``D`` is the number of datasets, ``k=len(lags)``, and
        ``N`` is the number of variables in each lag block.
    lags : array-like
        Lag values used to build ``covmats``. The first value must be zero.
    bias : torch.Tensor
        Bias correction values with shape ``(D, k-1)`` or broadcast-compatible
        with that shape. Pass zeros to compute uncorrected AIS.
    device : str or torch.device, default='cpu'
        Device used for computation.
    dtype : torch.dtype, default=torch.float32
        Floating point dtype used for computation.
    eps : float, default=1e-10
        Diagonal regularization added before determinant computations.

    Returns
    -------
    torch.Tensor
        AIS values with shape ``(D, k-1)``, one value for each non-zero lag.
        Values are clamped at zero after bias correction to avoid tiny negative
        numerical artifacts.
    """
    device = torch.device(device)
    lags_tensor = _as_lag_tensor(lags, device=device)
    covmats = _ensure_batched_covmat(covmats, device=device, dtype=dtype)
    B, D, N = _validate_temporal_covmat(covmats, lags_tensor.numel())

    eye = torch.eye(D, device=device, dtype=dtype).unsqueeze(0)
    covmats = covmats + eps * eye

    present_idx = torch.arange(N, device=device)
    sigma_present = covmats[:, present_idx][:, :, present_idx]
    logdet_present = torch.logdet(sigma_present)

    lagged_blocks = torch.stack([
        covmats[:, i * N:(i + 1) * N, i * N:(i + 1) * N]
        for i in range(1, lags_tensor.numel())
    ], dim=1)

    joint_idx = torch.stack([
        torch.cat([present_idx, torch.arange(i * N, (i + 1) * N, device=device)])
        for i in range(1, lags_tensor.numel())
    ])
    batch_idx = torch.arange(B, device=device)[:, None, None, None]
    joint_idx_b = joint_idx.unsqueeze(0).expand(B, -1, -1)
    rows = joint_idx_b.unsqueeze(3).expand(-1, -1, -1, 2 * N)
    cols = joint_idx_b.unsqueeze(2).expand(-1, -1, 2 * N, -1)
    joint_blocks = covmats[batch_idx, rows, cols]

    logdet_lagged = torch.logdet(lagged_blocks)
    logdet_joint = torch.logdet(joint_blocks)
    ais_values = 0.5 * (
        logdet_lagged + logdet_present.unsqueeze(1) - logdet_joint
    ) - bias.to(device=device, dtype=dtype)

    return torch.clamp(ais_values, min=0.0)


@torch.no_grad()
def batch_local_ais(
    Xg: torch.Tensor,
    covmats: torch.Tensor,
    lags: Union[List[int], torch.Tensor, np.ndarray],
    *,
    eps: float = 1e-10,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute local AIS from precomputed lagged Gaussian-copula data.

    This low-level kernel evaluates the pointwise log-ratio between the
    marginal Gaussian density of the present state and the Gaussian conditional
    density of the present state given each lagged state.

    Parameters
    ----------
    Xg : torch.Tensor
        Gaussian-copula normalized lagged data with shape ``(D, T_eff, k*N)``
        or ``(T_eff, k*N)``. The column order must match
        ``[X(t), X(t-lag_1), ..., X(t-lag_k)]``. ``T_eff`` is the effective
        number of samples after temporal embedding: ``T_eff = T - max(lags)``.
    covmats : torch.Tensor
        Covariance matrices matching ``Xg``, with shape ``(D, k*N, k*N)`` or
        ``(k*N, k*N)``.
    lags : array-like
        Lag values used to build ``Xg`` and ``covmats``. The first value must
        be zero.
    eps : float, default=1e-10
        Diagonal regularization for Cholesky decompositions.
    device : str or torch.device, default='cpu'
        Device used for computation.
    dtype : torch.dtype, default=torch.float32
        Floating point dtype used for computation.

    Returns
    -------
    torch.Tensor
        Local AIS values with shape ``(D, T_eff, k-1)``. ``T_eff`` is
        ``T - max(lags)``, and the last dimension excludes the zero lag.
    """
    device = torch.device(device)
    lags_tensor = _as_lag_tensor(lags, device=device)
    Xg = torch.as_tensor(Xg, dtype=dtype, device=device)
    if Xg.ndim == 2:
        Xg = Xg.unsqueeze(0)
    if Xg.ndim != 3:
        raise ValueError('Xg must have shape (D, T_eff, k*N) or (T_eff, k*N)')

    covmats = _ensure_batched_covmat(covmats, device=device, dtype=dtype)
    B, _, N = _validate_temporal_covmat(covmats, lags_tensor.numel())
    if Xg.shape[0] != B or Xg.shape[2] != covmats.shape[1]:
        raise ValueError('Xg and covmats are not compatible')

    _, T_eff, _ = Xg.shape
    k = lags_tensor.numel()
    eye_N = torch.eye(N, device=device, dtype=dtype)

    def block(i: int, j: int) -> torch.Tensor:
        return covmats[:, i * N:(i + 1) * N, j * N:(j + 1) * N]

    sigma_present = block(0, 0) + eps * eye_N
    sigma_lagged = torch.stack([block(i, i) for i in range(1, k)], dim=1) + eps * eye_N
    sigma_cross = torch.stack([block(0, i) for i in range(1, k)], dim=1)
    sigma_cross_t = sigma_cross.transpose(-1, -2)

    chol_present = torch.linalg.cholesky(sigma_present)
    logdet_present = 2.0 * torch.log(
        torch.diagonal(chol_present, dim1=-2, dim2=-1)
    ).sum(-1)

    chol_lagged = torch.linalg.cholesky(sigma_lagged)
    lagged_solve = torch.cholesky_solve(sigma_cross_t, chol_lagged)
    sigma_cond = sigma_present.unsqueeze(1) - torch.matmul(sigma_cross, lagged_solve) + eps * eye_N
    chol_cond = torch.linalg.cholesky(sigma_cond)
    logdet_cond = 2.0 * torch.log(
        torch.diagonal(chol_cond, dim1=-2, dim2=-1)
    ).sum(-1)

    x_present = Xg[:, :, :N]
    x_lagged = torch.stack([Xg[:, :, i * N:(i + 1) * N] for i in range(1, k)], dim=2)

    present_solve = torch.cholesky_solve(x_present.transpose(1, 2), chol_present)
    quad_present = (x_present.transpose(1, 2) * present_solve).sum(1)

    lagged_rhs = x_lagged.permute(0, 2, 3, 1)
    lagged_inv = torch.cholesky_solve(lagged_rhs, chol_lagged)
    mu = torch.matmul(sigma_cross, lagged_inv).permute(0, 3, 1, 2)

    diff = x_present.unsqueeze(2) - mu
    diff_t = diff.permute(0, 2, 3, 1)
    cond_solve = torch.cholesky_solve(diff_t, chol_cond)
    quad_cond = (diff_t * cond_solve).sum(2).permute(0, 2, 1)

    return (
        0.5 * (logdet_present[:, None, None] - logdet_cond[:, None, :])
        + 0.5 * (quad_present[:, :, None] - quad_cond)
    ).reshape(B, T_eff, k - 1)


def extract_time_delayed_subcovmats(
    covmats: torch.Tensor,
    idxs: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """
    Extract time-delayed covariance blocks for variable subsets.

    This helper is intentionally time-delayed-specific. THOI has ordinary
    n-plet covariance extractors, but AIS subsets need to select the same
    original variables from every lag block of a covariance matrix whose
    columns are ordered as ``[X(t), X(t-lag_1), ...]``.

    Parameters
    ----------
    covmats : torch.Tensor
        Full lagged covariance matrix with shape ``(k*N, k*N)`` or a batch with
        shape ``(D, k*N, k*N)``.
    idxs : torch.Tensor
        Variable subsets before lag expansion. Shape can be ``(g,)`` for one
        subset or ``(S, g)`` for multiple subsets.
    N : int
        Number of original variables in each lag block.

    Returns
    -------
    torch.Tensor
        If ``covmats`` is unbatched, returns ``(S, k*g, k*g)``. If batched,
        returns ``(D, S, k*g, k*g)``. Lag block order is preserved.
    """
    idxs = torch.as_tensor(idxs, dtype=torch.long, device=covmats.device)
    if idxs.ndim == 1:
        idxs = idxs.unsqueeze(0)

    S, g = idxs.shape
    batched = covmats.ndim == 3
    D = covmats.shape[-1]
    k = D // N
    if k * N != D:
        raise ValueError('covmats dimensionality is not compatible with N')

    lag_offsets = torch.arange(k, device=covmats.device).view(k, 1, 1) * N
    base = idxs.unsqueeze(0) + lag_offsets
    flat = base.permute(1, 0, 2).reshape(S, k * g)
    rows = flat.unsqueeze(2).expand(-1, -1, k * g)
    cols = flat.unsqueeze(1).expand(-1, k * g, -1)

    if not batched:
        return covmats[rows, cols]

    B = covmats.shape[0]
    rows_b = rows.unsqueeze(0).expand(B, -1, -1, -1)
    cols_b = cols.unsqueeze(0).expand(B, -1, -1, -1)
    batch_idx = torch.arange(B, device=covmats.device).view(B, 1, 1, 1)
    batch_idx = batch_idx.expand(-1, S, k * g, k * g)
    return covmats[batch_idx, rows_b, cols_b]


@torch.no_grad()
def ais(
    X: Optional[TensorLikeArray] = None,
    shifts: Union[List[int], torch.Tensor, np.ndarray] = None,
    *,
    covmats: Optional[torch.Tensor] = None,
    T: Optional[int] = None,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
    bias_correction: bool = True,
    batch_size_D: Optional[int] = None,
    precomputed: bool = False,
) -> torch.Tensor:
    """
    Compute Active Information Storage (AIS) for each lag after zero.

    AIS is computed after Gaussian-copula normalization of a temporal embedding
    ``[X(t), X(t-shift_1), ..., X(t-shift_k)]``. For repeated analyses on the
    same data, call :func:`precompute_temporal_embedding` once and then call
    this function with ``precomputed=True`` and the returned ``covmats``.

    Parameters
    ----------
    X : array-like, optional
        Raw timeseries with shape ``(T, N)``, batched timeseries with shape
        ``(D, T, N)``, or a list of ``(T, N)`` arrays. When
        ``precomputed=True``, this can be the precomputed normalized lagged data
        and is only used to infer ``T`` if ``T`` is not provided.
    shifts : array-like
        Lag values including zero as the first element.
    covmats : torch.Tensor, optional
        Precomputed lagged covariance matrices with shape ``(D, k*N, k*N)`` or
        ``(k*N, k*N)``. Required when ``precomputed=True``.
    T : int, optional
        Effective sample count used for bias correction when using precomputed
        covariances. This is ``T_eff``, defined as ``T - max(shifts)``.
    device : str or torch.device, default='cpu'
        Device used for preprocessing and computation.
    dtype : torch.dtype, default=torch.float32
        Floating point dtype used for computation.
    bias_correction : bool, default=True
        Whether to subtract the Gaussian AIS finite-sample bias.
    batch_size_D : int or None, default=None
        Dataset batching passed to ``gaussian_copula_covmat`` when
        ``precomputed=False``.
    precomputed : bool, default=False
        If True, skip temporal embedding/Gaussian-copula preprocessing and use
        ``covmats`` directly.

    Returns
    -------
    torch.Tensor
        AIS values with shape ``(D, k-1)``, one value per non-zero shift.
    """
    device = torch.device(device)
    lags_tensor = _as_lag_tensor(shifts, device=device)
    k = lags_tensor.numel()

    if precomputed:
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        covmats_t = _ensure_batched_covmat(covmats, device=device, dtype=dtype)
        B, _, N = _validate_temporal_covmat(covmats_t, k)
        T_eff = T
        if T_eff is None and X is not None:
            X_tensor = torch.as_tensor(X, dtype=dtype, device=device)
            T_eff = X_tensor.shape[0] if X_tensor.ndim == 2 else X_tensor.shape[1]
    else:
        if X is None:
            raise ValueError('X must be provided when precomputed=False')
        _, covmats_t, T_eff, N, lags_tensor = precompute_temporal_embedding(
            X,
            lags_tensor,
            device=device,
            dtype=dtype,
            batch_size_D=batch_size_D,
        )
        B = covmats_t.shape[0]

    if bias_correction and T_eff is not None:
        bias = gaussian_ais_bias_correction(N, T_eff, device=device, dtype=dtype)
        bias = bias.unsqueeze(0).expand(B, k - 1)
    else:
        bias = torch.zeros(B, k - 1, device=device, dtype=dtype)

    return compute_ais(
        covmats_t,
        lags_tensor,
        bias,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def local_ais(
    X: TensorLikeArray,
    shifts: Union[List[int], torch.Tensor, np.ndarray],
    *,
    covmats: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-10,
    batch_size_D: Optional[int] = None,
    precomputed: bool = False,
) -> torch.Tensor:
    """
    Compute local, time-resolved Active Information Storage.

    Local AIS keeps the temporal dimension instead of averaging over samples.
    It uses the same temporal embedding and Gaussian-copula normalization as
    :func:`ais`; the average over time matches uncorrected AIS up to numerical
    precision.

    Parameters
    ----------
    X : array-like
        Raw timeseries with shape ``(T, N)``, batched timeseries with shape
        ``(D, T, N)``, or a list of ``(T, N)`` arrays. When
        ``precomputed=True``, this must be the normalized lagged data returned
        by :func:`precompute_temporal_embedding`.
    shifts : array-like
        Lag values including zero as the first element.
    covmats : torch.Tensor, optional
        Precomputed lagged covariance matrices matching ``X``. Required when
        ``precomputed=True``.
    device : str or torch.device, default='cpu'
        Device used for preprocessing and computation.
    dtype : torch.dtype, default=torch.float32
        Floating point dtype used for computation.
    eps : float, default=1e-10
        Diagonal regularization for Cholesky decompositions.
    batch_size_D : int or None, default=None
        Dataset batching passed to ``gaussian_copula_covmat`` when
        ``precomputed=False``.
    precomputed : bool, default=False
        If True, skip temporal embedding/Gaussian-copula preprocessing.

    Returns
    -------
    torch.Tensor
        Local AIS values with shape ``(D, T_eff, k-1)``, where
        ``T_eff = T - max(shifts)``.
    """
    device = torch.device(device)
    lags_tensor = _as_lag_tensor(shifts, device=device)

    if precomputed:
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        Xg = torch.as_tensor(X, dtype=dtype, device=device)
        covmats_t = _ensure_batched_covmat(covmats, device=device, dtype=dtype)
    else:
        Xg, covmats_t, _, _, lags_tensor = precompute_temporal_embedding(
            X,
            lags_tensor,
            device=device,
            dtype=dtype,
            batch_size_D=batch_size_D,
        )

    return batch_local_ais(
        Xg,
        covmats_t,
        lags_tensor,
        eps=eps,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def ais_subset(
    X: Optional[TensorLikeArray] = None,
    shifts: Union[List[int], torch.Tensor, np.ndarray] = None,
    idxs: Union[List[int], torch.Tensor, np.ndarray] = None,
    *,
    covmats: Optional[torch.Tensor] = None,
    T: Optional[int] = None,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
    bias_correction: bool = True,
    batch_size_D: Optional[int] = None,
    precomputed: bool = False,
) -> torch.Tensor:
    """
    Compute AIS for one or more subsets of variables.

    This wrapper mirrors :func:`ais`, but restricts the present and lagged
    states to selected variables. It is useful when users want group-wise AIS
    without manually rebuilding reduced temporal embeddings. When
    ``precomputed=True``, a full-system temporal covariance can be reused and
    the matching subset blocks are extracted internally.

    Parameters
    ----------
    X : array-like, optional
        Raw timeseries with shape ``(T, N)``, batched timeseries with shape
        ``(D, T, N)``, or a list of ``(T, N)`` arrays. When
        ``precomputed=True``, this can be the normalized lagged data returned
        by :func:`precompute_temporal_embedding` and is only used to infer
        ``T`` if ``T`` is not provided.
    shifts : array-like
        Lag values including zero as the first element.
    idxs : array-like
        Original variable indices to include. Shape can be ``(g,)`` for one
        subset or ``(S, g)`` for multiple subsets of equal size.
    covmats : torch.Tensor, optional
        Full lagged covariance matrices with shape ``(D, k*N, k*N)`` or
        ``(k*N, k*N)``. Required when ``precomputed=True``.
    T : int, optional
        Effective sample count used for bias correction when using precomputed
        covariances. This is ``T_eff``, defined as ``T - max(shifts)``.
    device : str or torch.device, default='cpu'
        Device used for preprocessing and computation.
    dtype : torch.dtype, default=torch.float32
        Floating point dtype used for computation.
    bias_correction : bool, default=True
        Whether to subtract the Gaussian AIS finite-sample bias.
    batch_size_D : int or None, default=None
        Dataset batching passed to ``gaussian_copula_covmat`` when
        ``precomputed=False``.
    precomputed : bool, default=False
        If True, use ``covmats`` directly and skip temporal
        embedding/Gaussian-copula preprocessing.

    Returns
    -------
    torch.Tensor
        If one subset is provided, returns AIS with shape ``(D, k-1)``. If
        multiple subsets are provided, returns ``(D, S, k-1)``.
    """
    device = torch.device(device)
    lags_tensor = _as_lag_tensor(shifts, device=device)
    k = lags_tensor.numel()

    if idxs is None:
        raise ValueError('idxs must be provided')
    idxs_tensor = torch.as_tensor(idxs, dtype=torch.long, device=device)
    if idxs_tensor.ndim == 1:
        idxs_tensor = idxs_tensor.unsqueeze(0)
    S, g = idxs_tensor.shape

    if precomputed:
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        covmats_full = _ensure_batched_covmat(covmats, device=device, dtype=dtype)
        B, _, N = _validate_temporal_covmat(covmats_full, k)
        T_eff = T
        if T_eff is None and X is not None:
            X_tensor = torch.as_tensor(X, dtype=dtype, device=device)
            T_eff = X_tensor.shape[0] if X_tensor.ndim == 2 else X_tensor.shape[1]
    else:
        if X is None:
            raise ValueError('X must be provided when precomputed=False')
        _, covmats_full, T_eff, N, lags_tensor = precompute_temporal_embedding(
            X,
            lags_tensor,
            device=device,
            dtype=dtype,
            batch_size_D=batch_size_D,
        )
        B = covmats_full.shape[0]

    subcovs = extract_time_delayed_subcovmats(covmats_full, idxs_tensor, N)
    cov_sub_batch = subcovs.reshape(B * S, g * k, g * k)

    if bias_correction and T_eff is not None:
        bias = gaussian_ais_bias_correction(g, T_eff, device=device, dtype=dtype)
        bias = bias.unsqueeze(0).expand(B * S, k - 1)
    else:
        bias = torch.zeros(B * S, k - 1, device=device, dtype=dtype)

    values = compute_ais(
        cov_sub_batch,
        lags_tensor,
        bias,
        device=device,
        dtype=dtype,
    ).view(B, S, k - 1)

    return values.squeeze(1) if S == 1 else values


@torch.no_grad()
def local_ais_subset(
    X: TensorLikeArray,
    shifts: Union[List[int], torch.Tensor, np.ndarray],
    idxs: Union[List[int], torch.Tensor, np.ndarray],
    *,
    covmats: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-10,
    batch_size_D: Optional[int] = None,
    precomputed: bool = False,
) -> torch.Tensor:
    """
    Compute local AIS for one or more subsets of variables.

    This convenience wrapper is useful when a full-system temporal embedding
    has already been precomputed. It extracts the matching lagged columns and
    covariance blocks for each requested subset, avoiding repeated
    Gaussian-copula preprocessing and reducing the chance of user-side indexing
    mistakes.

    Parameters
    ----------
    X : array-like
        Raw data when ``precomputed=False``. When ``precomputed=True``, this
        must be the normalized lagged data returned by
        :func:`precompute_temporal_embedding`, with temporal length ``T_eff``.
        ``T_eff`` is defined as ``T - max(shifts)``.
    shifts : array-like
        Lag values including zero as the first element.
    idxs : array-like
        Original variable indices to include. Shape can be ``(g,)`` for one
        subset or ``(S, g)`` for multiple subsets of equal size.
    covmats : torch.Tensor, optional
        Full lagged covariance matrices. Required when ``precomputed=True``.
    device : str or torch.device, default='cpu'
        Device used for preprocessing and computation.
    dtype : torch.dtype, default=torch.float32
        Floating point dtype used for computation.
    eps : float, default=1e-10
        Diagonal regularization for Cholesky decompositions.
    batch_size_D : int or None, default=None
        Dataset batching passed to ``gaussian_copula_covmat`` when
        ``precomputed=False``.
    precomputed : bool, default=False
        If True, use the provided full-system temporal embedding and covariance
        matrices instead of recomputing them.

    Returns
    -------
    torch.Tensor
        If one subset is provided, returns ``(D, T_eff, k-1)``. If multiple
        subsets are provided, returns ``(D, S, T_eff, k-1)``. ``T_eff`` is
        ``T - max(shifts)``.
    """
    device = torch.device(device)
    lags_tensor = _as_lag_tensor(shifts, device=device)
    k = lags_tensor.numel()

    idxs_tensor = torch.as_tensor(idxs, dtype=torch.long, device=device)
    if idxs_tensor.ndim == 1:
        idxs_tensor = idxs_tensor.unsqueeze(0)
    S, g = idxs_tensor.shape

    if precomputed:
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        Xg = torch.as_tensor(X, dtype=dtype, device=device)
        if Xg.ndim == 2:
            Xg = Xg.unsqueeze(0)
        covmats_full = _ensure_batched_covmat(covmats, device=device, dtype=dtype)
        B, _, N = _validate_temporal_covmat(covmats_full, k)
    else:
        Xg, covmats_full, _, N, lags_tensor = precompute_temporal_embedding(
            X,
            lags_tensor,
            device=device,
            dtype=dtype,
            batch_size_D=batch_size_D,
        )
        B = Xg.shape[0]

    if Xg.shape[0] != B:
        raise ValueError('X and covmats batch dimensions are not compatible')

    T_eff = Xg.shape[1]
    lag_offsets = torch.arange(k, device=device, dtype=torch.long).view(k, 1) * N
    positions = (lag_offsets.unsqueeze(1) + idxs_tensor.unsqueeze(0))
    positions = positions.permute(1, 0, 2).reshape(S, k * g)

    X_expanded = Xg.unsqueeze(1).expand(B, S, T_eff, Xg.shape[2])
    gather_idx = positions.unsqueeze(0).unsqueeze(2).expand(B, S, T_eff, k * g)
    selected_data = torch.gather(X_expanded, dim=3, index=gather_idx)
    data_batch = selected_data.reshape(B * S, T_eff, k * g)

    subcovs = extract_time_delayed_subcovmats(covmats_full, idxs_tensor, N)
    cov_sub_batch = subcovs.reshape(B * S, g * k, g * k)
    values = batch_local_ais(
        data_batch,
        cov_sub_batch,
        lags_tensor,
        eps=eps,
        device=device,
        dtype=dtype,
    ).view(B, S, T_eff, k - 1)

    return values.squeeze(1) if S == 1 else values


@torch.no_grad()
def batch_compute_tdmi_xcorr(
    covmats: torch.Tensor,
    lags: Union[List[int], torch.Tensor, np.ndarray],
    *,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute TDMI and cross-correlation from lagged covariance matrices.

    Parameters
    ----------
    covmats : torch.Tensor
        Lagged covariance matrices with shape ``(D, k*N, k*N)`` or
        ``(k*N, k*N)``. Columns must be ordered by lag blocks:
        ``[X(t), X(t-lag_1), ..., X(t-lag_k)]``.
    lags : array-like
        Lag values used to build ``covmats``. The first value must be zero.
    device : str or torch.device, default='cpu'
        Device used for computation.
    dtype : torch.dtype, default=torch.float32
        Floating point dtype used for computation.
    eps : float, default=1e-10
        Diagonal/correlation denominator regularization.

    Returns
    -------
    tdmi : torch.Tensor
        Time-delayed mutual information with shape ``(D, k, N, N)`` where
        ``tdmi[d, tau, i, j] = I(X_i(t-lag_tau); X_j(t))``.
    xcorr : torch.Tensor
        Matching lagged Pearson correlations with shape ``(D, k, N, N)``.
    """
    device = torch.device(device)
    lags_tensor = _as_lag_tensor(lags, device=device)
    covmats = _ensure_batched_covmat(covmats, device=device, dtype=dtype)
    _, D, N = _validate_temporal_covmat(covmats, lags_tensor.numel())
    k = lags_tensor.numel()

    covmats = covmats + eps * torch.eye(D, device=device, dtype=dtype).unsqueeze(0)
    sigma_lagged_present = torch.stack([
        covmats[:, i * N:(i + 1) * N, :N]
        for i in range(k)
    ], dim=1)

    std_present = torch.sqrt(torch.diagonal(covmats[:, :N, :N], dim1=1, dim2=2))
    std_lagged = torch.stack([
        torch.sqrt(torch.diagonal(covmats[:, i * N:(i + 1) * N, i * N:(i + 1) * N], dim1=1, dim2=2))
        for i in range(k)
    ], dim=1)
    denom = std_lagged.unsqueeze(-1) * std_present[:, None, None, :]

    xcorr = sigma_lagged_present / (denom + eps)
    xcorr = torch.clamp(xcorr, min=-0.999999, max=0.999999)
    tdmi_values = -0.5 * torch.log1p(-xcorr.pow(2))

    return tdmi_values, xcorr


def build_full_tdmi(
    tdmi_values: torch.Tensor,
    *,
    mask_diagonal: bool = True,
) -> torch.Tensor:
    """
    Build a bidirectional TDMI tensor from non-negative-lag TDMI values.

    Parameters
    ----------
    tdmi_values : torch.Tensor
        TDMI values for non-negative lags with shape ``(D, k, N, N)``. The
        convention must match :func:`batch_compute_tdmi_xcorr`:
        ``tdmi_values[d, tau, i, j] = I(X_i(t-lag_tau); X_j(t))``.
    mask_diagonal : bool, default=True
        If True, set same-variable entries to NaN in the output because the
        bidirectional source/target interpretation is intended for
        cross-variable pairs.

    Returns
    -------
    torch.Tensor
        Full bidirectional TDMI with shape ``(D, N, N, 2*k - 1)``. The last
        axis is ordered from negative to positive lag, with the zero-lag slice
        in the center.
    """
    if tdmi_values.ndim != 4 or tdmi_values.shape[-1] != tdmi_values.shape[-2]:
        raise ValueError('tdmi_values must have shape (D, k, N, N)')

    _, k, N, _ = tdmi_values.shape
    backward = torch.flip(tdmi_values.transpose(2, 3), dims=[1])[:, :-1, :, :]
    full = torch.cat([backward, tdmi_values], dim=1).permute(0, 2, 3, 1)

    if not mask_diagonal:
        return full

    diagonal = torch.eye(N, device=tdmi_values.device, dtype=torch.bool)
    return full.masked_fill(diagonal.unsqueeze(0).unsqueeze(-1), float('nan'))


@torch.no_grad()
def tdmi(
    X: Optional[TensorLikeArray] = None,
    lags: Union[List[int], torch.Tensor, np.ndarray] = None,
    *,
    covmats: Optional[torch.Tensor] = None,
    T: Optional[int] = None,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
    bias_correction: bool = True,
    return_full: bool = True,
    batch_size_D: Optional[int] = None,
    precomputed: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Compute Time-Delayed Mutual Information (TDMI) and cross-correlation.

    Parameters
    ----------
    X : array-like, optional
        Raw timeseries with shape ``(T, N)``, batched timeseries with shape
        ``(D, T, N)``, or a list of ``(T, N)`` arrays. Ignored when
        ``precomputed=True`` except for optional ``T`` inference.
    lags : array-like
        Lag values including zero as the first element.
    covmats : torch.Tensor, optional
        Precomputed lagged covariance matrices with shape ``(D, k*N, k*N)`` or
        ``(k*N, k*N)``. Required when ``precomputed=True``.
    T : int, optional
        Effective sample count used for TDMI bias correction when using
        precomputed covariances. This is ``T_eff``, defined as
        ``T - max(lags)``.
    device : str or torch.device, default='cpu'
        Device used for preprocessing and computation.
    dtype : torch.dtype, default=torch.float32
        Floating point dtype used for computation.
    bias_correction : bool, default=True
        Whether to subtract Gaussian scalar MI finite-sample bias.
    return_full : bool, default=True
        Whether to also return a bidirectional ``(D, N, N, 2*k-1)`` TDMI tensor.
    batch_size_D : int or None, default=None
        Dataset batching passed to ``gaussian_copula_covmat`` when
        ``precomputed=False``.
    precomputed : bool, default=False
        If True, skip temporal embedding/Gaussian-copula preprocessing.

    Returns
    -------
    tuple
        If ``return_full=False``, returns ``(tdmi, xcorr)``. If
        ``return_full=True``, returns ``(tdmi, xcorr, full_tdmi)``.
        ``tdmi`` and ``xcorr`` have shape ``(D, k, N, N)``; ``full_tdmi`` has
        shape ``(D, N, N, 2*k-1)``.
    """
    device = torch.device(device)
    lags_tensor = _as_lag_tensor(lags, device=device)

    if precomputed:
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        covmats_t = _ensure_batched_covmat(covmats, device=device, dtype=dtype)
        T_eff = T
        if T_eff is None and X is not None:
            X_tensor = torch.as_tensor(X, dtype=dtype, device=device)
            T_eff = X_tensor.shape[0] if X_tensor.ndim == 2 else X_tensor.shape[1]
    else:
        if X is None:
            raise ValueError('X must be provided when precomputed=False')
        _, covmats_t, T_eff, _, lags_tensor = precompute_temporal_embedding(
            X,
            lags_tensor,
            device=device,
            dtype=dtype,
            batch_size_D=batch_size_D,
        )

    tdmi_values, xcorr_values = batch_compute_tdmi_xcorr(
        covmats_t,
        lags_tensor,
        device=device,
        dtype=dtype,
    )

    if bias_correction and T_eff is not None:
        tdmi_values = tdmi_values - gaussian_mi_bias_correction(T_eff, device=device, dtype=dtype)

    if return_full:
        return tdmi_values, xcorr_values, build_full_tdmi(tdmi_values)
    return tdmi_values, xcorr_values
