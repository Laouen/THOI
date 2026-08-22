from typing import List, Optional, Union

import torch

from thoi.commons import _normalize_input_data
from thoi.heuristics.scoring import _evaluate_nplets
from thoi.typing import TensorLikeArray


_METRICS = ('tc', 'dtc', 'o', 's')


def _to_correlation(covmats: torch.Tensor) -> torch.Tensor:
    """Standardize covariance matrices while preserving Gaussian HOI values."""
    diagonal = torch.diagonal(covmats, dim1=-2, dim2=-1)
    scale = diagonal.clamp_min(torch.finfo(covmats.dtype).tiny).sqrt()
    corrmats = covmats / (scale.unsqueeze(-1) * scale.unsqueeze(-2))
    corrmats = corrmats.clone()
    torch.diagonal(corrmats, dim1=-2, dim2=-1).fill_(1.0)
    return corrmats


def _project_hypersimplex(
    values: torch.Tensor,
    orders: torch.Tensor,
    iterations: int = 25,
) -> torch.Tensor:
    """Project rows onto {z in [0,1]^N : sum(z)=order}."""
    _, N = values.shape
    orders = orders.to(dtype=values.dtype)
    projected = torch.empty_like(values)

    full = orders >= N
    projected[full] = 1.0

    active = ~full
    if torch.any(active):
        x = values[active]
        k = orders[active]
        lower = (x - 1.0).amin(dim=1)
        upper = x.amax(dim=1)

        for _ in range(iterations):
            shift = (lower + upper) / 2.0
            mass = (x - shift.unsqueeze(1)).clamp(0.0, 1.0).sum(dim=1)
            too_large = mass > k
            lower = torch.where(too_large, shift, lower)
            upper = torch.where(too_large, upper, shift)

        shift = (lower + upper) / 2.0
        projected[active] = (x - shift.unsqueeze(1)).clamp(0.0, 1.0)

    return projected


def _mean_metric_gradient(
    corrmats: torch.Tensor,
    z: torch.Tensor,
    metric: str,
    batch_size_D: Optional[int],
) -> torch.Tensor:
    """Analytic mean Gaussian-HOI gradient across datasets."""
    D, N = corrmats.shape[:2]
    batch_size_D = D if batch_size_D is None else batch_size_D

    # The analytic formulas have finite boundary limits but appear as 0/0 at
    # exact zeros. A tiny interior evaluation recovers those limits stably.
    boundary_eps = torch.finfo(z.dtype).eps ** 0.5
    z_eval = z.clamp_min(boundary_eps)
    q = z_eval.sqrt()

    eye = torch.eye(N, dtype=z.dtype, device=z.device)
    gradient = torch.zeros_like(z)

    for start in range(0, D, batch_size_D):
        corr_batch = corrmats[start:start + batch_size_D]
        centered = corr_batch - eye
        relaxed = (
            eye
            + q[:, None, :, None]
            * centered[None, :, :, :]
            * q[:, None, None, :]
        )

        Db = relaxed.shape[1]
        relaxed = relaxed.reshape(-1, N, N)
        chol = torch.linalg.cholesky(relaxed)
        precision = torch.cholesky_solve(
            eye.expand(relaxed.shape[0], -1, -1), chol
        ).reshape(z.shape[0], Db, N, N)

        diag_precision = torch.diagonal(precision, dim1=-2, dim2=-1)
        weighted_row_norm = (
            precision.square() / diag_precision.unsqueeze(-2)
        ).sum(dim=-1)
        denominator = 2.0 * z_eval[:, None, :]

        if metric == 'tc':
            current = (diag_precision - 1.0) / denominator
        elif metric == 'dtc':
            current = (weighted_row_norm - diag_precision) / denominator
        elif metric == 'o':
            current = (
                2.0 * diag_precision - 1.0 - weighted_row_norm
            ) / denominator
        else:  # metric == 's'
            current = (weighted_row_norm - 1.0) / denominator

        gradient += current.sum(dim=1)

    return gradient / D


def _optimize_mixed_orders(
    corrmats: torch.Tensor,
    orders: torch.Tensor,
    metric: str,
    largest: bool,
    repeat_batch_size: int,
    batch_size_D: Optional[int],
    max_iterations: int,
    learning_rate: float,
) -> torch.Tensor:
    """Optimize independent cardinalities together in mixed-order batches."""
    trajectories, N = orders.shape[0], corrmats.shape[-1]
    z = _project_hypersimplex(
        torch.rand(trajectories, N, dtype=corrmats.dtype, device=corrmats.device),
        orders,
    )
    direction = 1.0 if largest else -1.0

    for _ in range(max_iterations):
        for start in range(0, trajectories, repeat_batch_size):
            stop = min(start + repeat_batch_size, trajectories)
            current = z[start:stop]
            gradient = _mean_metric_gradient(
                corrmats,
                current,
                metric,
                batch_size_D,
            )
            scale = gradient.abs().amax(dim=1, keepdim=True).clamp_min(
                torch.finfo(gradient.dtype).eps
            )
            step = current + direction * learning_rate * gradient / scale
            z[start:stop] = _project_hypersimplex(step, orders[start:stop])

    return z


@torch.no_grad()
def differentiable_hoi(
    X: TensorLikeArray,
    initial_order: int = 3,
    order: Optional[int] = None,
    *,
    covmat_precomputed: bool = False,
    T: Optional[Union[int, List[int]]] = None,
    repeat: int = 4,
    batch_size: int = 1000000,
    repeat_batch_size: int = 64,
    batch_size_D: Optional[int] = None,
    device: torch.device = torch.device('cpu'),
    metric: str = 'o',
    largest: bool = False,
    max_iterations: int = 20,
    learning_rate: float = 0.5,
):
    """Optimize Gaussian TC, DTC, O-information, or S-information n-plets.

    The method optimizes a continuous fixed-cardinality relaxation directly on
    the hypersimplex. All requested cardinalities and random restarts are
    independent but processed in mixed-order batches. The relaxed gradient is
    normalized per trajectory, followed by an Euclidean projection back onto
    the hypersimplex. Final n-plets are obtained with top-k rounding and scored
    with THOI's exact discrete estimator.

    For multiple datasets, the optimized objective is the mean selected metric
    across datasets, matching THOI's standard string-metric semantics.

    Parameters
    ----------
    X : TensorLikeArray
        Raw data or precomputed covariance matrix/matrices.
    initial_order : int, default=3
        Smallest interaction order to optimize.
    order : int or None, default=None
        Largest interaction order. If None, optimize through N.
    covmat_precomputed : bool, default=False
        Whether X contains covariance matrices.
    T : int or list of int, optional
        Sample counts used by THOI's exact finite-sample bias correction.
    repeat : int, default=4
        Number of independent trajectories per interaction order.
    batch_size : int, default=1000000
        Batch size used for final exact discrete scoring.
    repeat_batch_size : int, default=64
        Number of relaxed trajectories evaluated together.
    batch_size_D : int or None, default=None
        Number of datasets processed together in the relaxed gradient.
    device : torch.device, default=cpu
        Device used for optimization and scoring.
    metric : {'tc', 'dtc', 'o', 's'}, default='o'
        Gaussian higher-order interaction metric to optimize.
    largest : bool, default=False
        Maximize the metric when True; minimize it when False.
    max_iterations : int, default=20
        Number of projected-gradient iterations.
    learning_rate : float, default=0.5
        Step size after per-trajectory max-norm gradient normalization.

    Returns
    -------
    best_nplets : list[torch.Tensor]
        One tensor per order from initial_order through order. Each tensor has
        shape (repeat, current_order), sorted from best to worst exact score.
    best_scores : torch.Tensor
        Exact discrete scores with shape (repeat, n_orders).
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
    if repeat < 1:
        raise ValueError('repeat must be at least 1')
    if repeat_batch_size < 1:
        raise ValueError('repeat_batch_size must be at least 1')
    if max_iterations < 1:
        raise ValueError('max_iterations must be at least 1')
    if learning_rate <= 0:
        raise ValueError('learning_rate must be positive')

    covmats, _, N, T = _normalize_input_data(
        X,
        covmat_precomputed,
        T,
        device,
        batch_size_D=batch_size_D,
    )
    if not covmats.dtype.is_floating_point:
        covmats = covmats.to(torch.get_default_dtype())

    order = N if order is None else order
    if not 3 <= initial_order <= order <= N:
        raise ValueError('orders must satisfy 3 <= initial_order <= order <= N')

    corrmats = _to_correlation(covmats)
    optimized_orders = torch.arange(
        initial_order, order + 1, device=device, dtype=torch.long
    )
    trajectory_orders = optimized_orders.repeat_interleave(repeat)

    z = _optimize_mixed_orders(
        corrmats,
        trajectory_orders,
        metric,
        largest,
        repeat_batch_size,
        batch_size_D,
        max_iterations,
        learning_rate,
    )

    best_nplets = []
    best_scores = []
    for order_idx, current_order in enumerate(optimized_orders.tolist()):
        start = order_idx * repeat
        stop = start + repeat
        nplets = z[start:stop].topk(current_order, dim=1).indices
        scores = _evaluate_nplets(
            covmats,
            T,
            nplets,
            metric,
            batch_size=batch_size,
            device=device,
        )
        ranking = torch.argsort(scores, descending=largest)
        best_nplets.append(nplets[ranking])
        best_scores.append(scores[ranking])

    return best_nplets, torch.stack(best_scores, dim=1)
