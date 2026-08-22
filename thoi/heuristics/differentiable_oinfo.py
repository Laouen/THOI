from typing import List, Optional, Union

import torch

from thoi.commons import _normalize_input_data
from thoi.heuristics.scoring import _evaluate_nplets
from thoi.typing import TensorLikeArray


@torch.no_grad()
def _fixed_cardinality_sigmoid(theta: torch.Tensor,
                               order: torch.Tensor,
                               temperature: float,
                               iterations: int = 7) -> torch.Tensor:
    """Map logits to the interior of the fixed-cardinality hypersimplex."""
    batch_size, N = theta.shape
    order = torch.as_tensor(order, dtype=theta.dtype, device=theta.device).reshape(batch_size)

    fraction = (order / N).clamp(1e-6, 1 - 1e-6)
    shift = temperature * torch.log(fraction / (1 - fraction)) - theta.mean(dim=1)

    for _ in range(iterations):
        z = torch.sigmoid((theta + shift[:, None]) / temperature)
        error = z.sum(dim=1) - order
        derivative = (z * (1 - z)).sum(dim=1) / temperature
        shift -= (error / derivative.clamp_min(1e-8)).clamp(-5 * temperature, 5 * temperature)

    return torch.sigmoid((theta + shift[:, None]) / temperature)


@torch.no_grad()
def _mean_oinfo_gradient(corrmats: torch.Tensor,
                         z: torch.Tensor,
                         batch_size_D: Optional[int] = None) -> torch.Tensor:
    """Analytic gradient of mean relaxed Gaussian O-information across datasets."""
    D, N, _ = corrmats.shape
    batch_size = z.shape[0]
    batch_size_D = D if batch_size_D is None else min(batch_size_D, D)

    eye = torch.eye(N, dtype=z.dtype, device=z.device)
    z = z.clamp_min(1e-7)
    q = torch.sqrt(z)
    gradient = torch.zeros_like(z)

    for start in range(0, D, batch_size_D):
        corr = corrmats[start:start + batch_size_D]
        current_D = corr.shape[0]
        centered = corr - eye

        A = eye[None, None] + centered[:, None] * (
            q[None, :, :, None] * q[None, :, None, :]
        )
        A = A.reshape(current_D * batch_size, N, N)

        chol = torch.linalg.cholesky(A)
        precision = torch.cholesky_solve(
            eye.expand(current_D * batch_size, -1, -1), chol
        )
        diag_precision = torch.diagonal(precision, dim1=-2, dim2=-1)
        weighted_row_norm = (
            precision.square() / diag_precision[:, None, :]
        ).sum(dim=-1)

        current_gradient = (
            diag_precision - 0.5 - 0.5 * weighted_row_norm
        ) / z.repeat(current_D, 1)

        gradient += current_gradient.view(current_D, batch_size, N).sum(dim=0)

    return gradient / D


def _optimize_logits(corrmats: torch.Tensor,
                     theta: torch.Tensor,
                     order: torch.Tensor,
                     *,
                     largest: bool,
                     iterations: int,
                     learning_rate: float,
                     temperature: float,
                     batch_size_D: Optional[int]) -> torch.Tensor:
    """Optimize a batch of logits with an analytic gradient and Adam."""
    theta = theta.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=learning_rate)

    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            z = _fixed_cardinality_sigmoid(theta, order, temperature)
            gradient_z = _mean_oinfo_gradient(corrmats, z, batch_size_D)
            if not largest:
                gradient_z = -gradient_z

            slope = z * (1 - z) / temperature
            correction = (
                (slope * gradient_z).sum(dim=1, keepdim=True)
                / slope.sum(dim=1, keepdim=True).clamp_min(1e-8)
            )
            gradient_theta = slope * (gradient_z - correction)
            theta.grad = -gradient_theta

        optimizer.step()

    return theta.detach()


def _optimize_logits_batched(corrmats: torch.Tensor,
                             theta: torch.Tensor,
                             order: torch.Tensor,
                             *,
                             largest: bool,
                             iterations: int,
                             learning_rate: float,
                             temperature: float,
                             batch_size_D: Optional[int],
                             repeat_batch_size: int) -> torch.Tensor:
    """Chunk independent optimization trajectories to control peak memory."""
    optimized = []
    for start in range(0, theta.shape[0], repeat_batch_size):
        end = min(start + repeat_batch_size, theta.shape[0])
        optimized.append(
            _optimize_logits(
                corrmats,
                theta[start:end],
                order[start:end],
                largest=largest,
                iterations=iterations,
                learning_rate=learning_rate,
                temperature=temperature,
                batch_size_D=batch_size_D,
            )
        )
    return torch.cat(optimized, dim=0)


@torch.no_grad()
def _score_logits(covmats: torch.Tensor,
                  T: Optional[List[int]],
                  theta: torch.Tensor,
                  order: int,
                  *,
                  temperature: float,
                  batch_size: int,
                  device: torch.device):
    """Round relaxed solutions and score them with THOI's standard O-information path."""
    order_tensor = torch.full(
        (theta.shape[0],), order, dtype=theta.dtype, device=device
    )
    z = _fixed_cardinality_sigmoid(theta, order_tensor, temperature)
    nplets = torch.topk(z, order, dim=1).indices
    scores = _evaluate_nplets(
        covmats,
        T,
        nplets,
        metric='o',
        batch_size=batch_size,
        device=device,
    )
    return nplets, scores


def _continuation_paths(initial_order: int, order: int):
    """Return the non-redundant warm-start paths used by the optimizer."""
    middle = min(order, max(initial_order, order // 2))
    high = max(initial_order, order - 3)
    anchors = sorted(set((initial_order, middle, high)))

    paths = set()
    if initial_order < middle:
        paths.add((initial_order, middle))
    if middle > initial_order:
        paths.add((middle, initial_order))
    if middle < order:
        paths.add((middle, order))
    if high > middle:
        paths.add((high, middle))
    if high < order:
        paths.add((high, order))

    return anchors, sorted(paths)


@torch.no_grad()
def differentiable_oinfo(X: TensorLikeArray,
                         initial_order: int = 3,
                         order: Optional[int] = None,
                         *,
                         covmat_precomputed: bool = False,
                         T: Optional[Union[int, List[int]]] = None,
                         repeat: int = 4,
                         batch_size: int = 1000000,
                         repeat_batch_size: int = 1000000,
                         batch_size_D: Optional[int] = None,
                         device: torch.device = torch.device('cpu'),
                         largest: bool = False,
                         max_iterations: int = 40,
                         continuation_iterations: int = 4,
                         learning_rate: float = 0.08,
                         temperature: float = 0.25):
    """Find high- or low-O-information n-plets using differentiable optimization.

    The search uses a continuous fixed-cardinality relaxation of Gaussian
    O-information, an analytic gradient, batched Adam optimization, and warm
    starts between neighboring interaction orders. For multiple datasets, the
    optimized objective is the mean O-information across datasets, matching
    THOI's standard string-metric behavior.

    Parameters
    ----------
    X : TensorLikeArray
        Multivariate data or precomputed covariance matrix/matrices, following
        the same input conventions as :func:`thoi.heuristics.greedy.greedy`.
    initial_order : int, optional
        Lowest interaction order to return. Default is 3.
    order : int, optional
        Highest interaction order to return. If None, uses all variables.
    covmat_precomputed : bool, optional
        Treat ``X`` as covariance matrices. Default is False.
    T : int or list of int, optional
        Sample sizes for precomputed covariance matrices. Used by THOI when
        scoring the final discrete n-plets.
    repeat : int, optional
        Number of parallel optimization trajectories. Default is 4.
    batch_size : int, optional
        Batch size used by THOI to score final discrete n-plets.
    repeat_batch_size : int, optional
        Maximum number of optimization trajectories processed together.
    batch_size_D : int, optional
        Number of datasets processed together. Also controls Gaussian copula
        covariance batching when raw data are provided.
    device : torch.device, optional
        Device used for computation. Default is CPU.
    largest : bool, optional
        Maximize O-information if True, minimize it if False. Default is False.
    max_iterations : int, optional
        Adam iterations at each anchor order. Default is 40.
    continuation_iterations : int, optional
        Adam iterations after each warm-start step in interaction order.
        Default is 4.
    learning_rate : float, optional
        Adam learning rate. Default is 0.08.
    temperature : float, optional
        Temperature of the fixed-cardinality sigmoid relaxation. Default is
        0.25.

    Returns
    -------
    best_nplets : list of torch.Tensor
        One tensor per interaction order, from ``initial_order`` through
        ``order``. Each tensor has shape ``(repeat, current_order)`` and contains
        the best discrete n-plets found at that order.
    best_scores : torch.Tensor
        O-information scores with shape ``(repeat, n_orders)``. For multiple
        datasets these are mean scores across datasets, exactly as in THOI's
        standard O-information scoring.

    Notes
    -----
    This optimizer is specific to Gaussian O-information and therefore does not
    expose THOI's generic ``metric``/callable interface used by greedy and
    simulated annealing.
    """
    covmats, _, N, T = _normalize_input_data(
        X, covmat_precomputed, T, device, batch_size_D=batch_size_D
    )

    order = N if order is None else order
    assert 3 <= initial_order <= order <= N
    assert repeat > 0
    assert repeat_batch_size > 0

    # The relaxed selector assumes unit diagonal. This conversion does not
    # change Gaussian O-information and allows arbitrary covariance inputs.
    std = torch.sqrt(torch.diagonal(covmats, dim1=-2, dim2=-1))
    corrmats = covmats / (std[:, :, None] * std[:, None, :])

    anchors, paths = _continuation_paths(initial_order, order)
    candidates = {current_order: [] for current_order in range(initial_order, order + 1)}
    candidate_scores = {current_order: [] for current_order in range(initial_order, order + 1)}

    # Optimize every anchor in the same leading batch.
    theta = 0.15 * torch.randn(len(anchors) * repeat, N, device=device, dtype=corrmats.dtype)
    anchor_orders = torch.tensor(
        [anchor for anchor in anchors for _ in range(repeat)],
        device=device,
        dtype=theta.dtype,
    )
    theta = _optimize_logits_batched(
        corrmats,
        theta,
        anchor_orders,
        largest=largest,
        iterations=max_iterations,
        learning_rate=learning_rate,
        temperature=temperature,
        batch_size_D=batch_size_D,
        repeat_batch_size=repeat_batch_size,
    )

    anchor_theta = {}
    for i, anchor in enumerate(anchors):
        current_theta = theta[i * repeat:(i + 1) * repeat].clone()
        anchor_theta[anchor] = current_theta
        nplets, scores = _score_logits(
            covmats, T, current_theta, anchor,
            temperature=temperature, batch_size=batch_size, device=device,
        )
        candidates[anchor].append(nplets)
        candidate_scores[anchor].append(scores)

    # Advance every active continuation path together at each wave.
    active = [
        {
            'theta': anchor_theta[start].clone(),
            'order': start,
            'target': target,
            'step': 1 if target > start else -1,
        }
        for start, target in paths
    ]

    while active:
        next_active = []
        for path in active:
            next_order = path['order'] + path['step']
            if ((path['step'] > 0 and next_order <= path['target']) or
                    (path['step'] < 0 and next_order >= path['target'])):
                next_active.append({**path, 'order': next_order})

        if not next_active:
            break

        theta = torch.cat([path['theta'] for path in next_active], dim=0)
        current_orders = torch.tensor(
            [path['order'] for path in next_active for _ in range(repeat)],
            device=device,
            dtype=theta.dtype,
        )
        theta = _optimize_logits_batched(
            corrmats,
            theta,
            current_orders,
            largest=largest,
            iterations=continuation_iterations,
            learning_rate=learning_rate,
            temperature=temperature,
            batch_size_D=batch_size_D,
            repeat_batch_size=repeat_batch_size,
        )

        offset = 0
        for path in next_active:
            path['theta'] = theta[offset:offset + repeat].clone()
            offset += repeat
            nplets, scores = _score_logits(
                covmats, T, path['theta'], path['order'],
                temperature=temperature, batch_size=batch_size, device=device,
            )
            candidates[path['order']].append(nplets)
            candidate_scores[path['order']].append(scores)

        active = next_active

    best_nplets = []
    best_scores = []
    for current_order in range(initial_order, order + 1):
        nplets = torch.cat(candidates[current_order], dim=0)
        scores = torch.cat(candidate_scores[current_order], dim=0)
        top = torch.topk(scores, min(repeat, scores.numel()), largest=largest).indices
        best_nplets.append(nplets[top])
        best_scores.append(scores[top])

    return best_nplets, torch.stack(best_scores, dim=1)
