from typing import Union, Callable, List, Optional
from tqdm import trange

import torch
from functools import partial

from thoi.typing import TensorLikeArray
from thoi.measures.gaussian_copula import multi_order_measures
from thoi.collectors import batch_to_tensor, concat_batched_tensors
from thoi.heuristics.commons import _get_valid_candidates
from thoi.heuristics.scoring import _evaluate_nplets
from thoi.commons import _normalize_input_data

@torch.no_grad()
def greedy(X: TensorLikeArray,
           initial_order: int=3,
           order: Optional[int]=None,
           *,
           covmat_precomputed: bool=False,
           T: Optional[Union[int, List[int]]]=None,
           repeat: int=10,
           batch_size: int=1000000,
           repeat_batch_size: int=1000000,
           device: torch.device=torch.device('cpu'),
           metric: Union[str,Callable]='o',
           largest: bool=False):
    """
    Greedy algorithm to find the best order of n-plets to maximize the metric for a given multivariate series or covariance matrices.

    Parameters
    ----------
    X : TensorLikeArray
        The input data to compute the n-plets. It can be a list of 2D numpy arrays or tensors of shape:
        1. (T, N) where T is the number of samples if X are multivariate series.
        2. A list of 2D covariance matrices with shape (N, N).
    initial_order : int, optional
        The initial order to start the greedy algorithm. Default is 3.
    order : int, optional
        The final order to stop the greedy algorithm. If None, it will be set to N.
    covmat_precomputed : bool, optional
        A boolean flag to indicate if the input data is a list of covariance matrices or multivariate series. Default is False.
    T : int or list of int, optional
        A list of integers indicating the number of samples for each multivariate series. Default is None.
    repeat : int, optional
        The number of repetitions to do to obtain different solutions starting from less optimal initial solutions. Default is 10.
    batch_size : int, optional
        The batch size to use for the computation. Default is 1,000,000.
    repeat_batch_size : int, optional
        The batch size for repeating the computation. Default is 1,000,000.
    device : torch.device, optional
        The device to use for the computation. Default is 'cpu'.
    metric : Union[str, Callable], optional
        The metric to evaluate. One of 'tc', 'dtc', 'o', 's' or a callable function. Default is 'o'.
    largest : bool, optional
        A flag to indicate if the metric is to be maximized or minimized. Default is False.

    Returns
    -------
    best_nplets : torch.Tensor
        The n-plets with the best score found with shape (repeat, order).
    best_scores : torch.Tensor
        The best scores for the best n-plets with shape (repeat,).

    Notes
    -----
    - The function uses a greedy algorithm to iteratively find the best n-plets that maximize or minimize the specified metric.
    - The initial solutions are computed using the `multi_order_measures` function.
    - The function iterates over the remaining orders to get the best solution for each order.
    """

    covmats, D, N, T = _normalize_input_data(X, covmat_precomputed, T, device)

    # Compute initial solutions
    batch_data_collector = partial(batch_to_tensor, top_k=repeat, metric=metric, largest=largest)
    batch_aggregation = partial(concat_batched_tensors, top_k=repeat, metric=None, largest=largest)

    # |repeat| x |initial_order|, |repeat|
    _, current_solution, current_scores = multi_order_measures(covmats,
                                                               covmat_precomputed=True,
                                                               T=T,
                                                               min_order=initial_order,
                                                               max_order=initial_order,
                                                               batch_size=batch_size,
                                                               device=device,
                                                               batch_data_collector=batch_data_collector,
                                                               batch_aggregation=batch_aggregation)

    # send current solution to the device
    current_solution = current_solution.to(device).contiguous()

    # Set the order to the maximum order if not specified
    order = order if order is not None else N

    # Iterate over the remaining orders to get the best solution for each order
    best_scores = [current_scores]
    for _ in trange(initial_order, order, leave=False, desc='Order'):
        
        # |repeat|, |repeat|
        best_candidate, best_score = _next_order_greedy(covmats, T, current_solution,
                                                       metric=metric,
                                                       largest=largest,
                                                       batch_size=batch_size,
                                                       repeat_batch_size=repeat_batch_size,
                                                       device=device)
        # |order - initial_order| x |repeat|
        best_scores.append(best_score)
        
        # |repeat| x |order|
        current_solution = torch.cat((current_solution, best_candidate.unsqueeze(1)) , dim=1)
    
    # |repeat| x |order|, |repeat| x |order - initial_order|
    return current_solution, torch.stack(best_scores).T


def _create_all_solutions(initial_solution: torch.Tensor, valid_candidates: torch.Tensor) -> torch.Tensor:
    """
    Concatenates initial_solution with each valid_candidate to create all_solutions.

    Parameters
    ----------
    initial_solution : torch.Tensor
        Tensor of shape (batch_size, o) containing selected indices. Note: o < N and all elements must be unique.
    valid_candidates : torch.Tensor
        Tensor of shape (batch_size, T)

    Returns
    -------
    all_solutions : torch.Tensor
        Tensor of shape (batch_size, T, O + 1)
    """
    T = valid_candidates.shape[1]

    # Expand initial_solution to a new dimension at position 1 (for T)
    # |batch_size| x |1| x |O|
    initial_expanded = initial_solution.unsqueeze(1).expand(-1, T, -1)

    # Reshape valid_candidate to (batch_size, T, 1) for concatenation
    # |batch_size| x |T| x |1|
    valid_reshaped = valid_candidates.unsqueeze(2)

    # Concatenate along the last dimension to get (batch_size, T, O + 1)
    # |batch_size| x |T| x |O + 1|
    all_solutions = torch.cat([initial_expanded, valid_reshaped], dim=2)

    return all_solutions


def _next_order_greedy(covmats: torch.Tensor,
                      T: Optional[List[int]],
                      initial_solution: torch.Tensor,
                      metric: Union[str,Callable],
                      largest: bool,
                      batch_size: int=1000000,
                      repeat_batch_size: int=1000000,
                      device: torch.device=torch.device('cpu')):
    """
    Greedy algorithm to find the best candidate to add to the current solution.

    Parameters
    ----------
    covmats : torch.Tensor
        The covariance matrix or matrices with shape (D, N, N).
    T : list of int, optional
        The number of samples for each multivariate series.
    initial_solution : torch.Tensor
        The initial solution with shape (batch_size, order).
    metric : Union[str, Callable]
        The metric to evaluate. One of 'tc', 'dtc', 'o', 's' or a callable function.
    largest : bool
        A flag to indicate if the metric is to be maximized or minimized.
    batch_size : int, optional
        The batch size to use for the computation. Default is 1,000,000.
    repeat_batch_size : int, optional
        The batch size for repeating the computation. Default is 1,000,000.
    device : torch.device, optional
        The device to use for the computation. Default is 'cpu'.

    Returns
    -------
    best_candidates : torch.Tensor
        The best candidates to add to the current solution with shape (batch_size).
    best_score : torch.Tensor
        The best score for the best candidates with shape (batch_size).

    Notes
    -----
    - The function iterates over the valid candidates to find the best candidate to add to the current solution.
    - The best candidate is the one that maximizes or minimizes the specified metric.
    """

    # Get parameters attributes
    N = covmats.shape[1]
    total_size, order = initial_solution.shape

    # Initial valid candidates to iterate one by one
    # |total_size| x |N-order|
    valid_candidates = _get_valid_candidates(initial_solution, N, device)

    best_candidates = []
    best_scores = []

    for start in trange(0, total_size, repeat_batch_size, desc='Batch repeat', leave=False):
        end = min(start + repeat_batch_size, total_size)
        batch_initial_solution = initial_solution[start:end]
        batch_valid_candidates = valid_candidates[start:end]

        # |repeat_batch_size| x |N-order| x |order+1|
        all_solutions = _create_all_solutions(batch_initial_solution, batch_valid_candidates)
        
        # |repeat_batch_size x N-order| x |order+1|
        all_solutions = all_solutions.view(-1, order+1)
        
        # |repeat_batch_size x N-order|
        batch_best_score = _evaluate_nplets(covmats, T,
                                            all_solutions,
                                            metric,
                                            batch_size=batch_size,
                                            device=device)
        
        # |repeat_batch_size| x |N-order|
        batch_best_score = batch_best_score.view(end - start, N - order)
        
        if not largest:
            batch_best_score = -batch_best_score
        
        # get for each batch item the best score over the second dimension
        
        # |repeat_batch_size|
        max_idxs = torch.argmax(batch_best_score, dim=1)
        batch_best_candidates = batch_valid_candidates[torch.arange(end - start), max_idxs]
        batch_best_score = batch_best_score[torch.arange(end - start), max_idxs]
        
        # If minimizing, then return score to its original sign
        if not largest:
            batch_best_score = -batch_best_score

        best_candidates.append(batch_best_candidates)
        best_scores.append(batch_best_score)

    return torch.cat(best_candidates), torch.cat(best_scores)