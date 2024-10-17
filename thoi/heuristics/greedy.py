from typing import Union, Callable, List, Optional
from tqdm import trange

import numpy as np
import torch
from functools import partial

from thoi.measures.gaussian_copula import multi_order_measures
from thoi.collectors import batch_to_tensor, concat_batched_tensors
from thoi.heuristics.commons import _get_valid_candidates
from thoi.heuristics.scoring import _evaluate_nplets
from thoi.commons import _normalize_input_data

@torch.no_grad()
def greedy(X: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
           initial_order: int=3,
           order: Optional[int]=None,
           *,
           covmat_precomputed: bool=False,
           T: Optional[Union[int, List[int]]]=None,
           repeat: int=10,
           use_cpu: bool=False,
           batch_size: int=1000000,
           metric: Union[str,Callable]='o',
           largest: bool=False):

    '''
    Brief: Greedy algorithm to find the best order of nplets to maximize the metric for a given multivariate series or covariance matrices
    
    Parameters:
    - X (Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]): The input data to compute the nplets. It can be a list of 2D numpy arrays or tensors of shape: 1. (T, N) where T is the number of samples if X are multivariate series. 2. a list of 2D covariance matrices with shape (N, N).
    - covmat_precomputed (bool): A boolean flag to indicate if the input data is a list of covariance matrices or multivariate series.
    - T (Optional[Union[int, List[int]]]): A list of integers indicating the number of samples for each multivariate series.
    '''

    covmats, D, N, T, device = _normalize_input_data(X, covmat_precomputed, T, use_cpu)

    # Compute initial solutions
    batch_data_collector = partial(batch_to_tensor, top_k=repeat, metric=metric, largest=largest)
    batch_aggregation = partial(concat_batched_tensors, top_k=repeat, metric=None, largest=largest)

    # |repeat| x |initial_order|
    _, current_solution, current_scores = multi_order_measures(covmats,
                                                               covmat_precomputed=True,
                                                               T=T,
                                                               min_order=initial_order,
                                                               max_order=initial_order,
                                                               batch_size=batch_size,
                                                               use_cpu=use_cpu,
                                                               batch_data_collector=batch_data_collector,
                                                               batch_aggregation=batch_aggregation)
    
    # send current solution to the device
    current_solution = current_solution.to(device).contiguous()

    # Set the order to the maximum order if not specified
    order = order if order is not None else N

    # Iterate over the remaining orders to get the best solution for each order
    best_scores = [current_scores]
    for _ in trange(initial_order, order, leave=False, desc='Order'):
        best_candidate, best_score = _next_order_greedy(covmats, T, current_solution,
                                                       metric=metric,
                                                       largest=largest,
                                                       use_cpu=use_cpu)
        best_scores.append(best_score)

        current_solution = torch.cat((current_solution, best_candidate.unsqueeze(1)) , dim=1)
    
    return current_solution, torch.stack(best_scores).T


def _create_all_solutions(initial_solution: torch.Tensor, valid_candidates: torch.Tensor) -> torch.Tensor:
    """
    Concatenates initial_solution with each valid_candidate to create all_solutions.

    Parameters:
    - initial_solution (torch.Tensor): Tensor of shape (batch_size, o) containing selected indices. Note: o < N and all elements must be unique.
    - valid_candidate (torch.Tensor): Tensor of shape (batch_size, T)

    Returns:
    - all_solutions (torch.Tensor): Tensor of shape (batch_size, T, O + 1)
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
                      metric:Union[str,Callable],
                      largest:bool,
                      use_cpu:bool):
    
    '''
    Brief: Greedy algorithm to find the best candidate to add to the current solution
    
    Parameters:
    - covmats (torch.Tensor): The covariance matrix or matrixes with shape (D, N, N)
    - T (List[int]): The number of samples for each multivariate series
    - initial_solution (torch.Tensor): The initial solution with shape (batch_size, order)
    - metric (Union[str,Callable]): The metric to evaluate. One of tc, dtc, o, s or a callable function
    - largest (bool): A flag to indicate if the metric is to be maximized or minimized
    - use_cpu (bool): A flag to indicate if the computation should be done on the CPU
    
    Returns:
    - best_candidates (torch.Tensor): The best candidates to add to the current solution with shape (batch_size)
    - best_score (torch.Tensor): The best score for the best candidates with shape (batch_size)
    '''

    # Get parameters attributes
    device = covmats.device
    N = covmats.shape[1]
    batch_size, order = initial_solution.shape

    # Initial valid candidates to iterate one by one
    # |batch_size| x |N-order|
    valid_candidates = _get_valid_candidates(initial_solution, N, device)
    
    # |batch_size| x |N-order| x |order+1|
    all_solutions = _create_all_solutions(initial_solution, valid_candidates)
    
    # |batch_size x N-order| x |order+1|
    all_solutions = all_solutions.view(batch_size*(N-order), order+1)
    
    # |batch_size x N-order|
    best_score = _evaluate_nplets(covmats, T, all_solutions, metric, use_cpu=use_cpu)
    
    # |batch_size| x |N-order|
    best_score = best_score.view(batch_size, N-order)
    
    if not largest:
        best_score = -best_score
    
    # get for each batch item the best score over the second dimention
    
    # |batch_size|
    max_idxs = torch.argmax(best_score, dim=1)
    best_candidates = valid_candidates[torch.arange(batch_size), max_idxs]
    best_score = best_score[torch.arange(batch_size), max_idxs]
    
    # If minimizing, then return score to its original sign
    if not largest:
        best_score = -best_score

    return best_candidates, best_score