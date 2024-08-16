import numpy as np
from thoi.measures.utils import gaussian_copula
from tqdm import trange
import torch
from functools import partial

from thoi.measures.gaussian_copula import multi_order_measures
from thoi.collectors import batch_to_tensor, concat_tensors
from thoi.heuristics.scoring import _evaluate_nplet


def greedy(X:np.ndarray,
           order:int,
           initial_order:int=3,
           repeat:int=10,
           use_cpu:bool=False,
           batch_size:int=1000000,
           metric:str='o',
           largest:bool=False):


    assert metric in ['tc', 'dtc', 'o', 's'], f'metric must be one of tc, dtc, o or s. invalid value: {metric}'

    current_solution = multi_order_measures(
        X, min_order=initial_order, max_order=initial_order, batch_size=batch_size, use_cpu=use_cpu,
        batch_data_collector=partial(batch_to_tensor, top_k=repeat, metric=metric, largest=largest),
        batch_aggregation=partial(concat_tensors, top_k=repeat, metric=metric, largest=largest)
    )[-1]

    T, N = X.shape

    covmat = torch.tensor(gaussian_copula(X)[1])

    # Make device cpu if not cuda available or cuda if available
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')

    covmat = covmat.to(device).contiguous()
    current_solution = current_solution.to(device).contiguous()

    best_scores = [_evaluate_nplet(covmat, T, current_solution, metric, use_cpu=use_cpu)]
    for _ in trange(initial_order, order, leave=False, desc='Order'):
        best_candidate, best_score = next_order_greedy(covmat, T, current_solution,
                                                       metric=metric,
                                                       largest=largest,
                                                       use_cpu=use_cpu)
        best_scores.append(best_score)

        current_solution = torch.cat((current_solution, best_candidate.unsqueeze(1)) , dim=1)

    
    return current_solution, torch.stack(best_scores).T


def next_order_greedy(covmat: torch.tensor,
                      T: int,
                      initial_solution: torch.tensor,
                      metric:str='o',
                      largest:bool=False,
                      use_cpu:bool=False):
    
    assert metric in ['tc', 'dtc', 'o', 's'], f'metric must be one of tc, dtc, o or s. invalid value: {metric}'

    # Get parameters attributes
    device = covmat.device
    N = covmat.shape[0]
    batch_size = initial_solution.shape[0]

    # Initial valid candidates to iterate one by one
    # |batch_size| x |N-order|
    all_elements = torch.arange(N, device=device)
    valid_candidates = torch.stack([
        all_elements[~torch.isin(all_elements, initial_solution[b])]
        for b in torch.arange(batch_size, device=device)
    ]).contiguous()

    # Current_solution constructed by adding first element of valid_candidate to input solution
    current_solution = torch.cat((initial_solution, valid_candidates[:, [0]]) , dim=1)
    
    # Start best solution first_candidate
    # |batch_size| x |order+1|
    best_candidates = valid_candidates[:, 0]
    # |batch_size|
    best_score = _evaluate_nplet(covmat, T, current_solution, metric, use_cpu=use_cpu)
    
    if not largest:
        best_score = -best_score

    # Iterate candidate from 1 since candidate 0 is already in the initial best solution 
    for i_cand in trange(1,valid_candidates.shape[1], leave=False, desc=f'Candidates'):

        # Update current solution to the next candidate inplace to avoid memory overhead
        current_candidates = valid_candidates[:, i_cand]
        current_solution[:, -1] = current_candidates

        # Calculate score of new solution
        # |batch_size|
        new_score = _evaluate_nplet(covmat, T, current_solution, metric, use_cpu=use_cpu)

        # if minimizing, then maximize the inverted score
        if not largest:
            new_score = -new_score

        # Determine if we should accept the new solution
        # new_score is bigger (more optimal) than best_score
        # |batch_size|
        new_global_maxima = new_score > best_score
        
        # update best solution based on accpetance criteria
        # |batch_size| x |order|
        best_candidates = torch.where(
            new_global_maxima,
            current_candidates,
            best_candidates
        )

        # |batch_size|
        best_score = torch.where(
            new_global_maxima,
            new_score,
            best_score
        )
    
    # If minimizing, then return score to its original sign
    if not largest:
        best_score = -best_score

    return best_candidates, best_score
