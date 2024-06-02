import numpy as np
from tqdm import trange
import torch
from functools import partial

from thoi.measures.gaussian_copula import nplets_measures, gaussian_copula, multi_order_measures
from thoi.collectors import batch_to_tensor, concat_tensors


def gc_oinfo(covmat: torch.tensor, T:int, batched_nplets: torch.tensor):

    """
        X (torch.tensor): The covariance matrix with shape (n_variables, n_variables)
        batched_nplets (torch.tensor): The nplets to calculate the inverse of the oinformation with shape (batch_size, order)
    """
    # |batch_size| x |4 = (tc, dtc, o, s)|
    batched_res = nplets_measures(covmat, batched_nplets, T=T, covmat_precomputed=True)
    
    # Return minus the o information score to make it an maximum optimization
    # |batch_size|
    return batched_res[:,2].flatten()


def greedy(X:np.ndarray, initial_order:int, max_order:int, repeat:int=10, use_cpu:bool=False, batch_size:int=1000000):

    current_solution = multi_order_measures(
        X, initial_order, initial_order, batch_size=batch_size, use_cpu=use_cpu,
        batch_data_collector=partial(batch_to_tensor, top_k=repeat),
        batch_aggregation=partial(concat_tensors, top_k=repeat)
    )[-1]

    T, N = X.shape

    covmat = torch.tensor(gaussian_copula(X)[1])

    # make device cpu if not cuda available or cuda if available
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')

    covmat = covmat.to(device)
    current_solution = current_solution.to(device)

    best_scores = [gc_oinfo(covmat, T, current_solution)]
    for _ in trange(initial_order, max_order, leave=False, desc='Order'):
        best_candidate, best_score = next_order_greedy(covmat, T, current_solution)
        best_scores.append(best_score)

        current_solution = torch.cat((current_solution, best_candidate.unsqueeze(1)) , dim=1)

    
    return current_solution, torch.stack(best_scores).T


def next_order_greedy(covmat: torch.tensor,
                      T: int,
                      initial_solution: torch.tensor):

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
    ])

    # current_solution constructed by adding first element of valid_candidate to input solution
    current_solution = torch.cat((initial_solution, valid_candidates[:, [0]]) , dim=1)
    
    # start best solution first_candidate
    # |batch_size| x |order+1|
    best_candidates = valid_candidates[:, 0]
    # |batch_size|
    best_score = gc_oinfo(covmat, T, current_solution)

    # iterate candidate from 1 since candidate 0 is already in the initial best solution 
    for i_cand in trange(1,valid_candidates.shape[1], leave=False, desc=f'Candidates'):

        # Update current solution to the next candidate inplace to avoid memory overhead
        current_candidates = valid_candidates[:, i_cand]
        current_solution[:, -1] = current_candidates

        # Calculate score of new solution
        # |batch_size|
        new_score = gc_oinfo(covmat, T, current_solution)

        # Determine if we should accept the new solution
        # |batch_size|
        new_global_maxima = new_score < best_score
        
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

    return best_candidates, best_score
