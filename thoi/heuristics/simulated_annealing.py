from typing import Optional, List, Union, Callable
import numpy as np
from tqdm import trange
import torch
import logging

from thoi.heuristics.scoring import _evaluate_nplets
from thoi.heuristics.commons import _get_valid_candidates
from thoi.commons import _normalize_input_data

@torch.no_grad()
def random_sampler(N:int, order:int, repeat:int, device:Optional[torch.device]=None):

    device = torch.device('cpu') if device is None else device

    return torch.stack([
        torch.randperm(N, device=device)[:order]
        for _ in range(repeat)
    ])

@torch.no_grad()
def simulated_annealing(X: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
                        order: Optional[int]=None,
                        *,
                        covmat_precomputed: bool=False,
                        T: Optional[Union[int, List[int]]]=None,
                        initial_solution: Optional[torch.Tensor] = None,
                        repeat: int = 10,
                        batch_size: int = 1000000,
                        device: torch.device = torch.device('cpu'),
                        max_iterations: int = 1000,
                        early_stop: int = 100,
                        initial_temp: float = 100.0,
                        cooling_rate: float = 0.99,
                        metric: Union[str,Callable]='o', # tc, dtc, o, s
                        largest: bool = False,
                        verbose: int = logging.INFO):
    
    logging.basicConfig(
        level=verbose,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    covmats, D, N, T = _normalize_input_data(X, covmat_precomputed, T, device)

    # Compute current solution
    # |batch_size| x |order|
    if initial_solution is None:
        current_solution = random_sampler(N, order, repeat, device)
    else:
        current_solution = initial_solution.to(device).contiguous()

    # |batch_size|
    current_energy = _evaluate_nplets(covmats, T,
                                      current_solution,
                                      metric,
                                      batch_size=batch_size,
                                      device=device)

    if not largest:
        current_energy = -current_energy

    # Initial valid candidates
    # |batch_size| x |N-order|
    valid_candidates = _get_valid_candidates(current_solution, N, device)

    # Set initial temperature
    temp = initial_temp

    # Best solution found
    # |batch_size| x |order|
    best_solution = current_solution.clone()
    # |batch_size|
    best_energy = current_energy.clone()
    
    # Repeat tensor for indexing the current_solution
    # |repeat| x |1|
    i_repeat = torch.arange(repeat)

    no_progress_count = 0
    pbar = trange(max_iterations, leave=False)
    for _ in pbar:
        
        # Get function name if metric is a function
        metric_name = metric.__name__ if callable(metric) else metric

        pbar.set_description(f'mean({metric_name}) = {(1 if largest else -1) * best_energy.mean()} - ES: {no_progress_count}')
        
        # Generate new solution by modifying the current solution.
        # Generate the random indexes to change.
        # |batch_size| x |order|
        i_sol = torch.randint(0, current_solution.shape[1], (repeat,), device=device)
        i_cand = torch.randint(0, valid_candidates.shape[1], (repeat,), device=device)

        # Update current values by new candidates and keep the original 
        # candidates to restore where the new solution is not accepted.
        current_candidates = current_solution[i_repeat, i_sol]
        new_candidates = valid_candidates[i_repeat, i_cand]
        current_solution[i_repeat, i_sol] = new_candidates

        # Calculate energy of new solution
        # |batch_size|
        new_energy = _evaluate_nplets(covmats, T,
                                      current_solution,
                                      metric,
                                      batch_size=batch_size,
                                      device=device)

        if not largest:
            new_energy = -new_energy

        # Calculate change in energy
        # delca_energy > 0 means new_energy is bigger (more optimal) than current_energy
        # |batch_size|
        delta_energy = new_energy - current_energy

        # Determine if we should accept the new solution
        # |batch_size|
        temp_probas = torch.rand(repeat, device=device) < torch.exp(delta_energy / temp)
        improves = delta_energy > 0
        accept_new_solution = torch.logical_or(improves, temp_probas)
        
        # Restore original values for rejected candidates
        # |batch_size| x |order|
        current_solution[i_repeat[~accept_new_solution], i_sol[~accept_new_solution]] = current_candidates[~accept_new_solution]
        
        # Update valid_candidate for the accepted answers as they are not longer valid candidates
        # |batch_size| x |N-order|
        valid_candidates[i_repeat[accept_new_solution], i_cand[accept_new_solution]] = current_candidates[accept_new_solution]

        # Update current energy for the accepted solutions
        # |batch_size|
        current_energy[accept_new_solution] = new_energy[accept_new_solution]

        new_global_maxima = (new_energy > best_energy)

        # |batch_size| x |order|
        best_solution[new_global_maxima] = current_solution[new_global_maxima]

        # |batch_size|
        best_energy[new_global_maxima] = new_energy[new_global_maxima]

        # Cool down
        temp *= cooling_rate

        # Early stop
        if torch.any(new_global_maxima):
            no_progress_count = 0
        else:
            no_progress_count += 1
        
        if no_progress_count >= early_stop:
            logging.info('Early stop reached')
            break

    # If minimizing, then return score to its real value
    if not largest:
        best_energy = -best_energy

    return best_solution, best_energy

