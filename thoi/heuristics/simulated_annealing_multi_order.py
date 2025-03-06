from typing import List, Union, Optional, Callable
import numpy as np
from tqdm import trange
import torch
import logging

from thoi.commons import _normalize_input_data
from thoi.heuristics.scoring import _evaluate_nplet_hot_encoded

def _random_solutions(repeat, N, device):
    # Create a tensor of random 0s and 1s
    current_solution = torch.randint(0, 2, (repeat, N), device=device)

    # Ensure at least 3 ones in each element
    # Randomly choose 3 unique positions for each batch element
    rand_indices = torch.rand(repeat, N, device=device).argsort(dim=1)[:, :3]

    # Set these positions to 1
    current_solution[torch.arange(repeat).unsqueeze(1), rand_indices] = 1

    return current_solution

@torch.no_grad()
def hot_encode_to_indexes(nplets):
    batch_size, N = nplets.shape
    non_zero_indices = nplets.nonzero(as_tuple=False)
    indices = non_zero_indices[:, 1].view(batch_size, -1)
    return indices

@torch.no_grad()
def simulated_annealing_multi_order(X: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
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
                                    step_size: int = 1,
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
        current_solution = _random_solutions(repeat, N, device)
    else:
        current_solution = initial_solution.to(device).contiguous()

    # |batch_size|
    current_energy = _evaluate_nplet_hot_encoded(covmats, T,
                                                 current_solution,
                                                 metric,
                                                 batch_size=batch_size,
                                                 device=device)

    if not largest:
        current_energy = -current_energy

    # Set initial temperature
    temp = initial_temp

    # Best solution found
    # |batch_size| x |order|
    best_solution = current_solution.clone()
    # |batch_size|
    best_energy = current_energy.clone()
    
    # Repeat tensor for indexing the current_solution
    # |repeat| x |step_size|
    i_repeat = torch.arange(repeat).unsqueeze(1).expand(-1, step_size)

    no_progress_count = 0
    pbar = trange(max_iterations, leave=True)
    for _ in pbar:

        # Get function name if metric is a function
        metric_name = metric.__name__ if callable(metric) else metric

        pbar.set_description(f'mean({metric_name}) = {(1 if largest else -1) * best_energy.mean()} - ES: {no_progress_count}')
        
        
        # |batch_size|
        #i_change = torch.randint(0, N, (repeat,), device=device)
        i_change = torch.stack([torch.randperm(N, device=device)[:step_size] for _ in range(repeat)])
        

        # Change values of selected elements
        current_solution[i_repeat, i_change] = 1 - current_solution[i_repeat, i_change]

        # Calculate energy of new solution
        # |batch_size|
        new_energy = _evaluate_nplet_hot_encoded(covmats, T,
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

        # valid solutions have at least three elements with 1. Non valid solution are not accepted
        valid_solutions = current_solution.sum(dim=1) >= 3
        accept_new_solution = torch.logical_and(accept_new_solution, valid_solutions)

        # revert changes of not accepted solutions in the rows of the mask accept_new_solution and in the indexes of i_change
        current_solution[i_repeat[~accept_new_solution], i_change[~accept_new_solution]] = 1 - current_solution[i_repeat[~accept_new_solution], i_change[~accept_new_solution]]

        # |batch_size|
        current_energy[accept_new_solution] = new_energy[accept_new_solution]

        # |batch_size|
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