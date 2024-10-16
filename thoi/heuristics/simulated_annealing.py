from typing import Optional, List, Union, Callable
import numpy as np
from tqdm import trange
import torch

from thoi.heuristics.scoring import _evaluate_nplets
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
                        use_cpu: bool = False,
                        max_iterations: int = 1000,
                        early_stop: int = 100,
                        initial_temp: float = 100.0,
                        cooling_rate: float = 0.99,
                        metric: Union[str,Callable]='o', # tc, dtc, o, s
                        largest: bool = False):
    
    covmats, D, N, T, device = _normalize_input_data(X, covmat_precomputed, T, use_cpu)

    # Compute current solution
    # |batch_size| x |order|
    if initial_solution is None:
        current_solution = random_sampler(N, order, repeat, device)
    else:
        current_solution = initial_solution.to(device).contiguous()

    # |batch_size|
    current_energy = _evaluate_nplets(covmats, T, current_solution, metric, use_cpu=use_cpu)

    if not largest:
        current_energy = -current_energy

    # Initial valid candidates
    all_elements = torch.arange(N, device=device)
    valid_candidates = torch.stack([
        all_elements[~torch.isin(all_elements, current_solution[b])]
        for b in torch.arange(repeat, device=device)
    ]).contiguous()

    # Set initial temperature
    temp = initial_temp

    # Best solution found
    # |batch_size| x |order|
    best_solution = current_solution.clone()
    # |batch_size|
    best_energy = current_energy.clone()

    no_progress_count = 0
    pbar = trange(max_iterations, leave=False)
    for _ in pbar:

        pbar.set_description(f'mean({metric.upper()}) = {(1 if largest else -1) * best_energy.mean()} - ES: {no_progress_count}')
        
        # Generate new solution by modifying the current solution
        # |batch_size| x |order|
        new_solution = current_solution.clone()
        i_sol = torch.randint(0, new_solution.shape[1], (repeat,), device=device)
        i_cand = torch.randint(0, valid_candidates.shape[1], (repeat,), device=device)

        # Swap current values for candidates 
        new_candidates = new_solution[torch.arange(repeat), i_sol]
        current_candidates = valid_candidates[torch.arange(repeat), i_cand]
        new_solution[torch.arange(repeat), i_sol] = current_candidates

        # Calculate energy of new solution
        # |batch_size|
        new_energy = _evaluate_nplets(covmats, T, new_solution, metric, use_cpu=use_cpu)

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
        accept_new_solution = torch.logical_and(improves, temp_probas)
        
        # batch update solutions
        # |batch_size| x |order|
        current_solution = torch.where(
            accept_new_solution.unsqueeze(1).expand((-1,order)), # match correct shape
            new_solution,
            current_solution
        )

        # |batch_size|
        current_energy = torch.where(
            accept_new_solution,
            new_energy,
            current_energy
        )

        # update final valid_candidate depending on accepted answers
        valid_candidates[torch.arange(repeat), i_cand] = torch.where(
            accept_new_solution,
            new_candidates,
            current_candidates
        )

        new_global_maxima = (new_energy > best_energy)

        # |batch_size| x |order|
        best_solution = torch.where(
            new_global_maxima.unsqueeze(1).expand((-1,order)), # match correct shape
            new_solution,
            best_solution
        )

        # |batch_size|
        best_energy = torch.where(
            new_global_maxima,
            new_energy,
            best_energy
        )

        # Cool down
        temp *= cooling_rate

        # Early stop
        if torch.any(new_global_maxima):
            no_progress_count = 0
        else:
            no_progress_count += 1
        
        if no_progress_count >= early_stop:
            print('Early stop reached')
            break

    # If minimizing, then return score to its real value
    if not largest:
        best_energy = -best_energy

    return best_solution, best_energy

