from typing import Optional
import numpy as np
from tqdm import trange
import torch

from thoi.measures.gaussian_copula import gaussian_copula
from thoi.heuristics.scoring import _evaluate_nplet


def reorder_by_hot_sum(current_solutions):
    # Calculate the sum of each vector
    sums = current_solutions.sum(dim=1)
    
    # Get the sorted indices based on the sums
    sorted_indices = torch.argsort(sums)
    
    # Reorder the current_solutions based on the sorted indices
    reordered_solutions = current_solutions[sorted_indices]
    reordered_sums = sums[sorted_indices]
    
    return reordered_solutions, reordered_sums, sorted_indices

def split_by_hot_size(current_solution):
    reordered_solutions, reordered_sums, sorted_indices = reorder_by_hot_sum(current_solution)

    # Find unique sums and their counts
    unique_sums, counts = torch.unique(reordered_sums, return_counts=True)
    
    split_solutions = []
    start_idx = 0
    for count in counts:
        split_solutions.append(reordered_solutions[start_idx:start_idx + count])
        start_idx += count
    
    return split_solutions, sorted_indices

def _evaluate_nplet_by_size(covmat: torch.tensor, T:int, batched_nplets: torch.tensor, metric:str, use_cpu:bool=False):
    split_nplets, sorted_indices = split_by_hot_size(batched_nplets)
    
    # Evaluate the splits
    evaluated_splits = torch.cat([
        _evaluate_nplet(covmat, T, nplets, metric, use_cpu)
        for nplets in split_nplets
    ])
    
    # Create a tensor to hold the results in the original order
    results = torch.zeros_like(evaluated_splits)
    
    # Place the evaluated scores back into the original positions
    results[sorted_indices] = evaluated_splits
    
    return results


def order_free_simulated_annealing(X: np.ndarray,
                                   initial_temp: float = 100.0,
                                   cooling_rate: float = 0.99,
                                   max_iterations: int = 1000,
                                   repeat: int = 10,
                                   use_cpu: bool = False,
                                   early_stop: int = 100,
                                   current_solution: Optional[torch.tensor] = None,
                                   metric: str = 'o', # tc, dtc, o, s
                                   largest: bool = False):

    assert metric in ['tc', 'dtc', 'o', 's'], f'metric must be one of tc, dtc, o or s. invalid value: {metric}'

    # make device cpu if not cuda available or cuda if available
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')

    T, N = X.shape

    covmat = torch.tensor(gaussian_copula(X)[1])
    covmat = covmat.to(device).contiguous()

    # generate a matrix with shape (repeat, N) of hot encoders for each element
    # each row is a vector of 0s and 1s where 1s are the elements in the solution
    current_solution = torch.randint(0, 2, (repeat, N), device=device)

    # |batch_size|
    current_energy = _evaluate_nplet_by_size(covmat, T, current_solution, metric, use_cpu=use_cpu)

    if not largest:
        current_energy = -current_energy

    # Set initial temperature
    temp = initial_temp

    # Best solution found
    # |batch_size| x |order|
    best_solution = current_solution
    # |batch_size|
    best_energy = current_energy

    no_progress_count = 0
    pbar = trange(max_iterations, leave=False)
    for _ in pbar:

        pbar.set_description(f'mean({metric.upper()}) = {(1 if largest else -1) * best_energy.mean()}')
        
        
        # |batch_size|
        i_change = torch.randint(0, N, (repeat,), device=device)

        # Change values of selected elements
        current_solution[torch.arange(repeat), i_change] = 1 - current_solution[torch.arange(repeat), i_change]

        # Calculate energy of new solution
        # |batch_size|
        new_energy = _evaluate_nplet_by_size(covmat, T, current_solution, metric, use_cpu=use_cpu)

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

        # revert changes of not accepted solutions in the rows of the mask accept_new_solution and in the indexes of i_change
        current_solution[~accept_new_solution, i_change[~accept_new_solution]] = 1 - current_solution[~accept_new_solution, i_change[~accept_new_solution]]

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
            print('Early stop reached')
            break

    # If minimizing, then return score to its real value
    if not largest:
        best_energy = -best_energy

    return best_solution, best_energy