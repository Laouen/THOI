from typing import List, Union, Optional, Callable
import numpy as np
from tqdm import trange
import torch
import logging

from thoi.commons import _normalize_input_data
from thoi.heuristics.scoring import _evaluate_nplet_hot_encoded

def _random_solutions(repeat, N, device):
    """
    Generate random solutions for n-plets with at least 3 ones in each element.

    Parameters
    ----------
    repeat : int
        The number of solutions to generate.
    N : int
        The number of variables.
    device : torch.device
        The device to use for the computation.

    Returns
    -------
    torch.Tensor
        A tensor of shape (repeat, N) containing the random solutions.
    """
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
    """
    Convert hot-encoded n-plets to index-based representation.

    Parameters
    ----------
    nplets : torch.Tensor
        The hot-encoded n-plets with shape (batch_size, N).

    Returns
    -------
    torch.Tensor
        The index-based representation of the n-plets with shape (batch_size, order).
    """
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
                                    verbose: int = logging.INFO) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulated annealing algorithm to find the best multi-order n-plets to maximize the metric for a given multivariate series or covariance matrices.

    Parameters
    ----------
    X : Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]
        The input data to compute the n-plets. It can be a list of 2D numpy arrays or tensors of shape:
        1. (T, N) where T is the number of samples if X are multivariate series.
        2. A list of 2D covariance matrices with shape (N, N).
    covmat_precomputed : bool, optional
        A boolean flag to indicate if the input data is a list of covariance matrices or multivariate series. Default is False.
    T : int or list of int, optional
        A list of integers indicating the number of samples for each multivariate series. Default is None.
    initial_solution : torch.Tensor, optional
        The initial solution with shape (repeat, N). If None, a random initial solution is generated.
    repeat : int, optional
        The number of repetitions to do to obtain different solutions starting from less optimal initial solutions. Default is 10.
    batch_size : int, optional
        The batch size to use for the computation. Default is 1,000,000.
    device : torch.device, optional
        The device to use for the computation. Default is 'cpu'.
    max_iterations : int, optional
        The maximum number of iterations for the simulated annealing algorithm. Default is 1000.
    early_stop : int, optional
        The number of iterations with no improvement to stop early. Default is 100.
    initial_temp : float, optional
        The initial temperature for the simulated annealing algorithm. Default is 100.0.
    cooling_rate : float, optional
        The cooling rate for the simulated annealing algorithm. Default is 0.99.
    step_size : int, optional
        The number of elements to change in each step. Default is 1.
    metric : Union[str, Callable], optional
        The metric to evaluate. One of 'tc', 'dtc', 'o', 's' or a callable function. Default is 'o'.
    largest : bool, optional
        A flag to indicate if the metric is to be maximized or minimized. Default is False.
    verbose : int, optional
        Logging verbosity level. Default is `logging.INFO`.

    Returns
    -------
    best_solution : torch.Tensor
        The n-plets with the best score found with shape (repeat, N).
    best_energy : torch.Tensor
        The best scores for the best n-plets with shape (repeat,).

    Notes
    -----
    - The function uses a simulated annealing algorithm to iteratively find the best n-plets that maximize or minimize the specified metric.
    - The initial solutions are computed using the `_random_solutions` function if not provided.
    - The function iterates over the remaining orders to get the best solution for each order.
    """
    
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