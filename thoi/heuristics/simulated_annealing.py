from typing import Optional, List, Union, Callable
import numpy as np
from tqdm import trange
import torch
import logging

from thoi.heuristics.scoring import _evaluate_nplets
from thoi.heuristics.commons import _get_valid_candidates
from thoi.commons import _normalize_input_data

@torch.no_grad()
def random_sampler(N: int, order: int, repeat: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate random samples of n-plets.

    Parameters
    ----------
    N : int
        The number of variables.
    order : int
        The order of the n-plets.
    repeat : int
        The number of samples to generate.
    device : torch.device, optional
        The device to use for the computation. Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        A tensor of shape (repeat, order) containing the random samples.
    """
    device = torch.device('cpu') if device is None else device

    return torch.stack([
        torch.randperm(N, device=device)[:order]
        for _ in range(repeat)
    ])

@torch.no_grad()
def simulated_annealing(X: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
                        order: Optional[int] = None,
                        *,
                        covmat_precomputed: bool = False,
                        T: Optional[Union[int, List[int]]] = None,
                        initial_solution: Optional[torch.Tensor] = None,
                        repeat: int = 10,
                        batch_size: int = 1000000,
                        device: torch.device = torch.device('cpu'),
                        max_iterations: int = 1000,
                        early_stop: int = 100,
                        initial_temp: float = 100.0,
                        cooling_rate: float = 0.99,
                        metric: Union[str, Callable] = 'o',  # tc, dtc, o, s
                        largest: bool = False,
                        verbose: int = logging.INFO) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulated annealing algorithm to find the best n-plets to maximize the metric for a given multivariate series or covariance matrices.

    Parameters
    ----------
    X : Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]
        The input data to compute the n-plets. It can be a list of 2D numpy arrays or tensors of shape:
        1. (T, N) where T is the number of samples if X are multivariate series.
        2. A list of 2D covariance matrices with shape (N, N).
    order : int, optional
        The order of the n-plets. If None, it will be set to N.
    covmat_precomputed : bool, optional
        A boolean flag to indicate if the input data is a list of covariance matrices or multivariate series. Default is False.
    T : int or list of int, optional
        A list of integers indicating the number of samples for each multivariate series. Default is None.
    initial_solution : torch.Tensor, optional
        The initial solution with shape (repeat, order). If None, a random initial solution is generated.
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
    metric : Union[str, Callable], optional
        The metric to evaluate. One of 'tc', 'dtc', 'o', 's' or a callable function. Default is 'o'.
    largest : bool, optional
        A flag to indicate if the metric is to be maximized or minimized. Default is False.
    verbose : int, optional
        Logging verbosity level. Default is `logging.INFO`.

    Returns
    -------
    best_solution : torch.Tensor
        The n-plets with the best score found with shape (repeat, order).
    best_energy : torch.Tensor
        The best scores for the best n-plets with shape (repeat,).

    Notes
    -----
    - The function uses a simulated annealing algorithm to iteratively find the best n-plets that maximize or minimize the specified metric.
    - The initial solutions are computed using the `random_sampler` function if not provided.
    - The function iterates over the remaining orders to get the best solution for each order.
    """
    
    logging.basicConfig(
        level=verbose,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    covmats, D, N, T = _normalize_input_data(X, covmat_precomputed, T, device)

    # Compute current solution
    # |repeat| x |order|
    if initial_solution is None:
        current_solution = random_sampler(N, order, repeat, device)
    else:
        current_solution = initial_solution.to(device).contiguous()

    # |repeat|
    current_energy = _evaluate_nplets(covmats, T,
                                      current_solution,
                                      metric,
                                      batch_size=batch_size,
                                      device=device)

    if not largest:
        current_energy = -current_energy

    # Initial valid candidates
    # |repeat| x |N-order|
    valid_candidates = _get_valid_candidates(current_solution, N, device)

    # Set initial temperature
    temp = initial_temp

    # Best solution found
    # |repeat| x |order|
    best_solution = current_solution.clone()
    # |repeat|
    best_energy = current_energy.clone()
    
    # Repeat tensor for indexing the current_solution
    # |repeat| x |1|
    i_repeat = torch.arange(repeat, device=device)

    no_progress_count = 0
    pbar = trange(max_iterations, leave=False)
    for _ in pbar:
        
        # Get function name if metric is a function
        metric_name = metric.__name__ if callable(metric) else metric

        pbar.set_description(f'mean({metric_name}) = {(1 if largest else -1) * best_energy.mean()} - ES: {no_progress_count}')
        
        # Generate new solution by modifying the current solution.
        # Generate the random indexes to change.
        # |repeat| x |order|
        i_sol = torch.randint(0, current_solution.shape[1], (repeat,), device=device)
        i_cand = torch.randint(0, valid_candidates.shape[1], (repeat,), device=device)

        # Update current values by new candidates and keep the original 
        # candidates to restore where the new solution is not accepted.
        current_candidates = current_solution[i_repeat, i_sol]
        new_candidates = valid_candidates[i_repeat, i_cand]
        current_solution[i_repeat, i_sol] = new_candidates

        # Calculate energy of new solution
        # |repeat|
        new_energy = _evaluate_nplets(covmats, T,
                                      current_solution,
                                      metric,
                                      batch_size=batch_size,
                                      device=device)

        if not largest:
            new_energy = -new_energy

        # Calculate change in energy
        # delca_energy > 0 means new_energy is bigger (more optimal) than current_energy
        # |repeat|
        delta_energy = new_energy - current_energy

        # Determine if we should accept the new solution
        # |repeat|
        temp_probas = torch.rand(repeat, device=device) < torch.exp(delta_energy / temp)
        improves = delta_energy > 0
        accept_new_solution = torch.logical_or(improves, temp_probas)
        
        # Restore original values for rejected candidates
        # |repeat| x |order|
        current_solution[i_repeat[~accept_new_solution], i_sol[~accept_new_solution]] = current_candidates[~accept_new_solution]
        
        # Update valid_candidate for the accepted answers as they are not longer valid candidates
        # |repeat| x |N-order|
        valid_candidates[i_repeat[accept_new_solution], i_cand[accept_new_solution]] = current_candidates[accept_new_solution]

        # Update current energy for the accepted solutions
        # |repeat|
        current_energy[accept_new_solution] = new_energy[accept_new_solution]

        new_global_maxima = (new_energy > best_energy)

        # |repeat| x |order|
        best_solution[new_global_maxima] = current_solution[new_global_maxima]

        # |repeat|
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

