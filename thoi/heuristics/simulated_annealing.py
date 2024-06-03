from typing import Optional
import numpy as np
from tqdm import trange
import torch
from functools import partial

from thoi.measures.gaussian_copula import nplets_measures, gaussian_copula, multi_order_measures
from thoi.collectors import batch_to_tensor, concat_tensors

def init_lower_order(X: np.ndarray, order:int, lower_order:int, repeat:int, use_cpu:bool, device:torch.device):
    N = X.shape[1]

    # |repeat| x |lower_order|
    current_solution = multi_order_measures(
        X, lower_order, lower_order, batch_size=repeat, use_cpu=use_cpu,
        batch_data_collector=partial(batch_to_tensor, top_k=repeat),
        batch_aggregation=partial(concat_tensors, top_k=repeat)
    )[-1].to(device)

    # |N|
    all_elements = torch.arange(N, device=device)
    
    # |repeat| x |order-lower_order|
    valid_candidates = [
        all_elements[~torch.isin(all_elements, current_solution[b])]
        for b in torch.arange(repeat, device=device)
    ]

    # |repeat| x |order-lower_order|
    valid_candidates = torch.stack([
        vd[torch.randperm(len(vd), device=device)[:order-lower_order]]
        for vd in valid_candidates
    ]).contiguous()
    
    # |repeat| x |order|
    return torch.cat((current_solution, valid_candidates) , dim=1).contiguous()

def random_sampler(N:int, order:int, repeat:int, device:torch.device=None):

    if device is None:
        device = torch.device('cpu')

    return torch.stack([
        torch.randperm(N, device=device)[:order]
        for _ in range(repeat)
    ])


def _gc_oinfo_energy(covmat: torch.tensor, T:int, batched_nplets: torch.tensor):

    """
        X (torch.tensor): The covariance matrix with shape (n_variables, n_variables)
        batched_nplets (torch.tensor): The nplets to calculate the inverse of the oinformation with shape (batch_size, order)
    """
    # |batch_size| x |4 = (tc, dtc, o, s)|
    batched_res = nplets_measures(covmat, batched_nplets, T=T, covmat_precomputed=True)
    
    # Return minus the o information score to make it an maximum optimization (energy)
    # |batch_size|
    return -batched_res[:,2].flatten()


# TODO: add optimization value option as parameter
def simulated_annealing(X: np.ndarray, 
                        order: int,
                        initial_temp:float=100.0,
                        cooling_rate:float=0.99,
                        max_iterations:int=1000,
                        repeat:int=10,
                        use_cpu:bool=False,
                        init_method:str='random', # lower_order, 'random', 'precumputed', 'precomputed_lower_order';
                        lower_order:int=None,
                        early_stop:int=100,
                        current_solution: Optional[torch.tensor]=None):

    lower_order = order-1 if lower_order is None else lower_order
    assert init_method != 'lower_order' or lower_order < order, 'Init from optima lower order cannot start from a lower_order higher than the order to compute.' 

    # make device cpu if not cuda available or cuda if available
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')

    T, N = X.shape

    covmat = torch.tensor(gaussian_copula(X)[1])
    covmat = covmat.to(device).contiguous()

    # Initialize random solution
    # |batch_size| x |order|
    if init_method == 'random':
        current_solution = random_sampler(N, order, repeat, device)
    elif init_method == 'lower_order':
        current_solution = init_lower_order(X, order, lower_order, repeat, use_cpu, device)
    elif init_method == 'precomputed':
        assert current_solution is not None, 'current_solution must be a torch tensor'

    # |batch_size|
    current_energy = _gc_oinfo_energy(covmat, T, current_solution)

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
    best_solution = current_solution
    # |batch_size|
    best_energy = current_energy

    no_progress_count = 0
    pbar = trange(max_iterations, leave=False)
    for _ in pbar:

        pbar.set_description(f'mean(best_energy = {best_energy.mean()}')
        
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
        new_energy = _gc_oinfo_energy(covmat, T, new_solution)

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

    return best_solution, best_energy
