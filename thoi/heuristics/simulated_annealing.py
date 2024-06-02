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
    return -batched_res[:,2].flatten()


def simulated_annealing(X: np.ndarray, 
                        order: int,
                        initial_temp:float=100.0,
                        cooling_rate:float=0.99,
                        max_iterations:int=1000,
                        repeat:int=10,
                        use_cpu:bool=False,
                        init_method:str='lower_order',
                        lower_order:int=1):

    # make device cpu if not cuda available or cuda if available
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')

    T, N = X.shape

    covmat = torch.tensor(gaussian_copula(X)[1])
    covmat.to(device)

    # Initialize random solution
    if init_method == 'random':
        # |batch_size| x |order|
        current_solution = torch.stack([
            torch.randperm(N, device=device)[:order]
            for _ in range(repeat)
        ])

    elif init_method == 'lower_order':
        current_solution = multi_order_measures(
            X, order-lower_order, order-lower_order, batch_size=repeat, use_cpu=use_cpu,
            batch_data_collector=partial(batch_to_tensor, top_k=repeat),
            batch_aggregation=partial(concat_tensors, top_k=repeat)
        )[-1].to(device)
        all_elements = torch.arange(N, device=device)
        valid_candidates = [
            all_elements[~torch.isin(all_elements, current_solution[b])]
            for b in torch.arange(repeat, device=device)
        ]
        valid_candidates = torch.stack([
            vd[torch.randperm(len(vd), device=device)][:lower_order]
            for vd in valid_candidates
        ])
        current_solution = torch.cat((current_solution, valid_candidates) , dim=1)

    # |batch_size|
    current_energy = gc_oinfo(covmat, T, current_solution)

    # Initial valid candidates
    all_elements = torch.arange(N, device=device)
    valid_candidates = torch.stack([
        all_elements[~torch.isin(all_elements, current_solution[b])]
        for b in torch.arange(repeat, device=device)
    ])

    # Set initial temperature
    temp = initial_temp

    # Best solution found
    # |batch_size| x |order|
    best_solution = current_solution
    # |batch_size|
    best_energy = current_energy

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
        new_energy = gc_oinfo(covmat, T, new_solution)

        # Calculate change in energy
        # delca_score > 0 means new_soce is bigger (more optimal) than current_score
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

    return best_solution, best_energy
