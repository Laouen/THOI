import numpy as np
import math
import torch

from thoi.measures.gaussian_copula import nplets_measures, gaussianCopula


def gc_oinfo(covmat: np.ndarray, T:int, batched_nplets: np.ndarray):

    """
        X (np.ndarray): The data with shape T = n_samples, N = n_variables
        batched_nplets (np.ndarray): The nplets to calculate the inverse of the oinformation with shape batch_size, order
    """
    # |batch_size| x |4 = (tc, dtc, o, s)|
    batched_res = nplets_measures(covmat, batched_nplets, T=T, covmat_precomputed=True)
    
    # Return minus the o information score to make it an maximum optimization
    # |batch_size|
    return -batched_res[:,2].flatten()


def simulated_annealing(X: np.ndarray, order: int, initial_temp:float=100.0, cooling_rate:float=0.99, max_iterations:int=1000, repeat:int=10, use_cpu:bool=False):
    # make device cpu if not cuda available or cuda if available
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')

    T, N = X.shape

    covmat = torch.tensor(gaussianCopula(X)[1])
    covmat.to(device)

    # Initialize random solution
    # |batch_size| x |order|
    current_solution = torch.randint(0, N, size=(repeat, order))
    # |batch_size|
    current_energy = gc_oinfo(covmat, T, current_solution)

    # Set initial temperature
    temp = initial_temp

    # Best solution found
    # |batch_size| x |order|
    best_solution = current_solution
    # |batch_size|
    best_energy = current_energy

    for _ in range(max_iterations):
        
        # Generate new solution by modifying the current solution
        # |batch_size| x |order|
        new_solution = current_solution.clone()
        i = torch.randint(0, order, (repeat,))
        candidates = torch.tensor([
            np.random.choice([x for x in range(N) if x not in new_solution[b].tolist()])
            for b in range(repeat)
        ])
        new_solution[torch.arange(repeat), i] = candidates

        # Calculate energy of new solution
        # |batch_size|
        new_energy = gc_oinfo(covmat, T, new_solution)

        # Calculate change in energy
        # delca_score > 0 means new_soce is bigger (more optimal) than current_score
        # |batch_size|
        delta_energy = new_energy - current_energy

        # Determine if we should accept the new solution
        # |batch_size|
        temp_probas = torch.rand(repeat) < math.exp(delta_energy / temp)
        improves = delta_energy > 0
        
        # batch update solutions
        # |batch_size| x |order|
        current_solution = torch.where(
            torch.logical_and(improves, temp_probas).unsqueeze(1),
            new_solution,
            current_solution
        )

        # |batch_size|
        current_energy = torch.where(
            torch.logical_and(improves, temp_probas).unsqueeze(1),
            new_energy,
            current_energy
        )

        # |batch_size| x |order|
        best_solution = torch.where(
            (new_energy > best_energy).unsqueeze(1),
            new_solution,
            best_solution
        )

        # |batch_size|
        best_energy = torch.where(
            (new_energy > best_energy).unsqueeze(1),
            new_energy,
            best_energy
        )

        # Cool down
        temp *= cooling_rate

    return best_solution, best_energy
