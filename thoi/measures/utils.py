from typing import List, Union, Optional
import numpy as np
import scipy as sp
import torch

from thoi.measures.constants import TWOPIE


def _all_min_1_ids(N, device=torch.device('cpu')):
    base_tensor = torch.arange(N, device=device).unsqueeze(0).repeat(N, 1)  # Shape: (N, N)
    mask = base_tensor != torch.arange(N, device=device).unsqueeze(1)  # Shape: (N, N)
    result = base_tensor[mask].view(N, N - 1)  # Shape: (N, N-1)
    return result


def _gaussian_entropy_bias_correction(N,T):
    """Computes the bias of the entropy estimator of a 
    N-dimensional multivariate gaussian with T sample"""
    psiterms = sp.special.psi((T - np.arange(1,N+1))/2)
    return torch.tensor((N*np.log(2/(T-1)) + np.sum(psiterms))/2)


def _gaussian_entropy_estimation(cov_det, n_variables):
    return 0.5 * (n_variables*torch.log(TWOPIE) + torch.log(cov_det))


def _get_bias_correctors(T: Optional[List[int]], order: int, batch_size: int, D: int, device: torch.device):
    if T is not None:
        bc1 = torch.tensor([_gaussian_entropy_bias_correction(1,t) for t in T], device=device)
        bcN = torch.tensor([_gaussian_entropy_bias_correction(order,t) for t in T], device=device)
        bcNmin1 = torch.tensor([_gaussian_entropy_bias_correction(order-1,t) for t in T], device=device)
    else:
        bc1 = torch.tensor([0] * D, device=device)
        bcN = torch.tensor([0] * D, device=device)
        bcNmin1 = torch.tensor([0] * D, device=device)

    # Make bc1 a tensor with [*bc1, *bc1, ...] bach_size times to broadcast with the batched data
    bc1 = bc1.repeat(batch_size)
    bcN = bcN.repeat(batch_size)
    bcNmin1 = bcNmin1.repeat(batch_size)

    return bc1, bcN, bcNmin1


