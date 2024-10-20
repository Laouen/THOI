import numpy as np
import scipy as sp
import torch

from thoi.measures.constants import LOGTWOPIE, TWOPIE


def _all_min_1_ids(N: int, device: torch.device=torch.device('cpu')):
    # |N| x |N|
    base_tensor = torch.arange(N, device=device).unsqueeze(0).repeat(N, 1)
    # |N| x |N|
    mask = base_tensor != torch.arange(N, device=device).unsqueeze(1)
    # |N| x |N-1|
    return base_tensor[mask].view(N, N - 1)


def _gaussian_entropy_bias_correction(N,T):
    """Computes the bias of the entropy estimator of a 
    N-dimensional multivariate gaussian with T sample"""
    psiterms = sp.special.psi((T - np.arange(1,N+1))/2)
    return torch.tensor((N*np.log(2/(T-1)) + np.sum(psiterms))/2)


def _marginal_gaussian_entropies(covmats: torch.Tensor):
    D, N = covmats.shape[:2]
    return _univariate_gaussian_entropy(
        torch.diagonal(covmats, dim1=-2, dim2=-1))


def _univariate_gaussian_entropy(variances):
    return 0.5 * (torch.log(TWOPIE * variances))


def _multivariate_gaussian_entropy(covmats, N):
    return 0.5 * (N*LOGTWOPIE + torch.logdet(covmats))


def _get_single_exclusion_covmats(covmats: torch.Tensor, allmin1: torch.Tensor):

    batch_size, N, _ = covmats.shape

    # Step 1: Expand allmin1 to match the batch size
    # |batch_size| |N| |N-1|
    allmin1_expanded = allmin1.unsqueeze(0).expand(batch_size, -1, -1)

    # Step 2: Expand covmats to include the N dimension for variable exclusion
    # |batch_size| |N| |N| |N|
    covmats_expanded = covmats.unsqueeze(1).expand(-1, N, -1, -1)

    # Step 3: Gather the rows corresponding to the indices in allmin1
    # |batch_size| |N| |N-1| |N|
    indices_row = allmin1_expanded.unsqueeze(-1).expand(-1, -1, -1, N)
    gathered_rows = torch.gather(covmats_expanded, 2, indices_row)

    # Step 4: Gather the columns corresponding to the indices in allmin1
    # Shape of indices_col: (batch_size, N, N-1, N-1)
    # |batch_size| |N| |N-1| |N-1|
    indices_col = allmin1_expanded.unsqueeze(-2).expand(-1, -1, N-1, -1)
    covmats_sub = torch.gather(gathered_rows, 3, indices_col)

    # |bz| x |N| x |N-1| x |N-1|
    return covmats_sub
