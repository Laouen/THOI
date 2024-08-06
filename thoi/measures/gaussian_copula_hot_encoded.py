from typing import Optional, List, Union

import numpy as np
import torch

from thoi.measures.utils import _all_min_1_ids, _gaussian_entropy_bias_correction, _gaussian_entropy_estimation, gaussian_copula
from thoi.measures.constants import GAUS_ENTR_NORMAL


def _get_tc_dtc_from_batched_covmat(covmat: torch.Tensor, 
                                    nplets: torch.Tensor,
                                    allmin1: List[np.ndarray],
                                    bc1: float, 
                                    bcOrder: float, 
                                    bcOrdermin1: float):

    # covmat is a batch of covariance matrices
    # |bz| x |N| x |N|

    # nplets is a batch of nplets
    # |bz| x |N|

    N = covmat.shape[2]
    order = nplets.sum(dim=1).int()
    n_masked = N - order

    # |bz|
    batch_det = torch.linalg.det(covmat)
    # |bz| x |N|
    single_var_dets = torch.diagonal(covmat, dim1=-2, dim2=-1)
    # |bz| x |N|
    single_exclusion_dets = torch.stack([
        torch.linalg.det(covmat[:,ids].contiguous()[:,:,ids].contiguous())
        for ids in allmin1
    ], dim=1)

    # |bz|
    sys_ent = _gaussian_entropy_estimation(batch_det, N) - bcOrder - (n_masked * GAUS_ENTR_NORMAL)
    # This could be calculated once at the begining and then accessed here.
    # |bz| x |N|
    var_ents = _gaussian_entropy_estimation(single_var_dets, 1) - bc1
    var_ents = var_ents * nplets # Set as zero the elements to ignore (nplets = 0)
    # |bz| x |N|
    single_expclusion_ents = _gaussian_entropy_estimation(single_exclusion_dets, N-1) - bcOrdermin1[:,None] - (n_masked * GAUS_ENTR_NORMAL)[:,None]
    single_expclusion_ents = single_expclusion_ents * nplets

    # |bz|
    nplet_tc = torch.sum(var_ents, dim=1) - sys_ent
    # TODO: inf - inf return NaN in pytorch. Check how should I handle this.
    # |bz|
    nplet_dtc = torch.sum(single_expclusion_ents, dim=1) - (order-1.0)*sys_ent

    # |bz|
    nplet_o = nplet_tc - nplet_dtc
    # |bz|
    nplet_s = nplet_tc + nplet_dtc

    return nplet_tc, nplet_dtc, nplet_o, nplet_s


def nplets_measures_hot_encoded(X: Union[np.ndarray, torch.tensor],
                    nplets: Optional[Union[np.ndarray,torch.tensor]] = None,
                    T:Optional[int] = None,
                    covmat_precomputed:bool = False,
                    bias_correctors: Optional[torch.tensor] = None,
                    use_cpu:bool = False):

    # Handle different options for X parameter
    # Accept multivariate data or covariance matrix
    if covmat_precomputed:
        N1, N = X.shape
        assert N1 == N, 'Covariance matrix should be a squared matrix'
        covmat = X if torch.is_tensor(X) else torch.tensor(X)
    else:
        assert not torch.is_tensor(X), 'Not precomputed covariance should be numpys'
        T, N = X.shape
        covmat = torch.tensor(gaussian_copula(X)[1])

    # Handle different options for nplet parameter
    # Compute for the entire systems
    if nplets is None:
        nplets = torch.ones(N)

    # If nplets are not tensors, convert to tensor
    if not torch.is_tensor(nplets):
        nplets = torch.tensor(nplets)

    # If only nplet to calculate
    if len(nplets.shape) < 2:
        nplets = torch.unsqueeze(nplets, dim=0)

    # Process in correct device
    # Send elements to cuda if computing on GPU
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')
    covmat = covmat.to(device).contiguous()
    nplets = nplets.to(device).contiguous()

    # TODO: take identity matrices as parameter
    # Generate the covariance matrices for each nplet
    # |batch_size| x |order| x |order|
    identity_matrices = torch.eye(N, device=device).contiguous().unsqueeze(0).repeat(nplets.shape[0], 1, 1)
    diag_mask = (nplets == 0).unsqueeze(2).repeat(1, 1, N) * identity_matrices
    mask = nplets.unsqueeze(2) * nplets.unsqueeze(1)
    nplets_covmat = (mask * covmat) + diag_mask

    # Compute measures
    orders = nplets.sum(dim=1).int()
    allmin1 = _all_min_1_ids(N)
    
    # Compute bias corrector if from sampled data (T is not None)
    if bias_correctors is None:
        if T is not None:
            bias_correctors = torch.tensor([
                _gaussian_entropy_bias_correction(order,T)
                for order in range(1,N+1)
            ], device=device)
            
        else: 
            bias_correctors = torch.zeros(N, device=device)

    bc1 = bias_correctors[0]
    bcN = bias_correctors[orders-1]
    bcNmin1 = bias_correctors[orders-2]

    # Batch process all nplets at once
    # batch_res = (nplet_tc, nplet_dtc, nplet_o, nplet_s)
    results = torch.stack(_get_tc_dtc_from_batched_covmat(
        nplets_covmat, nplets, allmin1, bc1, bcN, bcNmin1
    )).T

    # If only one result, return is as value not list
    if results.shape[0] == 1:
        results = torch.squeeze(results, dim=0)

    return results
    
