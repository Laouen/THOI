from typing import Optional, List, Callable, Union

from tqdm import tqdm
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from thoi.typing import TensorLikeArray
from thoi.dataset import HotEncodedMultiOrderDataset
from thoi.collectors import batch_to_csv, concat_and_sort_csv
from thoi.measures.utils import _all_min_1_ids, \
                                _gaussian_entropy_bias_correction, \
                                _univariate_gaussian_entropy, \
                                _multivariate_gaussian_entropy, \
                                _get_single_exclusion_covmats
from thoi.measures.constants import GAUS_ENTR_NORMAL
from thoi.commons import _normalize_input_data


def _generate_nplets_covmants(covmats: torch.Tensor, nplets: torch.Tensor):

    # Ensure nplets is a float tensor
    # |batch_size| x |N|
    nplets = nplets.float()

    # Reshape covmats for broadcasting
    # |1| x |D| x |N| x |N|
    covmats_expanded = covmats.unsqueeze(0)

    # Create mask of the nplets
    # |batch_size| x |N| x |N|
    mask = nplets.unsqueeze(2) * nplets.unsqueeze(1)
    # |batch_size| x |1| x |N| x |N|
    mask = mask.unsqueeze(1)

    # Apply mask to covmats
    # |batch_size| x |D| x |N| x |N|
    masked_covmats = mask * covmats_expanded

    # Create diagonal mask for excluded variables
    # |batch_size| x |N|
    not_included = (nplets == 0).float()
    # |batch_size| x |N| x |N|
    diag_mask = torch.diag_embed(not_included)
    # |batch_size| x |1| x |N| x |N|
    diag_mask = diag_mask.unsqueeze(1)

    # Combine masked_covmats and diag_mask
    # |batch_size| x |D| x |N| x |N|
    nplets_covmat = masked_covmats + diag_mask
    
    return nplets_covmat


def _get_bias_correctors(T: Optional[List[int]], orders: torch.Tensor, batch_size: int, N: int, D: int, device: torch.device):
    
    
    # Compute bias corrector if from sampled data (T is not None)
    if T is not None:
        assert len(T) == D, 'T must have the same length as the number of datasets'
        # |N| x |D|
        bias_correctors = torch.tensor([
            [_gaussian_entropy_bias_correction(order,t) for t in T]
            for order in range(1,N+1)
        ], device=device)

    else:
        # |N| x |D|
        bias_correctors = torch.zeros((N,D), device=device)

    # |batch_size x D|
    bc1 = bias_correctors[0].repeat(batch_size)
    bcN = bias_correctors[orders-1].view(batch_size*D)
    bcNmin1 = bias_correctors[orders-2].view(batch_size*D)

    return bc1, bcN, bcNmin1


def _get_tc_dtc_from_batched_covmat(covmats: torch.Tensor, 
                                    nplets: torch.Tensor,
                                    orders: torch.Tensor,
                                    allmin1: List[np.ndarray],
                                    bc1: float, 
                                    bcOrder: float, 
                                    bcOrdermin1: float):
    '''
    Brief: Compute the total correlation (tc), dual total correlation (dtc), o-information (o) and s-information (s) for the given batch of covariance matrices.
    
    Parameters:
    - covmats (torch.Tensor): The covariance matrices with shape (batch_size*D, N, N)
    - nplets (torch.Tensor): The nplets to compute the measures with shape (batch_size, N)
    - orders (torch.Tensor): The order of each bached covmant (equivalent to covmat[...,0].sum(dim=1).int()), this must be provided to avoid multiple re calculations.
    - allmin1 (torch.Tensor): The indexes of marginal covariance matrices with shape (N, N-1)
    - bc1 (torch.Tensor): The bias corrector for the first order with shape (batch_size*D)
    - bcN (torch.Tensor): The bias corrector for the order with shape (batch_size*D)
    - bcNmin1 (torch.Tensor): The bias corrector for the order-1 with shape (batch_size*D)
    '''

    batch_size, N = covmats.shape[:2]
    D = batch_size // nplets.shape[0]
 
    # |batch_size|
    n_masked = N - orders
    
    # |batch_size| x |D| x |N|
    nplet_mask = nplets.unsqueeze(1).repeat(1, D, 1).view(batch_size, N)

    # Compute the sub covariance matrices for each variable and the system without that variable exclusion
    # |batch_size| x |N|
    single_var_covmats = torch.diagonal(covmats, dim1=-2, dim2=-1)
    # |batch_size| x |N| x |N-1| x |N-1|
    single_exclusion_covmats = _get_single_exclusion_covmats(covmats, allmin1)

    # Compute the entropy of the system, the variavbles and the system without the variable
    # |batch_size|
    sys_ent = _multivariate_gaussian_entropy(covmats, N) - bcOrder
    sys_ent.sub_(n_masked * GAUS_ENTR_NORMAL)
    # |batch_size| x |N|
    var_ents = _univariate_gaussian_entropy(single_var_covmats) - bc1.unsqueeze(1)
    var_ents.mul_(nplet_mask)

    # |batch_size| x |N|
    single_exclusion_ents = _multivariate_gaussian_entropy(single_exclusion_covmats, N-1) - bcOrdermin1.unsqueeze(1)
    single_exclusion_ents.sub_((n_masked * GAUS_ENTR_NORMAL).unsqueeze(1))
    single_exclusion_ents.mul_(nplet_mask)

    # |batch_size|
    nplet_tc = torch.sum(var_ents, dim=1) - sys_ent
    # TODO: inf - inf return NaN in pytorch. Check how should I handle this.
    # |batch_size|
    nplet_dtc = torch.sum(single_exclusion_ents, dim=1) - (orders-1.0)*sys_ent

    # |batch_size|
    nplet_o = nplet_tc - nplet_dtc
    # |batch_size|
    nplet_s = nplet_tc + nplet_dtc

    return nplet_tc, nplet_dtc, nplet_o, nplet_s


def _compute_nplets_measures_hot_encoded(covmats: torch.Tensor,
                                         T: Optional[int],
                                         N: int, D: int,
                                         nplets: Union[np.ndarray,torch.Tensor],
                                         allmin1: Optional[torch.Tensor] = None):
    batch_size = nplets.shape[0]
    device = covmats.device

    # Create the order of each nplet
    # |batch_size|
    orders = nplets.sum(dim=1).int()

    # |N| x |N-1|
    allmin1 = _all_min_1_ids(N, device=device) if allmin1 is None else allmin1

    # Create bias corrector values
    # |batch_size x D|, |batch_size x D|, |batch_size x D|
    bc1, bcN, bcNmin1 = _get_bias_correctors(T, orders, batch_size, N, D, device)

    # Create the covariance matrices for each nplet in the batch
    # |batch_size| x |D| x |N| x |N|
    nplets_covmat = _generate_nplets_covmants(covmats, nplets)

    # Pack covmats in a single batch
    # |batch_size x D| x |N| x |N|
    nplets_covmat = nplets_covmat.view(batch_size*D, N, N)

    # Batch process all nplets at once
    measures = _get_tc_dtc_from_batched_covmat(nplets_covmat,
                                               nplets,
                                               orders.repeat_interleave(D), # |batch_size x D|
                                               allmin1,
                                               bc1,
                                               bcN,
                                               bcNmin1)

    # Unpack results
    # |batch_size x D|, |batch_size x D|, |batch_size x D|, |batch_size x D|
    nplets_tc, nplets_dtc, nplets_o, nplets_s = measures
    
    return (
        nplets_tc.view(batch_size, D),
        nplets_dtc.view(batch_size, D),
        nplets_o.view(batch_size, D),
        nplets_s.view(batch_size, D)
    )

@torch.no_grad()
def nplets_measures_hot_encoded(X: TensorLikeArray,
                                nplets: Optional[TensorLikeArray] = None,
                                *,
                                covmat_precomputed: bool = False,
                                T: Optional[int] = None,
                                batch_size: int = 100000,
                                device: torch.device = torch.device('cpu')):
    '''
    Brief: Compute the higher order measurements (tc, dtc, o and s) for the given data matrices X over the nplets.
    
    Parameters:
    - X TensorLikeArray: The input data to compute the nplets. It can be a list of 2D numpy arrays or tensors of shape: 1. (T, N) where T is the number of samples if X are multivariate series. 2. a list of 2D covariance matrices with shape (N, N).
    - nplets (Optional[Union[np.ndarray,torch.Tensor]]): The nplets to calculate the measures with shape (batch_size, order)
    - covmat_precomputed (bool): A boolean flag to indicate if the input data is a list of covariance matrices or multivariate series.
    - T (Optional[Union[int, List[int]]]): A list of integers indicating the number of samples for each multivariate series.
    - device (torch.device): The device to use for the computation. Default is 'cpu'.
    - batch_size (int): Batch size for processing n-plets. Default is 100,000.
    
    Returns:
    - torch.Tensor: The measures for the nplets with shape (n_nplets, D, 4) where D is the number of matrices, n_nplets is the number of nplets to calculate over and 4 is the number of metrics (tc, dtc, o, s)
    '''

    covmats, D, N, T = _normalize_input_data(X, covmat_precomputed, T, device)

    if nplets is None:
        # If no nplets, then compute for the entire systems
        nplets = torch.ones(N, device=device).unsqueeze(0)
    else:
        # If nplets are not tensors, convert to tensor
        nplets = torch.as_tensor(nplets).to(device).contiguous()

    # nplets must be a batched tensor
    assert len(nplets.shape) == 2, 'nplets must be a batched tensor with shape (batch_size, order)'
    batch_size = min(batch_size, len(nplets))

    # Create DataLoader for nplets
    dataloader = DataLoader(nplets, batch_size=batch_size, shuffle=False)

    results = []
    for nplet_batch in tqdm(dataloader, desc='Processing n-plets', leave=False):

        # Batch process all nplets at once to obtain (tc, dtc, o, s)
        # |nplet_batch.shape[0]| x |D|, |nplet_batch.shape[0]| x |D|, |nplet_batch.shape[0] x D|, |nplet_batch.shape[0] x D|
        measures = _compute_nplets_measures_hot_encoded(covmats, T, N, D, nplet_batch)

        # Collect results
        results.append(torch.stack(measures, dim=-1))

    # Concatenate all results
    return torch.cat(results, dim=0)

@torch.no_grad()
def multi_order_measures_hot_encoded(X: TensorLikeArray,
                                     min_order: int=3,
                                     max_order: Optional[int]=None,
                                     *,
                                     covmat_precomputed: bool=False,
                                     T: Optional[int]=None,
                                     batch_size: int = 100000,
                                     device: torch.device = torch.device('cpu'),
                                     num_workers: int = 0,
                                     batch_aggregation: Optional[Callable[[any],any]] = None,
                                     batch_data_collector: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],any]] = None):
    """
    Compute multi-order Gaussian Copula (GC) measurements for the given data matrix X.
    The measurements computed are:
        * Total Correlation (TC)
        * Dual Total Correlation (DTC)
        * O-information (O)
        * S-information (S)

    Args:
        - X (np.ndarray or torch.Tensor): T samples x N variables matrix. if not covmat_precomputed, it should be a numpy array.
        - min_order (int): Minimum order to compute (default: 3).
        - max_order (Optional[int]): Maximum order to compute (default: None, will use N).
        - covmat_precomputed (bool): If True, X is a covariance matrix (default: False).
        - T (Optional[int]): Number of samples used to compute bias correction (default: None). This parameter is only used if covmat_precomputed is True.
        - batch_size (int): Batch size for DataLoader (default: 1000000).
        - device (torch.device): The device to use for the computation. Default is 'cpu'.
        - num_workers (int): Number of workers for DataLoader (default: 0).
        - batch_aggregation (Optional[Callable[[any],any]]): Function to aggregate the batched data (default: pd.concat).
        - batch_data_collector (Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],any]]): Function to collect the batched data (default: batch_to_csv).

    Returns:
        pd.DataFrame: DataFrame containing computed metrics.
    """
    
    covmats, D, N, T = _normalize_input_data(X, covmat_precomputed, T, device)

    max_order = N if max_order is None else max_order

    if batch_aggregation is None:
        batch_aggregation = concat_and_sort_csv

    if batch_data_collector is None:
        batch_data_collector = partial(batch_to_csv, indexing_method='hot_encoded', N=N)

    assert max_order <= N, f"max_order must be lower or equal than N. {max_order} > {N})"
    assert min_order <= max_order, f"min_order must be lower or equal than max_order. {min_order} > {max_order}"

    batch_size = batch_size // D

    # Create marginal indexes once to be reused
    # |N| x |N-1|
    allmin1 = _all_min_1_ids(N, device=device)

    dataset = HotEncodedMultiOrderDataset(N, min_order, max_order, device=device)
    dataloader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda' and dataset.device.type != 'cuda'
    )

    batched_data = []
    for bn, nplets in enumerate(tqdm(dataloader, total=len(dataloader), leave=False, desc='Batch')):
        
        # nplets = |batch_size| x |N|
        
        # Batch process all nplets at once to obtain (tc, dtc, o, s)
        # |batch_size| x |D|, |batch_size| x |D|, |batch_size| x |D|, |batch_size| x |D|
        measures = _compute_nplets_measures_hot_encoded(covmats, T, N, D, nplets, allmin1)

        batched_data.append(batch_data_collector(nplets, *measures, bn))

    return batch_aggregation(batched_data)