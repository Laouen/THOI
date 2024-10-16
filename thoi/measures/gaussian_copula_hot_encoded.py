from typing import Optional, List, Callable, Union

from tqdm.autonotebook import tqdm
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from thoi.typing import TensorLikeArray
from thoi.dataset import HotEncodedMultiOrderDataset
from thoi.collectors import batch_to_csv, concat_and_sort_csv
from thoi.measures.utils import _all_min_1_ids, _gaussian_entropy_bias_correction, _gaussian_entropy_estimation, _get_single_exclusion_covmats
from thoi.measures.constants import GAUS_ENTR_NORMAL
from thoi.commons import _normalize_input_data, _get_device


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

    # covmat is a batch of covariance matrices
    # |batch_size| x |N| x |N|

    # nplets is a batch of nplets
    # |batch_size| x |N|

    batch_size, N = covmats.shape[:2]
 
    # |batch_size|
    n_masked = N - orders
    
    # |batch_size|
    nplet_mask = nplets.repeat(batch_size // nplets.shape[0], 1)

    # Compute the sub covariance matrices for each variable and the system without that variable exclusion
    # |batch_size| x |N| x |1| x |1|
    single_var_covmats = torch.diagonal(covmats, dim1=-2, dim2=-1).view(batch_size, N, 1, 1)
    # |batch_size| x |N| x |N-1| x |N-1|
    single_exclusion_covmats = _get_single_exclusion_covmats(covmats, allmin1)

    # Compute the entropy of the system, the variavbles and the system without the variable
    # |batch_size|
    sys_ent = _gaussian_entropy_estimation(covmats, N) - bcOrder
    sys_ent.sub_(n_masked * GAUS_ENTR_NORMAL)
    # TODO: This could be calculated once at the begining and then accessed here.
    # |batch_size| x |N|
    var_ents = _gaussian_entropy_estimation(single_var_covmats, 1) - bc1.unsqueeze(1)
    var_ents.mul_(nplet_mask)

    # |batch_size| x |N|
    single_exclusion_ents = _gaussian_entropy_estimation(single_exclusion_covmats, N-1) - bcOrdermin1.unsqueeze(1)
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
                                         nplets: Optional[Union[np.ndarray,torch.Tensor]],
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
                                               orders.repeat(D),
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
                                covmat_precomputed: bool = False,
                                T: Optional[int] = None,
                                use_cpu: bool = False):

    covmats, D, N, T, device = _normalize_input_data(X, covmat_precomputed, T, use_cpu)

    if nplets is None:
        # If no nplets, then compute for the entire systems
        nplets = torch.ones(N, device=device).unsqueeze(0)
    else:
        # If nplets are not tensors, convert to tensor
        nplets = torch.as_tensor(nplets).to(device).contiguous()

    # nplets must be a batched tensor
    assert len(nplets.shape) == 2, 'nplets must be a batched tensor with shape (batch_size, order)'
    
    # Batch process all nplets at once to obtain (tc, dtc, o, s)
    # |batch_size| x |D|, |batch_size| x |D|, |batch_size x D|, |batch_size x D|
    measures = _compute_nplets_measures_hot_encoded(covmats, T, N, D, nplets)
    
    # |batch_size| x |D| x |4 = (tc, dtc, o, s)|
    return torch.stack(measures, dim=-1)


@torch.no_grad()
def multi_order_measures_hot_encoded(X: TensorLikeArray,
                                     covmat_precomputed: bool=False,
                                     T: Optional[int]=None,
                                     min_order: int=3,
                                     max_order: Optional[int]=None,
                                     batch_size: int = 1000000,
                                     use_cpu: bool = False,
                                     use_cpu_dataset: bool = True,
                                     batch_aggregation: Optional[Callable[[any],any]] = None,
                                     batch_data_collector: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],any]] = None,
                                     num_workers: int = 0):
    """
    Compute multi-order Gaussian Copula (GC) measurements for the given data matrix X.
    The measurements computed are:
        * Total Correlation (TC)
        * Dual Total Correlation (DTC)
        * O-information (O)
        * S-information (S)

    Args:
        X (np.ndarray or torch.Tensor): T samples x N variables matrix. if not covmat_precomputed, it should be a numpy array.
        covmat_precomputed (bool): If True, X is a covariance matrix (default: False).
        T (Optional[int]): Number of samples used to compute bias correction (default: None). This parameter is only used if covmat_precomputed is True.
        min_order (int): Minimum order to compute (default: 3).
        max_order (Optional[int]): Maximum order to compute (default: None, will use N).
        batch_size (int): Batch size for DataLoader (default: 1000000).
        use_cpu (bool): If true, it forces to use CPU even if GPU is available (default: False).
        batch_aggregation (Optional[Callable[[any],any]]): Function to aggregate the batched data (default: pd.concat).
        batch_data_collector (Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],any]]): Function to collect the batched data (default: batch_to_csv).

    Returns:
        pd.DataFrame: DataFrame containing computed metrics.
    """
    
    covmats, D, N, T, device = _normalize_input_data(X, covmat_precomputed, T, use_cpu)

    max_order = N if max_order is None else max_order

    if batch_aggregation is None:
        batch_aggregation = concat_and_sort_csv

    if batch_data_collector is None:
        batch_data_collector = partial(batch_to_csv, indexing_method='hot_encoded', N=N)

    assert max_order <= N, f"max_order must be lower or equal than N. {max_order} > {N})"
    assert min_order <= max_order, f"min_order must be lower or equal than max_order. {min_order} > {max_order}"

    batch_size = batch_size // D
    print('Effective batch size:', batch_size*D, 'for', D, 'datasets with batch size', batch_size, 'each')

    # Create marginal indexes once to be reused
    allmin1 = _all_min_1_ids(N, device=device)

    dataset = HotEncodedMultiOrderDataset(N, min_order, max_order, device=_get_device(use_cpu_dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda' and dataset.device.type != 'cuda'
    )

    batched_data = []
    for bn, nplets in enumerate(tqdm(dataloader, total=len(dataloader), leave=False, desc='Batch')):
        
        # |batch_size| x |N| x |N|
        nplets = nplets.to(device, non_blocking=True)
        
        # Batch process all nplets at once to obtain (tc, dtc, o, s)
        # |batch_size| x |D|, |batch_size| x |D|, |batch_size| x |D|, |batch_size| x |D|
        measures = _compute_nplets_measures_hot_encoded(covmats, T, N, D, nplets, allmin1)

        batched_data.append(batch_data_collector(nplets, *measures, bn))

    return batch_aggregation(batched_data)