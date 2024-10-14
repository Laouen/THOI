from typing import Optional, List, Callable, Union

from thoi.commons import gaussian_copula_covmat
from tqdm.autonotebook import tqdm
from functools import partial

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from thoi.dataset import HotEncodedMultiOrderDataset
from thoi.collectors import batch_to_csv
from thoi.measures.utils import _all_min_1_ids, _gaussian_entropy_bias_correction, _gaussian_entropy_estimation
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
    sys_ent = _gaussian_entropy_estimation(batch_det, N) - bcOrder
    sys_ent = sys_ent - (n_masked * GAUS_ENTR_NORMAL)
    # This could be calculated once at the begining and then accessed here.
    # |bz| x |N|
    var_ents = _gaussian_entropy_estimation(single_var_dets, 1) - bc1
    var_ents = var_ents * nplets # Set as zero the elements to ignore (nplets = 0)
    # |bz| x |N|
    single_exclusion_ents = _gaussian_entropy_estimation(single_exclusion_dets, N-1) - bcOrdermin1[:,None]
    single_exclusion_ents = single_exclusion_ents - (n_masked * GAUS_ENTR_NORMAL)[:,None]
    single_exclusion_ents = single_exclusion_ents * nplets

    # |bz|
    nplet_tc = torch.sum(var_ents, dim=1) - sys_ent
    # TODO: inf - inf return NaN in pytorch. Check how should I handle this.
    # |bz|
    nplet_dtc = torch.sum(single_exclusion_ents, dim=1) - (order-1.0)*sys_ent

    # |bz|
    nplet_o = nplet_tc - nplet_dtc
    # |bz|
    nplet_s = nplet_tc + nplet_dtc

    return nplet_tc, nplet_dtc, nplet_o, nplet_s

@torch.no_grad()
def nplets_measures_hot_encoded(X: Union[np.ndarray, torch.Tensor],
                    nplets: Optional[Union[np.ndarray,torch.Tensor]] = None,
                    T:Optional[int] = None,
                    covmat_precomputed:bool = False,
                    bias_correctors: Optional[torch.Tensor] = None,
                    use_cpu:bool = False):

    # Handle different options for X parameter
    # Accept multivariate data or covariance matrix
    if covmat_precomputed:
        N1, N = X.shape
        assert N1 == N, 'Covariance matrix should be a squared matrix'
        covmat = torch.as_tensor(X)
    else:
        assert not torch.is_tensor(X), 'Not precomputed covariance should be numpys'
        T, N = X.shape
        covmat = torch.from_numpy(gaussian_copula_covmat(X))

    # Handle different options for nplet parameter
    # Compute for the entire systems
    if nplets is None:
        nplets = torch.ones(N)

    # If nplets are not tensors, convert to tensor
    nplets = torch.as_tensor(nplets)

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
    # |batch_size| x |N| x |N|
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

@torch.no_grad()
def multi_order_measures_hot_encoded(X: Union[np.ndarray, torch.Tensor],
                         covmat_precomputed: bool=False,
                         T: Optional[int]=None,
                         min_order: int=3,
                         max_order: Optional[int]=None,
                         batch_size: int = 1000000,
                         use_cpu: bool = False,
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

    # Handle different options for X parameter. Accept multivariate data or covariance matrix
    if covmat_precomputed:
        N1, N = X.shape
        assert N1 == N, 'Covariance matrix should be a squared matrix'
        covmat = X
    else:
        assert not torch.is_tensor(X), 'Not precomputed covariance should be numpys'
        T, N = X.shape
        covmat = gaussian_copula_covmat(X)

    max_order = N if max_order is None else max_order

    if batch_aggregation is None:
        batch_aggregation = pd.concat

    if batch_data_collector is None:
        batch_data_collector = partial(batch_to_csv, indexing_method='hot_encoded', N=N)

    assert max_order <= N, f"max_order must be lower or equal than N. {max_order} > {N})"
    assert min_order <= max_order, f"min_order must be lower or equal than max_order. {min_order} > {max_order}"

    # make device cpu if not cuda available or cuda if available
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')

    # Create the reusable values:
    # * Identity matrix to add the independent variables
    # * All the combinations of N-1 elements
    identity_matrices = torch.eye(N, device=device).contiguous().unsqueeze(0).repeat(batch_size, 1, 1)
    allmin1 = _all_min_1_ids(N)
    
    # Compute bias corrector if from sampled data (T is not None)
    if T is not None:
        bias_correctors = torch.tensor([
            _gaussian_entropy_bias_correction(order,T)
            for order in range(1,N+1)
        ], device=device)

    else: 
        bias_correctors = torch.zeros(N, device=device)

    bc1 = bias_correctors[0]

    dataset = HotEncodedMultiOrderDataset(covmat, min_order, max_order)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0, 
        pin_memory=using_GPU
    )
    pbar = tqdm(dataloader, total=len(dataloader), leave=False, desc='Batch')
    batched_data = []
    for bn, nplets_idxs_hot_encoded in enumerate(pbar):
        # |batch_size| x |N| x |N|
        nplets_idxs_hot_encoded = nplets_idxs_hot_encoded.to(device)
        
        # last batch can have less items than batch_size
        if (nplets_idxs_hot_encoded.shape[0] < batch_size):
            identity_matrices = torch.eye(N, device=device).contiguous().unsqueeze(0).repeat(nplets_idxs_hot_encoded.shape[0], 1, 1)
        
        diag_mask = (nplets_idxs_hot_encoded == 0).unsqueeze(2).repeat(1, 1, N) * identity_matrices
        mask = nplets_idxs_hot_encoded.unsqueeze(2) * nplets_idxs_hot_encoded.unsqueeze(1)
        partition_covmat = (mask * covmat) + diag_mask
        
        # Compute measures
        orders = nplets_idxs_hot_encoded.sum(dim=1).int()
        
        bcN = bias_correctors[orders-1]
        bcNmin1 = bias_correctors[orders-2]
        
        # Batch process all nplets at once
        nplets_tc, nplets_dtc, nplets_o, nplets_s = _get_tc_dtc_from_batched_covmat(
            partition_covmat, nplets_idxs_hot_encoded, allmin1, bc1, bcN, bcNmin1
        )
        
        data = batch_data_collector(
            nplets_idxs_hot_encoded,
            nplets_tc,
            nplets_dtc,
            nplets_o,
            nplets_s,
            bn
        )
        batched_data.append(data)

    return batch_aggregation(batched_data)