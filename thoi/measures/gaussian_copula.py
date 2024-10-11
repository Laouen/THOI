from typing import Optional, Callable, Union, List

from tqdm.autonotebook import tqdm
from functools import partial

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from thoi.dataset import CovarianceDataset
from thoi.collectors import batch_to_csv, concat_and_sort_csv
from thoi.measures.utils import _all_min_1_ids, _gaussian_entropy_bias_correction, _gaussian_entropy_estimation, gaussian_copula, to_numpy

# TODO: make allmin1 a torch tensor as well
def _get_tc_dtc_from_batched_covmat(covmats: torch.tensor, allmin1: List[np.ndarray], bc1: torch.tensor, bcN: torch.tensor, bcNmin1: torch.tensor):

    # covmat is a batch of covariance matrices
    # |bz| x |N| x |N|

    n_variables = covmats.shape[2]

    # |bz|
    batch_det = torch.linalg.det(covmats)
    # |bz| x |N|
    single_var_dets = torch.diagonal(covmats, dim1=-2, dim2=-1)
    # |bz| x |N|
    single_exclusion_dets = torch.stack([
        torch.linalg.det(covmats[:,ids].contiguous()[:,:,ids].contiguous())
        for ids in allmin1
    ], dim=1)

    # |bz|
    sys_ent = _gaussian_entropy_estimation(batch_det, n_variables) - bcN
    # This could be calculated once at the begining and then accessed here.
    # |bz| x |N|
    var_ents = _gaussian_entropy_estimation(single_var_dets, 1) - bc1.unsqueeze(1)
    # |bz| x |N|
    single_exclusion_ents = _gaussian_entropy_estimation(single_exclusion_dets, n_variables-1) - bcNmin1.unsqueeze(1)

    # |bz|
    nplet_tc = torch.sum(var_ents, dim=1) - sys_ent
    # TODO: inf - inf return NaN in pytorch. Check how should I handle this.
    # |bz|
    nplet_dtc = torch.sum(single_exclusion_ents, dim=1) - (n_variables-1.0)*sys_ent

    # |bz|
    nplet_o = nplet_tc - nplet_dtc
    # |bz|
    nplet_s = nplet_tc + nplet_dtc

    return nplet_tc, nplet_dtc, nplet_o, nplet_s


def nplets_measures(X: Union[np.ndarray, torch.tensor],
                    nplets: Optional[Union[np.ndarray,torch.tensor]] = None,
                    T:Optional[int] = None,
                    covmat_precomputed:bool = False,
                    use_cpu:bool = False):

    # check the number of dimensions of X
    assert len(X.shape) <= 3, 'X must be a 2D or 3D matrix'

    if len(X.shape) == 2:
        # add a single dimension to the data
        X = X.unsqueeze(0) if torch.is_tensor(X) else np.expand_dims(X, axis=0)

    # Handle different options for X parameter
    # Accept multivariate data or covariance matrix
    if covmat_precomputed:
        _, N1, N = X.shape
        assert N1 == N, 'Precomputed covariance matrix should be a squared matrix'
        covmats = X if torch.is_tensor(X) else torch.tensor(X)
    else:
        _, T, N = X.shape
        covmats = torch.tensor([
            gaussian_copula(X_i)[1]
            for X_i in to_numpy(X)
        ])

    # Handle different options for nplet parameter
    # Compute for the entire systems
    if nplets is None:
        nplets = torch.arange(N).unsqueeze(0)

    # If nplets are not tensors, convert to tensor
    if not torch.is_tensor(nplets):
        nplets = torch.tensor(nplets)

    # nplets must be a batched tensor
    assert len(nplets.shape) == 2, 'nplets must be a batched tensor with shape (batch_size, order)'

    # Process in correct device
    # Send elements to cuda if computing on GPU
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')
    covmats = covmats.to(device).contiguous()
    nplets = nplets.to(device).contiguous()

    # Generate the covariance matrices for each nplet
    # |batch_size| x |order| x |order|
    nplets_covmat = torch.stack([
        covmat[nplet_idxs][:,nplet_idxs]
        for covmat in covmats
        for nplet_idxs in nplets
    ])

    # Compute measures
    _, order = nplets.shape
    allmin1 = _all_min_1_ids(order)
    
    # Compute bias corrector if from sampled data (T is not None)
    if T is not None:
        bc1 = _gaussian_entropy_bias_correction(1,T)
        bcN = _gaussian_entropy_bias_correction(order,T)
        bcNmin1 = _gaussian_entropy_bias_correction(order-1,T)
    else: 
        bc1 = 0
        bcN = 0
        bcNmin1 = 0

    # Batch process all nplets at once
    # batch_res = (nplet_tc, nplet_dtc, nplet_o, nplet_s)
    results = torch.stack(_get_tc_dtc_from_batched_covmat(
        nplets_covmat, allmin1, bc1, bcN, bcNmin1
    )).T

    return results


def multi_order_measures(X: Union[np.ndarray, torch.tensor, List[np.ndarray], List[torch.tensor]],
                         covmat_precomputed: bool=False,
                         T: Optional[Union[int, List[int]]]=None,
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
        X (np.ndarray or torch.tensor): (T samples x N variables) or (D datas x T samples x N variables) matrix.
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

    # Force data to be a list of data
    if isinstance(X, (np.ndarray, torch.Tensor)):
        X = [X]

    if isinstance(T, int):
        T = [T] * len(X)

    # check the number of dimensions of X
    assert all([len(x.shape) == 2 for x in X]), 'X must be all 2D timseseries or covariance matrices'

    # Handle different options for X parameter. Accept multivariate data or covariance matrix
    if covmat_precomputed:
        assert all([x.shape[0] == x.shape[1] == X[0].shape[0] for x in X]), 'All covariance matrices should be square and of the same dimension'
        N = X[0].shape[1]
        covmats = torch.stack([x if torch.is_tensor(x) else torch.tensor(x) for x in X])
    else:
        T = [x.shape[0] for x in X]
        N = X[0].shape[1]
        covmats = torch.stack([torch.tensor(gaussian_copula(to_numpy(x))[1]) for x in X])
    
    D = covmats.shape[0]

    max_order = N if max_order is None else max_order

    if batch_aggregation is None:
        batch_aggregation = concat_and_sort_csv

    if batch_data_collector is None:
        batch_data_collector = partial(batch_to_csv, N=N)

    assert max_order <= N, f"max_order must be lower or equal than N. {max_order} > {N})"
    assert min_order <= max_order, f"min_order must be lower or equal than max_order. {min_order} > {max_order}"

    # make device cpu if not cuda available or cuda if available
    using_GPU = (not use_cpu) and torch.cuda.is_available()
    device = torch.device('cuda' if using_GPU else 'cpu')

    batch_size = batch_size // D
    print('Effective batch size:', batch_size*D, 'for', D, 'datasets with batch size', batch_size, 'each')

    # To compute using pytorch, we need to compute each order separately
    batched_data = []
    for order in tqdm(range(min_order, max_order+1), leave=False, desc='Order', disable=(min_order==max_order)):

        # Calculate constant values valid for all n-plets of the current order
        allmin1 = _all_min_1_ids(order)
        if T is not None:
            bc1 = torch.stack([_gaussian_entropy_bias_correction(1,t) for t in T])
            bcN = torch.stack([_gaussian_entropy_bias_correction(order,t) for t in T])
            bcNmin1 = torch.stack([_gaussian_entropy_bias_correction(order-1,t) for t in T])
        else:
            bc1 = torch.stack([0] * len(X))
            bcN = torch.stack([0] * len(X))
            bcNmin1 = torch.stack([0] * len(X))
        
        # Make bc1 a tensor with [*bc1, *bc1, ...] bach_size times to broadcast with the batched data
        bc1 = bc1.repeat(batch_size)
        bcN = bcN.repeat(batch_size)
        bcNmin1 = bcNmin1.repeat(batch_size)
        
        # Generate dataset iterable
        dataset = CovarianceDataset(covmats, order)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0, 
            pin_memory=using_GPU
        )

        # calculate measurments for each batch
        for bn, (partition_idxs, partition_covmats) in enumerate(tqdm(dataloader, total=len(dataloader), leave=False, desc='Batch')):
            curr_bs = partition_covmats.shape[0]
            partition_covmats = partition_covmats.to(device).view(curr_bs*D, order, order)

            # Compute measures
            res = _get_tc_dtc_from_batched_covmat(partition_covmats,
                                                  allmin1[:curr_bs*D],
                                                  bc1[:curr_bs*D],
                                                  bcN[:curr_bs*D],
                                                  bcNmin1[:curr_bs*D])

            # Unpack results
            nplets_tc, nplets_dtc, nplets_o, nplets_s = res

            # Collect batch data
            data = batch_data_collector(partition_idxs,
                                        nplets_o.view(curr_bs, D),
                                        nplets_s.view(curr_bs, D),
                                        nplets_tc.view(curr_bs, D),
                                        nplets_dtc.view(curr_bs, D),
                                        bn)

            # Append to batched data
            batched_data.append(data)

    # Aggregate all data
    return batch_aggregation(batched_data)
