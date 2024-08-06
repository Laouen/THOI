from typing import Optional, Callable, Union

from tqdm.autonotebook import tqdm
from functools import partial

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from thoi.dataset import CovarianceDataset
from thoi.collectors import batch_to_csv
from thoi.measures.utils import _all_min_1_ids, _gaussian_entropy_bias_correction, _gaussian_entropy_estimation, gaussian_copula


def _get_tc_dtc_from_batched_covmat(covmat: torch.Tensor, allmin1, bc1: float, bcN: float, bcNmin1: float):

    # covmat is a batch of covariance matrices
    # |bz| x |N| x |N|

    n_variables = covmat.shape[2]

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
    sys_ent = _gaussian_entropy_estimation(batch_det, n_variables) - bcN
    # This could be calculated once at the begining and then accessed here.
    # |bz| x |N|
    var_ents = _gaussian_entropy_estimation(single_var_dets, 1) - bc1
    # |bz| x |N|
    single_expclusion_ents = _gaussian_entropy_estimation(single_exclusion_dets, n_variables-1) - bcNmin1

    # |bz|
    nplet_tc = torch.sum(var_ents, dim=1) - sys_ent
    # TODO: inf - inf return NaN in pytorch. Check how should I handle this.
    # |bz|
    nplet_dtc = torch.sum(single_expclusion_ents, dim=1) - (n_variables-1.0)*sys_ent

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
        nplets = torch.arange(N)

    # If nplets are not tensors, convert to tensor
    if not torch.is_tensor(nplets):
        nplets = torch.tensor(nplets)

    # nplets must be a batched tensor
    assert len(nplets.shape) == 2, 'nplets must be a batched tensor with shape (batch_size, order)'

    # Process in correct device
    # Send elements to cuda if computing on GPU
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')
    covmat = covmat.to(device).contiguous()
    nplets = nplets.to(device).contiguous()

    # Generate the covariance matrices for each nplet
    # |batch_size| x |order| x |order|
    nplets_covmat = torch.stack([
        covmat[nplet_idxs][:,nplet_idxs]
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
    

def multi_order_measures(X: Union[np.ndarray, torch.tensor],
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
        X (np.ndarray or torch.tensor): T samples x N variables matrix. if not covmat_precomputed, it should be a numpy array.
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
        covmat = gaussian_copula(X)[1]

    max_order = N if max_order is None else max_order

    if batch_aggregation is None:
        batch_aggregation = pd.concat

    if batch_data_collector is None:
        batch_data_collector = partial(batch_to_csv, N=N)

    assert max_order <= N, f"max_order must be lower or equal than N. {max_order} > {N})"
    assert min_order <= max_order, f"min_order must be lower or equal than max_order. {min_order} > {max_order}"

    # make device cpu if not cuda available or cuda if available
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')

    # To compute using pytorch, we need to compute each order separately
    batched_data = []
    pbar_order = tqdm(range(min_order, max_order+1), leave=False, desc='Order', disable=(min_order==max_order))
    for order in pbar_order:

        # Calculate constant values valid for all n-plets of the current order
        allmin1 = _all_min_1_ids(order)
        if T is not None:
            bc1 = _gaussian_entropy_bias_correction(1,T)
            bcN = _gaussian_entropy_bias_correction(order,T)
            bcNmin1 = _gaussian_entropy_bias_correction(order-1,T)
        else:
            bc1 = 0
            bcN = 0
            bcNmin1 = 0

        # Generate dataset iterable
        dataset = CovarianceDataset(covmat, N, order)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0, 
            pin_memory=using_GPU
        )

        # calculate measurments for each batch
        pbar = tqdm(dataloader, total=len(dataloader), leave=False, desc='Batch')
        for bn, (partition_idxs, partition_covmat) in enumerate(pbar):
            partition_covmat = partition_covmat.to(device)
            nplets_tc, nplets_dtc, nplets_o, nplets_s = _get_tc_dtc_from_batched_covmat(
                partition_covmat, allmin1, bc1, bcN, bcNmin1
            )

            data = batch_data_collector(
                partition_idxs,
                nplets_o,
                nplets_s,
                nplets_tc,
                nplets_dtc,
                bn
            )
            batched_data.append(data)

    return batch_aggregation(batched_data)