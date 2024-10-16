from typing import Optional, Callable, Union, List

from tqdm.autonotebook import tqdm
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from thoi.typing import TensorLikeArray
from thoi.commons import _normalize_input_data, _get_device
from thoi.dataset import CovarianceDataset
from thoi.collectors import batch_to_csv, concat_and_sort_csv
from thoi.measures.utils import _all_min_1_ids, _gaussian_entropy_estimation, \
                                _gaussian_entropy_bias_correction, \
                                _get_single_exclusion_covmats


def _generate_nplets_covmants(covmats: torch.Tensor, nplets: torch.Tensor):
    
    batch_size, order = nplets.shape
    D, N = covmats.shape[:2]
    
    # Step 1: Expand nplets to match the dimensions needed for batch indexing
    # nplets_expanded will be of shape (batch_size, D, order)
    nplets_expanded = nplets.unsqueeze(1).expand(-1, D, -1)

    # Step 2: Prepare covmats for batch indexing
    # We need to gather elements along the N dimension
    # First, expand covmats to shape (batch_size, D, N, N)
    covmats_expanded = covmats.unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Step 3: Gather the rows
    # indices_row will be of shape (batch_size, D, order, N)
    indices_row = nplets_expanded.unsqueeze(-1).expand(-1, -1, -1, N)
    gathered_rows = torch.gather(covmats_expanded, 2, indices_row)

    # Step 4: Gather the columns
    # indices_col will be of shape (batch_size, D, order, order)
    indices_col = nplets_expanded.unsqueeze(-2).expand(-1, -1, order, -1)
    nplets_covmat = torch.gather(gathered_rows, 3, indices_col)
    
    return nplets_covmat


def _get_bias_correctors(T: Optional[List[int]], order: int, batch_size: int, D: int, device: torch.device):
    if T is not None:
        # |batch_size|
        bc1 = torch.tensor([_gaussian_entropy_bias_correction(1,t) for t in T], device=device)
        bcN = torch.tensor([_gaussian_entropy_bias_correction(order,t) for t in T], device=device)
        bcNmin1 = torch.tensor([_gaussian_entropy_bias_correction(order-1,t) for t in T], device=device)
    else:
        # |batch_size|
        bc1 = torch.tensor([0] * D, device=device)
        bcN = torch.tensor([0] * D, device=device)
        bcNmin1 = torch.tensor([0] * D, device=device)

    # |batch_size x D|
    bc1 = bc1.repeat(batch_size)
    bcN = bcN.repeat(batch_size)
    bcNmin1 = bcNmin1.repeat(batch_size)

    return bc1, bcN, bcNmin1


def _get_tc_dtc_from_batched_covmat(covmats: torch.Tensor, allmin1: torch.Tensor, bc1: torch.Tensor, bcN: torch.Tensor, bcNmin1: torch.Tensor):

    # covmat is a batch of covariance matrices
    # |batch_size| x |N| x |N|

    batch_size, N = covmats.shape[:2]

    # Compute the sub covariance matrices for each variable and the system without that variable exclusion
    # |batch_size| x |N| x |1| x |1|
    single_var_covmats = torch.diagonal(covmats, dim1=-2, dim2=-1).view(batch_size, N, 1, 1)
    # |batch_size| x |N| x |N-1| x |N-1|
    single_exclusion_covmats = _get_single_exclusion_covmats(covmats, allmin1)

    # Compute the entropy of the system, the variavbles and the system without the variable
    # |batch_size|
    sys_ent = _gaussian_entropy_estimation(covmats, N) - bcN
    # TODO: This could be calculated once at the begining and then accessed here.
    # |batch_size| x |N|
    var_ents = _gaussian_entropy_estimation(single_var_covmats, 1) - bc1.unsqueeze(1)
    # |batch_size| x |N|
    single_exclusion_ents = _gaussian_entropy_estimation(single_exclusion_covmats, N-1) - bcNmin1.unsqueeze(1)

    # |batch_size|
    nplet_tc = torch.sum(var_ents, dim=1) - sys_ent
    # TODO: inf - inf return NaN in pytorch. Check how should I handle this.
    # |batch_size|
    nplet_dtc = torch.sum(single_exclusion_ents, dim=1) - (N-1.0)*sys_ent

    # |batch_size|
    nplet_o = nplet_tc - nplet_dtc
    # |batch_size|
    nplet_s = nplet_tc + nplet_dtc

    return nplet_tc, nplet_dtc, nplet_o, nplet_s

@torch.no_grad()
def nplets_measures(X: Union[TensorLikeArray],
                    nplets: Optional[TensorLikeArray] = None,
                    covmat_precomputed: bool = False,
                    T: Optional[Union[int, List[int]]] = None,
                    use_cpu: bool = False):
    
    '''
    Brief: Compute the higher order measurements (tc, dtc, o and s) for the given data matrices X over the nplets.
    
    Parameters:
    - X (Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]): The input data to compute the nplets. It can be a list of 2D numpy arrays or tensors of shape: 1. (T, N) where T is the number of samples if X are multivariate series. 2. a list of 2D covariance matrices with shape (N, N).
    - covmat_precomputed (bool): A boolean flag to indicate if the input data is a list of covariance matrices or multivariate series.
    - T (Optional[Union[int, List[int]]]): A list of integers indicating the number of samples for each multivariate series.
    - nplets (Optional[Union[np.ndarray,torch.Tensor]]): The nplets to calculate the measures with shape (batch_size, order)
    - use_cpu (bool): A boolean flag to indicate if the computation should be done on the CPU.
    
    Returns:
    - torch.Tensor: The measures for the nplets with shape (n_nplets, D, 4) where D is the number of matrices, n_nplets is the number of nplets to calculate over and 4 is the number of metrics (tc, dtc, o, s)
    '''
    
    covmats, D, N, T, device = _normalize_input_data(X, covmat_precomputed, T, use_cpu)

    # If no nplets, then compute for the entire systems
    if nplets is None:
        nplets = torch.arange(N, device=device).unsqueeze(0)
    else:
        # If nplets are not tensors, convert to tensor
        nplets = torch.as_tensor(nplets).to(device).contiguous()
        
    # nplets must be a batched tensor
    assert len(nplets.shape) == 2, 'nplets must be a batched tensor with shape (batch_size, order)'
    batch_size, order = nplets.shape

    # Create marginal indexes
    allmin1 = _all_min_1_ids(order, device=device)

    # Create bias corrector values
    # |batch_size x D|, |batch_size x D|, |batch_size x D|
    bc1, bcN, bcNmin1 = _get_bias_correctors(T, order, batch_size, D, device)

    # Create the covariance matrices for each nplet in the batch
    # |batch_size| x |D| x |N| x |N|
    nplets_covmats = _generate_nplets_covmants(covmats, nplets)

    # Pack covmat in a single batch
    # |batch_size x D| x |order| x |order|
    nplets_covmats = nplets_covmats.view(batch_size*D, order, order)

    # Batch process all nplets at once
    measures = _get_tc_dtc_from_batched_covmat(nplets_covmats,
                                               allmin1,
                                               bc1,
                                               bcN,
                                               bcNmin1)

    # Unpack results
    # |batch_size x D|, |batch_size x D|, |batch_size x D|, |batch_size x D|
    nplets_tc, nplets_dtc, nplets_o, nplets_s = measures

    # |batch_size| x |D| x |4 = (tc, dtc, o, s)|
    return torch.stack([nplets_tc.view(batch_size, D),
                        nplets_dtc.view(batch_size, D),
                        nplets_o.view(batch_size, D),
                        nplets_s.view(batch_size, D)], dim=-1)

@torch.no_grad()
def multi_order_measures(X: TensorLikeArray,
                         covmat_precomputed: bool=False,
                         T: Optional[Union[int, List[int]]]=None,
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

    Parameters:
        X (np.ndarray or torch.Tensor): (T samples x N variables) or (D datas x T samples x N variables) matrix.
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
        batch_data_collector = partial(batch_to_csv, N=N)

    assert max_order <= N, f"max_order must be lower or equal than N. {max_order} > {N})"
    assert min_order <= max_order, f"min_order must be lower or equal than max_order. {min_order} > {max_order}"

    # Ensure that final batch_size is smaller than the original batch_size 
    batch_size = batch_size // D

    # To compute using pytorch, we need to compute each order separately
    batched_data = []
    for order in tqdm(range(min_order, max_order+1), leave=False, desc='Order', disable=(min_order==max_order)):

        # Calculate constant values valid for all n-plets of the current order
        allmin1 = _all_min_1_ids(order, device=device)
        bc1, bcN, bcNmin1 = _get_bias_correctors(T, order, batch_size, D, device)

        # Generate dataset iterable
        dataset = CovarianceDataset(N, order, device=_get_device(use_cpu_dataset))
        dataloader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda' and dataset.device.type != 'cuda'
        )

        # calculate measurments for each batch
        for bn, nplets in enumerate(tqdm(dataloader, total=len(dataloader), leave=False, desc='Batch')):
            curr_batch_size = nplets.shape[0]

            # Send nplets to the device in case it is not there
            nplets = nplets.to(device)

            # Create the covariance matrices for each nplet in the batch
            # |curr_batch_size| x |D| x |N| x |N|
            nplets_covmats = _generate_nplets_covmants(covmats, nplets)

            # Pack covmats in a single batch
            # |curr_batch_size x D| x |N| x |N|
            nplets_covmats = nplets_covmats.view(curr_batch_size*D, order, order)

            # Batch process all nplets at once
            res = _get_tc_dtc_from_batched_covmat(nplets_covmats,
                                                  allmin1,
                                                  bc1[:curr_batch_size*D],
                                                  bcN[:curr_batch_size*D],
                                                  bcNmin1[:curr_batch_size*D])

            # Unpack results
            # |curr_batch_size x D|, |curr_batch_size x D|, |curr_batch_size x D|, |curr_batch_size x D|
            nplets_tc, nplets_dtc, nplets_o, nplets_s = res

            # Collect batch data
            data = batch_data_collector(nplets,
                                        nplets_tc.view(curr_batch_size, D),
                                        nplets_dtc.view(curr_batch_size, D),
                                        nplets_o.view(curr_batch_size, D),
                                        nplets_s.view(curr_batch_size, D),
                                        bn)

            # Append to batched data
            batched_data.append(data)

    # Aggregate all data
    return batch_aggregation(batched_data)
