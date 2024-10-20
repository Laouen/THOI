from typing import Optional, Callable, Union, List

from tqdm import tqdm
from functools import partial
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from thoi.typing import TensorLikeArray
from thoi.commons import _normalize_input_data
from thoi.dataset import CovarianceDataset
from thoi.collectors import batch_to_csv, concat_and_sort_csv
from thoi.measures.gaussian_copula_hot_encoded import nplets_measures_hot_encoded
from thoi.measures.utils import _all_min_1_ids, \
                                _multivariate_gaussian_entropy, \
                                _univariate_gaussian_entropy, \
                                _marginal_gaussian_entropies, \
                                _gaussian_entropy_bias_correction, \
                                _get_single_exclusion_covmats

def _indices_to_hot_encoded(nplets_idxs, N):
    """
    Converts a list of index lists to a multi-hot encoded tensor using one_hot and masking.

    Parameters:
    - nplets_idxs: List of lists of indices.
    - N: Number of classes.

    Returns:
    - multi_hot: torch.Tensor of shape (batch_size, N)
    """
    return torch.stack([
        F.one_hot(torch.as_tensor(lst), num_classes=N).sum(dim=0)
        for lst in nplets_idxs
    ])


def _generate_nplets_marginal_entropies(marginal_entropies: torch.Tensor, nplets: torch.Tensor):
    
    D, N = marginal_entropies.shape
    batch_size, order = nplets.shape
    
    # Expand entropies
    # |batch_size| x |D| x |N|
    entropies_expanded = marginal_entropies.unsqueeze(0).expand(batch_size, D, N)

    # Expand nplets and repeat them across the D dimensions
    # |batch_size| x |D| x |order|
    nplets_expanded = nplets.unsqueeze(1).expand(batch_size, D, order)

    # Gather the entropies based on nplets indices
    # |batch_size| x |D| x |order|
    nplets_marginal_entropies = torch.gather(entropies_expanded, dim=2, index=nplets_expanded)

     # |batch_size| x |D| x |order|
    return nplets_marginal_entropies
    

def _generate_nplets_covmants(covmats: torch.Tensor, nplets: torch.Tensor):
    
    batch_size, order = nplets.shape
    D, N = covmats.shape[:2]
    
    # Expand nplets to match the dimensions needed for batch indexing
    # |batch_size| x |D| x |order|
    nplets_expanded = nplets.unsqueeze(1).expand(-1, D, -1)

    # Prepare covmats for batch indexing to gather elements along the N dimension
    # |batch_size| x |D| x |N| x |N|
    covmats_expanded = covmats.unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Gather the rows
    # |batch_size| x |D| x |order| x |N|
    indices_row = nplets_expanded.unsqueeze(-1).expand(-1, -1, -1, N)
    gathered_rows = torch.gather(covmats_expanded, 2, indices_row)

    # Gather the columns
    # |batch_size| x |D| x |order| x |order|
    indices_col = nplets_expanded.unsqueeze(-2).expand(-1, -1, order, -1)
    nplets_covmat = torch.gather(gathered_rows, 3, indices_col)

    # |batch_size| x |D| x |order| x |order|
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


def _get_tc_dtc_from_batched_covmat(covmats: torch.Tensor,
                                    allmin1: torch.Tensor,
                                    bc1: torch.Tensor,
                                    bcN: torch.Tensor,
                                    bcNmin1: torch.Tensor,
                                    marginal_entropies: Optional[torch.Tensor] = None):

    '''
    Brief: Compute the total correlation (tc), dual total correlation (dtc), o-information (o) and s-information (s) for the given batch of covariance matrices.
    
    Parameters:
    - covmats (torch.Tensor): The covariance matrices with shape (batch_size, N, N)
    - allmin1 (torch.Tensor): The indexes of marginal covariance matrices with shape (batch_size, N, N-1)
    - bc1 (torch.Tensor): The bias corrector for the first order with shape (batch_size)
    - bcN (torch.Tensor): The bias corrector for the order with shape (batch_size)
    - bcNmin1 (torch.Tensor): The bias corrector for the order-1 with shape (batch_size)
    - marginal_entropies (Optional[torch.Tensor]): The marginal entropies for each variable with shape (batch_size, N). If None, it will be dynamically computed
    '''

    N = covmats.shape[1]

    # Compute the entire system entropy
    # |batch_size|
    sys_ent = _multivariate_gaussian_entropy(covmats, N) - bcN
    
    # Compute the single variables entropy
    # |batch_size| x |N|
    if marginal_entropies is None:
        single_var_variances = torch.diagonal(covmats, dim1=-2, dim2=-1)
        marginal_entropies = _univariate_gaussian_entropy(single_var_variances)
    marginal_entropies.sub_(bc1.unsqueeze(1))
    
    # Compute the single exclusion entropies
    # |batch_size| x |N|
    single_exclusion_covmats = _get_single_exclusion_covmats(covmats, allmin1)
    single_exclusion_ents = _multivariate_gaussian_entropy(single_exclusion_covmats, N-1) - bcNmin1.unsqueeze(1)

    # |batch_size|
    nplet_tc = torch.sum(marginal_entropies, dim=1) - sys_ent
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
                    *,
                    covmat_precomputed: bool = False,
                    T: Optional[Union[int, List[int]]] = None,
                    device: torch.device = torch.device('cpu'),
                    verbose: int = logging.INFO):
    
    '''
    Brief: Compute the higher order measurements (tc, dtc, o and s) for the given data matrices X over the nplets.
    
    Parameters:
    - X (Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]): The input data to compute the nplets. It can be a list of 2D numpy arrays or tensors of shape: 1. (T, N) where T is the number of samples if X are multivariate series. 2. a list of 2D covariance matrices with shape (N, N).
    - nplets (Optional[Union[np.ndarray,torch.Tensor]]): The nplets to calculate the measures with shape (batch_size, order)
    - covmat_precomputed (bool): A boolean flag to indicate if the input data is a list of covariance matrices or multivariate series.
    - T (Optional[Union[int, List[int]]]): A list of integers indicating the number of samples for each multivariate series.
    - device (torch.device): The device to use for the computation. Default is 'cpu'.
    - hot_encoded (bool): A boolean flag to indicate if to convert nplets to hot encoded and use the hot encoded version or if to divide by orders and process as indexes.
    
    Returns:
    - torch.Tensor: The measures for the nplets with shape (n_nplets, D, 4) where D is the number of matrices, n_nplets is the number of nplets to calculate over and 4 is the number of metrics (tc, dtc, o, s)
    '''
    
    logging.basicConfig(
        level=verbose,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    covmats, D, N, T = _normalize_input_data(X, covmat_precomputed, T, device)
    
    # If nplets is a list of nplets with different orders, then use hot encoding to compute multiorder measures
    if isinstance(nplets, list) and not all([len(nplet) == len(nplets[0]) for nplet in nplets]):
        logging.warning('Using hot encoding to compute multi-order measures as nplets have different orders')
        nplets = _indices_to_hot_encoded(nplets, N)
        return nplets_measures_hot_encoded(covmats, nplets, covmat_precomputed=True, T=T)
    
    # If nplets are not tensors, convert to tensor
    nplets = torch.as_tensor(nplets).to(device).contiguous()
        
    # nplets must be a batched tensor
    assert len(nplets.shape) == 2, 'nplets must be a batched tensor with shape (batch_size, order)'
    batch_size, order = nplets.shape

    # Create marginal indexes
    # |N| x |N-1|
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
                         min_order: int=3,
                         max_order: Optional[int]=None,
                         *,
                         covmat_precomputed: bool=False,
                         T: Optional[Union[int, List[int]]]=None,
                         batch_size: int = 1000000,
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

    Parameters:
        - X (np.ndarray or torch.Tensor): (T samples x N variables) or (D datas x T samples x N variables) matrix.
        - min_order (int): Minimum order to compute (default: 3).
        - max_order (Optional[int]): Maximum order to compute (default: None, will use N).
        - covmat_precomputed (bool): If True, X is a covariance matrix (default: False).
        - T (Optional[int]): Number of samples used to compute bias correction (default: None). This parameter is only used if covmat_precomputed is True.
        - batch_size (int): Batch size for DataLoader (default: 1000000).
        - device (torch.device): Device to use for the computation (default: 'cpu').
        - num_workers (int): Number of workers for DataLoader (default: 0).
        - batch_aggregation (Optional[Callable[[any],any]]): Function to aggregate the batched data (default: pd.concat).
        - batch_data_collector (Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],any]]): Function to collect the batched data (default: batch_to_csv).

    Returns:
        pd.DataFrame: DataFrame containing computed metrics.
    """

    covmats, D, N, T = _normalize_input_data(X, covmat_precomputed, T, device)
    
    # For each dataset, precompute the single variable marginal gaussian entropies
    # |D| x |N|
    marginal_entropies = _marginal_gaussian_entropies(covmats)

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
        # |N| x |N-1|
        allmin1 = _all_min_1_ids(order, device=device)
        
        # Create the bias corrector for the current order
        # |batch_size x D|, |batch_size x D|, |batch_size x D|
        bc1, bcN, bcNmin1 = _get_bias_correctors(T, order, batch_size, D, device)

        # Generate dataset iterable
        dataset = CovarianceDataset(N, order, device=device)
        dataloader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            num_workers=num_workers
        )

        # calculate measurments for each batch
        for bn, nplets in enumerate(tqdm(dataloader, total=len(dataloader), leave=False, desc='Batch')):
            # Batch_size can vary in the last batch, then effective batch size is the current batch size
            curr_batch_size = nplets.shape[0]

            # Send nplets to the device in case it is not there
            # |curr_batch_size| x |order|
            nplets = nplets.to(device)

            # Create the covariance matrices for each nplet in the batch
            # |curr_batch_size| x |D| x |order| x |order|
            nplets_covmats = _generate_nplets_covmants(covmats, nplets)
            
            # Create the marginal entropies sampling from the marginal_entropies tensor
            # |curr_batch_size| x |D| x |order|
            nplets_marginal_entropies = _generate_nplets_marginal_entropies(marginal_entropies, nplets)

            # Pack covmats and marginal entropies in a single batch
            # |curr_batch_size x D| x |N| x |N|
            nplets_covmats = nplets_covmats.view(curr_batch_size*D, order, order)
            nplets_marginal_entropies = nplets_marginal_entropies.view(curr_batch_size*D, order)

            # Batch process all nplets at once
            measures = _get_tc_dtc_from_batched_covmat(nplets_covmats,
                                                       allmin1,
                                                       bc1[:curr_batch_size*D],
                                                       bcN[:curr_batch_size*D],
                                                       bcNmin1[:curr_batch_size*D],
                                                       nplets_marginal_entropies)

            # Unpack results
            # |curr_batch_size x D|, |curr_batch_size x D|, |curr_batch_size x D|, |curr_batch_size x D|
            nplets_tc, nplets_dtc, nplets_o, nplets_s = measures

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
