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
                    verbose: int = logging.INFO,
                    batch_size: int = 1000000):
    
    """
    Compute higher-order measures (TC, DTC, O, S) for specified n-plets in the given data matrices X.

    The computed measures are:
        - **Total Correlation (TC)**
        - **Dual Total Correlation (DTC)**
        - **O-information (O)**
        - **S-information (S)**

    Parameters
    ----------
    X : TensorLikeArray
        Input data, which can be one of the following:
        - A single torch.Tensor or np.ndarray with shape (T, N).
        - A sequence (e.g., list) of torch.Tensor or np.ndarray, each with shape (T, N), representing multiple datasets.
        - A sequence of sequences, where each inner sequence is an array-like object of shape (T, N).
        If `covmat_precomputed` is True, X should be:
        - A single torch.Tensor or np.ndarray covariance matrix with shape (N, N).
        - A sequence of covariance matrices, each with shape (N, N).

    nplets : TensorLikeArray, optional
        The n-plets to calculate the measures, with shape `(n_nplets, order)`. If `None`, all possible n-plets of the given order are considered.

    covmat_precomputed : bool, optional
        If True, X is treated as covariance matrices instead of raw data. Default is False.

    T : int or list of int, optional
        Number of samples used to compute bias correction. This parameter is used only if `covmat_precomputed` is True.
        If X is a sequence of covariance matrices, T should be a list of sample sizes corresponding to each matrix.
        If T is None and `covmat_precomputed` is True, bias correction is not applied. Default is None.

    device : torch.device, optional
        Device to use for computation. Default is torch.device('cpu').

    verbose : int, optional
        Logging verbosity level. Default is `logging.INFO`.

    batch_size : int, optional
        Batch size for processing n-plets. Default is 1,000,000.

    Returns
    -------
    torch.Tensor
        Tensor containing the computed measures for each n-plet with shape `(n_nplets, D, 4)`

    Where
    -----
    D : int
        Number of datasets. If `X` is a single dataset, `D = 1`.

    N : int
        Number of variables (features) in each dataset.

    T : int
        Number of samples in each dataset (if applicable).

    order : int
        The size of the n-plets being analyzed.

    n_nplets : int
        Number of n-plets processed.

    Examples
    --------
    **Compute measures for all possible 3-plets in a single dataset:**

    ```python
    import torch
    import numpy as np

    # Sample data matrix with 100 samples and 5 variables
    X = np.random.randn(100, 5)

    # Compute measures for all 3-plets
    measures = nplets_measures(X, nplets=None, covmat_precomputed=False, T=100)
    ```

    **Compute measures for specific n-plets in multiple datasets:**

    ```python
    import torch
    import numpy as np

    # Sample data matrices for 2 datasets, each with 100 samples and 5 variables
    X1 = np.random.randn(100, 5)
    X2 = np.random.randn(100, 5)
    X = [X1, X2]

    # Define specific n-plets to analyze
    nplets = torch.tensor([[0, 1, 2], [1, 2, 3]])

    # Compute measures for the specified n-plets
    measures = nplets_measures(X, nplets=nplets, covmat_precomputed=False, T=[100, 100])
    ```

    **Compute measures with precomputed covariance matrices:**

    ```python
    import torch
    import numpy as np

    # Precompute covariance matrices for 2 datasets
    covmat1 = np.cov(np.random.randn(100, 5), rowvar=False)
    covmat2 = np.cov(np.random.randn(100, 5), rowvar=False)
    X = [covmat1, covmat2]

    # Number of samples for each covariance matrix
    T = [100, 100]

    # Define specific n-plets to analyze
    nplets = torch.tensor([[0, 1], [2, 3]])

    # Compute measures using precomputed covariance matrices
    measures = nplets_measures(X, nplets=nplets, covmat_precomputed=True, T=T)
    ```

    **Notes**
    -----
    - If `nplets` is `None`, the function considers all possible n-plets of the specified order within the datasets.
    - Ensure that the length of `T` matches the number of datasets when `covmat_precomputed` is `True` and `X` is a sequence of covariance matrices.
    - The function is optimized for batch processing using PyTorch tensors, facilitating efficient computations on large datasets.
    
    References
    ----------
    .. [1] Rosas, Fernando E., et al. "Quantifying high-order interdependencies via multivariate extensions of the mutual information." Physical Review E 100.3 (2019): 032305.

    """
    
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
    elif nplets is None:
        nplets = torch.arange(N, device=device).unsqueeze(0)
    
    # If nplets are not tensors, convert to tensor
    nplets = torch.as_tensor(nplets).to(device).contiguous()
        
    # nplets must be a batched tensor
    assert len(nplets.shape) == 2, 'nplets must be a batched tensor with shape (batch_size, order)'
    batch_size = min(batch_size, len(nplets))
    order = nplets.shape[1]

    # Create marginal indexes
    # |N| x |N-1|
    allmin1 = _all_min_1_ids(order, device=device)

    # Create bias corrector values
    # |batch_size x D|, |batch_size x D|, |batch_size x D|
    bc1, bcN, bcNmin1 = _get_bias_correctors(T, order, batch_size, D, device)

    # Create DataLoader for nplets
    dataloader = DataLoader(nplets, batch_size=batch_size, shuffle=False)

    results = []
    for nplet_batch in tqdm(dataloader, desc='Processing n-plets', leave=False):
        curr_batch_size = nplet_batch.shape[0]

        # Create the covariance matrices for each nplet in the batch
        # |curr_batch_size| x |D| x |order| x |order|
        nplets_covmats = _generate_nplets_covmants(covmats, nplet_batch)
        
        # Pack covmats in a single batch
        # |curr_batch_size x D| x |order| x |order|
        nplets_covmats = nplets_covmats.view(curr_batch_size * D, order, order)

        # Batch process all nplets at once
        measures = _get_tc_dtc_from_batched_covmat(nplets_covmats,
                                                   allmin1,
                                                   bc1[:curr_batch_size * D],
                                                   bcN[:curr_batch_size * D],
                                                   bcNmin1[:curr_batch_size * D])

        # Unpack results
        # |curr_batch_size x D|, |curr_batch_size x D|, |curr_batch_size x D|, |curr_batch_size x D|
        nplets_tc, nplets_dtc, nplets_o, nplets_s = measures

        # Collect results
        results.append(torch.stack([nplets_tc.view(curr_batch_size, D),
                                    nplets_dtc.view(curr_batch_size, D),
                                    nplets_o.view(curr_batch_size, D),
                                    nplets_s.view(curr_batch_size, D)], dim=-1))

    # Concatenate all results
    return torch.cat(results, dim=0)

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
    Compute multi-order measures (TC, DTC, O, S) for the given data matrix X.

    The measurements computed are:
        - Total Correlation (TC)
        - Dual Total Correlation (DTC)
        - O-information (O)
        - S-information (S)

    Parameters
    ----------
    X : TensorLikeArray
        Input data, which can be one of the following:
        - A single torch.Tensor or np.ndarray with shape (T, N).
        - A sequence (e.g., list) of torch.Tensor or np.ndarray, each with shape (T, N), representing multiple datasets.
        - A sequence of sequences, where each inner sequence is an array-like object of shape (T, N).
        If `covmat_precomputed` is True, X should be:
        - A single torch.Tensor or np.ndarray covariance matrix with shape (N, N).
        - A sequence of covariance matrices, each with shape (N, N).

    min_order : int, optional
        Minimum order to compute. Default is 3. Note: 3 <= min_order <= max_order <= N.
    max_order : int, optional
        Maximum order to compute. If None, uses N (number of variables). Default is None. Note: min_order <= max_order <= N.
    covmat_precomputed : bool, optional
        If True, X is treated as covariance matrices instead of raw data. Default is False.
    T : int or list of int, optional
        Number of samples used to compute bias correction. This parameter is used only if `covmat_precomputed` is True.
        If X is a sequence of covariance matrices, T should be a list of sample sizes corresponding to each matrix.
        If T is None and `covmat_precomputed` is True, bias correction is not applied. Default is None.
    batch_size : int, optional
        Batch size for DataLoader. Default is 1,000,000.
    device : torch.device, optional
        Device to use for computation. Default is torch.device('cpu').
    num_workers : int, optional
        Number of workers for DataLoader. Default is 0.
    batch_aggregation : callable, optional
        Function to aggregate the collected batch data into the final result.
        It should accept a list of outputs from `batch_data_collector` and return the final aggregated result.
        The return type of this function determines the return type of `multi_order_measures`.
        By default, it uses `concat_and_sort_csv`, which concatenates CSV data and sorts it, returning a pandas DataFrame. 
        For more information see :ref:`collectors__concat_and_sort_csv`
    batch_data_collector : callable, optional
        Function to process and collect data from each batch.
        It should accept the following parameters:
            - nplets: torch.Tensor of n-plet indices, shape (batch_size, order)
            - nplets_tc: torch.Tensor of total correlation values, shape (batch_size, D)
            - nplets_dtc: torch.Tensor of dual total correlation values, shape (batch_size, D)
            - nplets_o: torch.Tensor of O-information values, shape (batch_size, D)
            - nplets_s: torch.Tensor of S-information values, shape (batch_size, D)
            - batch_number: int, the current batch number
        The output of `batch_data_collector` must be compatible with the input expected by `batch_aggregation`.
        By default, it uses `batch_to_csv`, which collects data into CSV. For more information see :ref:`collectors__batch_to_csv`

    Returns
    -------
    Any
        The aggregated result of the computed measures. The exact type depends on the `batch_aggregation` function used.
        By default, it returns a pandas DataFrame containing the computed metrics (DTC, TC, O, S), the n-plets indexes, 
        the order and the dataset information.
    
    Where
    -----
    D : int
        Number of datasets. If X is a single dataset, D = 1.
    N : int
        Number of variables (features) in each dataset.
    T : int
        Number of samples in each dataset (if applicable).
    order : int
        The size of the n-plets being analyzed, ranging from `min_order` to `max_order`.
    batch_size : int
        Number of n-plets processed in each batch.

    Notes
    -----
    - The default `batch_data_collector` and `batch_aggregation` functions are designed to work together.
      If you provide custom functions, ensure that the output of `batch_data_collector` is compatible with the input of `batch_aggregation`.
    - Ensure that the length of `T` matches the number of datasets when `covmat_precomputed` is `True` and `X` is a sequence of covariance matrices.
    - The function computes measures for all combinations of variables of orders ranging from `min_order` to `max_order`.
    - The function is optimized for batch processing using PyTorch tensors, facilitating efficient computations on large datasets.

    Examples
    --------
    Using default batch data collector and aggregation:

    >>> result = multi_order_measures(X, min_order=3, max_order=5)

    Using custom batch data collector and aggregation:

    >>> def custom_batch_data_collector(nplets, tc, dtc, o, s, batch_number):
    ...     # Custom processing
    ...     return custom_data
    ...
    >>> def custom_batch_aggregation(batch_data_list):
    ...     # Custom aggregation
    ...     return final_result
    ...
    >>> result = multi_order_measures(
    ...     X,
    ...     min_order=3,
    ...     max_order=5,
    ...     batch_data_collector=custom_batch_data_collector,
    ...     batch_aggregation=custom_batch_aggregation
    ... )

    References
    ----------
    .. [1] Rosas, Fernando E., et al. "Quantifying high-order interdependencies via multivariate extensions of the mutual information." Physical Review E 100.3 (2019): 032305.

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
