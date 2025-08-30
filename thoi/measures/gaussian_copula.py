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

    """
    Compute the total correlation (TC), dual total correlation (DTC), o-information (O), and s-information (S) for the given batch of covariance matrices.

    Parameters
    ----------
    covmats : torch.Tensor
        The covariance matrices with shape (batch_size, N, N).
    allmin1 : torch.Tensor
        The indexes of marginal covariance matrices with shape (batch_size, N, N-1).
    bc1 : torch.Tensor
        The bias corrector for the first order with shape (batch_size).
    bcN : torch.Tensor
        The bias corrector for the order with shape (batch_size).
    bcNmin1 : torch.Tensor
        The bias corrector for the order-1 with shape (batch_size).
    marginal_entropies : Optional[torch.Tensor], optional
        The marginal entropies for each variable with shape (batch_size, N). If None, it will be dynamically computed.

    Returns
    -------
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        The computed measures: total correlation (TC), dual total correlation (DTC), o-information (O), and s-information (S).
    """

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

    Notes
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
    batch_size = max(batch_size // D, 1)

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


# ============================================================================
# LOCAL (TIME-RESOLVED) MEASURES IMPLEMENTATION
# ============================================================================

def _gaussian_entropy_bias(N: int, T: int, device='cpu', dtype=torch.float64) -> torch.Tensor:
    """Alias for _gaussian_entropy_bias_correction with consistent naming."""
    from thoi.measures.utils import _gaussian_entropy_bias_correction
    return _gaussian_entropy_bias_correction(N, T).to(device=device, dtype=dtype)

def gaussian_tc_bias_correction(K: int, T: int, device='cpu', dtype=torch.float64) -> torch.Tensor:
    """Bias for TC = sum H(X_i) - H(X_1..X_K)."""
    return K * _gaussian_entropy_bias(1, T, device, dtype) - _gaussian_entropy_bias(K, T, device, dtype)

def gaussian_dtc_bias_correction(K: int, T: int, device='cpu', dtype=torch.float64) -> torch.Tensor:
    """Bias for DTC = sum_i H(X_{-i}) - (K-1) H(X_1..X_K)."""
    return K * _gaussian_entropy_bias(K - 1, T, device, dtype) - (K - 1) * _gaussian_entropy_bias(K, T, device, dtype)

def _batched_chol_logdet(S, eps=1e-10):
    """Compute Cholesky decomposition and log determinant for batched matrices."""
    k = S.shape[-1]
    L = torch.linalg.cholesky(S + eps*torch.eye(k, device=S.device, dtype=S.dtype))
    logdet = 2.0*torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    return L, logdet  # L:[B,k,k], logdet:[B]

def _quad_from_chol(L, X):
    """Compute quadratic form using Cholesky decomposition."""
    # L:[B,k,k], X:[B,Lc,k] -> [B,Lc]
    ZT = torch.cholesky_solve(X.transpose(1,2), L)  # [B,k,Lc]
    return (X.transpose(1,2) * ZT).sum(1)           # [B,Lc]

def _gather_nplet_covs(covmats, nplets):
    """Extract covariance submatrices for specified n-plets."""
    B, K = nplets.shape
    D, N, _ = covmats.shape
    device, dtype = covmats.device, covmats.dtype
    d_idx = torch.arange(D, device=device).view(1,D,1,1).expand(B,D,K,K)
    r_idx = nplets[:,None,:,None].expand(B,D,K,K)
    c_idx = nplets[:,None,None,:].expand(B,D,K,K)
    sub = covmats[d_idx, r_idx, c_idx]   # [B,D,K,K]
    return sub.reshape(B*D, K, K)

def _gather_nplet_data(data, nplets, t0, t1):
    """Extract data subsets for specified n-plets and time range."""
    # slice only [t0:t1] to save memory
    B,K = nplets.shape
    D,T,N = data.shape
    Lc = t1-t0
    device = data.device
    d_idx = torch.arange(D, device=device).view(1,D,1,1).expand(B,D,Lc,K)
    t_idx = torch.arange(t0,t1, device=device).view(1,1,Lc,1).expand(B,D,Lc,K)
    n_idx = nplets[:,None,None,:].expand(B,D,Lc,K)
    sub = data[d_idx, t_idx, n_idx]  # [B,D,Lc,K]
    return sub.reshape(B*D, Lc, K)

def _leave_one_out_stats(S, Xc, Lj, logdet_j, eps=1e-10):
    """
    Compute leave-one-out statistics for DTC calculation.
    
    This function computes the sum of all leave-one-out negative log-likelihoods
    which is needed for DTC and S calculations. Uses direct computation of each
    leave-one-out subsystem for accuracy.
    
    Parameters
    ----------
    S : torch.Tensor
        Covariance matrices, shape [B, K, K]
    Xc : torch.Tensor  
        Data chunk, shape [B, Lc, K]
    Lj : torch.Tensor
        Cholesky decomposition of S
    logdet_j : torch.Tensor
        Log determinant of S
        
    Returns
    -------
    logdet_mi_sum : torch.Tensor
        Sum of log determinants for all leave-one-out subsystems, shape [B]
    qmi_sum : torch.Tensor
        Sum of quadratic forms for all leave-one-out subsystems, shape [B, Lc]
    """
    B, K, _ = S.shape
    Lc = Xc.shape[1]
    
    # Initialize accumulators
    logdet_mi_sum = torch.zeros(B, device=S.device, dtype=S.dtype)
    qmi_sum = torch.zeros(B, Lc, device=S.device, dtype=S.dtype)
    
    # Compute each leave-one-out subsystem directly
    for i in range(K):
        # Indices for leave-one-out (excluding i-th variable)
        loo_indices = torch.cat([torch.arange(i), torch.arange(i+1, K)])
        
        # Extract leave-one-out covariance matrices and data
        S_loo = S[:, loo_indices][:, :, loo_indices]  # [B, K-1, K-1]
        Xc_loo = Xc[:, :, loo_indices]                # [B, Lc, K-1]
        
        # Use Cholesky decomposition for numerical stability (like traditional THOI)
        try:
            L_loo = torch.linalg.cholesky(S_loo + eps * torch.eye(K-1, device=S.device, dtype=S.dtype))
            logdet_loo = 2 * torch.sum(torch.log(torch.diagonal(L_loo, dim1=-2, dim2=-1)), dim=-1)
        except:
            # Fallback to regular logdet if Cholesky fails
            logdet_loo = torch.logdet(S_loo + eps * torch.eye(K-1, device=S.device, dtype=S.dtype))
            
        logdet_mi_sum += logdet_loo
        
        # Compute quadratic form using Cholesky solve (more stable than inverse)
        # Solve L @ y = x for each time point, then compute ||y||^2
        try:
            # L_loo: [B, K-1, K-1], Xc_loo: [B, Lc, K-1]
            # We need to solve for each time point: L @ y = x^T
            Xc_loo_t = Xc_loo.transpose(-1, -2)  # [B, K-1, Lc]
            y = torch.linalg.solve_triangular(L_loo, Xc_loo_t, upper=False)  # [B, K-1, Lc]
            q_loo = torch.sum(y * y, dim=1)  # [B, Lc] - removed .T which was causing problems
        except:
            # Fallback to inverse method
            S_loo_inv = torch.inverse(S_loo + eps * torch.eye(K-1, device=S.device, dtype=S.dtype))
            temp = torch.bmm(Xc_loo, S_loo_inv)
            q_loo = torch.sum(temp * Xc_loo, dim=2)  # [B, Lc]
            
        qmi_sum += q_loo

    return logdet_mi_sum, qmi_sum

@torch.no_grad()
def local_nplets_measures(
    data, nplets, covmats, *,
    device='cpu', dtype=torch.float32,
    batch_size=100000, time_chunk=2048, eps=1e-10
):
    """
    Compute local (time-resolved) higher-order information measures.
    
    This function computes local versions of TC, DTC, O, and S for each time point
    using the corrected negative log-likelihood approach that ensures theoretical
    consistency (TC ≥ 0) and proper convergence to traditional measures.
    
    IMPORTANT: Input data must be normalized using gaussian_copula_cov_opt first
    to ensure marginal gaussianity required for the Gaussian copula approach.
    
    Parameters
    ----------
    data : torch.Tensor or array-like
        Normalized input data with shape (D, T, N) or (T, N) for single dataset.
        This should be the output of gaussian_copula_cov_opt (normalized Gaussian data).
    nplets : torch.Tensor or array-like
        N-plets to analyze, shape (B, K) where B is number of n-plets, K is order
    covmats : torch.Tensor
        Covariance matrices corresponding to the normalized data, shape (D, N, N).
        This should be the covariance output of gaussian_copula_cov_opt.
    device : str, default='cpu'
        Device for computation
    dtype : torch.dtype, default=torch.float32
        Data type for computation
    batch_size : int, default=100000
        Batch size for n-plet processing
    time_chunk : int, default=2048
        Time chunk size for memory optimization
    eps : float, default=1e-10
        Numerical stability epsilon
        
    Returns
    -------
    torch.Tensor
        Local measures with shape [B, D, T, 4] where last dimension is (TC, DTC, O, S)
    """
    # Normalize inputs - data should already be normalized from gaussian_copula_cov_opt
    if hasattr(data, 'dim') and data.dim()==3:
        D,T,N = data.shape
        X = data.to(device=device, dtype=dtype)
    elif isinstance(data,(list,tuple)):
        D = len(data); T,N = data[0].shape
        X = torch.stack([torch.as_tensor(d,device=device,dtype=dtype) for d in data],0)
    else:
        X = torch.as_tensor(data,device=device,dtype=dtype).unsqueeze(0)
        D,T,N = 1,X.shape[1],X.shape[2]

    # Covmats must be provided (from gaussian_copula_cov_opt)
    covmats = torch.as_tensor(covmats,device=device,dtype=dtype)
    if covmats.dim()==2: 
        covmats = covmats.unsqueeze(0)

    nplets = torch.as_tensor(nplets,device=device,dtype=torch.long).contiguous()
    B_total,K = nplets.shape

    out = torch.empty(B_total,D,T,4,device=device,dtype=dtype)

    for start in range(0,B_total,batch_size):
        npl = nplets[start:start+batch_size]
        B = npl.shape[0]

        # Gather covariance matrices for this batch of nplets
        S = _gather_nplet_covs(covmats,npl)          # [B*D,K,K]
        
        # Compute Cholesky decomposition and joint log-determinant
        Lj, logdet_j = _batched_chol_logdet(S,eps=eps)
        
        # Get diagonal elements (variances) for univariate calculations
        var = torch.diagonal(S,dim1=-2,dim2=-1)      # [B*D,K]

        for t0 in range(0,T,time_chunk):
            t1 = min(T,t0+time_chunk)
            Xc = _gather_nplet_data(X,npl,t0,t1)     # [B*D,Lc,K]
            Lc = t1-t0

            # CORRECTED IMPLEMENTATION: Proper negative log-likelihood calculations
            # Following the validated approach that ensures TC ≥ 0 and convergence
            
            # Joint NLL: -log p(x_t) for the full n-plet
            # NLL = 0.5 * [K*log(2π) + log|Σ| + x_t^T Σ^{-1} x_t]
            qj = _quad_from_chol(Lj, Xc)                                    # [B*D, Lc]
            joint_nll = 0.5 * (K * np.log(2 * np.pi) + logdet_j.unsqueeze(1) + qj)  # [B*D, Lc]
            
            # Univariate NLLs: -log p(x_i_t) for each variable
            # For each variable: NLL_i = 0.5 * [log(2π) + log(σ_i²) + (x_i_t)²/σ_i²]
            q_uni = (Xc**2 / var.unsqueeze(1))                             # [B*D, Lc, K]
            logdet_uni_per_var = torch.log(var)                            # [B*D, K]
            uni_nll = 0.5 * (np.log(2 * np.pi) + 
                           logdet_uni_per_var.unsqueeze(1) + q_uni)        # [B*D, Lc, K]
            
            # Sum of univariate NLLs: Σᵢ[-log p(x_i_t)]
            uni_nll_sum = uni_nll.sum(dim=2)                               # [B*D, Lc]
            
            # TC local: Sum of univariate entropies minus joint entropy
            # TC = H(X₁) + H(X₂) + ... + H(Xₙ) - H(X₁,X₂,...,Xₙ)
            # Since H = -E[log p] and for local: TC = uni_nll_sum - joint_nll
            tc_loc = uni_nll_sum - joint_nll                               # [B*D, Lc]
            
            # For DTC, O, S: need leave-one-out calculations for K≥3
            if K >= 3:
                # Leave-one-out NLLs: -log p(x_{-i}_t) for each subset
                logdet_mi, qmi = _leave_one_out_stats(S, Xc, Lj, logdet_j, eps)
                # Correct constant term: K leave-one-out systems, each with (K-1) variables
                # Total constant: K * (K-1) * log(2π)
                loo_nll = 0.5 * (K * (K-1) * np.log(2 * np.pi) + 
                               logdet_mi.unsqueeze(1) + qmi)               # [B*D, Lc]
                
                # DTC local: DTC = sum_i H(X_{-i}) - (K-1) H(X)
                # In terms of NLL: DTC = loo_nll - (K-1) * joint_nll
                # (previous implementation subtracted only joint_nll, which is incorrect)
                dtc_loc = loo_nll - (K - 1) * joint_nll                      # [B*D, Lc]
                
                # O: TC minus DTC (redundancy)
                o_loc = tc_loc - dtc_loc                                   # [B*D, Lc]
                
                # S: TC plus DTC (synergy)  
                s_loc = tc_loc + dtc_loc                                   # [B*D, Lc]
                
            else:
                # For K=2, DTC=TC and O=0, S=TC+DTC=2×TC
                dtc_loc = tc_loc
                o_loc = torch.zeros_like(tc_loc)
                s_loc = tc_loc + dtc_loc  # S = TC + DTC = TC + TC = 2×TC

            # Store results in output tensor
            out[start:start+B,:,t0:t1,0] = tc_loc.view(B,D,Lc)
            out[start:start+B,:,t0:t1,1] = dtc_loc.view(B,D,Lc)
            out[start:start+B,:,t0:t1,2] = o_loc.view(B,D,Lc)
            out[start:start+B,:,t0:t1,3] = s_loc.view(B,D,Lc)

    return out

@torch.no_grad()
def local_multi_order_measures(
    X: TensorLikeArray,
    min_order: int = 3,
    max_order: Optional[int] = None, 
    *, 
    device: str = 'cpu', 
    dtype: torch.dtype = torch.float32,
    batch_size: int = 100000, 
    time_chunk: int = 4096, 
    eps: float = 1e-10
) -> dict:
    """
    Compute local measures for every order in [min_order, max_order] on the full set of variables.
    
    This function provides a wrapper around local_nplets_measures to compute local
    information theory measures for all possible n-plets of specified orders.
    
    IMPORTANT: Input data X must be normalized using normalize_input_data first
    to ensure marginal gaussianity required for the Gaussian copula approach.
    
    Parameters
    ----------
    X : TensorLikeArray
        Input data which can be:
        - A single torch.Tensor or np.ndarray with shape (T, N)
        - A list/sequence of torch.Tensor or np.ndarray each with shape (T, N)
        Must be normalized using normalize_input_data first.
    min_order : int, default=3
        Minimum order of interactions to compute
    max_order : int, optional
        Maximum order of interactions to compute. If None, uses N (number of variables)
    device : str, default='cpu'
        Device for computation
    dtype : torch.dtype, default=torch.float32
        Data type for computation
    batch_size : int, default=100000
        Batch size for n-plet processing
    time_chunk : int, default=4096
        Time chunk size for memory optimization
    eps : float, default=1e-10
        Numerical stability epsilon
        
    Returns
    -------
    dict
        Dictionary with keys as orders and values as tensors [C(N,order), D, T, 4]
        where last dimension is (TC, DTC, O, S)
    """
    from ..commons import gaussian_copula_cov_opt
    
    # Ensure X is in the correct format for gaussian_copula_cov_opt (needs 3D: D, T, N)
    if isinstance(X, np.ndarray):
        if X.ndim == 2:
            # Single dataset: (T, N) -> (1, T, N)
            X = X[np.newaxis, :, :]
        X_tensor = torch.tensor(X, dtype=dtype)
    elif isinstance(X, torch.Tensor):
        if X.ndim == 2:
            # Single dataset: (T, N) -> (1, T, N) 
            X = X.unsqueeze(0)
        X_tensor = X.to(dtype=dtype)
    elif isinstance(X, (list, tuple)):
        # Multiple datasets: [(T, N), ...] -> (D, T, N)
        X_tensor = torch.stack([torch.tensor(x, dtype=dtype) for x in X])
    else:
        X_tensor = torch.tensor(X, dtype=dtype)
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(0)
    
    # Normalize input data and get covariance matrices using gaussian_copula_cov_opt
    # This ensures proper Gaussian distributions with the correct covariance structure
    normalized_data, covmats = gaussian_copula_cov_opt(X_tensor, return_xg=True)
    
    # Determine data dimensions
    if hasattr(normalized_data, 'dim') and normalized_data.dim()==3:
        D, T, N = normalized_data.shape
    elif isinstance(normalized_data, (list,tuple)):
        T, N = normalized_data[0].shape
        D = len(normalized_data)
    else:
        T, N = normalized_data.shape
        D = 1
        
    if max_order is None: 
        max_order = N

    out = {}
    for K in range(min_order, max_order+1):
        # All n-plets of this order: use lexicographic ordering
        nplets = torch.combinations(torch.arange(N, device=device), r=K, with_replacement=False)
        local_measures = local_nplets_measures(
            normalized_data, nplets, covmats, device=device,
            batch_size=batch_size, time_chunk=time_chunk, eps=eps, dtype=dtype
        )
        out[K] = local_measures
    
    return out


def time_averaged_local_measures(
    X: TensorLikeArray,
    min_order: int = 3,
    max_order: Optional[int] = None, 
    *, 
    device: str = 'cpu', 
    dtype: torch.dtype = torch.float32,
    batch_size: int = 100000, 
    time_chunk: int = 4096, 
    eps: float = 1e-10,
    bias_correction: bool = True
) -> dict:
    """
    Compute time-averaged local measures with optional bias correction.
    
    This function computes local measures and then averages them over time,
    applying bias correction to the averaged results (not to individual local measures).
    
    Parameters
    ----------
    X : TensorLikeArray
        Input data which can be:
        - A single torch.Tensor or np.ndarray with shape (T, N)
        - A list/sequence of torch.Tensor or np.ndarray each with shape (T, N)
        Must be normalized using normalize_input_data first.
    min_order : int, default=3
        Minimum order of interactions to compute
    max_order : int, optional
        Maximum order of interactions to compute. If None, uses N (number of variables)
    device : str, default='cpu'
        Device for computation
    dtype : torch.dtype, default=torch.float32
        Data type for computation
    batch_size : int, default=100000
        Batch size for n-plet processing  
    time_chunk : int, default=4096
        Chunk size for temporal processing
    eps : float, default=1e-10
        Small value for numerical stability
    bias_correction : bool, default=True
        Whether to apply bias correction after temporal averaging
        
    Returns
    -------
    dict
        Dictionary mapping order -> tensor of averaged measures with shape
        (n_combinations, n_samples, 4) where last dim is [TC, DTC, O, S]
    """
    
    # First get local measures
    local_results = local_multi_order_measures(
        X,
        min_order=min_order,
        max_order=max_order,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        time_chunk=time_chunk,
        eps=eps
    )

    # Time-average the results and apply bias correction AFTER temporal averaging
    averaged_results = {}

    for K, measures in local_results.items():
        # measures shape: [n_combinations, D, T, 4]
        # Average over time dimension
        time_averaged = measures.mean(dim=2)  # -> [n_combinations, D, 4]

        if bias_correction and K >= 2:
            # Number of samples used to compute bias correction is the temporal length
            T_samples = measures.shape[2]

            # Compute corrections using the helper functions defined above
            tc_correction = gaussian_tc_bias_correction(K, T_samples, device=time_averaged.device, dtype=time_averaged.dtype)
            dtc_correction = gaussian_dtc_bias_correction(K, T_samples, device=time_averaged.device, dtype=time_averaged.dtype)

            # Apply corrections (broadcasting will handle scalar or tensor shapes)
            corrected = time_averaged.clone()
            corrected[..., 0] = time_averaged[..., 0] - tc_correction
            corrected[..., 1] = time_averaged[..., 1] - dtc_correction

            # Recompute O and S from the corrected TC/DTC to keep theoretical relations
            corrected[..., 2] = corrected[..., 0] - corrected[..., 1]  # O = TC - DTC
            corrected[..., 3] = corrected[..., 0] + corrected[..., 1]  # S = TC + DTC

            averaged_results[K] = corrected
        else:
            averaged_results[K] = time_averaged

    return averaged_results
