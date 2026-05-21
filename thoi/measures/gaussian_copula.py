from typing import Optional, Callable, Union, List, Tuple

from tqdm import tqdm
from functools import partial
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import scipy.special as sp

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

def gaussian_tc_bias_correction(K: int, T: int, device='cpu', dtype=torch.float64) -> torch.Tensor:
    """Bias for TC = sum H(X_i) - H(X_1..X_K)."""
    from thoi.measures.utils import _gaussian_entropy_bias_correction
    return K * _gaussian_entropy_bias_correction(1, T).to(device=device, dtype=dtype) - \
           _gaussian_entropy_bias_correction(K, T).to(device=device, dtype=dtype)

def gaussian_dtc_bias_correction(K: int, T: int, device='cpu', dtype=torch.float64) -> torch.Tensor:
    """Bias for DTC = sum_i H(X_{-i}) - (K-1) H(X_1..X_K)."""
    from thoi.measures.utils import _gaussian_entropy_bias_correction
    return K * _gaussian_entropy_bias_correction(K - 1, T).to(device=device, dtype=dtype) - \
           (K - 1) * _gaussian_entropy_bias_correction(K, T).to(device=device, dtype=dtype)

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
            q_loo = torch.sum(y * y, dim=1)  # [B, Lc]: sum over K-1 dimension for quadratic form
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
    eps: float = 1e-10,
    covmats: Optional[torch.Tensor] = None,
    precomputed: bool = False
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
        When precomputed=False: Must be normalized using normalize_input_data first.
        When precomputed=True: Must be already normalized data with shape (D, T, N).
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
    covmats : torch.Tensor, optional
        Pre-computed covariance matrices with shape [D, N, N].
        Required when precomputed=True, optional otherwise.
    precomputed : bool, default=False
        If True, assumes X is already normalized data and covmats are provided.
        If False, performs full preprocessing including gaussian_copula_cov_opt.
        
    Returns
    -------
    dict
        Dictionary with keys as orders and values as tensors [C(N,order), D, T, 4]
        where last dimension is (TC, DTC, O, S)
    """
    from ..commons import gaussian_copula_cov_opt
    
    if precomputed:
        # Use provided normalized data and covariance matrices
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True because normalized data requires corresponding covariance matrices')
        
        # Ensure X is properly formatted normalized data
        normalized_data = torch.as_tensor(X, dtype=dtype, device=device)
        if normalized_data.dim() == 2:
            # Single dataset: (T, N) -> (1, T, N)
            normalized_data = normalized_data.unsqueeze(0)
        elif normalized_data.dim() != 3:
            raise ValueError('When precomputed=True, X must be normalized data with shape (D, T, N) or (T, N)')
        
        # Ensure covmats is properly formatted
        covmats = torch.as_tensor(covmats, dtype=dtype, device=device)
        if covmats.dim() == 2:
            # Single covariance matrix: (N, N) -> (1, N, N)
            covmats = covmats.unsqueeze(0)
        elif covmats.dim() != 3:
            raise ValueError('When precomputed=True, covmats must have shape (D, N, N) or (N, N)')
        
        # Check compatibility
        D, T, N = normalized_data.shape
        if covmats.shape != (D, N, N):
            raise ValueError(f'covmats shape {covmats.shape} is not compatible with normalized_data shape {normalized_data.shape}')
    else:
        # Standard preprocessing path: normalize input data using gaussian_copula_cov_opt
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
        else:
            T, N = normalized_data.shape
        
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
    bias_correction: bool = True,
    covmats: Optional[torch.Tensor] = None,
    precomputed: bool = False
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
        When precomputed=False: Must be normalized using normalize_input_data first.
        When precomputed=True: Must be already normalized data with shape (D, T, N).
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
    covmats : torch.Tensor, optional
        Pre-computed covariance matrices with shape [D, N, N].
        Required when precomputed=True, optional otherwise.
    precomputed : bool, default=False
        If True, assumes X is already normalized data and covmats are provided.
        If False, performs full preprocessing including gaussian_copula_cov_opt.
        
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
        eps=eps,
        covmats=covmats,
        precomputed=precomputed
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


# ============================================================================
# TIME-DELAYED MEASURES - YOUR HIGH-PERFORMANCE FUNCTIONS
# ============================================================================

def generate_stacked_lagged_batches(data_list: List[torch.Tensor], shifts: torch.Tensor) -> torch.Tensor:
    """
    Stack shifted versions of multivariate time series.

    Parameters
    ----------
    data_list : list of L tensors, each of shape (T, N)
    shifts    : tensor of lags (e.g., torch.tensor([0, 1, 2, 5]))

    Returns
    -------
    Tensor of shape (L, T - max(shifts), len(shifts) * N)
    """
    L = len(data_list)
    T, N = data_list[0].shape
    device = data_list[0].device
    shifts = shifts.to(device)
    max_shift = int(shifts.max().item())

    X_tensor = torch.stack(data_list)  # (L, T, N)

    # Build shifted indices
    idx_base = torch.arange(T - max_shift, device=device)
    idx = idx_base.unsqueeze(0) + shifts.view(-1, 1)  # (len(shifts), T - max_shift)

    # Apply shifts
    lagged = X_tensor[:, idx, :]  # (L, len(shifts), T - max_shift, N)
    lagged = lagged.permute(0, 2, 1, 3).reshape(L, T - max_shift, -1)

    return lagged  # (L, T - max_shift, len(shifts) * N)


@torch.no_grad()
def batch_local_ais_torch(
    data_batch: torch.Tensor,   # [L, T, k*N] output of generate_stacked_lagged_batches
    cov_batch:  torch.Tensor,   # [L, k*N, k*N] covariance per block, same order as shifts
    shifts:     torch.Tensor,   # [k] with shifts; shifts[0] must be 0
    eps: float = 1e-10,
    device: str = 'cpu',
):
    """
    Returns:
      i_local : [L, T, k-1]  (nats) Local Active Information Storage (AIS) per lag (excluding shift 0)
      auc     : [L, T]       (nats·lag) Trapezoidal integral over shifts[1:]
    """
    L, T, D = data_batch.shape
    k = shifts.numel()
    assert D % k == 0, "D must be a multiple of len(shifts)"
    N = D // k

    data_batch = data_batch.to(device)
    cov_batch  = cov_batch.to(device)
    eyeN = torch.eye(N, device=device)

    # --- helpers for blocks (0 = present, 1..k-1 = past according to 'shifts')
    def block(i, j):
        return cov_batch[:, i*N:(i+1)*N, j*N:(j+1)*N]  # [L,N,N]

    Sigma_t  = block(0, 0) + eps * eyeN                          # [L,N,N]
    Sigma_y  = torch.stack([block(s, s) for s in range(1, k)], 1) + eps * eyeN  # [L,k-1,N,N]
    Sigma_ty = torch.stack([block(0, s) for s in range(1, k)], 1)              # [L,k-1,N,N]
    Sigma_yt = Sigma_ty.transpose(-1, -2)                                       # [L,k-1,N,N]

    # --- Cholesky and logdets
    L_t = torch.linalg.cholesky(Sigma_t)                                        # [L,N,N]
    logdet_t = 2.0 * torch.log(torch.diagonal(L_t, dim1=-2, dim2=-1)).sum(-1)  # [L]

    L_y = torch.linalg.cholesky(Sigma_y)                                        # [L,k-1,N,N]
    # Σ_{t|y} = Σ_t - Σ_ty Σ_y^{-1} Σ_yt
    M = torch.cholesky_solve(Sigma_yt, L_y)                                     # [L,k-1,N,N]
    Sigma_cond = Sigma_t.unsqueeze(1) - torch.matmul(Sigma_ty, M) + eps * eyeN  # [L,k-1,N,N]
    L_cond = torch.linalg.cholesky(Sigma_cond)                                  # [L,k-1,N,N]
    logdet_cond = 2.0 * torch.log(torch.diagonal(L_cond, dim1=-2, dim2=-1)).sum(-1)  # [L,k-1]

    # --- data x_t and past y_{τ}
    x = data_batch[:, :, :N]                                                    # [L,T,N]
    y = torch.stack([data_batch[:, :, s*N:(s+1)*N] for s in range(1, k)], 2)    # [L,T,k-1,N]

    # x^T Σ_t^{-1} x
    z_marg = torch.cholesky_solve(x.transpose(1, 2), L_t)                       # [L,N,T]
    quad_marg = (x.transpose(1, 2) * z_marg).sum(1)                              # [L,T]

    # μ = Σ_ty Σ_y^{-1} y
    y_rhs = y.permute(0, 2, 3, 1)                                               # [L,k-1,N,T]
    y_inv = torch.cholesky_solve(y_rhs, L_y)                                     # [L,k-1,N,T]
    mu = torch.matmul(Sigma_ty, y_inv).permute(0, 3, 1, 2)                       # [L,T,k-1,N]

    # (x-μ)^T Σ_{t|y}^{-1} (x-μ)
    diff = x.unsqueeze(2) - mu                                                  # [L,T,k-1,N]
    diff_T = diff.permute(0, 2, 3, 1)                                           # [L,k-1,N,T]
    z_cond = torch.cholesky_solve(diff_T, L_cond)                               # [L,k-1,N,T]
    quad_cond = (diff_T * z_cond).sum(2).permute(0, 2, 1)                        # [L,T,k-1]

    # i_local(t,τ) = 0.5[log|Σ_t| - log|Σ_{t|y_τ}|] + 0.5[x^T Σ_t^{-1} x - (x-μ)^T Σ_{t|y}^{-1}(x-μ)]
    i_local = 0.5 * (logdet_t[:, None, None] - logdet_cond[:, None, :]) + 0.5 * (quad_marg[:, :, None] - quad_cond)  # [L,T,k-1]

    return i_local


def gaussian_mi_bias_correction(T: int) -> torch.Tensor:
    """Compute the bias correction for Gaussian mutual information"""
    return torch.tensor((sp.psi((T-1)/2) - sp.psi((T-2)/2)) / 2)


def gaussian_ais_bias_correction(N: int, T: int, device='cpu') -> torch.Tensor:
    if T is None:
        return torch.tensor(0.0, device=device)
    def bias_H(n):
        psi_terms = torch.from_numpy(sp.psi((T - np.arange(1, n + 1)) / 2)).to(device)
        return 0.5 * (n * torch.log(torch.tensor(2.0/(T-1), device=device)) + psi_terms.sum())
    return bias_H(N) + bias_H(N) - bias_H(2 * N)


def compute_ais(
    cov_batch: torch.Tensor,
    lags: torch.Tensor,
    bias: torch.Tensor,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, D, _ = cov_batch.shape
    k = len(lags)
    N = D // (k)
    cov_batch = cov_batch + 1e-10 * torch.eye(D, device=device).unsqueeze(0)
    idx_t = torch.arange(N, device=device)
    Σ_t = cov_batch[:, idx_t][:, :, idx_t]
    Σ_lag = torch.stack([
        cov_batch[:, τ*N:(τ+1)*N, τ*N:(τ+1)*N]
        for τ in range(1, k)
    ], dim=1)
    joint_idx = torch.stack([
        torch.cat([idx_t, torch.arange(τ*N,(τ+1)*N,device=device)])
        for τ in range(1, k)
    ])
    batch_idx = torch.arange(B, device=device)[:,None,None,None]
    ji = joint_idx.unsqueeze(0).expand(B,-1,-1)
    rows = ji.unsqueeze(3).expand(-1,-1,-1,2*N)
    cols = ji.unsqueeze(2).expand(-1,-1,2*N,-1)
    Σ_joint = cov_batch[batch_idx, rows, cols]
    log_det_past  = torch.logdet(Σ_lag)
    log_det_joint = torch.logdet(Σ_joint)
    log_det_t     = torch.logdet(Σ_t)
    ais = 0.5*(log_det_past + log_det_t.unsqueeze(1) - log_det_joint) - bias
    ais = torch.clamp(ais, min=0.0)
    return ais 


def extract_subcov_batch_vec(
    cov_full: torch.Tensor,
    idxs: torch.Tensor,        # shape: (S, g)
    N_total: int,   
) -> torch.Tensor:
    """
    Extract sub-covariances from full lagged matrix using arbitrary lag positions.

    Supports both a single covariance matrix of shape (D, D) and a batch
    of covariance matrices of shape (B, D, D). When given a batch it returns
    a tensor of shape (B, S, g*k, g*k); when given a single matrix it returns
    (S, g*k, g*k).

    Parameters
    ----------
    cov_full : (D, D) or (B, D, D)
    idxs     : (S, g)   
    N_total  : number of channels

    Returns
    -------
    (S, g·(k), g·(k)) covariance blocks or (B, S, g·(k), g·(k)) for batched input
    """
    S, g = idxs.shape
    # Determine whether input is batched
    batched = cov_full.dim() == 3
    if batched:
        B, D, _ = cov_full.shape
        device = cov_full.device
    else:
        D = cov_full.size(0)
        device = cov_full.device

    # infer how many lags we have
    L = D // N_total  # must be integer

    # block offsets = [0, N_total, 2*N_total, ..., (L-1)*N_total]
    lag_offsets = torch.arange(L, device=device).view(L, 1, 1) * N_total  # (L,1,1)

    # para cada subset y cada lag sumamos offset
    # base: (L, S, g)
    base = idxs.unsqueeze(0) + lag_offsets

    # aplanamos a (S, L*g)
    flat = base.permute(1, 0, 2).reshape(S, L * g)  # (S, L*g)

    # construct row/column indices (S, L*g, L*g)
    rows = flat.unsqueeze(2).expand(-1, -1, L * g)
    cols = flat.unsqueeze(1).expand(-1, L * g, -1)

    if not batched:
        # extract for single matrix -> (S, L*g, L*g)
        return cov_full[rows, cols]
    else:
        # For batched input, build a batch index to perform elementwise advanced indexing
        # rows/cols -> expand to (B, S, L*g, L*g)
        rows_b = rows.unsqueeze(0).expand(B, -1, -1, -1)
        cols_b = cols.unsqueeze(0).expand(B, -1, -1, -1)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(-1, S, L * g, L * g)
        # cov_full is (B, D, D); advanced indexing returns (B, S, L*g, L*g)
        return cov_full[batch_idx, rows_b, cols_b]


@torch.no_grad()
def batch_compute_tdmi_xcorr_torch(
    cov_batch: torch.Tensor,
    lags: torch.Tensor,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute time‐delayed mutual information (TDMI) with bias‐correction and
    cross‐correlation (XCORR) for all variable pairs across specified lags,
    by extracting both from a single time‐embedded covariance batch.

    Parameters
    ----------
    cov_batch : torch.Tensor, shape (B, D, D)
        Batch of full lagged covariance matrices. D = N_vars * n_lags.
    lags      : torch.Tensor, shape (k,)
        Lag indices (including zero).
    device    : str, default 'cpu'
        Device on which to perform the computations.

    Returns
    -------
    tdmi  : torch.Tensor, shape (B, k, N, N)
        Bias‐corrected time‐delayed mutual information for each variable‐pair at each lag.
    xcorr : torch.Tensor, shape (B, k, N, N)
        Time‐delayed Pearson correlation coefficients for each pair at each lag.
    """
    B, D, _ = cov_batch.shape
    k = lags.numel()
    N = D // k

    # ensure numerical stability and move to device
    cov_batch = cov_batch.to(device) + 1e-10 * torch.eye(D, device=device).unsqueeze(0)

    # Cov(X_{t-τ}, X_t) for each lag τ
    Sigma_lagged = torch.stack([
        cov_batch[:, τ * N:(τ + 1) * N, :N]
        for τ in range(k)
    ], dim=1)  # → (B, k, N, N)

    # Covariance & variance at time t
    Sigma_t = cov_batch[:, :N, :N]               # (B, N, N)
    var_t   = torch.diagonal(Sigma_t, offset=0, dim1=1, dim2=2)       # (B, N)
    std_t   = torch.sqrt(var_t)                   # (B, N)

    # ----- cross‐correlation -----
    denom_corr = std_t.unsqueeze(2) * std_t.unsqueeze(1)       # (B, N, N)
    denom_corr = denom_corr.unsqueeze(1).expand(-1, k, -1, -1)  # (B, k, N, N)
    xcorr = Sigma_lagged / (denom_corr + 1e-10)
    xcorr = torch.clamp(xcorr, -0.999999, 0.999999)

   
    # compute Gaussian mutual information: I = -½ log(1 - ρ²)
    tdmi = -0.5 * torch.log1p(-xcorr.pow(2))

    return tdmi, xcorr


def build_full_tdmi(
    measure: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized construction of full, symmetric TDMI series for all batches
    and variable pairs, returning a torch.Tensor.

    Parameters
    ----------
    measure : torch.Tensor, shape (B, k, N, N)
        Batch‐dimension TDMI tensor for non‐negative lags:
        measure[b, τ, i, j] = I(X_i(t); X_j(t−τ)).

    Returns
    -------
    full_tdmi : torch.Tensor, shape (B, N, N, 2*k - 1)
        Combined TDMI for lags from −(k−1) to +(k−1):
        full_tdmi[b, i, j, :] = [I(j→i, backwards), …, I(i→j, forwards)].
        Diagonal (i==j) entries are set to NaN.
    """
    _, k, N, _ = measure.shape
    # L = 2*k - 1

    # Forward: (B, k, N, N)
    m_fwd = measure

    # Backward: swap i/j, reverse lag axis, drop duplicate zero‐lag
    m_bwd = measure.transpose(2, 3)           # (B, k, N, N)
    m_bwd = torch.flip(m_bwd, dims=[1])       # reverse along lag dim → (B, k, N, N)
    m_bwd = m_bwd[:, :-1, :, :]               # drop last frame → (B, k-1, N, N)

    # Concatenate backward + forward → (B, 2k-1, N, N)
    m_full = torch.cat([m_bwd, m_fwd], dim=1)

    # Permute to (B, N, N, 2k-1)
    full_tdmi = m_full.permute(0, 2, 3, 1)

    # Mask diagonal entries
    eye = torch.eye(N, device=measure.device, dtype=torch.bool)
    mask = eye.unsqueeze(0).unsqueeze(-1)     # → (1, N, N, 1), broadcastable
    full_tdmi = full_tdmi.masked_fill(mask, float('nan'))

    return full_tdmi


# ============================================================================
# WRAPPER FUNCTIONS FOR AIS AND TDMI
# ============================================================================

@torch.no_grad()
def local_ais(
    X: TensorLikeArray,
    shifts: Union[List[int], torch.Tensor, np.ndarray],
    *,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-10,
    covmats: Optional[torch.Tensor] = None,
    precomputed: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function for computing Local Active Information Storage (AIS).
    
    This function takes raw input data, applies the necessary preprocessing 
    (stacked lagging and Gaussian copula normalization), and computes the 
    local AIS using your optimized batch_local_ais_torch function.
    
    Parameters
    ----------
    X : TensorLikeArray
        Input data which can be:
        - A single torch.Tensor or np.ndarray with shape (T, N)
        - A list/sequence of torch.Tensor or np.ndarray each with shape (T, N)
    shifts : array-like
        Lag values including 0. Must start with 0 (present time).
        Example: [0, 1, 2, 5] for lags 0, 1, 2, and 5 time steps.
    device : str, default='cpu'
        Device for computation
    dtype : torch.dtype, default=torch.float32
        Data type for computation
    eps : float, default=1e-10
        Numerical stability epsilon
        
    Returns
    -------
    i_local : torch.Tensor
        Local AIS values with shape [L, T, k-1] where k-1 excludes lag 0
    auc : torch.Tensor  
        Area under curve (integral) over lags with shape [L, T]
    """
    from ..commons import gaussian_copula_cov_opt

    # Minimal parsing of shifts and validation
    shifts_tensor = torch.as_tensor(shifts, dtype=torch.int32, device=device)
    if shifts_tensor.numel() == 0 or shifts_tensor[0].item() != 0:
        raise ValueError("shifts must start with 0 (present time)")

    # Helper to ensure covmats is a batched tensor
    def _ensure_batched_cov(cov):
        c = torch.as_tensor(cov, dtype=dtype, device=device)
        if c.dim() == 2:
            c = c.unsqueeze(0)
        return c

    # PRECOMPUTED branch: expect covmats and stacked normalized data
    if precomputed:
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        covmats = _ensure_batched_cov(covmats)

        normalized_data = torch.as_tensor(X, dtype=dtype, device=device)
        if normalized_data.dim() == 2:
            normalized_data = normalized_data.unsqueeze(0)
        if normalized_data.dim() != 3:
            raise ValueError('When precomputed=True, X must be stacked lagged normalized data with shape (B, T_eff, D)')

        # Basic compatibility check
        if covmats.shape[1] != normalized_data.shape[2] or covmats.shape[2] != normalized_data.shape[2]:
            raise ValueError('covmats dimensionality is not compatible with provided stacked X')

    # NOT precomputed: build lagged stacks and normalize once
    if not precomputed:
        # Normalize input into a list of (T,N) tensors
        if isinstance(X, (torch.Tensor, np.ndarray)):
            if hasattr(X, 'dim') and X.dim() == 2:
                data_list = [torch.as_tensor(X, dtype=dtype, device=device)]
            elif isinstance(X, np.ndarray) and X.ndim == 2:
                data_list = [torch.tensor(X, dtype=dtype, device=device)]
            else:
                raise ValueError("Single dataset must have shape (T, N)")
        elif isinstance(X, (list, tuple)):
            data_list = [torch.as_tensor(x, dtype=dtype, device=device) for x in X]
        else:
            raise ValueError("X must be array-like or list of array-like")

        lagged_data = generate_stacked_lagged_batches(data_list, shifts_tensor)
        # Only compute normalization/covmats if not provided by precomputed branch
        normalized_data, covmats = gaussian_copula_cov_opt(lagged_data, return_xg=True)

    # Single unified call to the optimized function
    return batch_local_ais_torch(normalized_data, covmats, shifts_tensor, eps=eps, device=device)


@torch.no_grad()
def ais(
    X: TensorLikeArray,
    shifts: Union[List[int], torch.Tensor, np.ndarray],
    *,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    bias_correction: bool = True,
    covmats: Optional[torch.Tensor] = None,
    T: Optional[int] = None,
    precomputed: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function for computing Active Information Storage (AIS) directly 
    (non-local version) using the whole dataset.
    
    This function computes AIS using determinant-based calculations rather than
    the local (time-resolved) approach, providing average AIS values across lags.
    
    Parameters
    ----------
    X : TensorLikeArray
        Input data which can be:
        - A single torch.Tensor or np.ndarray with shape (T, N)
        - A list/sequence of torch.Tensor or np.ndarray each with shape (T, N)
    shifts : array-like
        Lag values including 0. Must start with 0 (present time).
        Example: [0, 1, 2, 5] for lags 0, 1, 2, and 5 time steps.
    device : str, default='cpu'
        Device for computation
    dtype : torch.dtype, default=torch.float32
        Data type for computation
    bias_correction : bool, default=True
        Whether to apply bias correction to the results
        
    Returns
    -------
    ais : torch.Tensor
        AIS values with shape [B, k-1] where k-1 excludes lag 0
    """
    from ..commons import gaussian_copula_cov_opt

    # Minimal parsing of shifts and validation
    shifts_tensor = torch.as_tensor(shifts, dtype=torch.int32, device=device)
    if shifts_tensor.numel() == 0 or shifts_tensor[0].item() != 0:
        raise ValueError("shifts must start with 0 (present time)")
    k = shifts_tensor.numel()

    def _ensure_batched_cov(cov):
        c = torch.as_tensor(cov, dtype=dtype, device=device)
        if c.dim() == 2:
            c = c.unsqueeze(0)
        return c

    def _build_bias(B, N_vars, T_samples):
        if bias_correction and T_samples is not None:
            bias_corr = gaussian_ais_bias_correction(N_vars, T_samples, device=device)
            return bias_corr.unsqueeze(0).expand(B, k - 1).to(device)
        elif bias_correction and T_samples is None:
            return torch.zeros(B, k - 1, device=device)
        else:
            return torch.zeros(B, k - 1, device=device)

    # PRECOMPUTED: use provided covmats (and optional X for T inference)
    if precomputed:
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        covmats = _ensure_batched_cov(covmats)

        B, D, _ = covmats.shape
        if D % k != 0:
            raise ValueError('covmats dimensionality is not compatible with provided shifts')
        N_vars = D // k

        # Infer temporal samples for bias correction (if possible)
        if X is not None:
            X_tensor = torch.as_tensor(X, dtype=dtype, device=device)
            if X_tensor.dim() == 2:
                T_samples = X_tensor.shape[0]
            elif X_tensor.dim() == 3:
                T_samples = X_tensor.shape[1]
            else:
                T_samples = T
        else:
            T_samples = T

        # Build bias tensor for precomputed path
        bias_tensor = _build_bias(B, N_vars, T_samples)
    else:
        # NOT precomputed: build lagged stacks and normalize once
        if isinstance(X, (torch.Tensor, np.ndarray)):
            if hasattr(X, 'dim') and X.dim() == 2:
                data_list = [torch.as_tensor(X, dtype=dtype, device=device)]
            elif isinstance(X, np.ndarray) and X.ndim == 2:
                data_list = [torch.tensor(X, dtype=dtype, device=device)]
            else:
                raise ValueError("Single dataset must have shape (T, N)")
        elif isinstance(X, (list, tuple)):
            data_list = [torch.as_tensor(x, dtype=dtype, device=device) for x in X]
        else:
            raise ValueError("X must be array-like or list of array-like")

        lagged_data = generate_stacked_lagged_batches(data_list, shifts_tensor)
        normalized_data, covmats = gaussian_copula_cov_opt(lagged_data, return_xg=True)
        # Prepare bias using actual temporal length if not set by precomputed branch    
        T_samples = normalized_data.shape[1]
        N_vars = lagged_data.shape[2] // k
        B = covmats.shape[0]
        # Build bias tensor once and call compute_ais a single time
        bias_tensor = _build_bias(B, N_vars, T_samples)




    return compute_ais(covmats, shifts_tensor, bias_tensor, device=device)


@torch.no_grad()
def ais_subset(
    X: Optional[TensorLikeArray] = None,
    shifts: Union[List[int], torch.Tensor, np.ndarray] = None,
    idxs: Union[List[int], torch.Tensor, np.ndarray] = None,
    *,
    covmats: Optional[torch.Tensor] = None,
    T: Optional[int] = None,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    bias_correction: bool = True,
    precomputed: bool = False,
) -> torch.Tensor:
    """
    Compute AIS for a specific subset of variables using extract_subcov_batch_vec.

    Parameters
    ----------
    X : optional
        Raw data (single array (T,N) or list of arrays). If provided, `covmats` is ignored.
    shifts : array-like
        Lags including 0. Must start with 0.
    idxs : array-like
        Indices of variables in the original N to include in the subset (e.g. [0,2,3]).
    covmats : optional
        Precomputed full lagged covariance matrix/matrices with shape [B, D, D] or [D, D].
    T : int, optional
        Number of samples used to compute bias correction when providing covmats directly.
    device, dtype, bias_correction : same as other wrappers.

    Returns
    -------
    torch.Tensor
        AIS values with shape [B, k-1] (or [B, S, k-1] if multiple index-sets provided)
    """
    from ..commons import gaussian_copula_cov_opt
    if shifts is None:
        raise ValueError('shifts must be provided')
    if idxs is None and covmats is None and X is None:
        raise ValueError('Either idxs and X or covmats must be provided')

    # Prepare shifts tensor on device
    shifts_tensor = torch.as_tensor(shifts, dtype=torch.long, device=device)
    if shifts_tensor.numel() == 0 or shifts_tensor[0].item() != 0:
        raise ValueError('shifts must start with 0 (present time)')

    k = shifts_tensor.numel()

    # If precomputed=True we expect covmats to be passed and X (if passed) to be
    # already the stacked/normalized representation (shape: B, T_eff, D).
    covmats_full = None
    normalized_data = None

    if precomputed:
        # Use provided covmats (required in precomputed mode)
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        covmats_full = torch.as_tensor(covmats, dtype=dtype, device=device)

        # If X is provided, check compatibility: X should be stacked (B, T_eff, D)
        if X is not None:
            X_tensor = torch.as_tensor(X, dtype=dtype, device=device)
            if X_tensor.dim() == 2:
                # single dataset provided as (T_eff, D) -> make batched
                X_tensor = X_tensor.unsqueeze(0)
            if X_tensor.dim() != 3:
                raise ValueError('When precomputed=True, X must be stacked lagged data with shape (B, T_eff, D)')
            # ensure covmats D matches X D
            if covmats_full.dim() == 2:
                covmats_full = covmats_full.unsqueeze(0)
            if covmats_full.shape[1] != X_tensor.shape[2] or covmats_full.shape[2] != X_tensor.shape[2]:
                raise ValueError('covmats dimensionality is not compatible with provided stacked X')
            normalized_data = X_tensor
    else:
        # If raw data provided, compute lagged covmats first
        if X is not None:
            # Normalize input to list format
            if isinstance(X, (torch.Tensor, np.ndarray)):
                if hasattr(X, 'ndim') and X.ndim == 2:
                    data_list = [torch.as_tensor(X, dtype=dtype, device=device)]
                else:
                    raise ValueError('Single dataset must have shape (T, N)')
            elif isinstance(X, (list, tuple)):
                data_list = [torch.as_tensor(x, dtype=dtype, device=device) for x in X]
            else:
                raise ValueError('X must be array-like or list of array-like')

            # Generate stacked lagged batches and compute covariance matrices
            lagged_data = generate_stacked_lagged_batches(data_list, shifts_tensor)
            normalized_data, covmats_full = gaussian_copula_cov_opt(lagged_data, return_xg=True)

        # If covmats argument provided use it (overrides X)
        if covmats is not None:
            covmats_full = torch.as_tensor(covmats, dtype=dtype, device=device)

    if covmats_full is None:
        raise ValueError('Unable to obtain covariance matrices from X or covmats')

    # Ensure batched shape (B, D, D)
    if covmats_full.dim() == 2:
        covmats_full = covmats_full.unsqueeze(0)
    B, D, _ = covmats_full.shape

    # Prepare idxs tensor: support 1D (single set) or 2D (multiple sets)
    idxs_tensor = torch.as_tensor(idxs, dtype=torch.long, device=device)
    if idxs_tensor.dim() == 1:
        idxs_tensor = idxs_tensor.unsqueeze(0)  # (S=1, g)
    S, g = idxs_tensor.shape

    # infer N_total (channels per lag)
    N_total = D // k
    if N_total * k != D:
        raise ValueError('covmats dimensionality is not compatible with provided shifts')

    # Vectorized extraction: returns (B, S, g*k, g*k)
    subcovs = extract_subcov_batch_vec(covmats_full, idxs_tensor, N_total)
    # subcovs should be (B, S, gk, gk)
    if subcovs.dim() == 3:
        # single matrix returned (S, gk, gk) -> make batched
        subcovs = subcovs.unsqueeze(0)

    # reshape to (B*S, gk, gk) for compute_ais
    gk = g * k
    cov_sub_batch = subcovs.reshape(B * S, gk, gk)

    # Determine number of temporal samples for bias correction
    if normalized_data is not None:
        T_samples = normalized_data.shape[1]
    else:
        T_samples = T

    # Prepare bias tensor per batch element
    if bias_correction and T_samples is not None:
        bias_corr = gaussian_ais_bias_correction(g, T_samples, device=device)
        bias_tensor = bias_corr.unsqueeze(0).expand(B * S, k - 1).to(device)
    elif bias_correction and T_samples is None:
        # cannot compute bias without T: fallback to zeros
        bias_tensor = torch.zeros(B * S, k - 1, device=device)
    else:
        bias_tensor = torch.zeros(B * S, k - 1, device=device)

    # Compute AIS on the extracted subcovariances (fully vectorized)
    ais_vals = compute_ais(cov_sub_batch, shifts_tensor, bias_tensor, device=device)

    # Reshape back to (B, S, k-1) and squeeze single-S if needed
    ais_vals = ais_vals.view(B, S, -1)
    if S == 1:
        ais_vals = ais_vals.squeeze(1)

    return ais_vals


@torch.no_grad()
def local_ais_subset(
    X: Optional[TensorLikeArray] = None,
    shifts: Union[List[int], torch.Tensor, np.ndarray] = None,
    idxs: Union[List[int], torch.Tensor, np.ndarray] = None,
    *,
    covmats: Optional[torch.Tensor] = None,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-10,
    precomputed: bool = False,
) -> torch.Tensor:
    """
    Compute local AIS for subsets of variables (vectorized, batched, no loops).

    This function expects raw data `X` in the same format used by `local_ais`:
    - a single (T, N) array/tensor or a list of (T, N) arrays -> will be treated
      as B datasets. It will build the lagged stacked representation using
      `generate_stacked_lagged_batches` and normalize it with
      `gaussian_copula_cov_opt`.

    Parameters
    ----------
    X : required
        Raw data (single T×N array or list of T×N arrays). Normalized data is
        derived internally.
    shifts : array-like
        Lag values including 0 (must start with 0).
    idxs : array-like (S, g) or (g,)
        Indices of variables to include in each subset. Can be a single set
        (g,) or multiple sets (S, g).
    covmats : optional
        If provided, overrides covariances computed from X when extracting
        sub-block covariances. Note: X is still required because local AIS
        needs time-resolved normalized data.
    Returns
    -------
    torch.Tensor
        Local AIS values with shape [B, S, T, k-1] (or [B, T, k-1] when S==1).
    """
    from ..commons import gaussian_copula_cov_opt

    # Minimal argument validation
    if shifts is None or idxs is None:
        raise ValueError('shifts and idxs must be provided')

    shifts_tensor = torch.as_tensor(shifts, dtype=torch.long, device=device)
    if shifts_tensor.numel() == 0 or shifts_tensor[0].item() != 0:
        raise ValueError('shifts must start with 0')

    k = shifts_tensor.numel()

    covmats_full = None
    normalized_data = None

    if precomputed:
        # precomputed expects stacked/normalized X and covmats provided
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        covmats_full = torch.as_tensor(covmats, dtype=dtype, device=device)

        if X is None:
            raise ValueError('X (stacked normalized data) must be provided when precomputed=True')
        X_tensor = torch.as_tensor(X, dtype=dtype, device=device)
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(0)
        if X_tensor.dim() != 3:
            raise ValueError('When precomputed=True, X must be stacked lagged normalized data with shape (B, T_eff, D)')

        # Ensure covmats batch dims
        if covmats_full.dim() == 2:
            covmats_full = covmats_full.unsqueeze(0)

        # Check compatibility between X and covmats
        if covmats_full.shape[1] != X_tensor.shape[2] or covmats_full.shape[2] != X_tensor.shape[2]:
            raise ValueError('covmats dimensionality is not compatible with provided stacked X')

        normalized_data = X_tensor
        B, T_eff, D = normalized_data.shape
    else:
        # Prepare data list and build lagged stacked batches
        if X is None:
            raise ValueError('X (raw data) must be provided for local AIS (time-resolved)')
        if isinstance(X, (torch.Tensor, np.ndarray)):
            if hasattr(X, 'ndim') and X.ndim == 2:
                data_list = [torch.as_tensor(X, dtype=dtype, device=device)]
            else:
                raise ValueError('Single dataset must have shape (T, N)')
        elif isinstance(X, (list, tuple)):
            data_list = [torch.as_tensor(x, dtype=dtype, device=device) for x in X]
        else:
            raise ValueError('X must be array-like or list of array-like')

        lagged_data = generate_stacked_lagged_batches(data_list, shifts_tensor)  # [B, T_eff, k*N]
        normalized_data, covmats_full = gaussian_copula_cov_opt(lagged_data, return_xg=True)

        # covmats argument overrides internal covmats for extraction if provided
        if covmats is not None:
            covmats_full = torch.as_tensor(covmats, dtype=dtype, device=device)

        # Ensure batched covmats
        if covmats_full.dim() == 2:
            covmats_full = covmats_full.unsqueeze(0)
        B, T_eff, D = normalized_data.shape

    # idxs -> (S, g)
    idxs_tensor = torch.as_tensor(idxs, dtype=torch.long, device=device)
    if idxs_tensor.dim() == 1:
        idxs_tensor = idxs_tensor.unsqueeze(0)
    S, g = idxs_tensor.shape

    # infer channels per lag
    N_total = D // k

    # build per-subset column indices: (S, k, g) -> (S, k*g)
    lag_offsets = (torch.arange(k, device=device, dtype=torch.long).view(k, 1) * N_total)  # (k,1)
    # positions: (k, S, g)
    positions = (lag_offsets.unsqueeze(1) + idxs_tensor.unsqueeze(0))  # (k,S,g)
    positions = positions.permute(1, 0, 2).reshape(S, k * g)  # (S, k*g)

    # Gather selected columns from normalized_data in a batched, vectorized way
    # normalized_data: (B, T_eff, D) -> expand to (B, S, T_eff, D)
    Xexp = normalized_data.unsqueeze(1).expand(B, S, T_eff, D)
    # indices for gather -> (B, S, T_eff, k*g)
    idx_gather = positions.unsqueeze(0).unsqueeze(2).expand(B, S, T_eff, k * g)
    data_selected = torch.gather(Xexp, dim=3, index=idx_gather)  # (B, S, T_eff, k*g)

    # reshape to (B*S, T_eff, k*g) to match batch_local_ais_torch input convention
    data_batch = data_selected.reshape(B * S, T_eff, k * g)

    # Extract covariance sub-blocks for each subset: (B, S, gk, gk)
    subcovs = extract_subcov_batch_vec(covmats_full, idxs_tensor, N_total)
    if subcovs.dim() == 3:
        subcovs = subcovs.unsqueeze(0)
    cov_sub_batch = subcovs.reshape(B * S, g * k, g * k)

    # Compute local AIS on the batched subsets
    i_local = batch_local_ais_torch(
        data_batch,      # [B*S, T_eff, k*g]
        cov_sub_batch,   # [B*S, k*g, k*g]
        shifts_tensor,   # shifts
        eps=eps,
        device=device,
    )  # returns [B*S, T_eff, k-1]

    # Reshape back to (B, S, T_eff, k-1)
    i_local = i_local.view(B, S, T_eff, -1)
    if S == 1:
        i_local = i_local.squeeze(1)  # -> (B, T_eff, k-1)

    return i_local


@torch.no_grad()
def tdmi(
    X: Optional[TensorLikeArray] = None,
    lags: Union[List[int], torch.Tensor, np.ndarray] = None,
    *,
    covmats: Optional[torch.Tensor] = None,
    T: Optional[int] = None,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    bias_correction: bool = True,
    return_full: bool = True,
    precomputed: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Wrapper function for computing Time-Delayed Mutual Information (TDMI).
    
    This function takes raw input data, applies the necessary preprocessing 
    (stacked lagging and Gaussian copula normalization), and computes the 
    TDMI using your optimized batch_compute_tdmi_xcorr_torch function.
    
    Parameters
    ----------
    X : TensorLikeArray, optional
        Input data which can be:
        - A single torch.Tensor or np.ndarray with shape (T, N)
        - A list/sequence of torch.Tensor or np.ndarray each with shape (T, N)
        When precomputed=True, X can be None (only covmats needed).
    lags : array-like
        Lag values including 0. Must start with 0 (present time).
        Example: [0, 1, 2, 5] for lags 0, 1, 2, and 5 time steps.
    covmats : torch.Tensor, optional
        Pre-computed covariance matrices with shape [B, D, D] where D = N*k.
        Required when precomputed=True, optional otherwise.
    T : int, optional
        Number of time samples in the original data. Required when precomputed=True
        for proper bias correction computation. When precomputed=False, this is
        inferred from the input data.
    device : str, default='cpu'
        Device for computation
    dtype : torch.dtype, default=torch.float32
        Data type for computation
    bias_correction : bool, default=True
        Whether to apply bias correction to the TDMI results
    return_full : bool, default=True
        Whether to return the full bidirectional TDMI matrix
    precomputed : bool, default=False
        If True, assumes covmats are provided and skips data preprocessing.
        If False, performs full preprocessing from raw data X.
        
    Returns
    -------
    tdmi : torch.Tensor
        TDMI values with shape [B, k, N, N] for forward lags
    xcorr : torch.Tensor
        Cross-correlation values with shape [B, k, N, N]
    full_tdmi : torch.Tensor, optional
        Full bidirectional TDMI matrix with shape [B, N, N, 2*k-1] if return_full=True
    """
    from ..commons import gaussian_copula_cov_opt
    
    # Minimal argument validation
    if lags is None:
        raise ValueError('lags must be provided')
    
    lags_tensor = torch.as_tensor(lags, dtype=torch.int32, device=device)
    if lags_tensor.numel() == 0 or lags_tensor[0].item() != 0:
        raise ValueError('lags must start with 0 (present time)')
    
    k = lags_tensor.numel()
    
    def _build_bias(B, N_vars, T_samples):
        """Helper to build bias correction tensor for TDMI."""
        if not bias_correction:
            return 0.0
        bias_val = gaussian_mi_bias_correction(T_samples)
        return bias_val
    
    # Prepare covmats and bias based on precomputed flag
    if precomputed:
        # Use provided covmats
        if covmats is None:
            raise ValueError('covmats must be provided when precomputed=True')
        if T is None:
            raise ValueError('T (number of time samples) must be provided when precomputed=True')
        
        covmats_final = torch.as_tensor(covmats, dtype=dtype, device=device)
        if covmats_final.dim() == 2:
            covmats_final = covmats_final.unsqueeze(0)
        
        B, D, _ = covmats_final.shape
        N_vars = D // k
        T_samples = T
    else:
        # Full preprocessing from raw data
        if X is None:
            raise ValueError('X must be provided when precomputed=False')
        
        # Normalize input to list format for generate_stacked_lagged_batches
        if isinstance(X, (torch.Tensor, np.ndarray)):
            if hasattr(X, 'dim') and X.dim() == 2:
                data_list = [torch.as_tensor(X, dtype=dtype, device=device)]
            elif isinstance(X, np.ndarray) and X.ndim == 2:
                data_list = [torch.tensor(X, dtype=dtype, device=device)]
            else:
                raise ValueError("Single dataset must have shape (T, N)")
        elif isinstance(X, (list, tuple)):
            data_list = [torch.as_tensor(x, dtype=dtype, device=device) for x in X]
        else:
            raise ValueError("X must be array-like or list of array-like")
        
        # Generate stacked lagged batches
        lagged_data = generate_stacked_lagged_batches(data_list, lags_tensor)
        
        # Apply Gaussian copula normalization to get proper covariance structure
        normalized_data, covmats_final = gaussian_copula_cov_opt(lagged_data, return_xg=True)
        
        B, T_samples, D = normalized_data.shape
        N_vars = D // k
    
    # Single call to the optimized TDMI function
    bias_corr = _build_bias(B, N_vars, T_samples)
    tdmi_vals, xcorr_vals = batch_compute_tdmi_xcorr_torch(
        covmats_final,    # cov_batch
        lags_tensor,      # lags
        device=device     # device
    )
    
    # Apply bias correction
    tdmi_vals = tdmi_vals - bias_corr
    
    if return_full:
        # Build full bidirectional TDMI matrix
        full_tdmi = build_full_tdmi(tdmi_vals)
        return tdmi_vals, xcorr_vals, full_tdmi
    else:
        return tdmi_vals, xcorr_vals
