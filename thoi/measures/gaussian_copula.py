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

_VALID_GAUSSIAN_MODES = ('full', 'fast', 'only_o')


def _validate_gaussian_mode(mode: str) -> str:
    if mode not in _VALID_GAUSSIAN_MODES:
        valid_modes = ', '.join(repr(valid_mode) for valid_mode in _VALID_GAUSSIAN_MODES)
        raise ValueError(f"mode must be one of {valid_modes}; got {mode!r}")
    return mode

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
        F.one_hot(torch.as_tensor(lst, dtype=torch.long), num_classes=N).sum(dim=0)
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
    

def _generate_nplets_covmats(covmats: torch.Tensor, nplets: torch.Tensor):
    
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


def _get_bias_correctors(T: Optional[List[int]], order: int, batch_size: int, D: int, device: torch.device, dtype: torch.dtype):
    if T is not None:
        # |batch_size|
        bc1 = torch.tensor([_gaussian_entropy_bias_correction(1,t) for t in T], device=device, dtype=dtype)
        bcN = torch.tensor([_gaussian_entropy_bias_correction(order,t) for t in T], device=device, dtype=dtype)
        bcNmin1 = torch.tensor([_gaussian_entropy_bias_correction(order-1,t) for t in T], device=device, dtype=dtype)
    else:
        # |batch_size|
        bc1 = torch.tensor([0] * D, device=device, dtype=dtype)
        bcN = torch.tensor([0] * D, device=device, dtype=dtype)
        bcNmin1 = torch.tensor([0] * D, device=device, dtype=dtype)

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


def _get_fast_tc_dtc_from_batched_covmat(covmats: torch.Tensor,
                                         order: int,
                                         bc1: torch.Tensor,
                                         bcN: torch.Tensor,
                                         bcNmin1: torch.Tensor):
    """
    Compute Gaussian TC, DTC, O, and S from covariance and precision matrices.

    This is algebraically equivalent to the entropy-based estimator, but avoids
    the single-exclusion entropy calculations required by the full path.
    """

    chol = torch.linalg.cholesky(covmats)
    diag_cov = torch.diagonal(covmats, dim1=-2, dim2=-1)
    eye = torch.eye(order, dtype=covmats.dtype, device=covmats.device)
    eye = eye.expand(covmats.shape[0], order, order)
    inv_chol = torch.linalg.solve_triangular(chol, eye, upper=False)
    diag_precision = (inv_chol * inv_chol).sum(dim=1)
    logdet_cov = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=1)

    nplet_tc = 0.5 * (torch.log(diag_cov).sum(dim=1) - logdet_cov)
    nplet_tc += bcN - order * bc1

    nplet_dtc = 0.5 * (logdet_cov + torch.log(diag_precision).sum(dim=1))
    nplet_dtc += (order - 1) * bcN - order * bcNmin1

    nplet_o = nplet_tc - nplet_dtc
    nplet_s = nplet_tc + nplet_dtc

    return nplet_tc, nplet_dtc, nplet_o, nplet_s


def _get_o_from_batched_covmat(covmats: torch.Tensor,
                               order: int,
                               bc1: torch.Tensor,
                               bcN: torch.Tensor,
                               bcNmin1: torch.Tensor):
    """
    Compute only Gaussian O-information from covariance and precision matrices.

    This keeps the only_o path cheaper than the full fast path by avoiding TC and
    DTC materialization.
    """

    chol = torch.linalg.cholesky(covmats)
    diag_cov = torch.diagonal(covmats, dim1=-2, dim2=-1)
    eye = torch.eye(order, dtype=covmats.dtype, device=covmats.device)
    eye = eye.expand(covmats.shape[0], order, order)
    inv_chol = torch.linalg.solve_triangular(chol, eye, upper=False)
    diag_precision = (inv_chol * inv_chol).sum(dim=1)
    logdet_cov = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=1)

    nplet_o = 0.5 * (
        torch.log(diag_cov).sum(dim=1)
        - torch.log(diag_precision).sum(dim=1)
        - 2.0 * logdet_cov
    )
    nplet_o += -order * bc1 + order * bcNmin1 - (order - 2) * bcN

    return nplet_o

@torch.no_grad()
def nplets_measures(X: Union[TensorLikeArray],
                    nplets: Optional[TensorLikeArray] = None,
                    *,
                    covmat_precomputed: bool = False,
                    T: Optional[Union[int, List[int]]] = None,
                    device: torch.device = torch.device('cpu'),
                    verbose: int = logging.INFO,
                    batch_size: int = 1000000,
                    batch_size_D: Optional[int] = None,
                    mode: str = 'full'):
    
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
    batch_size_D : int or None, optional
        Number of datasets to process per batch during Gaussian copula covariance computation.
        Reduces peak memory when D is large. Default is None (all datasets at once).
    mode : {'full', 'fast', 'only_o'}, optional
        Computation mode. ``'full'`` uses the standard entropy-based estimator.
        ``'fast'`` uses compact Gaussian covariance/precision formulas to compute TC and DTC,
        then computes O and S from their difference and sum. ``'only_o'`` computes only
        O-information with the direct Gaussian precision-matrix estimator; the returned tensor
        keeps the standard `(tc, dtc, o, s)` layout with `tc`, `dtc`, and `s` filled with NaN.
        Default is ``'full'``.

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
    mode = _validate_gaussian_mode(mode)
    is_only_o = mode == 'only_o'
    is_fast = mode == 'fast'
    
    covmats, D, N, T = _normalize_input_data(X, covmat_precomputed, T, device, batch_size_D=batch_size_D)
    
    # If nplets is a list of nplets with different orders, then use hot encoding to compute multiorder measures
    if isinstance(nplets, list) and not all([len(nplet) == len(nplets[0]) for nplet in nplets]):
        if mode != 'full':
            raise NotImplementedError(f"mode={mode!r} is not implemented for mixed-order nplets lists")
        logging.warning('Using hot encoding to compute multi-order measures as nplets have different orders')
        nplets = _indices_to_hot_encoded(nplets, N)
        return nplets_measures_hot_encoded(covmats, nplets, covmat_precomputed=True, T=T)
    elif nplets is None:
        nplets = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0)
    else:
        nplets = torch.as_tensor(nplets, device=device, dtype=torch.long).contiguous()
        
    # nplets must be a batched tensor
    assert len(nplets.shape) == 2, 'nplets must be a batched tensor with shape (batch_size, order)'
    batch_size = min(batch_size, len(nplets))
    order = nplets.shape[1]

    # Create marginal indexes
    # |N| x |N-1|
    allmin1 = None if mode != 'full' else _all_min_1_ids(order, device=device)

    # Create bias corrector values
    # |batch_size x D|, |batch_size x D|, |batch_size x D|
    bc1, bcN, bcNmin1 = _get_bias_correctors(T, order, batch_size, D, device, covmats.dtype)

    # Create DataLoader for nplets
    dataloader = DataLoader(nplets, batch_size=batch_size, shuffle=False)

    results = []
    for nplet_batch in tqdm(dataloader, desc='Processing n-plets', leave=False):
        curr_batch_size = nplet_batch.shape[0]

        # Create the covariance matrices for each nplet in the batch
        # |curr_batch_size| x |D| x |order| x |order|
        nplets_covmats = _generate_nplets_covmats(covmats, nplet_batch)
        
        # Pack covmats in a single batch
        # |curr_batch_size x D| x |order| x |order|
        nplets_covmats = nplets_covmats.view(curr_batch_size * D, order, order)

        if is_only_o:
            nplets_o = _get_o_from_batched_covmat(
                nplets_covmats,
                order,
                bc1[:curr_batch_size * D],
                bcN[:curr_batch_size * D],
                bcNmin1[:curr_batch_size * D],
            )
            nplets_tc = torch.full_like(nplets_o, torch.nan)
            nplets_dtc = torch.full_like(nplets_o, torch.nan)
            nplets_s = torch.full_like(nplets_o, torch.nan)
        elif is_fast:
            measures = _get_fast_tc_dtc_from_batched_covmat(
                nplets_covmats,
                order,
                bc1[:curr_batch_size * D],
                bcN[:curr_batch_size * D],
                bcNmin1[:curr_batch_size * D],
            )
            nplets_tc, nplets_dtc, nplets_o, nplets_s = measures
        else:
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


def _batch_processing_multi_order(
    N: int,
    min_order: int,
    max_order: int,
    batch_fn: Callable,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    batch_data_collector: Optional[Callable] = None,
    batch_aggregation: Optional[Callable] = None,
) -> dict:
    """Shared batch iteration engine for multi-order n-plet measures.

    For each order K in [min_order, max_order], lazily generates all C(N, K)
    n-plets via CovarianceDataset + DataLoader, then for each batch:
      1. calls ``batch_fn(nplets_batch, K)`` → batch_result
      2. calls ``batch_data_collector(nplets_batch, batch_result, batch_number)`` → item
      3. collects all items and calls ``batch_aggregation(items)`` → order_result

    Parameters
    ----------
    N : int
        Total number of variables.
    min_order, max_order : int
        Inclusive range of orders to process.
    batch_fn : callable
        ``(nplets: Tensor[B, K], K: int) -> Any`` — core computation per n-plet batch.
    batch_size : int
        Maximum number of n-plets per DataLoader batch.
    device : torch.device
        Device on which n-plet index tensors are generated.
    num_workers : int, default 0
        DataLoader worker count.
    batch_data_collector : callable, optional
        ``(nplets: Tensor[B, K], batch_result: Any, bn: int) -> Any``
        Post-processes each batch result. Defaults to the identity.
    batch_aggregation : callable, optional
        ``(items: list[Any]) -> Any``
        Aggregates all collected items for one order.
        Defaults to ``torch.cat(items, dim=0)``.

    Returns
    -------
    dict
        ``{K: aggregated_result}`` for each order K in [min_order, max_order].
    """
    out = {}
    for K in tqdm(range(min_order, max_order + 1), leave=False, desc='Order',
                  disable=(min_order == max_order)):
        dataset = CovarianceDataset(N, K, device=device)
        dataloader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            num_workers=num_workers,
        )
        collected = []
        for bn, nplets in enumerate(tqdm(dataloader, total=len(dataloader),
                                         leave=False, desc='Batch')):
            nplets = nplets.to(device)
            batch_result = batch_fn(nplets, K)
            item = batch_data_collector(nplets, batch_result, bn) \
                if batch_data_collector is not None else batch_result
            collected.append(item)
        out[K] = batch_aggregation(collected) \
            if batch_aggregation is not None else torch.cat(collected, dim=0)
    return out


@torch.no_grad()
def multi_order_measures(X: TensorLikeArray,
                         min_order: int=3,
                         max_order: Optional[int]=None,
                         *,
                         covmat_precomputed: bool=False,
                         T: Optional[Union[int, List[int]]]=None,
                         batch_size: int = 1000000,
                         batch_size_D: Optional[int] = None,
                         device: torch.device = torch.device('cpu'),
                         num_workers: int = 0,
                         batch_aggregation: Optional[Callable[[any],any]] = None,
                         batch_data_collector: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],any]] = None,
                         mode: str = 'full'):
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
    batch_size_D : int or None, optional
        Number of datasets to process per batch during Gaussian copula covariance computation.
        Reduces peak memory when D is large. Default is None (all datasets at once).
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
    mode : {'full', 'fast', 'only_o'}, optional
        Computation mode. ``'full'`` uses the standard entropy-based estimator.
        ``'fast'`` uses compact Gaussian covariance/precision formulas to compute TC and DTC,
        then computes O and S from their difference and sum. ``'only_o'`` computes only
        O-information with the direct Gaussian precision-matrix estimator; collectors receive
        the standard `(tc, dtc, o, s)` tensors with `tc`, `dtc`, and `s` filled with NaN.
        Default is ``'full'``.

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

    mode = _validate_gaussian_mode(mode)
    is_only_o = mode == 'only_o'
    is_fast = mode == 'fast'

    covmats, D, N, T = _normalize_input_data(X, covmat_precomputed, T, device, batch_size_D=batch_size_D)

    # For each dataset, precompute the single variable marginal gaussian entropies
    # |D| x |N|
    marginal_entropies = None if mode != 'full' else _marginal_gaussian_entropies(covmats)

    max_order = N if max_order is None else max_order

    if batch_aggregation is None:
        batch_aggregation = concat_and_sort_csv
    if batch_data_collector is None:
        batch_data_collector = partial(batch_to_csv, N=N)

    assert max_order <= N, f"max_order must be lower or equal than N. {max_order} > {N})"
    assert min_order <= max_order, f"min_order must be lower or equal than max_order. {min_order} > {max_order}"

    # Ensure that final batch_size is smaller than the original batch_size
    batch_size = max(batch_size // D, 1)

    # Cache order-specific constants so they are computed once per order, not once per batch.
    _order_cache: dict = {}

    def _batch_fn(nplets, K):
        curr_B = nplets.shape[0]
        if K not in _order_cache:
            allmin1 = None if mode != 'full' else _all_min_1_ids(K, device=device)
            _order_cache[K] = (
                allmin1,
                _get_bias_correctors(T, K, batch_size, D, device, covmats.dtype),
            )
        allmin1, (bc1, bcN, bcNmin1) = _order_cache[K]

        # |curr_B| x |D| x |K| x |K|  →  |curr_B*D| x |K| x |K|
        nplets_covmats = _generate_nplets_covmats(covmats, nplets).view(curr_B * D, K, K)
        if is_only_o:
            o = _get_o_from_batched_covmat(
                nplets_covmats,
                K,
                bc1[:curr_B * D],
                bcN[:curr_B * D],
                bcNmin1[:curr_B * D],
            )
            tc = torch.full_like(o, torch.nan)
            dtc = torch.full_like(o, torch.nan)
            s = torch.full_like(o, torch.nan)
        elif is_fast:
            tc, dtc, o, s = _get_fast_tc_dtc_from_batched_covmat(
                nplets_covmats,
                K,
                bc1[:curr_B * D],
                bcN[:curr_B * D],
                bcNmin1[:curr_B * D],
            )
        else:
            # |curr_B| x |D| x |K|  →  |curr_B*D| x |K|
            nplets_marginal = _generate_nplets_marginal_entropies(marginal_entropies, nplets).view(curr_B * D, K)

            tc, dtc, o, s = _get_tc_dtc_from_batched_covmat(
                nplets_covmats, allmin1,
                bc1[:curr_B * D], bcN[:curr_B * D], bcNmin1[:curr_B * D],
                nplets_marginal,
            )
        # Return stacked Tensor[B, D, 4] so the collector can unpack uniformly.
        return torch.stack([
            tc.view(curr_B, D), dtc.view(curr_B, D),
            o.view(curr_B, D),  s.view(curr_B, D),
        ], dim=-1)

    def _internal_collector(nplets, result, bn):
        # Adapt internal (nplets, Tensor[B, D, 4], bn) to the public 6-arg signature.
        return batch_data_collector(
            nplets,
            result[:, :, 0], result[:, :, 1],
            result[:, :, 2], result[:, :, 3],
            bn,
        )

    per_order = _batch_processing_multi_order(
        N=N, min_order=min_order, max_order=max_order,
        batch_fn=_batch_fn,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        batch_data_collector=_internal_collector,
        batch_aggregation=lambda items: items,  # no per-order aggregation
    )

    # Flatten per-order lists into a single list in the same order as before,
    # then apply the public batch_aggregation once (identical semantics to the original).
    all_items = [item for items_for_order in per_order.values() for item in items_for_order]
    return batch_aggregation(all_items)
