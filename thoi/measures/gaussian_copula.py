from typing import Optional, Callable, Union, List

from tqdm import tqdm
import logging

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from thoi.typing import TensorLikeArray
from thoi.commons import _normalize_input_data
from thoi.measures.gaussian_copula_hot_encoded import nplets_measures_hot_encoded
from thoi.measures.utils import _gaussian_entropy_bias_correction
from thoi.batch_processing_multi_order import _batch_processing_multi_order

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

@torch.no_grad()
def nplets_measures(X: Union[TensorLikeArray],
                    nplets: Optional[TensorLikeArray] = None,
                    *,
                    covmat_precomputed: bool = False,
                    T: Optional[Union[int, List[int]]] = None,
                    device: torch.device = torch.device('cpu'),
                    verbose: int = logging.INFO,
                    batch_size: int = 1000000,
                    batch_size_D: Optional[int] = None):
    
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

    covmats, D, N, T = _normalize_input_data(X, covmat_precomputed, T, device, batch_size_D=batch_size_D)

    # If nplets is a list of nplets with different orders, then use hot encoding to compute multiorder measures
    if isinstance(nplets, list) and not all([len(nplet) == len(nplets[0]) for nplet in nplets]):
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

    # |batch_size x D|, |batch_size x D|, |batch_size x D|
    bc1, bcN, bcNmin1 = _get_bias_correctors(T, order, batch_size, D, device, covmats.dtype)

    dataloader = DataLoader(nplets, batch_size=batch_size, shuffle=False)

    results = []
    for nplet_batch in tqdm(dataloader, desc='Processing n-plets', leave=False):
        curr_batch_size = nplet_batch.shape[0]

        # |curr_batch_size| x |D| x |order| x |order|  →  |curr_batch_size*D| x |order| x |order|
        nplets_covmats = _generate_nplets_covmats(covmats, nplet_batch).view(curr_batch_size * D, order, order)

        nplets_tc, nplets_dtc, nplets_o, nplets_s = _get_fast_tc_dtc_from_batched_covmat(
            nplets_covmats, order,
            bc1[:curr_batch_size * D], bcN[:curr_batch_size * D], bcNmin1[:curr_batch_size * D],
        )

        results.append(torch.stack([nplets_tc.view(curr_batch_size, D),
                                    nplets_dtc.view(curr_batch_size, D),
                                    nplets_o.view(curr_batch_size, D),
                                    nplets_s.view(curr_batch_size, D)], dim=-1))

    return torch.cat(results, dim=0)

@torch.no_grad()
def multi_order_measures(X: TensorLikeArray,
                       min_order: int = 3,
                       max_order: Optional[int] = None,
                       *,
                       covmat_precomputed: bool = False,
                       T: Optional[Union[int, List[int]]] = None,
                       batch_size: int = 1000000,
                       batch_size_D: Optional[int] = None,
                       device: torch.device = torch.device('cpu'),
                       num_workers: int = 0,
                       offload_to_cpu: bool = True,
                       batch_data_collector: Optional[Callable] = None,
                       batch_aggregation: Optional[Callable] = None) -> dict:
    """
    Compute TC, DTC, O, and S using the fast Gaussian covariance/precision formulas.

    Algebraically equivalent to the full entropy-based estimator but avoids the
    single-exclusion entropy loop, making it faster for large orders.

    Parameters
    ----------
    X : TensorLikeArray
        Input data (shape (T, N) or list thereof) or precomputed covariance matrices
        (shape (N, N) or list thereof, when covmat_precomputed=True).
    min_order : int, default=3
        Minimum order of interactions to compute.
    max_order : int, optional
        Maximum order. Defaults to N.
    covmat_precomputed : bool, default=False
        If True, X is treated as covariance matrices instead of raw data.
    T : int or list of int, optional
        Sample sizes for bias correction (only used when covmat_precomputed=True).
    batch_size : int, default=1_000_000
        Maximum n-plets per batch.
    batch_size_D : int or None, default=None
        Number of datasets per batch during covariance computation.
    device : torch.device, default=cpu
        Computation device.
    num_workers : int, default=0
        DataLoader worker count.
    offload_to_cpu : bool, default=True
        When True (default), each batch is moved to CPU immediately after
        computation, keeping GPU memory usage proportional to a single batch.
        Set to False only if the GPU has enough memory to hold all results
        across all orders simultaneously; doing so avoids repeated small
        host-device transfers and can be faster in that case.
        Has no effect when a custom ``batch_data_collector`` is provided.
    batch_data_collector : callable, optional
        ``(nplets: Tensor[B, K], result: Tensor[B, D, 4], bn: int) -> Any``
        Post-processes each batch. Last dimension of result is (TC, DTC, O, S).
        When provided, ``offload_to_cpu`` is ignored.
    batch_aggregation : callable, optional
        ``(items: list[Any]) -> Any``
        Aggregates all collected items into the final result.
        Defaults to ``batched_results_to_dataframe`` → ``pd.DataFrame``.

    Returns
    -------
    pd.DataFrame or Any
        By default, a DataFrame with columns ``dataset``, ``tc``, ``dtc``,
        ``o``, ``s``, ``var_0 … var_{N-1}``, ``order``, sorted by ``dataset``.
        Returns whatever ``batch_aggregation`` produces when one is provided.
    """
    covmats, D, N, T = _normalize_input_data(X, covmat_precomputed, T, device, batch_size_D=batch_size_D)
    max_order = N if max_order is None else max_order
    batch_size = max(batch_size // D, 1)

    _order_cache: dict = {}

    def _batch_fn(nplets, K):
        curr_B = nplets.shape[0]
        if K not in _order_cache:
            _order_cache[K] = _get_bias_correctors(T, K, batch_size, D, device, covmats.dtype)
        bc1, bcN, bcNmin1 = _order_cache[K]
        nplets_covmats = _generate_nplets_covmats(covmats, nplets).view(curr_B * D, K, K)
        tc, dtc, o, s = _get_fast_tc_dtc_from_batched_covmat(
            nplets_covmats, K,
            bc1[:curr_B * D], bcN[:curr_B * D], bcNmin1[:curr_B * D],
        )
        return torch.stack([tc.view(curr_B, D), dtc.view(curr_B, D),
                            o.view(curr_B, D), s.view(curr_B, D)], dim=-1)

    return _batch_processing_multi_order(
        N=N, min_order=min_order, max_order=max_order,
        batch_fn=_batch_fn,
        batch_size=batch_size, device=device, num_workers=num_workers,
        offload_to_cpu=offload_to_cpu,
        batch_data_collector=batch_data_collector,
        batch_aggregation=batch_aggregation,
    )
