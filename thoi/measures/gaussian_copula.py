from typing import Optional, Callable

from tqdm.autonotebook import tqdm

import scipy as sp
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..dataset import CovarianceDataset


def gaussianCopula(X):
    """
    Transform the data into a Gaussian copula and compute the covariance matrix.
    
    Parameters:
    - X: A 2D numpy array of shape (T, N) where T is the number of samples and N is the number of variables.
    
    Returns:
    - X_gaussian: The data transformed into the Gaussian copula (same shape as the parameter input).
    - X_gaussian_covmat: The covariance matrix of the Gaussian copula transformed data.
    """

    assert X.ndim == 2, f'data must be 2D but got {X.ndim}D data input'

    T = X.shape[0]

    # Step 1 & 2: Rank the data and normalize the ranks
    sortid = np.argsort(X, axis=0) # sorting indices
    copdata = np.argsort(sortid, axis=0) # sorting sorting indices
    copdata = (copdata+1)/(T+1) # normalized indices in the [0,1] range 

    # Step 3: Apply the inverse CDF of the standard normal distribution
    X_gaussian = sp.special.ndtri(copdata) #uniform data to gaussian

    # Handle infinite values by setting them to 0 (optional and depends on use case)
    X_gaussian[np.isinf(X_gaussian)] = 0

    # Step 4: Compute the covariance matrix
    X_gaussian_covmat = np.cov(X_gaussian.T)

    return X_gaussian, X_gaussian_covmat


def _gaussian_entropy_bias_correction(N,T):
    """Computes the bias of the entropy estimator of a 
    N-dimensional multivariate gaussian with T sample"""
    psiterms = sp.special.psi((T - np.arange(1,N+1))/2)
    return (N*np.log(2/(T-1)) + np.sum(psiterms))/2


def _gaussian_entropy_estimation(cov_det, n_variables):
    return 0.5 * torch.log(torch.tensor(2 * torch.pi * torch.e).pow(n_variables) * cov_det)


def _all_min_1_ids(n_variables):
    return [np.setdiff1d(range(n_variables),x) for x in range(n_variables)]


def _get_tc_dtc_from_batched_covmat(covmat: torch.Tensor, allmin1, bc1: float, bcN: float, bcNmin1: float):

    # covmat is a batch of covariance matrices
    # |bz| x |N| x |N|

    n_variables = covmat.shape[2]

    # |bz|
    batch_det = torch.linalg.det(covmat)
    # |bz| x |N|
    single_var_dets = torch.diagonal(covmat, dim1=-2, dim2=-1)
    # |bz| x |N|
    single_exclusion_dets = torch.stack([torch.linalg.det(covmat[:,ids][:,:,ids]) for ids in allmin1], dim=1)

    # |bz|
    sys_ent = _gaussian_entropy_estimation(batch_det, n_variables) - bcN

    # This could be calculated once at the begining and then accessed here.
    var_ents = _gaussian_entropy_estimation(single_var_dets, 1) - bc1
    # |bz| x |N|
    ent_min_one = _gaussian_entropy_estimation(single_exclusion_dets, n_variables-1) - bcNmin1
    # |bz| x |N|

    # |bz|
    nplet_tc = torch.sum(var_ents, dim=1) - sys_ent
    # TODO: inf - inf return NaN in pytorch. Check how should I handle this.
    # |bz|
    nplet_dtc = torch.sum(ent_min_one, dim=1) - (n_variables-1.0)*sys_ent

    # |bz|
    nplet_o = nplet_tc - nplet_dtc
    # |bz|
    nplet_s = nplet_tc + nplet_dtc

    return nplet_tc, nplet_dtc, nplet_o, nplet_s


def nplet_measures(X: np.ndarray):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    T, N = np.shape(X)

    covmat = gaussianCopula(X)[1]
    covmat = torch.Tensor(np.expand_dims(covmat, axis=0))
    covmat.to(device)

    allmin1 = _all_min_1_ids(N)
    bc1 = _gaussian_entropy_bias_correction(1,T)
    bcN = _gaussian_entropy_bias_correction(N,T)
    bcNmin1 = _gaussian_entropy_bias_correction(N-1,T)

    nplet_tc, nplet_dtc, nplet_o, nplet_s = _get_tc_dtc_from_batched_covmat(
        covmat, allmin1, bc1, bcN, bcNmin1
    )

    return (
        nplet_tc.cpu().numpy()[0],
        nplet_dtc.cpu().numpy()[0],
        nplet_o.cpu().numpy()[0],
        nplet_s.cpu().numpy()[0]
    )


def multi_order_measures(X: np.ndarray,
                        min_order: int=3,
                        max_order: Optional[int]=None,
                        batch_size: int = 1000000,
                        use_cpu: bool = False,
                        batch_aggregation: Optional[Callable[[any],any]] = None,
                        batch_data_collector: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],any]] = None):
    """
    Compute multi-order Gaussian Copula (GC) measurements for the given data matrix X.

    Args:
        X (np.ndarray): T samples x N variables matrix.
        min_order (int): Minimum order to compute (default: 3).
        max_order (Optional[int]): Maximum order to compute (default: None, will use N).
        batch_size (int): Batch size for DataLoader (default: 1000000).
        use_cpu (bool): If true, it forces to use CPU even if GPU is available (default: False).

    Returns:
        pd.DataFrame: DataFrame containing computed metrics.
    """

    if batch_aggregation is None:
        batch_aggregation = lambda X: [x for x in X if x is not None]

    if batch_data_collector is None:
        batch_data_collector = lambda x: x

    T, N = X.shape
    max_order = N if max_order is None else max_order

    assert max_order <= N, f"max_order must be lower or equal than N. {max_order} > {N})"
    assert min_order <= max_order, f"min_order must be lower or equal than max_order. {min_order} > {max_order}"

    # make device cpu if not cuda available or cuda if available
    using_GPU = torch.cuda.is_available() and not use_cpu
    device = torch.device('cuda' if using_GPU else 'cpu')

    # Gaussian Copula of data
    covmat = gaussianCopula(X)[1]

    # To compute using pytorch, we need to compute each order separately
    batched_data = []
    pbar_order = tqdm(range(min_order, max_order+1), leave=False, desc='Order', disable=(min_order==max_order))
    for order in pbar_order:

        # Calculate constant values valid for all n-plets of the current order
        allmin1 = _all_min_1_ids(order)
        bc1 = _gaussian_entropy_bias_correction(1,T)
        bcN = _gaussian_entropy_bias_correction(order,T)
        bcNmin1 = _gaussian_entropy_bias_correction(order-1,T)

        # Generate dataset iterable
        dataset = CovarianceDataset(covmat, N, order)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
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
                partition_idxs.cpu().numpy(),
                nplets_o.cpu().numpy(),
                nplets_s.cpu().numpy(),
                nplets_tc.cpu().numpy(),
                nplets_dtc.cpu().numpy(),
                bn
            )
            batched_data.append(data)

    return batch_aggregation(batched_data)