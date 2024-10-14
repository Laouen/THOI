from typing import List, Optional, Union, Callable

from thoi.commons import _get_string_metric
from thoi.measures.gaussian_copula import nplets_measures
from thoi.measures.gaussian_copula_hot_encoded import nplets_measures_hot_encoded

from functools import partial
import torch


def _evaluate_nplets(covmats: torch.tensor,
                     T: Optional[List[int]],
                     batched_nplets: torch.tensor,
                     metric: Union[str, Callable],
                     use_cpu:bool):
    """
        covmats (torch.tensor): The covariance matrix or matrixes with shape (N, N) or (D, N, N)
        T (Optional[List[int]]): The number of samples for each multivariate series or None
        batched_nplets (torch.tensor): The nplets to calculate the inverse of the oinformation with shape (batch_size, order)
        metric (str): The metric to evaluate. One of tc, dtc, o, s or Callable
    """

    if len(covmats.shape) == 2:
        covmats = covmats.unsqueeze(0)
        
    metric_func = partial(_get_string_metric, metric=metric) if isinstance(metric, str) else metric

    # |bached_nplets| x |D| x |4 = (tc, dtc, o, s)|
    batched_measures = nplets_measures(covmats,
                                       T=T,
                                       covmat_precomputed=True,
                                       nplets=batched_nplets,
                                       use_cpu=use_cpu)
    
    # |batch_size|
    return metric_func(batched_measures).to(covmats.device)


def _evaluate_nplet_hot_encoded(covmat: torch.tensor,
                                T:int,
                                batched_nplets: torch.tensor,
                                metric:str,
                                use_cpu:bool):

    """
        X (torch.tensor): The covariance matrix with shape (n_variables, n_variables)
        batched_nplets (torch.tensor): The nplets to calculate the inverse of the oinformation with shape (batch_size, order)
        metric (str): The metric to evaluate. One of tc, dtc, o or s
    """

    # get the metric index to get from the result of nplets
    METRICS = ['tc', 'dtc', 'o', 's']
    metric_idx = METRICS.index(metric)

    # |batch_size| x |4 = (tc, dtc, o, s)|
    batched_res = nplets_measures_hot_encoded(covmat, batched_nplets, T=T, covmat_precomputed=True, use_cpu=use_cpu)

    # Return minus the o information score to make it an maximum optimization (energy)
    # |batch_size|
    res = batched_res[:,metric_idx].flatten()

    return res