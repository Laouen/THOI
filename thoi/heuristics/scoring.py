from typing import List, Optional, Union, Callable

from thoi.commons import _get_string_metric
from thoi.measures.gaussian_copula import nplets_measures
from thoi.measures.gaussian_copula_hot_encoded import nplets_measures_hot_encoded

from functools import partial
import torch


def _evaluate_nplets(covmats: torch.Tensor,
                     T: Optional[List[int]],
                     batched_nplets: torch.Tensor,
                     metric: Union[str, Callable],
                     use_cpu:bool):
    """
        covmats (torch.Tensor): The covariance matrix or matrixes with shape (N, N) or (D, N, N)
        T (Optional[List[int]]): The number of samples for each multivariate series or None
        batched_nplets (torch.Tensor): The nplets to calculate the inverse of the oinformation with shape (batch_size, order)
        metric (str): The metric to evaluate. One of tc, dtc, o, s or Callable
    """

    if len(covmats.shape) == 2:
        covmats = covmats.unsqueeze(0)
        
    metric_func = partial(_get_string_metric, metric=metric) if isinstance(metric, str) else metric

    # |bached_nplets| x |D| x |4 = (tc, dtc, o, s)|
    batched_measures = nplets_measures(covmats,
                                       nplets=batched_nplets,
                                       T=T,
                                       covmat_precomputed=True,
                                       use_cpu=use_cpu)
    
    # |batch_size|
    return metric_func(batched_measures).to(covmats.device)


def _evaluate_nplet_hot_encoded(covmats: torch.Tensor,
                                T:int,
                                batched_nplets: torch.Tensor,
                                metric:str,
                                use_cpu:bool):

    """
        covmats (torch.Tensor): The covariance matrix or matrixes with shape (N, N) or (D, N, N)
        T (Optional[List[int]]): The number of samples for each multivariate series or None
        batched_nplets (torch.Tensor): The nplets to calculate the inverse of the oinformation with shape (batch_size, order)
        metric (str): The metric to evaluate. One of tc, dtc, o, s or Callable
    """

    if len(covmats.shape) == 2:
        covmats = covmats.unsqueeze(0)

    metric_func = partial(_get_string_metric, metric=metric) if isinstance(metric, str) else metric

    # |bached_nplets| x |D| x |4 = (tc, dtc, o, s)|
    batched_measures = nplets_measures_hot_encoded(covmats,
                                                   nplets=batched_nplets,
                                                   T=T,
                                                   covmat_precomputed=True,
                                                   use_cpu=use_cpu)

    # |batch_size|
    return metric_func(batched_measures).to(covmats.device)