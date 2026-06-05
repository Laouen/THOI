import logging
from functools import partial
from typing import Callable, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from thoi.dataset import CovarianceDataset
from thoi.collectors import batched_results_to_dataframe


def _batch_processing_multi_order(
    N: int,
    min_order: int,
    max_order: int,
    batch_fn: Callable,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    offload_to_cpu: bool = True,
    batch_data_collector: Optional[Callable] = None,
    batch_aggregation: Optional[Callable] = None,
):
    """Shared batch iteration engine for multi-order n-plet measures.

    For each order K in [min_order, max_order], lazily generates all C(N, K)
    n-plets via CovarianceDataset + DataLoader, then for each batch:
      1. calls ``batch_fn(nplets_batch, K)`` → batch_result
      2. calls ``batch_data_collector(nplets_batch, batch_result, batch_number)`` → item
      3. appends item to a flat list across all orders
    Finally calls ``batch_aggregation(all_items)`` and returns the result.

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
    offload_to_cpu : bool, default True
        When True and ``batch_data_collector`` is None, each batch is moved to
        CPU immediately after computation so GPU memory stays proportional to a
        single batch.  Set to False only when the GPU has enough memory to hold
        all results simultaneously; this avoids repeated small host-device
        transfers and can be faster in that case.  Has no effect when a custom
        ``batch_data_collector`` is provided.
    batch_data_collector : callable, optional
        ``(nplets: Tensor[B, K], batch_result: Any, bn: int) -> Any``
        Post-processes each batch result.  When None, defaults to an identity
        that returns ``(nplets, batch_result)`` as a tuple, moving both to CPU
        first if ``offload_to_cpu=True``.
    batch_aggregation : callable, optional
        ``(items: list[Any]) -> Any``
        Aggregates all collected items (across every order) into the final result.
        When None, returns the raw flat list of collected items.
    """

    assert max_order <= N, f"max_order must be lower or equal than N. {max_order} > {N})"
    assert min_order <= max_order, f"min_order must be lower or equal than max_order. {min_order} > {max_order}"

    if batch_aggregation is None:
        batch_aggregation = partial(batched_results_to_dataframe, N=N)

    if batch_data_collector is None:
        if offload_to_cpu:
            _collector = lambda nplets, result, _: (nplets.cpu(), result.cpu())
        else:
            if device.type == 'cuda':
                logging.warning(
                    'offload_to_cpu=False: all batch results are accumulated on %s. '
                    'This can be faster when GPU memory is sufficient but may cause '
                    'out-of-memory errors for large N or many orders. '
                    'Set offload_to_cpu=True (default) to transfer each batch to CPU immediately.',
                    device,
                )
            _collector = lambda nplets, result, _: (nplets, result)
    else:
        _collector = batch_data_collector

    all_collected = []
    for K in tqdm(range(min_order, max_order + 1), leave=False, desc='Order',
                  disable=(min_order == max_order)):
        dataset = CovarianceDataset(N, K, device=device)
        dataloader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            num_workers=num_workers,
        )
        for bn, nplets in enumerate(tqdm(dataloader, total=len(dataloader),
                                         leave=False, desc='Batch')):
            nplets = nplets.to(device)
            batch_result = batch_fn(nplets, K)
            all_collected.append(_collector(nplets, batch_result, bn))

    if batch_aggregation is not None:
        return batch_aggregation(all_collected)
    return all_collected