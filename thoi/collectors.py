from typing import List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
import torch
from functools import partial

from thoi.commons import _get_string_metric

####################################################################################################
######################                     To dataframe                    #########################
####################################################################################################

def batch_to_csv(nplets_idxs: torch.Tensor,
                 nplets_tc: torch.Tensor,
                 nplets_dtc: torch.Tensor,
                 nplets_o: torch.Tensor,
                 nplets_s: torch.Tensor,
                 bn:int,
                 only_synergetic: bool=False,
                 columns: Optional[List[str]]=None,
                 N: Optional[int]=None,
                 sep: str='\t',
                 indexing_method: str='indexes', # indexes or hot_encoded
                 output_path: Optional[str]=None) -> Optional[pd.DataFrame]:
    """
    Convert batch results to a pandas DataFrame and optionally save to CSV.

    This function processes the measures computed for n-plets in a batch and converts them into a pandas DataFrame.
    It can also save the DataFrame to a CSV file if an output path is provided.

    Parameters
    ----------
    nplets_idxs : torch.Tensor
        Indices of the n-plets. Shape: (batch_size, order).
    nplets_tc : torch.Tensor
        Total correlation values. Shape: (batch_size, D).
    nplets_dtc : torch.Tensor
        Dual total correlation values. Shape: (batch_size, D).
    nplets_o : torch.Tensor
        O-information values. Shape: (batch_size, D).
    nplets_s : torch.Tensor
        S-information values. Shape: (batch_size, D).
    bn : int
        Batch number, used for identification in output files.
    only_synergetic : bool, optional
        If True, only includes n-plets with negative O-information (synergetic). Default is False.
    columns : list of str, optional
        Names of the variables (features). If None, variable names will be generated as 'var_0', 'var_1', ..., 'var_N-1'.
    N : int, optional
        Total number of variables. Required if `columns` is not provided.
    sep : str, optional
        Separator to use in the CSV file. Default is tab ('\\t').
    indexing_method : str, optional
        Method used to represent n-plets. Can be 'indexes' or 'hot_encoded'. Default is 'indexes'.
    output_path : str, optional
        Path to save the CSV file. If None, the DataFrame is returned instead of being saved.

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing the measures and variable information for the n-plets.
        Returns None if `output_path` is provided and the DataFrame is saved to a file.

    Where
    -----
    D : int
        Number of datasets. If measures are computed over multiple datasets, D > 1.
    N : int
        Number of variables (features).
    batch_size : int
        Number of n-plets in the batch.
    order : int
        Order of the n-plets (number of variables in each n-plet).

    Notes
    -----
    - The function can filter out n-plets with non-negative O-information if `only_synergetic` is True.
    - The resulting DataFrame includes the measures and a binary indicator for each variable indicating its presence in the n-plet.
    - The DataFrame also includes columns for 'order' and 'dataset'.

    Examples
    --------
    ```python
    # Sample inputs
    nplets_idxs = torch.tensor([[0, 1], [1, 2], [0, 2]])
    nplets_tc = torch.rand(3, 1)
    nplets_dtc = torch.rand(3, 1)
    nplets_o = torch.rand(3, 1)
    nplets_s = torch.rand(3, 1)
    bn = 0
    columns = ['A', 'B', 'C']
    N = 3

    # Convert batch to DataFrame
    df = batch_to_csv(
        nplets_idxs,
        nplets_tc,
        nplets_dtc,
        nplets_o,
        nplets_s,
        bn,
        columns=columns,
        N=N
    )
    ```
    """
    
    # nplets have shape |batch_size| x |D|
    
    assert columns is not None or N is not None, 'either columns or N must be defined'

    if N is None:
        N = len(columns)
    
    if columns is None:
        columns = [f'var_{i}' for i in range(N)]
    
    assert N == len(columns), f'N must be equal to len(columns). {N} != {len(columns)}'

    # If only_synergetic; remove nplets with nplet_o >= 0
    if only_synergetic:
        to_keep = torch.where(nplets_o < 0)[0]
        nplets_tc = nplets_tc[to_keep]
        nplets_dtc = nplets_dtc[to_keep]
        nplets_o = nplets_o[to_keep]
        nplets_s = nplets_s[to_keep]
        # TODO: Check if this is correct since to_keep has shape |batch_size| x |D| and nplets_idcx has shape |batch_size|
        nplets_idxs = nplets_idxs[to_keep.to(nplets_idxs.device)]
    
    bs, D = nplets_dtc.shape if len(nplets_dtc.shape) == 2 else (nplets_dtc.shape[0], 1)

    # One we removed not synergistic if not required on GPU, we move to CPU the
    # final results to save
    df_meas = pd.DataFrame({
        'tc': nplets_tc.detach().cpu().numpy().flatten(),
        'dtc': nplets_dtc.detach().cpu().numpy().flatten(),
        'o': nplets_o.detach().cpu().numpy().flatten(),
        's': nplets_s.detach().cpu().numpy().flatten()
    })
    
    # Create a DataFrame with the n-plets
    if indexing_method == 'indexes':
        nplets_idxs = nplets_idxs.detach().cpu().numpy()
        batch_size, order = nplets_idxs.shape
        bool_array = np.zeros((batch_size, N), dtype=bool)
        rows = np.arange(batch_size).reshape(-1, 1)
        bool_array[rows, nplets_idxs] = True
    else:
        bool_array = nplets_idxs.bool().detach().cpu().numpy()
        

    # Repeat the boolean array to match the number of datasets and store in a DataFrame
    bool_array = np.repeat(bool_array, D, axis=0)
    df_vars = pd.DataFrame(bool_array, columns=columns)

    # Concat both dataframes columns and store in disk
    df = pd.concat([df_meas, df_vars], axis=1)

    # Compute a column with the order
    df['order'] = df[columns].sum(axis=1)
    
    # Add dataset col at the first columns
    df.insert(0, 'dataset', np.tile(np.arange(D), bs))

    if output_path is not None:
        df.to_csv(output_path.format(order=order, bn=bn), index=False, sep=sep)
        return None # Don't return

    return df


def concat_and_sort_csv(batched_dataframes) -> pd.DataFrame:
    """
    Concatenate a list of DataFrames and sort them by the 'dataset' column.

    Parameters
    ----------
    batched_dataframes : list of pd.DataFrame
        List of DataFrames to concatenate and sort.

    Returns
    -------
    pd.DataFrame
        The concatenated and sorted DataFrame.

    Notes
    -----
    - The DataFrames are concatenated along the rows.
    - Sorting is performed using the 'dataset' column in ascending order.
    - The index is reset after sorting.

    Examples
    --------
    ```python
    df1 = pd.DataFrame({'dataset': [0, 0], 'value': [1, 2]})
    df2 = pd.DataFrame({'dataset': [1, 1], 'value': [3, 4]})
    combined_df = concat_and_sort_csv([df1, df2])
    ```
    """
    df = pd.concat(batched_dataframes)
    df = df.sort_values(by='dataset', kind='stable', ascending=True).reset_index(drop=True)
    return df


####################################################################################################
#######################                     To tensor                    ###########################
####################################################################################################


def top_k_nplets(nplets_idxs: torch.Tensor,
                 nplets_measures: torch.Tensor,
                 k: int,
                 metric: Union[str, Callable[[torch.Tensor], torch.Tensor]], # Add typing and documentation of the callable input
                 largest: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select the top-k n-plets based on a specified metric.

    Parameters
    ----------
    nplets_idxs : torch.Tensor
        Indices of the n-plets. Shape: (batch_size, order).
    nplets_measures : torch.Tensor
        Measures for each n-plet. Shape: (batch_size, D, 4).
    k : int
        Number of top n-plets to select.
    metric : string with value 'dtc', 'tc', 'o' or 's' or Callable
        Metric to use for ranking the n-plets. Can be a string specifying a measure ('tc', 'dtc', 'o', 's'),
        or a custom callable that takes `nplets_measures` and returns a tensor of values.
    largest : bool
        If True, selects n-plets with the largest metric values if false return n-plets with the smalest values. Default is False.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - Selected n-plets measures. Shape: (k, D, 4).
        - Selected n-plets indices. Shape: (k, order).
        - Metric values of the selected n-plets. Shape: (k,).

    Where
    -----
    D : int
        Number of datasets.
    order : int
        Order of the n-plets (number of variables in each n-plet).
    batch_size : int
        Number of n-plets in the batch.

    Notes
    -----
    - The function computes the specified metric for each n-plet and selects the top-k based on this metric.
    - The metric can be one of the predefined measures or a custom function.

    Examples
    --------
    ```python
    # Sample data
    nplets_idxs = torch.tensor([[0, 1], [1, 2], [0, 2]])
    nplets_measures = torch.rand(3, 1, 4)  # Assuming D=1
    k = 2
    metric = 'o'  # Use O-information for ranking

    # Get top-k n-plets
    top_measures, top_idxs, top_values = top_k_nplets(
        nplets_idxs, nplets_measures, k, metric, largest=False
    )
    ```
    """
    
    metric_func = partial(_get_string_metric, metric=metric) if isinstance(metric, str) else metric
    
    # |batch_size|
    values = metric_func(nplets_measures).to(nplets_measures.device)
    
    # |k|
    _, indices = torch.topk(values, k, largest=largest, sorted=True)

    # (|k| x |D| x |4|, |k| x |N|)
    return (
        nplets_measures[indices],
        nplets_idxs[indices.to(nplets_idxs.device)], # indices can be in CPU while the rest on GPU
        values[indices]
    )


def batch_to_tensor(nplets_idxs: torch.Tensor,
                    nplets_tc: torch.Tensor,
                    nplets_dtc: torch.Tensor,
                    nplets_o: torch.Tensor,
                    nplets_s: torch.Tensor,
                    bn: Optional[int]=None,
                    top_k: Optional[int]=None,
                    metric: Union[str, Callable[[torch.Tensor], torch.Tensor]]='o',
                    largest: bool=False)  -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Process batch measures and optionally select top-k n-plets.

    Parameters
    ----------
    nplets_idxs : torch.Tensor
        Indices of the n-plets. Shape: (batch_size, order).
    nplets_tc : torch.Tensor
        Total correlation values. Shape: (batch_size, D).
    nplets_dtc : torch.Tensor
        Dual total correlation values. Shape: (batch_size, D).
    nplets_o : torch.Tensor
        O-information values. Shape: (batch_size, D).
    nplets_s : torch.Tensor
        S-information values. Shape: (batch_size, D).
    bn : int, optional
        Batch number. Not used in the function but kept for compatibility.
    top_k : int, optional
        If provided, selects the top-k n-plets based on the specified metric.
    metric : string with value 'dtc', 'tc', 'o' or 's' or Callable, optional
        Metric to use for ranking if `top_k` is provided. Default is 'o' (O-information).
    largest : bool, optional
        If True, selects n-plets with the largest metric values if false return n-plets with the smalest values. Default is False.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        - n-plets measures. Shape: (batch_size or k, D, 4).
        - n-plets indices. Shape: (batch_size or k, order).
        - Metric values of the selected n-plets if `top_k` is provided, else None.

    Where
    -----
    D : int
        Number of datasets.
    order : int
        Order of the n-plets.
    batch_size : int
        Number of n-plets in the batch.

    Notes
    -----
    - If `top_k` is provided and less than `batch_size`, the function selects the top-k n-plets based on the metric.
    - The measures are stacked along the last dimension in the order: (tc, dtc, o, s).

    Examples
    --------
    ```python
    # Sample inputs
    nplets_idxs = torch.tensor([[0, 1], [1, 2], [0, 2]])
    nplets_tc = torch.rand(3, 1)
    nplets_dtc = torch.rand(3, 1)
    nplets_o = torch.rand(3, 1)
    nplets_s = torch.rand(3, 1)

    # Process batch without top-k selection
    measures, idxs, _ = batch_to_tensor(
        nplets_idxs, nplets_tc, nplets_dtc, nplets_o, nplets_s
    )

    # Process batch with top-2 selection based on O-information
    measures_topk, idxs_topk, metric_values = batch_to_tensor(
        nplets_idxs, nplets_tc, nplets_dtc, nplets_o, nplets_s, top_k=2, metric='o', largest=False
    )
    ```
    """

    # |batch_size| x |D|
    assert len(nplets_tc.shape) == len(nplets_dtc.shape) == len(nplets_o.shape) == len(nplets_s.shape) == 2, 'All nplets must be 2D tensors'

    # |batch_size| x |D| x |4 = (tc, dtc, o, s)|
    nplets_measures = torch.stack([nplets_tc,
                                   nplets_dtc,
                                   nplets_o,
                                   nplets_s], dim=-1)

    # If top_k is not None and len(nplets_o) > top_k, return the top_k nplets
    if (top_k is not None) and (nplets_measures.shape[0] > top_k):
        # |k x D x 4|, |k x N|, |k|
        return top_k_nplets(nplets_idxs,
                            nplets_measures,
                            top_k,
                            metric,
                            largest)

    metric_func = partial(_get_string_metric, metric=metric) if isinstance(metric, str) else metric
    
    # If not top_k or len(nplets_measuresa) > top_k  return the original values
    # |k x D x 4|, |k x N|, |k|
    values = metric_func(nplets_measures).to(nplets_measures.device)
    return (nplets_measures, nplets_idxs, values)


def concat_batched_tensors(batched_tensors: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
                           top_k: Optional[int] = None,
                           metric: Optional[Union[str, Callable]] = 'o',
                           largest: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Concatenate batched tensors and optionally select top-k n-plets.

    Parameters
    ----------
    batched_tensors : list of tuples
        Each tuple contains:
            - nplets_measures: torch.Tensor, shape (batch_size, D, 4)
            - nplets_idxs: torch.Tensor, shape (batch_size, order)
            - nplets_scores: torch.Tensor or None, shape (batch_size,)
    top_k : int, optional
        If provided, selects the top-k n-plets across all batches. Default is None.
    metric : str or Callable, optional
        Metric to use for ranking if `top_k` is provided. Default is 'o'.
    largest : bool, optional
        If True, selects n-plets with the largest metric values, if not, select n-plets with smallest values. Default is False.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        - Concatenated n-plets measures. Shape: (total_nplets or k, D, 4).
        - Concatenated n-plets indices. Shape: (total_nplets or k, order).
        - Metric values of the selected n-plets if `top_k` is provided, else None.

    Where
    -----
    D : int
        Number of datasets.
    order : int
        Order of the n-plets.
    total_nplets : int
        Total number of n-plets across all batches.

    Notes
    -----
    - If `top_k` is provided, the function selects the top-k n-plets across all batches.
    - If `top_k` is provided, nplets_scores must be provided in `batched_tensors`.

    Examples
    --------
    ```python
    # Suppose we have batched tensors from two batches
    batched_tensors = [
        (measures_batch1, idxs_batch1, None),
        (measures_batch2, idxs_batch2, None)
    ]

    # Concatenate without top-k selection
    measures_all, idxs_all, _ = concat_batched_tensors(batched_tensors)

    # Concatenate and select top-5 n-plets based on O-information
    measures_topk, idxs_topk, metric_values = concat_batched_tensors(
        batched_tensors, top_k=5, metric='o', largest=False
    )
    ```
    """
    
    # Unpac the batched tensors
    nplets_measures, nplets_idxs, nplets_scores = zip(*batched_tensors)
    
    # Concatenate each tuple of tensors
    nplets_measures = torch.cat(nplets_measures)
    nplets_idxs = torch.cat(nplets_idxs)
    nplets_scores = torch.cat(nplets_scores) if nplets_scores[0] is not None else None
    
    # If top_k is not None and len(nplets_o) > top_k, return the top_k nplets
    if (top_k is not None) and (nplets_measures.shape[0] > top_k):

        assert (metric is not None) or (nplets_scores[0] is not None), 'if no metric defined to use, then nplets_scores must be defined if top_k is not None'

        if metric is not None:
            # |k x D x 4|, |k x N|, |k|
            return top_k_nplets(nplets_idxs,
                                nplets_measures,
                                top_k,
                                metric,
                                largest)

        else:

            # |k|
            _, indices = torch.topk(nplets_scores, top_k, largest=largest, sorted=True)

            # (|k| x |D| x |4|, |k| x |N|)
            return (
                nplets_measures[indices],
                nplets_idxs[indices.cpu()],
                nplets_scores[indices]
            )

    # If not top_k or len(nplets_measuresa) > top_k  return the original values
    # |k x D x 4|, |k x N|
    return (nplets_measures, nplets_idxs, nplets_scores)