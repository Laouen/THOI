from typing import List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
import torch
from functools import partial

from thoi.commons import _get_string_metric

####################################################################################################
######################                     To dataframe                    #########################
####################################################################################################

def batch_to_csv(partition_idxs: torch.Tensor,
                 nplets_tc: torch.Tensor,
                 nplets_dtc: torch.Tensor,
                 nplets_o: torch.Tensor,
                 nplets_s: torch.Tensor,
                 bn:int,
                 only_synergestic: bool=False,
                 columns: Optional[List[str]]=None,
                 N: Optional[int]=None,
                 sep: str='\t',
                 indexing_method: str='indexes', # indexes or hot_encoded
                 output_path: Optional[str]=None):
    
    assert columns is not None or N is not None, 'either columns or N must be defined'

    if N is None:
        N = len(columns)
    
    if columns is None:
        columns = [f'var_{i}' for i in range(N)]
    
    assert N == len(columns), f'N must be equal to len(columns). {N} != {len(columns)}'

    # If only_synergestic; remove nplets with nplet_o >= 0
    if only_synergestic:
        to_keep = torch.where(nplets_o < 0)[0]
        nplets_tc = nplets_tc[to_keep]
        nplets_dtc = nplets_dtc[to_keep]
        nplets_o = nplets_o[to_keep]
        nplets_s = nplets_s[to_keep]
        partition_idxs = partition_idxs[to_keep.cpu()]
    
    bs, D = nplets_dtc.shape if len(nplets_dtc.shape) == 2 else (nplets_dtc.shape[0], 1)

    # One we removed not synergistic if not required on GPU, we move to CPU the
    # final results to save
    df_meas = pd.DataFrame({
        'tc': nplets_tc.cpu().numpy().flatten(),
        'dtc': nplets_dtc.cpu().numpy().flatten(),
        'o': nplets_o.cpu().numpy().flatten(),
        's': nplets_s.cpu().numpy().flatten()
    })

    # Create a DataFrame with the n-plets
    if indexing_method == 'indexes':
        batch_size, order = partition_idxs.shape
        bool_array = np.zeros((batch_size, N), dtype=bool)
        rows = np.arange(batch_size).reshape(-1, 1)
        bool_array[rows, partition_idxs] = True
        bool_array = np.repeat(bool_array, D, axis=0)
    else:
        bool_array = partition_idxs.bool().cpu().numpy()
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


def concat_and_sort_csv(batched_dataframes):
    df = pd.concat(batched_dataframes)
    df = df.sort_values(by='dataset', kind='stable', ascending=True).reset_index(drop=True)
    return df


def concatenate_csv(batched_dataframes, output_path, sep='\t'):

    df = pd.concat(batched_dataframes, axis=0)
    df.to_csv(output_path, index=False, sep=sep)

####################################################################################################
#######################                     To tensor                    ###########################
####################################################################################################


def top_k_nplets(nplets_measures: torch.tensor,
          partition_idxs: torch.tensor,
          k: int,
          metric: Union[str, Callable], # Add typing and documentation of the callable input
          largest: bool):
    
    metric_func = partial(_get_string_metric, metric=metric) if isinstance(metric, str) else metric
    
    # |batch_size|
    values = metric_func(nplets_measures)
    
    # |k|
    _, indices = torch.topk(values, k, largest=largest, sorted=True)

    # (|k| x |D| x |4|, |k| x |N|)
    return (
        nplets_measures[indices],
        partition_idxs[indices.cpu()]
    )


# TODO: This function is used in simulated_annealing, check if its used in other places and if there is modifications to be done on these places
def batch_to_tensor(partition_idxs: torch.tensor,
                    nplets_tc: torch.tensor,
                    nplets_dtc: torch.tensor,
                    nplets_o: torch.tensor,
                    nplets_s: torch.tensor,
                    bn: Optional[int]=None,
                    top_k: Optional[int]=None,
                    metric: Union[str,Callable]='o',
                    largest: bool=False):

    # TODO: make this function to accept both a single or a list of inputs to use it for batches and for concat all batches
    # |batch_size| x |D|
    assert nplets_tc.shape == nplets_dtc.shape == nplets_o.shape == nplets_s.shape == 2, 'All nplets must be 2D tensors'

    # |batch_size| x |D| x |4 = (tc, dtc, o, s)|
    nplets_measures = torch.stack([nplets_tc,
                                   nplets_dtc,
                                   nplets_o,
                                   nplets_s], dim=-1)

    # If top_k is not None and len(nplets_o) > top_k, return the top_k nplets
    if (top_k is not None) and (nplets_measures.shape[0] > top_k):
        # |k x D x 4|, |k x N|
        return top_k_nplets(nplets_measures,
                            partition_idxs,
                            top_k,
                            metric,
                            largest)

    # If not top_k or len(nplets_measuresa) > top_k  return the original values
    # |k x D x 4|, |k x N|
    return (nplets_measures, partition_idxs)

# TODO: This function is used in simulated_annealing, check if its used in other places and if there is modifications to be done on these places
def concat_batched_tensors(batched_tensors: List[Tuple[torch.tensor, torch.tensor]],
                           top_k: Optional[int] = None,
                           metric:str = 'o',
                           largest:bool = False):
    
    # Unpac the batched tensors
    nplets_measures, partition_idxs_tensors = zip(*batched_tensors)
    
    # Concatenate each tuple of tensors
    nplets_measures = torch.cat(nplets_measures)
    partition_idxs = torch.cat(partition_idxs_tensors)
    
    # If top_k is not None and len(nplets_o) > top_k, return the top_k nplets
    if (top_k is not None) and (nplets_measures.shape[0] > top_k):
        # |k x D x 4|, |k x N|
        return top_k_nplets(nplets_measures,
                            partition_idxs,
                            top_k,
                            metric,
                            largest)

    # If not top_k or len(nplets_measuresa) > top_k  return the original values
    # |k x D x 4|, |k x N|
    return (nplets_measures, partition_idxs)


# TODO: Check where this is used
def beast_mean_across_batch(partition_idxs:torch.Tensor,
                            nplets_tc:torch.Tensor,
                            nplets_dtc:torch.Tensor,
                            nplets_o:torch.Tensor,
                            nplets_s:torch.Tensor,
                            bn:int,
                            only_synergestic:bool=False,
                            top_k:Optional[int] = None,
                            metric:str = 'o',
                            largest:bool = False):
    '''
    @param partition_idxs: torch.Tensor of shape (n_data, repeat, order)
    @param nplets_tc: torch.Tensor of shape (n_data, repeat)
    @param nplets_dtc: torch.Tensor of shape (n_data, repeat)
    @param nplets_o: torch.Tensor of shape (n_data, repeat)
    @param nplets_s: torch.Tensor of shape (n_data, repeat)
    '''
    
    # Assert that that all partition_idxs have the same values
    assert torch.all(partition_idxs[0] == partition_idxs).item(), 'All partition_idxs must have the same values'
    
    # Remove the repeat dimension as is the same for all
    partition_idxs = partition_idxs[0]
    
    # Compute the mean across the n_data
    nplets_o = nplets_o.mean(axis=0)
    nplets_s = nplets_s.mean(axis=0)
    nplets_tc = nplets_tc.mean(axis=0)
    nplets_dtc = nplets_dtc.mean(axis=0)
    
    # If only_synergestic; remove nplets with nplet_o >= 0
    if only_synergestic:
        to_keep = torch.where(nplets_o < 0)[0]
        nplets_tc = nplets_tc[to_keep]
        nplets_dtc = nplets_dtc[to_keep]
        nplets_o = nplets_o[to_keep]
        nplets_s = nplets_s[to_keep]
        partition_idxs = partition_idxs[to_keep.cpu()]
    
    if top_k is not None and len(nplets_o) > top_k:

        values_idxs = ['tc','dtc','o','s'].index(metric)
        values = [nplets_tc, nplets_dtc, nplets_o, nplets_s][values_idxs]

        _, indices = torch.topk(values, top_k, largest=largest, sorted=True)

        nplets_tc = nplets_tc[indices]
        nplets_dtc = nplets_dtc[indices]
        nplets_o = nplets_o[indices]
        nplets_s = nplets_s[indices]
        partition_idxs = partition_idxs[indices.cpu()]
    
    return (
        nplets_tc,
        nplets_dtc,
        nplets_o,
        nplets_s,
        partition_idxs
    )