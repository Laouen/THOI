from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import torch

def batch_to_csv(partition_idxs: np.ndarray,
                 nplets_o: np.ndarray,
                 nplets_s: np.ndarray,
                 nplets_tc: np.ndarray,
                 nplets_dtc: np.ndarray,
                 bn:int,
                 only_synergestic: bool=False,
                 columns: Optional[List[str]]=None,
                 N: Optional[int]=None,
                 sep: str='\t',
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

    # One we removed not synergistic if not required on GPU, we move to CPU the
    # final results to save
    df_meas = pd.DataFrame({
        'tc': nplets_tc.cpu().numpy(),
        'dtc': nplets_dtc.cpu().numpy(),
        'o': nplets_o.cpu().numpy(),
        's': nplets_s.cpu().numpy()
    })

    # Create a DataFrame with the n-plets
    partition_idxs = partition_idxs
    batch_size, order = partition_idxs.shape
    bool_array = np.zeros((batch_size, N), dtype=bool)
    rows = np.arange(batch_size).reshape(-1, 1)
    bool_array[rows, partition_idxs] = True
    df_vars = pd.DataFrame(bool_array, columns=columns)

    # Concat both dataframes columns and store in disk
    df = pd.concat([df_meas, df_vars], axis=1)

    # Compute a column with the order
    df['order'] = df[columns].sum(axis=1)

    if output_path is not None:
        df.to_csv(output_path.format(order=order, bn=bn), index=False, sep=sep)
        return None # Don't return

    return df


def concatenate_csv(batched_dataframes, output_path, sep='\t'):

    df = pd.concat(batched_dataframes, axis=0)
    df.to_csv(output_path, index=False, sep=sep)


def batch_to_tensor(partition_idxs: np.ndarray,
                    nplets_o:np.ndarray,
                    nplets_s:np.ndarray,
                    nplets_tc:np.ndarray,
                    nplets_dtc:np.ndarray,
                    bn:int,
                    only_synergestic:bool=False,
                    top_k:Optional[int] = None,
                    metric:str = 'o',
                    largest:bool = False):
    
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


def concat_tensors(batched_tensors: List[Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]],
                   top_k: Optional[int] = None,
                   metric:str = 'o',
                   largest:bool = False):
    
    nplets_tc = torch.cat([batch[0] for batch in batched_tensors])
    nplets_dtc = torch.cat([batch[1] for batch in batched_tensors])
    nplets_o = torch.cat([batch[2] for batch in batched_tensors])
    nplets_s = torch.cat([batch[3] for batch in batched_tensors])
    partition_idxs = torch.cat([batch[4] for batch in batched_tensors])

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