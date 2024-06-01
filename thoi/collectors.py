from typing import List, Optional
import pandas as pd
import numpy as np

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

    # if only_synergestic; remove nplets with nplet_o >= 0
    if only_synergestic:
        to_keep = np.where(nplets_o < 0)[0]
        nplets_tc = nplets_tc[to_keep]
        nplets_dtc = nplets_dtc[to_keep]
        nplets_o = nplets_o[to_keep]
        nplets_s = nplets_s[to_keep]
        partition_idxs = partition_idxs[to_keep]

    df_meas = pd.DataFrame({
        'tc': nplets_tc,
        'dtc': nplets_dtc,
        'o': nplets_o,
        's': nplets_s
    })

    # Create a DataFrame with the n-plets
    batch_size, order = partition_idxs.shape
    bool_array = np.zeros((batch_size, N), dtype=bool)
    rows = np.arange(batch_size).reshape(-1, 1)
    bool_array[rows, partition_idxs] = True
    df_vars = pd.DataFrame(bool_array, columns=columns)

    # Concat both dataframes columns and store in disk
    df = pd.concat([df_meas, df_vars], axis=1)

    if output_path is not None:
        df.to_csv(output_path.format(order=order, bn=bn), index=False, sep=sep)
        return None # Don't return

    return df


def concatenate_csv(batched_dataframes, output_path, sep='\t'):

    df = pd.concat(batched_dataframes, axis=0)
    df.to_csv(output_path, index=False, sep=sep)