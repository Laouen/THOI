from typing import Union
from torch.utils.data import IterableDataset
from itertools import combinations
import math
import numpy as np
import torch

class CovarianceDataset(IterableDataset):
    def __init__(self,
                 covmat: torch.tensor,
                 partition_order: int):
        
        assert len(covmat.shape) == 3, 'The covariance matrix must be 3D. (n_data, n_variables, n_variables)'

        # Force covariance matrix to be contiguous in CPU to use the CPU memory to create the next batche
        self.covmats = covmat.cpu().contiguous()
        self.n_variables = self.covmats.shape[1]
        self.partition_order = partition_order
        self.partitions_generator = combinations(range(self.n_variables), self.partition_order)

    def __len__(self):
        """Returns the number of combinations of features of the specified order."""
        return self.covmats.shape[0] * math.comb(self.n_variables, self.partition_order)

    def __iter__(self):
        """
        Iterate over all combinations of features in the dataset.

        Yields:
            tuple: A tuple containing:
                - partition_idxs (list): The indices of the features in the current combination.
                - partition_covmat (np.ndarray): The submatrix of the covariance matrix corresponding to the current combination, shape (order, order).
        """
        for partition_idxs in self.partitions_generator:
            partition_idxs = torch.tensor(partition_idxs)

            # (|order|, |n_covmats x order|)
            yield partition_idxs, torch.stack([covmat[partition_idxs][:,partition_idxs] for covmat in self.covmats])


class HotEncodedMultiOrderDataset(IterableDataset):
    def __init__(self, covmat: Union[np.ndarray, torch.tensor], min_order: int, max_order: int):

        self.covmat = torch.tensor(covmat).contiguous()
        self.n_variables = self.covmat.shape[0]
        self.min_order = min_order
        self.max_order = max_order

    def __len__(self):
        """Returns the number of combinations of features for all order."""
        return np.sum([
            math.comb(self.n_variables, order)
            for order in range(self.min_order, self.max_order+1)
        ])

    def __iter__(self):
        """
        Iterate over all combinations of features in the dataset.

        Yields:
            tuple: A tuple containing:
                partition_idxs_hot_conded (np.ndarray): A hot encoded representation of the elements in the current combination.
                partition_covmat (np.ndarray): The masked matrix where the covariance matrix corresponding to the current combination is maintained and the other rows and columns are set as independent normal (0,1) variables.
        """
        for order in range(self.min_order, self.max_order+1):
            for partition_idxs in combinations(range(self.n_variables), order):
                partition_idxs_hot_encoded = torch.zeros(self.n_variables, dtype=torch.int)
                partition_idxs_hot_encoded[list(partition_idxs)] = 1
                
                yield partition_idxs_hot_encoded


class NpletsCovariancesDataset(IterableDataset):
    def __init__(self, covmat: torch.tensor, nplets: torch.tensor):

        assert covmat.device == nplets.device, 'covariance and nplets'

        self.covmat = covmat
        self.nplets = nplets

    def __len__(self):
        """Returns the number of combinations of features of the specified order."""
        return len(self.nplets)

    def __iter__(self):
        """
        Iterate over all combinations of features in the dataset.

        Yields:
            tuple: A tuple containing:
                partition_covmat (np.ndarray): The subcovmat of the covariance matrix corresponding to the current combination, shape (order, order).
        """
        for partition_idxs in self.nplets:
            yield self.covmat[partition_idxs][:,partition_idxs]
