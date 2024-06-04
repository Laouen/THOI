from typing import List
from torch.utils.data import IterableDataset
from itertools import combinations
import math
import numpy as np
import torch

class CovarianceDataset(IterableDataset):
    def __init__(self, covmat: np.ndarray, n_variables: int, partition_order: int):
        self.covmat = torch.tensor(covmat).contiguous()
        self.n_variables = n_variables
        self.partition_order = partition_order
        self.partitions_generator = combinations(range(self.n_variables), self.partition_order)

    def __len__(self):
        """Returns the number of combinations of features of the specified order."""
        return math.comb(self.n_variables, self.partition_order)

    def __iter__(self):
        """
        Iterate over all combinations of features in the dataset.

        Yields:
            tuple: A tuple containing:
                - partition_idxs (list): The indices of the features in the current combination.
                - partition_covmat (np.ndarray): The submatrix of the covariance matrix corresponding to the current combination, shape (order, order).
        """
        for partition_idxs in self.partitions_generator:
            # |order|
            partition_idxs = torch.tensor(partition_idxs)
            # |order| x |order|
            subcovmat = self.covmat[partition_idxs][:,partition_idxs]

            yield partition_idxs, subcovmat


class MultiCovarianceDataset(IterableDataset):
    def __init__(self, covmats: List[np.ndarray], n_variables: int, partition_order: int):
        self.datasets = [CovarianceDataset(covmat, n_variables, partition_order) for covmat in covmats]

    def __len__(self):
        return len(self.datasets[0])

    def __iter__(self):
        
        iterators = [iter(dataset) for dataset in self.datasets]
        for _ in range(len(self)):
            sub_covmats = []
            all_partition_idxs = []
            for it in iterators:
                partition_idxs, sub_covmat = next(it)
                sub_covmats.append(sub_covmat)
                all_partition_idxs.append(partition_idxs)
            
            # |sub| x |order| x |order|
            sub_covmats = torch.stack(sub_covmats)
            # |sub| x |order|
            all_partition_idxs = torch.stack(all_partition_idxs)

            yield all_partition_idxs, sub_covmats


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
