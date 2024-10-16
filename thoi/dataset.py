from torch.utils.data import IterableDataset
from itertools import combinations
import math
import numpy as np
import torch

class CovarianceDataset(IterableDataset):
    def __init__(self,
                 N: int,
                 order: int,
                 device: torch.device):

        # Depending on the resources, maybe its better to create the batches on CPU using RAM memory and/or in parallel
        self.device = device
        self.nplet_generator = combinations(range(N), order)
        self.length = math.comb(N, order)

    def __len__(self):
        """Returns the number of combinations of elements for the specified order."""
        return self.length

    def __iter__(self):
        """
        Returns the next nplet in the sequence.

        Returns:
            - nplets (torch.Tensor): The indices of the features in the current combination.
        """
        for nplet in self.nplet_generator:
            yield torch.tensor(nplet, device=self.device)


class HotEncodedMultiOrderDataset(IterableDataset):
    def __init__(self,
                 N: int,
                 min_order: int,
                 max_order: int,
                 device: torch.device):

        self.device = device
        self.N = N
        self.min_order = min_order
        self.max_order = max_order
        self.orders_length = [
            math.comb(self.N, order)
            for order in range(self.min_order, self.max_order+1)
        ]
        self.total_length = np.sum(self.orders_length)

    def __len__(self):
        return self.total_length

    def __iter__(self):
        """
        Iterate over all combinations of features in the dataset.

        Yields:
            tuple: A tuple containing:
                nplets_idxs_hot_conded (np.ndarray): A hot encoded representation of the elements in the current combination.
                partition_covmat (np.ndarray): The masked matrix where the covariance matrix corresponding to the current combination is maintained and the other rows and columns are set as independent normal (0,1) variables.
        """
        for order in range(self.min_order, self.max_order+1):
            for nplets in combinations(range(self.N), order):
                nplets_hot_encoded = torch.zeros(self.N, dtype=torch.int, device=self.device)
                nplets_hot_encoded[list(nplets)] = 1
                
                yield nplets_hot_encoded
