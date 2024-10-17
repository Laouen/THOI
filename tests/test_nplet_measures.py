# tests/test_multiorder_measures.py

import unittest
import numpy as np
import pandas as pd
import torch
import os

from itertools import combinations

from thoi.measures.gaussian_copula import nplets_measures
from thoi.measures.gaussian_copula_hot_encoded import nplets_measures_hot_encoded
from thoi.commons import gaussian_copula_covmat

# TODO: make this test for all combinations of use_cpu in [True, False] use_cpu_dataset in [True, False] and dataset_device in ['cpu', 'gpu']
class TestNpletsMeasures(unittest.TestCase):

    # make the constructor
    def setUp(self):
        # Read X from the X.tsv file in the tests folder
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data','X_random.tsv')
        self.X = pd.read_csv(file_path, sep='\t', header=None).values
        self.covmat = gaussian_copula_covmat(self.X)

        # get precomputed stats for each order and measure
        self.df_stats = pd.read_csv(
            os.path.join(current_dir, 'data','X_random__measures_stats.tsv'),
            sep='\t', index_col=0
        )

        self.cols_to_compare = ['tc', 'dtc', 'o', 's']

    def _validate_with_stats(self, res, order, rtol, atol):
        
        if isinstance(res, torch.Tensor):
            res = res.squeeze(1) # remove the dataset dimension
            df_desc = pd.DataFrame(res.detach().cpu().numpy(), columns=['tc', 'dtc', 'o', 's'])
            df_desc = df_desc.describe()
        else:
            df_desc = res
        
        df_desc = df_desc.sort_index()
        
        df_stats = self.df_stats[self.df_stats['order'] == order][self.cols_to_compare]
        df_stats = df_stats.sort_index()

        pd.testing.assert_frame_equal(df_desc, df_stats, rtol=rtol, atol=atol)

    def _validate_same_results_for_repeated_datasets(self, res, order, rtol, atol):

        dfs = []
        for idx in range(res.shape[1]):
            df_desc = pd.DataFrame(res[:,idx,:].detach().cpu().numpy(), columns=['tc', 'dtc', 'o', 's'])
            df_desc = df_desc.describe()
            df_desc = df_desc.sort_index()
            dfs.append(df_desc)

        for df in dfs[1:]:
            pd.testing.assert_frame_equal(df, dfs[0])
            self._validate_with_stats(df, order, rtol, atol)

    def test_nplets_measures_timeseries(self):
        for order in [3, 4, 5, 18, 19, 20]:
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(range(self.X.shape[1]), order)))
                res = nplets_measures(self.X, nplets, use_cpu=True)
                self._validate_with_stats(res, order, rtol=1e-16, atol=1e-7)
    
    def test_nplets_measures_precomputed(self):
        for order in [3, 4, 5, 18, 19, 20]:
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(range(self.X.shape[1]), order)))
                res = nplets_measures(self.covmat, nplets, covmat_precomputed=True, T=self.X.shape[0], use_cpu=True)
                self._validate_with_stats(res, order, rtol=1e-16, atol=1e-7)

    def test_nplets_measures_timeseries_hot_encoded(self):
        N = self.X.shape[1]
        for order in [3, 4, 5, 18, 19, 20]:
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(range(self.X.shape[1]), order)))
                batch_size = nplets.shape[0]
                nplets_hot_encoded = torch.zeros((batch_size, N), dtype=torch.int)
                nplets_hot_encoded[torch.arange(0,batch_size, dtype=int).view(-1,1), nplets] = 1
                res = nplets_measures_hot_encoded(self.X, nplets_hot_encoded, use_cpu=True)
                self._validate_with_stats(res, order, rtol=1e-7, atol=1e-4)

    def test_nplets_measures_timeseries_hot_encoded(self):
        N = self.X.shape[1]
        for order in [3, 4, 5, 18, 19, 20]:
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(range(self.X.shape[1]), order)))
                batch_size = nplets.shape[0]
                nplets_hot_encoded = torch.zeros((batch_size, N), dtype=torch.int)
                nplets_hot_encoded[torch.arange(0,batch_size, dtype=int).view(-1,1), nplets] = 1
                res = nplets_measures_hot_encoded(self.covmat, nplets_hot_encoded, covmat_precomputed=True, T=self.X.shape[0], use_cpu=True)
                self._validate_with_stats(res, order, rtol=1e-7, atol=1e-4)

    def test_multiple_times_same_datasets_timeseries(self):
        for order in [3, 4, 5, 18, 19, 20]:
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(range(self.X.shape[1]), order)))
                res = nplets_measures([self.X, self.X], nplets, use_cpu=True)
                self._validate_same_results_for_repeated_datasets(res, order, rtol=1e-7, atol=1e-4)
    
    def test_multiple_times_same_datasets_precomputed(self):
        for order in [3, 4, 5, 18, 19, 20]:
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(range(self.X.shape[1]), order)))
                res = nplets_measures([self.covmat, self.covmat], nplets, covmat_precomputed=True, T=self.X.shape[0], use_cpu=True)
                self._validate_same_results_for_repeated_datasets(res, order, rtol=1e-7, atol=1e-4)

if __name__ == '__main__':
    unittest.main()
