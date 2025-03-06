# tests/test_multiorder_measures.py

import unittest
import numpy as np
import pandas as pd
import torch
import os

from itertools import combinations

from thoi.collectors import batch_to_csv
from thoi.measures.gaussian_copula import nplets_measures
from thoi.measures.gaussian_copula_hot_encoded import nplets_measures_hot_encoded
from thoi.commons import gaussian_copula_covmat

# TODO: make this test for all combinations of devices in [cpu, gpu] and different input types
class TestNpletsMeasures(unittest.TestCase):

    # make the constructor
    def setUp(self):
        # Read X from the X.tsv file in the tests folder
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data','X_random.tsv')
        self.X = pd.read_csv(file_path, sep='\t', header=None).values
        self.covmat = gaussian_copula_covmat(self.X)

        # get precomputed multi order measures
        self.df_true = pd.read_csv(
            os.path.join(current_dir, 'data','X_random__multi_order_measures.tsv'),
            sep='\t'
        )

        self.cols_to_compare = ['tc', 'dtc', 'o', 's']

    def _as_sorted_dataframe(self, df, nplets=None):
        
        if isinstance(df, torch.Tensor):
            assert nplets is not None
            df = batch_to_csv(nplets, df[:,:,0], df[:,:,1], df[:,:,2], df[:,:,3], 0, N=10)
        
        df = df.reset_index(drop=True)
        N = self.X.shape[1]
        nplet_cols = [f'var_{i}' for i in range(N)]
        df.loc[np.arange(len(df)), 'nplet'] = df[nplet_cols].apply(lambda x: ''.join(x.values.astype(int).astype(str)), axis=1)
        
        df = df.sort_values(by=['order', 'nplet'])
        df = df.reset_index(drop=True)
        return df

    def _compare_with_ground_truth(self, res, nplets, rtol=1e-5, atol=1e-8):
        
        df_test = self._as_sorted_dataframe(res, nplets)
        
        df_true = self.df_true[self.df_true['order'] == nplets.shape[1]]
        df_true = self._as_sorted_dataframe(df_true)

        pd.testing.assert_frame_equal(df_test, df_true, rtol=rtol, atol=atol)

    def _validate_same_results_for_repeated_datasets(self, res, nplets, rtol, atol):

        # Every dataset should have the same results as the first one
        for idx in range(res.shape[1]):
            self._compare_with_ground_truth(res[:,idx,:].unsqueeze(1), nplets, rtol, atol)

    def test_nplets_measures_timeseries(self):
        full_nplet = range(self.X.shape[1])
        for order in range(3,11):
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(full_nplet, order)))
                res = nplets_measures(self.X, nplets)
                self._compare_with_ground_truth(res, nplets, rtol=1e-16, atol=1e-12)
    
    def test_nplets_measures_precomputed(self):
        full_nplet = range(self.X.shape[1])
        for order in range(3,11):
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(full_nplet, order)))
                res = nplets_measures(self.covmat, nplets, covmat_precomputed=True, T=self.X.shape[0])
                self._compare_with_ground_truth(res, nplets, rtol=1e-16, atol=1e-12)
    
    def test_multiple_times_same_datasets_timeseries(self):
        full_nplet = range(self.X.shape[1])
        for order in range(3,11):
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(full_nplet, order)))
                res = nplets_measures([self.X, self.X], nplets)
                self._validate_same_results_for_repeated_datasets(res, nplets, rtol=1e-16, atol=1e-7)
    
    def test_multiple_times_same_datasets_precomputed(self):
        full_nplet = range(self.X.shape[1])
        for order in range(3,11):
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(full_nplet, order)))
                res = nplets_measures([self.covmat, self.covmat], nplets, covmat_precomputed=True, T=self.X.shape[0])
                self._validate_same_results_for_repeated_datasets(res, nplets, rtol=1e-16, atol=1e-7)
    
    def test_batch_size_does_not_change_result(self):
        full_nplet = range(self.X.shape[1])
        
        nplets = torch.tensor([list(c) for i, c in enumerate(combinations(full_nplet, 3)) if i < 100000])
        
        # test for different batch sizes
        res = nplets_measures(self.X, nplets)
        res2 = nplets_measures(self.X, nplets, batch_size=10)
        res3 = nplets_measures(self.X, nplets, batch_size=100)
        res4 = nplets_measures(self.X, nplets, batch_size=1000)
        res5 = nplets_measures(self.X, nplets, batch_size=10000)
        res6 = nplets_measures(self.X, nplets, batch_size=100000)
        res7 = nplets_measures(self.X, nplets, batch_size=1000000) # this should do a single batch
        
        # check that the results are the same
        self.assertTrue(torch.allclose(res, res2, rtol=1e-16, atol=1e-12))
        self.assertTrue(torch.allclose(res, res3, rtol=1e-16, atol=1e-12))
        self.assertTrue(torch.allclose(res, res4, rtol=1e-16, atol=1e-12))
        self.assertTrue(torch.allclose(res, res5, rtol=1e-16, atol=1e-12))
        self.assertTrue(torch.allclose(res, res6, rtol=1e-16, atol=1e-12))
        self.assertTrue(torch.allclose(res, res7, rtol=1e-16, atol=1e-12))

    def test_nplets_measures_timeseries_hot_encoded(self):
        N = self.X.shape[1]
        full_nplet = range(N)
        for order in range(3,11):
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(full_nplet, order)))
                batch_size = nplets.shape[0]
                nplets_hot_encoded = torch.zeros((batch_size, N), dtype=torch.int)
                nplets_hot_encoded[torch.arange(0,batch_size, dtype=int).view(-1,1), nplets] = 1
                res = nplets_measures_hot_encoded(self.X, nplets_hot_encoded)
                self._compare_with_ground_truth(res, nplets, rtol=1e-8, atol=1e-4)

    def test_nplets_measures_precomputed_hot_encoded(self):
        N = self.X.shape[1]
        full_nplet = range(N)
        for order in range(3,11):
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(full_nplet, order)))
                batch_size = nplets.shape[0]
                nplets_hot_encoded = torch.zeros((batch_size, N), dtype=torch.int)
                nplets_hot_encoded[torch.arange(0,batch_size, dtype=int).view(-1,1), nplets] = 1
                res = nplets_measures_hot_encoded(self.covmat, nplets_hot_encoded, covmat_precomputed=True, T=self.X.shape[0])
                self._compare_with_ground_truth(res, nplets, rtol=1e-8, atol=1e-4)
    
    def test_multiple_times_same_dataset_timeseries_hot_encoded(self):
        N = self.X.shape[1]
        full_nplet = range(self.X.shape[1])
        for order in range(3,11):
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(full_nplet, order)))
                batch_size = nplets.shape[0]
                nplets_hot_encoded = torch.zeros((batch_size, N), dtype=torch.int)
                nplets_hot_encoded[torch.arange(0,batch_size, dtype=int).view(-1,1), nplets] = 1
                res = nplets_measures_hot_encoded([self.X, self.X], nplets_hot_encoded)
                self._validate_same_results_for_repeated_datasets(res, nplets, rtol=1e-8, atol=1e-4)

    def test_multiple_times_same_dataset_precomputed_hot_encoded(self):
        N = self.X.shape[1]
        full_nplet = range(self.X.shape[1])
        for order in range(3,11):
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(full_nplet, order)))
                batch_size = nplets.shape[0]
                nplets_hot_encoded = torch.zeros((batch_size, N), dtype=torch.int)
                nplets_hot_encoded[torch.arange(0,batch_size, dtype=int).view(-1,1), nplets] = 1
                res = nplets_measures_hot_encoded([self.covmat, self.covmat], nplets_hot_encoded, covmat_precomputed=True, T=self.X.shape[0])
                self._validate_same_results_for_repeated_datasets(res, nplets, rtol=1e-8, atol=1e-4)

    def test_batch_size_does_not_change_result_hot_encoded(self):
        full_nplet = range(self.X.shape[1])
        
        nplets = torch.tensor([list(c) for i, c in enumerate(combinations(full_nplet, 3)) if i < 100000])
        nplets_hot_encoded = torch.zeros((nplets.shape[0], self.X.shape[1]), dtype=torch.int)
        nplets_hot_encoded[torch.arange(0,nplets.shape[0], dtype=int).view(-1,1), nplets] = 1
        
        # test for different batch sizes
        res = nplets_measures_hot_encoded(self.X, nplets_hot_encoded)
        res2 = nplets_measures_hot_encoded(self.X, nplets_hot_encoded, batch_size=10)
        res3 = nplets_measures_hot_encoded(self.X, nplets_hot_encoded, batch_size=100)
        res4 = nplets_measures_hot_encoded(self.X, nplets_hot_encoded, batch_size=1000)
        res5 = nplets_measures_hot_encoded(self.X, nplets_hot_encoded, batch_size=10000)
        res6 = nplets_measures_hot_encoded(self.X, nplets_hot_encoded, batch_size=100000)
        res7 = nplets_measures_hot_encoded(self.X, nplets_hot_encoded, batch_size=1000000) # this should do a single batch
        
        # check that the results are the same
        self.assertTrue(torch.allclose(res, res2, rtol=1e-16, atol=1e-12))
        self.assertTrue(torch.allclose(res, res3, rtol=1e-16, atol=1e-12))
        self.assertTrue(torch.allclose(res, res4, rtol=1e-16, atol=1e-12))
        self.assertTrue(torch.allclose(res, res5, rtol=1e-16, atol=1e-12))
        self.assertTrue(torch.allclose(res, res6, rtol=1e-16, atol=1e-12))
        self.assertTrue(torch.allclose(res, res7, rtol=1e-16, atol=1e-12))

        
if __name__ == '__main__':
    unittest.main()
