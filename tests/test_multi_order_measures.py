# tests/test_multiorder_measures.py

import unittest
import numpy as np
import pandas as pd
import os

import torch

from thoi.measures.gaussian_copula import multi_order_measures
from thoi.measures.gaussian_copula_hot_encoded import multi_order_measures_hot_encoded
from thoi.commons import gaussian_copula_covmat

# TODO: make this test for all combinations of device in [cpu, cuda] and different input types
class TestMultiOrderMeasures(unittest.TestCase):

    # make the constructor
    def setUp(self):
        # Read X from the X.tsv file in the tests folder
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data','X_random.tsv')
        self.X = pd.read_csv(file_path, sep='\t', header=None).values
        _, cov = gaussian_copula_covmat(torch.as_tensor(self.X).unsqueeze(0))
        self.covmat = cov[0].numpy()

        # get precomputed multi order measures
        self.df_true = pd.read_csv(
            os.path.join(current_dir, 'data','X_random__multi_order_measures.tsv'),
            sep='\t'
        )

        self.cols_to_compare = ['tc', 'dtc', 'o', 's']

    def _as_sorted_dataframe(self, df):
        
        df = df.reset_index(drop=True)
        N = self.X.shape[1]
        nplet_cols = [f'var_{i}' for i in range(N)]
        df.loc[np.arange(len(df)), 'nplet'] = df[nplet_cols].apply(lambda x: ''.join(x.values.astype(int).astype(str)), axis=1)
        
        df = df.sort_values(by=['order', 'nplet'])
        df = df.reset_index(drop=True)
        return df

    def _compare_with_ground_truth(self, df_res, rtol=1e-5, atol=1e-8):
        
        df_test = self._as_sorted_dataframe(df_res)
        df_true = self._as_sorted_dataframe(self.df_true)

        pd.testing.assert_frame_equal(df_test, df_true, rtol=rtol, atol=atol)

    def _validate_same_results_for_repeated_datasets(self, df_res, rtol, atol):

        # Every dataset should have the same results as the first one
        for _, df_dataset in df_res.groupby('dataset'):
            df_dataset['dataset'] = 0 # set all datasets to 0 to compare
            self._compare_with_ground_truth(df_dataset, rtol, atol)

    def test_multiorder_measures_timeseries(self):
        df_res = multi_order_measures(self.X)
        self._compare_with_ground_truth(df_res, rtol=1e-12, atol=1e-12)

    def test_batch_size_D_does_not_change_result(self):
        df_res_no_batch = multi_order_measures(self.X)
        df_res_batch1   = multi_order_measures(self.X, batch_size_D=1)
        self._compare_with_ground_truth(df_res_no_batch, rtol=1e-12, atol=1e-12)
        self._compare_with_ground_truth(df_res_batch1,   rtol=1e-12, atol=1e-12)

    def test_multiorder_measures_precomputed_covmat(self):
        df_res = multi_order_measures(self.covmat, covmat_precomputed=True, T=self.X.shape[0])
        self._compare_with_ground_truth(df_res, rtol=1e-12, atol=1e-12)


    def test_multiorder_measures_timeseries_hot_encoded(self):
        df_res = multi_order_measures_hot_encoded(self.X, batch_size=200000)
        self._compare_with_ground_truth(df_res, rtol=1e-8, atol=1e-4)

    def test_multiorder_measures_precomputed_hot_encoded(self):
        df_res = multi_order_measures_hot_encoded(self.covmat, covmat_precomputed=True, T=self.X.shape[0])
        self._compare_with_ground_truth(df_res, rtol=1e-8, atol=1e-4)

    def test_multiple_times_same_datasets_timeseries(self):
        df_res = multi_order_measures([self.X, self.X])
        self._validate_same_results_for_repeated_datasets(df_res, rtol=1e-12, atol=1e-12)

    def test_multiple_times_same_datasets_precomputed(self):
        covmats = [self.covmat, self.covmat]
        df_res = multi_order_measures(covmats, covmat_precomputed=True, T=self.X.shape[0])
        self._validate_same_results_for_repeated_datasets(df_res, rtol=1e-12, atol=1e-12)

    def test_multiple_times_same_datasets_timeseries_hot_encoded(self):
        df_res = multi_order_measures_hot_encoded([self.X, self.X])
        self._validate_same_results_for_repeated_datasets(df_res, rtol=1e-8, atol=1e-4)
    
    def test_multiple_times_same_datasets_precomputed_hot_encoded(self):
        covmats = [self.covmat, self.covmat]
        df_res = multi_order_measures_hot_encoded(covmats, covmat_precomputed=True, T=self.X.shape[0])
        self._validate_same_results_for_repeated_datasets(df_res, rtol=1e-8, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
