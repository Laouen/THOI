# tests/test_multiorder_measures.py

import unittest
import numpy as np
import pandas as pd
import os

from thoi.measures.gaussian_copula import multi_order_measures
from thoi.measures.gaussian_copula_hot_encoded import multi_order_measures_hot_encoded
from thoi.commons import gaussian_copula_covmat

# TODO: make this test for all combinations of use_cpu in [True, False] use_cpu_dataset in [True, False] and dataset_device in ['cpu', 'gpu']
class TestMultiOrderMeasures(unittest.TestCase):

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

    def _validate_with_stats(self, df_res, atol):
        
        dfs = []
        for order in sorted(df_res['order'].unique()):
            df_desc = df_res[df_res['order'] == order].describe()
            df_desc['order'] = order
            dfs.append(df_desc)

        df_desc = pd.concat(dfs)

        for order in df_res['order'].unique():
            with self.subTest(order=order):
                df_desc_order = df_desc[df_desc['order'] == order][self.cols_to_compare]
                df_stats_order = self.df_stats[self.df_stats['order'] == order][self.cols_to_compare]

                # sort by index
                df_desc_order = df_desc_order.sort_index()
                df_stats_order = df_stats_order.sort_index()

                self.assertTrue(np.allclose(df_desc_order.values, df_stats_order.values, atol=atol, equal_nan=True))

    def _validate_same_results_for_repeated_datasets(self, df_res, atol):
    
        # check the results are the same if the same dataset is passed multiple times
        dfs = []
        for _, group in df_res.groupby('dataset'):
            # Remove dataset column as it is not relevant for the comparison and it will be different
            group.drop(columns=['dataset'], inplace=True)
            group = group.reset_index(drop=True)
            dfs.append(group)

        for df in dfs[1:]:
            pd.testing.assert_frame_equal(df, dfs[0])
            self._validate_with_stats(df, atol=atol)

    def test_multiorder_measures_timeseries(self):
        df_res = multi_order_measures(self.X, use_cpu=True)
        self._validate_with_stats(df_res, atol=1e-6)

    def test_multiorder_measures_precomputed_covmat(self):
        df_res = multi_order_measures(self.covmat, covmat_precomputed=True, T=self.X.shape[0], use_cpu=True)
        self._validate_with_stats(df_res, atol=1e-6)

    def test_multiorder_measures_timeseries_hot_encoded(self):
        df_res = multi_order_measures_hot_encoded(self.X, batch_size=200000, use_cpu=True)
        self._validate_with_stats(df_res, atol=1e-4)

    def test_multiorder_measures_precomputed_hot_encoded(self):
        df_res = multi_order_measures_hot_encoded(self.covmat, covmat_precomputed=True, T=self.X.shape[0], use_cpu=True)
        self._validate_with_stats(df_res, atol=1e-4)

    def test_multiple_times_same_datasets_timeseries(self):
        X = [self.X, self.X]
        df_res = multi_order_measures(X, use_cpu=True)
        self._validate_same_results_for_repeated_datasets(df_res, atol=1e-6)
    
    def test_multiple_times_same_datasets_precomputed(self):
        covmats = [self.covmat, self.covmat]
        df_res = multi_order_measures(covmats, covmat_precomputed=True, T=self.X.shape[0], use_cpu=True)
        self._validate_same_results_for_repeated_datasets(df_res, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
