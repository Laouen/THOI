# tests/test_measures.py

import unittest
import numpy as np
import pandas as pd
import os

from thoi.measures.gaussian_copula import multi_order_measures, gaussian_copula

class TestMultiOrderMeasures(unittest.TestCase):

    # make the constructor
    def setUp(self):
        # Read X from the X.tsv file in the tests folder
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data','X_random.tsv')
        self.X = pd.read_csv(file_path, sep='\t', header=None).values

        # get precomputed stats for each order and measure
        self.df_stats = pd.read_csv(
            os.path.join(current_dir, 'data','X_random__measures_stats.tsv'),
            sep='\t', index_col=0
        )

        self.cols_to_compare = ['tc', 'dtc', 'o', 's']

    def test_multiorder_measures_timeseries(self):

        df_res = multi_order_measures(self.X)

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

                self.assertTrue(np.allclose(df_desc_order.values, df_stats_order.values, atol=1e-6, equal_nan=True))

    def test_multiorder_measures_precomputed_covmat(self):

        T, N = self.X.shape
        covmat = gaussian_copula(self.X)[1]

        df_res = multi_order_measures(covmat, covmat_precomputed=True, T=T)

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

                self.assertTrue(np.allclose(df_desc_order.values, df_stats_order.values, atol=1e-6, equal_nan=True))

if __name__ == '__main__':
    unittest.main()
