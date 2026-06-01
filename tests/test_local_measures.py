# tests/test_local_measures.py

import unittest
from itertools import combinations
from math import comb
import numpy as np
import pandas as pd
import os
import torch

from thoi.commons import gaussian_copula_covmat
from thoi.measures.gaussian_copula import nplets_measures
from thoi.measures.gaussian_copula_local import (
    local_nplets_measures,
    local_multi_order_measures,
    time_averaged_local_measures,
)

_CURRENT_DIR = os.path.dirname(__file__)
_X_FULL = pd.read_csv(
    os.path.join(_CURRENT_DIR, 'data', 'X_random.tsv'),
    sep='\t', header=None,
).values
_DF_TRUE = pd.read_csv(
    os.path.join(_CURRENT_DIR, 'data', 'X_random__multi_order_measures.tsv'),
    sep='\t',
)

# Small slice used by shape/bias tests: 200 time points × 5 variables.
_X = _X_FULL[:200, :5]  # (T=200, N=5)


class TestLocalNpletsMeasures(unittest.TestCase):
    """Tests for local_nplets_measures output shapes and numerical properties."""

    def setUp(self):
        self.X = _X
        self.T, self.N = self.X.shape
        self.nplets = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    def _compute(self, X=None, nplets=None):
        X = self.X if X is None else X
        nplets = self.nplets if nplets is None else nplets
        return local_nplets_measures(X, nplets, device='cpu', dtype=torch.float64)

    def test_output_shape_single_dataset(self):
        result = self._compute()
        B = self.nplets.shape[0]  # 2
        self.assertEqual(result.shape, (B, 1, self.T, 4))

    def test_output_shape_multiple_datasets(self):
        # Stack two copies → D=2; use a single n-plet so the shape is unambiguous.
        X_multi = np.stack([self.X, self.X], axis=0)  # (2, T, N)
        nplet = self.nplets[:1]  # (1, 3)
        result = self._compute(X=X_multi, nplets=nplet)
        self.assertEqual(result.shape, (1, 2, self.T, 4))

    def test_output_is_finite(self):
        result = self._compute()
        self.assertTrue(
            torch.isfinite(result).all(),
            "local_nplets_measures produced non-finite values",
        )

    def test_identical_datasets_give_identical_results(self):
        """Two copies of the same dataset must produce the same measures."""
        X_multi = np.stack([self.X, self.X], axis=0)
        result = self._compute(X=X_multi, nplets=self.nplets[:1])
        # result shape: (1, 2, T, 4) — D=2 are the two copies
        torch.testing.assert_close(result[:, 0, :, :], result[:, 1, :, :])

    def test_values_match_local_multi_order(self):
        """local_nplets_measures must agree exactly with local_multi_order_measures.

        Both functions call _local_single_batch_from_xg, but via different paths:
        local_nplets_measures goes through _local_nplets_from_xg while
        local_multi_order_measures calls _local_single_batch_from_xg directly.
        Any modification to one path that is not reflected in the other is caught here.
        """
        order = 3
        nplets = torch.tensor(list(combinations(range(self.N), order)))
        nplets_out = local_nplets_measures(
            self.X, nplets, device='cpu', dtype=torch.float32,
        )  # (C(N,3), 1, T, 4)
        multi_out = local_multi_order_measures(
            self.X, min_order=order, max_order=order,
            device='cpu', dtype=torch.float32,
        )  # {3: (C(N,3), 1, T, 4)}
        torch.testing.assert_close(nplets_out, multi_out[order], atol=0, rtol=0)


class TestTimeAveragedLocalMeasures(unittest.TestCase):
    """Tests for local_multi_order_measures and time_averaged_local_measures."""

    def setUp(self):
        self.X = _X

    def test_manual_average_matches_no_bias_wrapper(self):
        """mean(dim=2) of local_multi_order_measures == time_averaged with bias_correction=False."""
        local_results = local_multi_order_measures(
            self.X, min_order=3, max_order=3,
            device='cpu', dtype=torch.float64,
        )
        averaged_no_bias = time_averaged_local_measures(
            self.X, min_order=3, max_order=3,
            device='cpu', dtype=torch.float64,
            bias_correction=False,
        )
        manual_avg = local_results[3].mean(dim=2)  # [n_comb, D, 4]
        self.assertTrue(
            torch.allclose(manual_avg, averaged_no_bias[3], rtol=0, atol=1e-12),
            "Manual time average does not match time_averaged_local_measures(bias_correction=False)",
        )

    def test_bias_correction_changes_results(self):
        """Results with and without bias correction must differ."""
        no_bias = time_averaged_local_measures(
            self.X, min_order=3, max_order=3,
            device='cpu', dtype=torch.float64,
            bias_correction=False,
        )
        with_bias = time_averaged_local_measures(
            self.X, min_order=3, max_order=3,
            device='cpu', dtype=torch.float64,
            bias_correction=True,
        )
        max_diff = torch.abs(with_bias[3] - no_bias[3]).max().item()
        self.assertGreater(
            max_diff, 1e-6,
            "Bias correction had no effect on results",
        )

    def test_time_averaged_tc_and_s_non_negative(self):
        """After time averaging, TC and S must be >= 0 for the Gaussian copula."""
        result = time_averaged_local_measures(
            self.X, min_order=3, max_order=3,
            device='cpu', dtype=torch.float64,
            bias_correction=True,
        )
        tc = result[3][:, :, 0]  # [n_comb, D]
        s  = result[3][:, :, 3]  # [n_comb, D]
        self.assertTrue(
            (tc >= -1e-10).all(),
            f"Time-averaged TC should be non-negative; min={tc.min().item():.4f}",
        )
        self.assertTrue(
            (s >= -1e-10).all(),
            f"Time-averaged S should be non-negative; min={s.min().item():.4f}",
        )

    def test_output_shape(self):
        """time_averaged_local_measures must return (C(N,K), D, 4) per order."""
        N = self.X.shape[1]  # 5
        result = time_averaged_local_measures(
            self.X, min_order=3, max_order=4,
            device='cpu', dtype=torch.float64,
            bias_correction=False,
        )
        for order in [3, 4]:
            with self.subTest(order=order):
                self.assertIn(order, result)
                self.assertEqual(result[order].shape, (comb(N, order), 1, 4))


class TestLocalMatchesOriginalMeasures(unittest.TestCase):
    """Direct comparison: time-averaged local measures == original measures.

    For each order K, generates all C(N, K) n-plets, calls nplets_measures
    (the original implementation) and time_averaged_local_measures (local
    implementation averaged over time + bias-corrected), and asserts they agree.
    No ground-truth file is involved — this tests the two code paths directly.
    """

    N_SUB = 5
    MIN_ORDER = 3
    MAX_ORDER = 5
    ATOL = 1e-5

    def setUp(self):
        self.X = _X_FULL[:, : self.N_SUB]

    def test_nplets_measures_matches_time_averaged_local(self):
        """nplets_measures == time_averaged_local_measures for every order."""
        for order in range(self.MIN_ORDER, self.MAX_ORDER + 1):
            with self.subTest(order=order):
                nplets = torch.tensor(list(combinations(range(self.N_SUB), order)))
                original = nplets_measures(self.X, nplets)   # (C(N,K), 1, 4)
                local = time_averaged_local_measures(
                    self.X,
                    min_order=order,
                    max_order=order,
                    device='cpu',
                    dtype=torch.float32,
                    bias_correction=True,
                )                                             # {order: (C(N,K), 1, 4)}
                torch.testing.assert_close(
                    original.float(),
                    local[order],
                    atol=self.ATOL,
                    rtol=0,
                )

    def test_multiple_datasets_nplets_measures_matches_local(self):
        """Consistency holds for D > 1 datasets simultaneously."""
        order = 3
        nplets = torch.tensor(list(combinations(range(self.N_SUB), order)))
        original = nplets_measures([self.X, self.X], nplets)   # (C(N,K), 2, 4)
        local = time_averaged_local_measures(
            [self.X, self.X],
            min_order=order,
            max_order=order,
            device='cpu',
            dtype=torch.float32,
            bias_correction=True,
        )
        torch.testing.assert_close(
            original.float(),
            local[order],
            atol=self.ATOL,
            rtol=0,
        )


class TestTimeAveragedLocalVsGroundTruth(unittest.TestCase):
    """Verify time_averaged_local_measures against the pre-computed ground truth.

    Uses the first N_SUB=5 variables with the full time series so that the
    Gaussian copula normalisation (marginal per variable) and the bias correction
    (function of K and T only) match exactly what was used to build the truth file.
    The rows for n-plets restricted to those 5 variables can therefore serve as
    the reference.  The tolerance is set to 1e-5 (float32 compute vs float64
    ground truth; observed max error ~1.4e-6).
    """

    N_SUB = 5           # number of variables used from the dataset
    MIN_ORDER = 3
    MAX_ORDER = 5       # = N_SUB, covers all possible orders
    ATOL = 1e-5

    def setUp(self):
        self.X = _X_FULL[:, : self.N_SUB]          # (T=50 000, N_SUB)

        # Keep only rows for dataset 0 whose n-plet uses only variables 0..N_SUB-1.
        higher_var_cols = [f'var_{i}' for i in range(self.N_SUB, 10)]
        mask = (
            (_DF_TRUE[higher_var_cols] == 0).all(axis=1)
            & (_DF_TRUE['dataset'] == 0)
        )
        self.df_true = _DF_TRUE[mask].copy().reset_index(drop=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run(self, X=None, **kwargs):
        defaults = dict(
            min_order=self.MIN_ORDER,
            max_order=self.MAX_ORDER,
            device='cpu',
            dtype=torch.float32,
            bias_correction=True,
        )
        defaults.update(kwargs)
        return time_averaged_local_measures(self.X if X is None else X, **defaults)

    def _result_to_sorted_df(self, result):
        """Convert a result dict {order: Tensor[C(N,K), D, 4]} to a sorted DataFrame.

        The n-plet variable membership is hot-encoded over all 10 original
        columns (var_0..var_9) so the 'nplet' sort key matches the ground truth.
        D=0 is always used (caller should extract a single-dataset slice first).
        """
        rows = []
        for order, measures in result.items():
            for nplet_idx, nplet in enumerate(combinations(range(self.N_SUB), order)):
                row = {f'var_{i}': int(i in set(nplet)) for i in range(10)}
                row['order'] = order
                row['dataset'] = 0
                tc, dtc, o, s = measures[nplet_idx, 0, :].float().tolist()
                row.update(tc=tc, dtc=dtc, o=o, s=s)
                rows.append(row)
        df = pd.DataFrame(rows)
        nplet_cols = [f'var_{i}' for i in range(10)]
        df['nplet'] = df[nplet_cols].apply(
            lambda x: ''.join(x.values.astype(int).astype(str)), axis=1
        )
        return df.sort_values(['order', 'nplet']).reset_index(drop=True)

    def _sorted_ground_truth(self):
        df = self.df_true.copy()
        nplet_cols = [f'var_{i}' for i in range(10)]
        df['nplet'] = df[nplet_cols].apply(
            lambda x: ''.join(x.values.astype(int).astype(str)), axis=1
        )
        return df.sort_values(['order', 'nplet']).reset_index(drop=True)

    def _compare_with_ground_truth(self, result):
        df_res = self._result_to_sorted_df(result)
        df_true = self._sorted_ground_truth()
        cols = ['tc', 'dtc', 'o', 's']
        pd.testing.assert_frame_equal(
            df_res[cols],
            df_true[cols].reset_index(drop=True),
            atol=self.ATOL,
            rtol=0,
            check_dtype=False,
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_timeseries_matches_ground_truth(self):
        """time_averaged_local_measures on raw timeseries matches saved ground truth."""
        result = self._run()
        self._compare_with_ground_truth(result)

    def test_precomputed_xg_covmats_matches_ground_truth(self):
        """Passing pre-normalised Xg + covmats gives the same result as raw timeseries."""
        X_t = torch.as_tensor(self.X).unsqueeze(0)           # (1, T, N_SUB)
        Xg, covmats = gaussian_copula_covmat(X_t, return_xg=True)
        result = self._run(X=Xg.squeeze(0), covmats=covmats.squeeze(0))
        self._compare_with_ground_truth(result)

    def test_multiple_identical_datasets_match_ground_truth(self):
        """With D identical datasets every dataset's results match the ground truth."""
        result = self._run(X=[self.X, self.X])
        for d in range(2):
            with self.subTest(dataset=d):
                # Re-pack the D-slice as (n_comb, 1, 4) so _result_to_sorted_df works.
                result_d = {k: v[:, d : d + 1, :] for k, v in result.items()}
                self._compare_with_ground_truth(result_d)


if __name__ == '__main__':
    unittest.main()
