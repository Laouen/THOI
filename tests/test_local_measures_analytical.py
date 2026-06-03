# tests/test_local_measures_pointwise_ground_truth.py
#
# Exact pointwise ground-truth test for local HOI measures.
#
# Mathematical basis
# ------------------
# For data X ~ N(0, C) where C is a *correlation* matrix (diagonal = 1),
# the local measures at a point x are defined analytically as:
#
#   nll_joint(x)     = 0.5 * (K*log(2π) + log|C| + x^T C^{-1} x)
#   nll_i(x_i)       = 0.5 * (log(2π) + x_i²)          [var_i = 1]
#   nll_{-i}(x_{-i}) = 0.5 * ((K-1)*log(2π) + log|C_{-i}| + x_{-i}^T C_{-i}^{-1} x_{-i})
#
#   tc_loc(x)  = Σ_i nll_i(x_i)        - nll_joint(x)
#   dtc_loc(x) = Σ_i nll_{-i}(x_{-i}) - (K-1)*nll_joint(x)
#   o_loc(x)   = tc_loc(x) - dtc_loc(x)
#   s_loc(x)   = tc_loc(x) + dtc_loc(x)
#
# Strategy
# --------
# We bypass the Gaussian copula step by passing pre-normalised data Xg and
# the theoretical correlation matrix C directly to local_nplets_measures.
# This means the function computes measures using exactly C (not an estimate)
# and exactly Xg (not rank-transformed data), so the output must match the
# analytic formula above to near floating-point precision (< 1e-9).
#
# Notes on sign
# -------------
# Individual pointwise local TC, DTC, and S-info CAN be negative — only
# their expectations (E[tc_loc] = TC, etc.) are guaranteed non-negative.

import unittest
import numpy as np
import torch

from thoi.measures.gaussian_copula_local import local_nplets_measures


# ── analytic ground truth ──────────────────────────────────────────────────────

def _analytic_local_measures(x: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Exact pointwise local HOI measures for x given correlation matrix C.

    Parameters
    ----------
    x : (K,) array
        Observation in the Gaussian-normalised space.
    C : (K, K) correlation matrix (diagonal = 1)

    Returns
    -------
    (4,) array : [tc_loc, dtc_loc, o_loc, s_loc]
    """
    K = len(x)
    Cinv = np.linalg.inv(C)
    logdet = np.log(np.linalg.det(C))
    log2pi = np.log(2 * np.pi)

    nll_joint = 0.5 * (K * log2pi + logdet + x @ Cinv @ x)
    nll_marginals_sum = 0.5 * (K * log2pi + float(np.sum(x ** 2)))

    tc = nll_marginals_sum - nll_joint

    loo_nll_sum = 0.0
    for i in range(K):
        idx = [j for j in range(K) if j != i]
        C_loo = C[np.ix_(idx, idx)]
        x_loo = x[idx]
        nll_loo = 0.5 * (
            (K - 1) * log2pi
            + np.log(np.linalg.det(C_loo))
            + x_loo @ np.linalg.inv(C_loo) @ x_loo
        )
        loo_nll_sum += nll_loo

    dtc = loo_nll_sum - (K - 1) * nll_joint
    return np.array([tc, dtc, tc - dtc, tc + dtc])


# ── test cases ─────────────────────────────────────────────────────────────────

class TestLocalMeasuresPointwiseGroundTruth(unittest.TestCase):
    """Verify local_nplets_measures against exact analytic values point-by-point.

    We bypass the Gaussian copula by providing pre-normalised data (Xg) together
    with the theoretical correlation matrix.  The function must then reproduce
    the analytic local HOI measures to near floating-point precision.
    """

    # Correlation matrix used as the covariance of the Gaussian system
    C = np.array([
        [1.00, 0.60, 0.20],
        [0.60, 1.00, 0.30],
        [0.20, 0.30, 1.00],
    ], dtype=np.float64)

    # Deterministic evaluation points (diverse: correlated, anti-correlated,
    # origin, extremes, negative values)
    X_PTS = np.array([
        [ 0.5, -0.3,  0.8],   # mixed signs
        [ 1.0,  1.0,  1.0],   # aligned with positive correlation
        [-1.0,  0.5, -0.5],   # negative dominant
        [ 0.0,  0.0,  0.0],   # origin — non-trivial because C != I
        [ 2.0, -2.0,  1.0],   # large magnitude, anti-correlated
        [-0.1,  0.9, -0.7],   # moderate, mixed
    ], dtype=np.float64)

    ATOL = 1e-8   # limited by eps=1e-10 Cholesky regularisation in the code

    @classmethod
    def setUpClass(cls):
        K = cls.C.shape[0]

        # Analytic ground truth for every evaluation point
        cls.gt = np.stack([_analytic_local_measures(x, cls.C) for x in cls.X_PTS])
        # shape: (n_pts, 4)

        # Feed data directly as Xg (bypass copula) with the theoretical covmat
        Xg_t   = torch.tensor(cls.X_PTS, dtype=torch.float64)   # (n_pts, K)
        C_t    = torch.tensor(cls.C,     dtype=torch.float64)    # (K, K)
        nplets = torch.arange(K).unsqueeze(0)                     # [[0,1,2]]

        res = local_nplets_measures(
            Xg_t,
            nplets=nplets,
            covmats=C_t,
            device='cpu',
            dtype=torch.float64,
        )  # (1, 1, n_pts, 4)

        cls.computed = res[0, 0, :, :].numpy()  # (n_pts, 4)

    # ------------------------------------------------------------------
    # Pointwise match against the analytic formula
    # ------------------------------------------------------------------

    def _check_measure(self, measure_idx: int, name: str):
        for pt_idx, x in enumerate(self.X_PTS):
            with self.subTest(point=pt_idx, measure=name):
                computed_val = self.computed[pt_idx, measure_idx]
                analytic_val = self.gt[pt_idx, measure_idx]
                self.assertAlmostEqual(
                    computed_val, analytic_val, delta=self.ATOL,
                    msg=f"x={x}: {name} computed={computed_val:.8f}, "
                        f"analytic={analytic_val:.8f}",
                )

    def test_tc_loc_matches_analytic(self):
        """Local TC matches the analytic NLL formula at every evaluation point."""
        self._check_measure(0, 'TC')

    def test_dtc_loc_matches_analytic(self):
        """Local DTC matches the analytic NLL formula at every evaluation point."""
        self._check_measure(1, 'DTC')

    def test_o_loc_matches_analytic(self):
        """Local O-information matches the analytic formula at every point."""
        self._check_measure(2, 'O')

    def test_s_loc_matches_analytic(self):
        """Local S-information matches the analytic formula at every point."""
        self._check_measure(3, 'S')

    # ------------------------------------------------------------------
    # Internal consistency (O = TC - DTC, S = TC + DTC) — algebraic
    # ------------------------------------------------------------------

    def test_o_equals_tc_minus_dtc(self):
        """O = TC - DTC must hold exactly at every point."""
        for pt_idx in range(len(self.X_PTS)):
            with self.subTest(point=pt_idx):
                tc  = self.computed[pt_idx, 0]
                dtc = self.computed[pt_idx, 1]
                o   = self.computed[pt_idx, 2]
                self.assertAlmostEqual(o, tc - dtc, delta=1e-12)

    def test_s_equals_tc_plus_dtc(self):
        """S = TC + DTC must hold exactly at every point."""
        for pt_idx in range(len(self.X_PTS)):
            with self.subTest(point=pt_idx):
                tc  = self.computed[pt_idx, 0]
                dtc = self.computed[pt_idx, 1]
                s   = self.computed[pt_idx, 3]
                self.assertAlmostEqual(s, tc + dtc, delta=1e-12)

    # ------------------------------------------------------------------
    # Sign properties of pointwise values
    # ------------------------------------------------------------------

    def test_local_measures_can_be_negative(self):
        """Pointwise local TC, DTC, and S-info are not constrained to be >= 0.

        Only their expectations equal TC, DTC, S, which are >= 0.
        This test asserts that negative values actually occur so that any
        future regression that wrongly clamps them to zero is caught.
        """
        tc_vals  = self.computed[:, 0]
        dtc_vals = self.computed[:, 1]
        s_vals   = self.computed[:, 3]
        self.assertTrue(
            (tc_vals < 0).any(),
            "Expected at least one negative pointwise TC value among the test points",
        )
        self.assertTrue(
            (dtc_vals < 0).any(),
            "Expected at least one negative pointwise DTC value among the test points",
        )
        self.assertTrue(
            (s_vals < 0).any(),
            "Expected at least one negative pointwise S-info value among the test points",
        )

    # ------------------------------------------------------------------
    # Sensitivity: changing the correlation matrix changes the values
    # ------------------------------------------------------------------

    def test_independent_system_has_zero_tc_pointwise(self):
        """When C = I (independent variables), tc_loc = 0 at every point.

        nll_joint = sum nll_i when variables are independent, so tc_loc = 0
        exactly.  This provides an additional analytic anchor.
        """
        K = self.C.shape[0]
        C_identity = np.eye(K)

        Xg_t   = torch.tensor(self.X_PTS, dtype=torch.float64)
        C_id_t = torch.tensor(C_identity, dtype=torch.float64)
        nplets  = torch.arange(K).unsqueeze(0)

        res_id = local_nplets_measures(
            Xg_t, nplets=nplets, covmats=C_id_t,
            device='cpu', dtype=torch.float64,
        )
        tc_id = res_id[0, 0, :, 0].numpy()

        np.testing.assert_allclose(
            tc_id, 0.0, atol=1e-8,
            err_msg="TC must be zero pointwise when C = I (independent variables)",
        )

    def test_different_covmat_gives_different_values(self):
        """Changing C must change the computed local measures.

        Verifies that the covmats argument is actually used in the computation.
        """
        K = self.C.shape[0]
        C_identity = np.eye(K)

        Xg_t   = torch.tensor(self.X_PTS, dtype=torch.float64)
        C_id_t = torch.tensor(C_identity, dtype=torch.float64)
        nplets  = torch.arange(K).unsqueeze(0)

        res_id = local_nplets_measures(
            Xg_t, nplets=nplets, covmats=C_id_t,
            device='cpu', dtype=torch.float64,
        )
        computed_id = res_id[0, 0, :, :].numpy()

        max_diff = np.abs(self.computed[:, 0] - computed_id[:, 0]).max()
        self.assertGreater(
            max_diff, 1e-3,
            "Different correlation matrices must produce different local TC values",
        )


class TestLocalMeasuresSimulatedData(unittest.TestCase):
    """Verify local_nplets_measures on data simulated from a known distribution.

    This mirrors the real-world use case: we have empirical data, we run the
    full pipeline (Gaussian copula normalisation + covariance estimation +
    local measures), and we check that the results converge to the analytic
    ground truth as sample size grows.

    The ground truth is the same analytic formula as above, evaluated at the
    *true* correlation matrix C.  With finite T there will be estimation error,
    so the tolerance is wider and scales with 1/sqrt(T).
    """

    # Same correlation matrix as the pointwise test for easy cross-referencing
    C = np.array([
        [1.00, 0.60, 0.20],
        [0.60, 1.00, 0.30],
        [0.20, 0.30, 1.00],
    ], dtype=np.float64)

    T = 50_000   # large enough to keep finite-sample error well below ATOL
    ATOL = 0.01  # ~2–3 sigma of the Monte-Carlo estimation error at T=50k
    SEED = 42

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(cls.SEED)
        K = cls.C.shape[0]

        # Draw T samples from N(0, C)
        L = np.linalg.cholesky(cls.C)
        cls.X = rng.standard_normal((cls.T, K)) @ L.T  # (T, K)

        # Run the full pipeline: copula + covmat estimation + local measures
        cls.nplets = torch.arange(K).unsqueeze(0)   # [[0, 1, 2]]
        res = local_nplets_measures(
            cls.X,
            nplets=cls.nplets,
            device='cpu',
            dtype=torch.float64,
        )  # (1, 1, T, 4)
        cls.local_vals = res[0, 0, :, :].numpy()    # (T, 4)

        # Analytic ground truth at each simulated point using the TRUE C
        # (not the estimated one) — this is what the estimator should converge to
        Cinv   = np.linalg.inv(cls.C)
        logdet = np.log(np.linalg.det(cls.C))
        cls.gt_per_point = np.stack([
            _analytic_local_measures(x, cls.C) for x in cls.X
        ])  # (T, 4)

    # ------------------------------------------------------------------
    # Time-averaged local measures converge to the analytic mean
    # ------------------------------------------------------------------

    def test_mean_tc_converges_to_analytic(self):
        """E[tc_loc] = TC must hold up to finite-sample error."""
        tc_analytic_mean = self.gt_per_point[:, 0].mean()
        tc_estimated_mean = self.local_vals[:, 0].mean()
        self.assertAlmostEqual(
            tc_estimated_mean, tc_analytic_mean, delta=self.ATOL,
            msg=f"Mean TC: estimated={tc_estimated_mean:.5f}, "
                f"analytic={tc_analytic_mean:.5f}",
        )

    def test_mean_dtc_converges_to_analytic(self):
        """E[dtc_loc] = DTC must hold up to finite-sample error."""
        dtc_analytic_mean = self.gt_per_point[:, 1].mean()
        dtc_estimated_mean = self.local_vals[:, 1].mean()
        self.assertAlmostEqual(
            dtc_estimated_mean, dtc_analytic_mean, delta=self.ATOL,
            msg=f"Mean DTC: estimated={dtc_estimated_mean:.5f}, "
                f"analytic={dtc_analytic_mean:.5f}",
        )

    def test_mean_o_converges_to_analytic(self):
        """E[o_loc] = O must hold up to finite-sample error."""
        o_analytic_mean = self.gt_per_point[:, 2].mean()
        o_estimated_mean = self.local_vals[:, 2].mean()
        self.assertAlmostEqual(
            o_estimated_mean, o_analytic_mean, delta=self.ATOL,
            msg=f"Mean O: estimated={o_estimated_mean:.5f}, "
                f"analytic={o_analytic_mean:.5f}",
        )

    def test_mean_s_converges_to_analytic(self):
        """E[s_loc] = S must hold up to finite-sample error."""
        s_analytic_mean = self.gt_per_point[:, 3].mean()
        s_estimated_mean = self.local_vals[:, 3].mean()
        self.assertAlmostEqual(
            s_estimated_mean, s_analytic_mean, delta=self.ATOL,
            msg=f"Mean S: estimated={s_estimated_mean:.5f}, "
                f"analytic={s_analytic_mean:.5f}",
        )

    # ------------------------------------------------------------------
    # Distribution of local values matches the analytic distribution
    # ------------------------------------------------------------------

    def test_median_o_loc_close_to_analytic_median(self):
        """The median of simulated o_loc should match the analytic median."""
        median_analytic = np.median(self.gt_per_point[:, 2])
        median_estimated = np.median(self.local_vals[:, 2])
        self.assertAlmostEqual(
            median_estimated, median_analytic, delta=self.ATOL,
            msg=f"Median O: estimated={median_estimated:.5f}, "
                f"analytic={median_analytic:.5f}",
        )

    # ------------------------------------------------------------------
    # Output properties under the full pipeline
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """local_nplets_measures must return (1, 1, T, 4) for a single n-plet."""
        K = self.C.shape[0]
        res = local_nplets_measures(
            self.X, nplets=self.nplets, device='cpu', dtype=torch.float64,
        )
        self.assertEqual(res.shape, (1, 1, self.T, 4))

    def test_all_values_finite(self):
        """No NaN or Inf in any of the T local-measure evaluations."""
        self.assertTrue(
            np.isfinite(self.local_vals).all(),
            "local_nplets_measures produced non-finite values on simulated data",
        )

    def test_algebraic_consistency_pointwise(self):
        """O = TC - DTC and S = TC + DTC must hold at every simulated point."""
        tc  = self.local_vals[:, 0]
        dtc = self.local_vals[:, 1]
        o   = self.local_vals[:, 2]
        s   = self.local_vals[:, 3]
        np.testing.assert_allclose(o, tc - dtc, atol=1e-12,
                                   err_msg="O != TC - DTC on simulated data")
        np.testing.assert_allclose(s, tc + dtc, atol=1e-12,
                                   err_msg="S != TC + DTC on simulated data")


if __name__ == '__main__':
    unittest.main()
