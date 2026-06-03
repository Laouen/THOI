# tests/test_gaussian_copula_covmat.py

import unittest
import torch

from thoi.commons import gaussian_copula_covmat


# 4×4 correlation matrix used as ground truth throughout.
# Varied off-diagonals; symmetric, PSD, diagonal = 1.
_TRUE_CORR = torch.tensor([
    [1.00, 0.80, 0.30, 0.10],
    [0.80, 1.00, 0.20, 0.05],
    [0.30, 0.20, 1.00, 0.60],
    [0.10, 0.05, 0.60, 1.00],
], dtype=torch.float64)

_T = 100_000   # large enough for correlation recovery; runs in < 0.1 s
_SEED = 42


def _make_gaussian_data(true_corr, T, seed):
    """Sample X ~ N(0, true_corr) via Cholesky, returned as (1, T, N)."""
    torch.manual_seed(seed)
    L = torch.linalg.cholesky(true_corr)
    X = torch.randn(T, true_corr.shape[0], dtype=true_corr.dtype) @ L.T
    return X.unsqueeze(0)  # (1, T, N)


class TestGaussianCopulaCovariance(unittest.TestCase):
    """Tests for gaussian_copula_covmat using data generated from a known correlation matrix.

    Data is drawn from N(0, _TRUE_CORR).  Because the Gaussian copula transform
    is rank-based and marginal, applying it to data that is already jointly Gaussian
    should recover the generating correlation matrix up to finite-sample error.
    """

    @classmethod
    def setUpClass(cls):
        cls.X = _make_gaussian_data(_TRUE_CORR, _T, _SEED)   # (1, T, 4)
        _, cls.covmat = gaussian_copula_covmat(cls.X)          # (1, 4, 4)

    # ------------------------------------------------------------------
    # Shape and structural properties
    # ------------------------------------------------------------------

    def test_output_shape(self):
        self.assertEqual(self.covmat.shape, (1, 4, 4))

    def test_return_xg_shape(self):
        Xg, covmats = gaussian_copula_covmat(self.X, return_xg=True)
        self.assertEqual(Xg.shape, self.X.shape)
        self.assertEqual(covmats.shape, (1, 4, 4))

    def test_symmetric(self):
        torch.testing.assert_close(self.covmat[0], self.covmat[0].T, atol=0, rtol=0)

    def test_diagonal_is_one(self):
        """Copula transform produces unit-variance marginals; diagonal → 1 as T → ∞."""
        diag = torch.diagonal(self.covmat[0])
        torch.testing.assert_close(diag, torch.ones_like(diag), atol=1e-3, rtol=0)

    # ------------------------------------------------------------------
    # Ground-truth recovery
    # ------------------------------------------------------------------

    def test_recovers_known_correlation_matrix(self):
        """Estimated correlations must match the generating matrix within sampling error.

        With T=100_000 the theoretical standard error for any of the correlations
        is < 0.004, so atol=0.01 is a ~2× safety margin.
        """
        torch.testing.assert_close(
            self.covmat[0],
            _TRUE_CORR.to(self.covmat.dtype),
            atol=0.01,
            rtol=0,
        )

    # ------------------------------------------------------------------
    # API / parameter invariants
    # ------------------------------------------------------------------

    def test_batch_size_D_does_not_change_result(self):
        """batch_size_D must not affect the computed covariance."""
        D = 4
        X_multi = self.X.expand(D, -1, -1).contiguous()
        _, cov_full   = gaussian_copula_covmat(X_multi)
        _, cov_batch1 = gaussian_copula_covmat(X_multi, batch_size_D=1)
        torch.testing.assert_close(cov_full, cov_batch1, atol=1e-12, rtol=0)

    def test_multiple_identical_datasets_give_identical_covmats(self):
        """D copies of the same data must produce identical covariance matrices."""
        D = 3
        X_multi = self.X.expand(D, -1, -1).contiguous()
        _, covmats = gaussian_copula_covmat(X_multi)
        for d in range(1, D):
            torch.testing.assert_close(covmats[0], covmats[d], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
