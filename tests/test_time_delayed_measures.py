import unittest

import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov

from thoi.measures.time_delayed import (
    ais,
    ais_subset,
    generate_stacked_lagged_batches,
    local_ais,
    local_ais_subset,
    precompute_temporal_embedding,
    tdmi,
)


def _spectral_radius(matrix):
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def _toy_var1_model():
    """Three-variable VAR(1) with self memory, cross-transfer, and correlated noise."""
    transition_template = np.array(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    A = 0.55 * transition_template / _spectral_radius(transition_template)
    Q = np.array(
        [
            [1.00, 0.25, -0.15],
            [0.25, 1.00, 0.20],
            [-0.15, 0.20, 1.00],
        ],
        dtype=float,
    )
    return A, Q


def _simulate_var1(A, Q, T=160, burn=200, seed=0):
    """Simulate empirical data from X(t) = A X(t-1) + eps(t), eps ~ N(0, Q)."""
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    noise = rng.multivariate_normal(np.zeros(N), Q, size=T + burn)
    X = np.zeros((T + burn, N), dtype=float)

    for t in range(1, T + burn):
        X[t] = A @ X[t - 1] + noise[t]

    return torch.as_tensor(X[burn:], dtype=torch.float64)


def _var1_data(T=160, seed=0):
    A, Q = _toy_var1_model()
    return _simulate_var1(A, Q, T=T, seed=seed)


def _gaussian_entropy(cov, idx):
    idx = np.atleast_1d(idx).astype(int)
    sub_cov = cov[np.ix_(idx, idx)]
    sign, logdet = np.linalg.slogdet(sub_cov)
    if sign <= 0:
        raise ValueError('Sub-covariance is not positive definite')
    return 0.5 * logdet


def _gaussian_mi(cov, a, b):
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    return (
        _gaussian_entropy(cov, a)
        + _gaussian_entropy(cov, b)
        - _gaussian_entropy(cov, np.concatenate([a, b]))
    )


def _stationary_covariance(A, Q):
    return solve_discrete_lyapunov(A, Q)


def _var1_present_past_covariance(A, Q):
    sigma = _stationary_covariance(A, Q)
    cross = A @ sigma
    return np.block([[sigma, cross], [cross.T, sigma]])


def _correlation_matrix(cov):
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)


def _analytic_lag1_measures(A, Q):
    joint_cov = _var1_present_past_covariance(A, Q)
    N = A.shape[0]
    present = np.arange(N)
    past = np.arange(N, 2 * N)

    return {
        'ais_full': _gaussian_mi(joint_cov, present, past),
        'ais_parts': np.array([
            _gaussian_mi(joint_cov, [i], [N + i])
            for i in range(N)
        ]),
        'tdmi': np.array([
            [
                _gaussian_mi(joint_cov, [N + i], [j])
                for j in range(N)
            ]
            for i in range(N)
        ]),
        'xcorr': _correlation_matrix(joint_cov)[N:, :N],
    }


def _analytic_local_ais_from_embedding(X_lagged, covmats):
    """
    Compute local AIS as log p(X_0 | X_1) - log p(X_0) from Gaussian covariances.

    This helper follows THOI's lagged column convention for ``lags=[0, 1]``:
    the first block is ``X(t)`` and the second block is ``X(t-1)``.
    """
    X_lagged = torch.as_tensor(X_lagged, dtype=torch.float64)
    covmats = torch.as_tensor(covmats, dtype=torch.float64)
    if X_lagged.ndim == 2:
        X_lagged = X_lagged.unsqueeze(0)
    if covmats.ndim == 2:
        covmats = covmats.unsqueeze(0)

    B, _, D = X_lagged.shape
    N = D // 2
    x0 = X_lagged[:, :, :N]
    x1 = X_lagged[:, :, N:]

    sigma_00 = covmats[:, :N, :N]
    sigma_11 = covmats[:, N:, N:]
    sigma_01 = covmats[:, :N, N:]
    sigma_10 = covmats[:, N:, :N]

    chol_00 = torch.linalg.cholesky(sigma_00)
    chol_11 = torch.linalg.cholesky(sigma_11)
    sigma_11_inv_10 = torch.cholesky_solve(sigma_10, chol_11)
    sigma_cond = sigma_00 - torch.matmul(sigma_01, sigma_11_inv_10)
    chol_cond = torch.linalg.cholesky(sigma_cond)

    x1_inv = torch.cholesky_solve(x1.transpose(1, 2), chol_11)
    cond_mean = torch.matmul(sigma_01, x1_inv).transpose(1, 2)

    def _logpdf(x, mean, chol):
        diff = x - mean
        solved = torch.cholesky_solve(diff.transpose(1, 2), chol)
        quad = (diff.transpose(1, 2) * solved).sum(1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        return -0.5 * (N * np.log(2 * np.pi) + logdet[:, None] + quad)

    return (
        _logpdf(x0, cond_mean, chol_cond)
        - _logpdf(x0, torch.zeros(B, x0.shape[1], N, dtype=torch.float64), chol_00)
    ).unsqueeze(-1)


def _analytic_raw_var1_local_ais(X, A, Q):
    """Analytic local AIS for the raw Gaussian VAR(1), using the true covariance."""
    sigma = _stationary_covariance(A, Q)
    # THOI's [0, 1] embedding is [X(t), X(t-1)], so the cross block is A Sigma.
    cross = A @ sigma
    joint_cov = np.block([[sigma, cross], [cross.T, sigma]])
    lagged_raw = torch.cat([X[1:], X[:-1]], dim=1).unsqueeze(0)
    return _analytic_local_ais_from_embedding(
        lagged_raw,
        torch.as_tensor(joint_cov, dtype=torch.float64).unsqueeze(0),
    )


class TestTemporalEmbedding(unittest.TestCase):
    def setUp(self):
        self.X = _var1_data()
        self.lags = [0, 1, 3]

    def test_lagged_embedding_shape(self):
        data_list = [self.X, self.X + 0.1]
        lagged = generate_stacked_lagged_batches(data_list, self.lags)
        self.assertEqual(lagged.shape, (2, self.X.shape[0] - max(self.lags), len(self.lags) * self.X.shape[1]))

    def test_precompute_temporal_embedding_shape(self):
        Xg, covmats, T_eff, N, lags = precompute_temporal_embedding(
            self.X,
            self.lags,
            dtype=torch.float64,
        )
        self.assertEqual(Xg.shape, (1, self.X.shape[0] - max(self.lags), len(self.lags) * self.X.shape[1]))
        self.assertEqual(covmats.shape, (1, len(self.lags) * self.X.shape[1], len(self.lags) * self.X.shape[1]))
        self.assertEqual(T_eff, self.X.shape[0] - max(self.lags))
        self.assertEqual(N, self.X.shape[1])
        torch.testing.assert_close(lags, torch.tensor(self.lags))


class TestAIS(unittest.TestCase):
    def setUp(self):
        self.X = _var1_data()
        self.lags = [0, 1, 2]

    def test_ais_shape_and_precomputed_match(self):
        raw = ais(self.X, self.lags, dtype=torch.float64, bias_correction=False)
        Xg, covmats, T_eff, _, _ = precompute_temporal_embedding(
            self.X,
            self.lags,
            dtype=torch.float64,
        )
        precomputed = ais(
            Xg,
            self.lags,
            covmats=covmats,
            T=T_eff,
            dtype=torch.float64,
            bias_correction=False,
            precomputed=True,
        )
        self.assertEqual(raw.shape, (1, len(self.lags) - 1))
        torch.testing.assert_close(raw, precomputed, atol=0, rtol=0)

    def test_bias_correction_changes_ais(self):
        no_bias = ais(self.X, self.lags, dtype=torch.float64, bias_correction=False)
        with_bias = ais(self.X, self.lags, dtype=torch.float64, bias_correction=True)
        self.assertGreater(torch.abs(no_bias - with_bias).max().item(), 1e-8)

    def test_list_and_batched_inputs(self):
        data_list = [self.X, self.X]
        batched = torch.stack(data_list)
        from_list = ais(data_list, self.lags, dtype=torch.float64, bias_correction=False)
        from_batched = ais(batched, self.lags, dtype=torch.float64, bias_correction=False)
        self.assertEqual(from_list.shape, (2, len(self.lags) - 1))
        torch.testing.assert_close(from_list, from_batched, atol=0, rtol=0)

    def test_ais_subset_matches_full_for_all_variables(self):
        full = ais(self.X, self.lags, dtype=torch.float64, bias_correction=False)
        subset = ais_subset(
            self.X,
            self.lags,
            idxs=[0, 1, 2],
            dtype=torch.float64,
            bias_correction=False,
        )
        torch.testing.assert_close(full, subset, atol=0, rtol=0)


class TestAnalyticVAR1Measures(unittest.TestCase):
    """Compare empirical THOI estimates against analytic Gaussian VAR(1) values."""

    @classmethod
    def setUpClass(cls):
        cls.A, cls.Q = _toy_var1_model()
        cls.X = _simulate_var1(cls.A, cls.Q, T=40_000, burn=1_000, seed=123)
        cls.lags = [0, 1]
        cls.truth = _analytic_lag1_measures(cls.A, cls.Q)

    def test_full_system_ais_matches_analytic_var1(self):
        value = ais(
            self.X,
            self.lags,
            dtype=torch.float64,
            bias_correction=False,
        )[0, 0].item()
        self.assertAlmostEqual(value, self.truth['ais_full'], delta=2e-3)

    def test_single_variable_ais_matches_analytic_var1(self):
        values = np.array([
            ais_subset(
                self.X,
                self.lags,
                idxs=[i],
                dtype=torch.float64,
                bias_correction=False,
            )[0, 0].item()
            for i in range(3)
        ])
        np.testing.assert_allclose(values, self.truth['ais_parts'], atol=2e-3, rtol=0)

    def test_tdmi_and_xcorr_match_analytic_var1(self):
        tdmi_values, xcorr_values = tdmi(
            self.X,
            self.lags,
            dtype=torch.float64,
            bias_correction=False,
            return_full=False,
        )
        np.testing.assert_allclose(
            tdmi_values[0, 1].numpy(),
            self.truth['tdmi'],
            atol=2e-3,
            rtol=0,
        )
        np.testing.assert_allclose(
            xcorr_values[0, 1].numpy(),
            self.truth['xcorr'],
            atol=1e-2,
            rtol=0,
        )

    def test_local_ais_matches_gaussian_embedding_ground_truth(self):
        Xg, covmats, _, _, _ = precompute_temporal_embedding(
            self.X,
            self.lags,
            dtype=torch.float64,
        )
        local = local_ais(
            Xg,
            self.lags,
            covmats=covmats,
            dtype=torch.float64,
            precomputed=True,
        )
        truth = _analytic_local_ais_from_embedding(Xg, covmats)
        torch.testing.assert_close(local, truth, atol=1e-8, rtol=0)

    def test_local_ais_raw_var1_analytic_error_is_small(self):
        local = local_ais(self.X, self.lags, dtype=torch.float64)
        raw_truth = _analytic_raw_var1_local_ais(self.X, self.A, self.Q)
        abs_error = torch.abs(local - raw_truth)

        mean_error = abs_error.mean().item()
        p95_error = torch.quantile(abs_error.flatten(), 0.95).item()
        average_error = abs(local.mean().item() - raw_truth.mean().item())

        self.assertLess(mean_error, 2e-2, f'mean local AIS error={mean_error:.6f}')
        self.assertLess(p95_error, 6e-2, f'p95 local AIS error={p95_error:.6f}')
        self.assertLess(average_error, 2e-3, f'average local AIS error={average_error:.6f}')


class TestLocalAIS(unittest.TestCase):
    def setUp(self):
        self.X = _var1_data(T=140)
        self.lags = [0, 1, 2]

    def test_local_ais_shape_and_precomputed_match(self):
        raw = local_ais(self.X, self.lags, dtype=torch.float64)
        Xg, covmats, _, _, _ = precompute_temporal_embedding(
            self.X,
            self.lags,
            dtype=torch.float64,
        )
        precomputed = local_ais(
            Xg,
            self.lags,
            covmats=covmats,
            dtype=torch.float64,
            precomputed=True,
        )
        self.assertEqual(raw.shape, (1, self.X.shape[0] - max(self.lags), len(self.lags) - 1))
        torch.testing.assert_close(raw, precomputed, atol=0, rtol=0)

    def test_local_average_matches_ais_without_bias(self):
        local = local_ais(self.X, self.lags, dtype=torch.float64).mean(dim=1)
        global_ais = ais(self.X, self.lags, dtype=torch.float64, bias_correction=False)
        torch.testing.assert_close(local, global_ais, atol=1e-10, rtol=1e-8)

    def test_local_subset_matches_full_for_all_variables(self):
        full = local_ais(self.X, self.lags, dtype=torch.float64)
        subset = local_ais_subset(
            self.X,
            self.lags,
            idxs=[0, 1, 2],
            dtype=torch.float64,
        )
        torch.testing.assert_close(full, subset, atol=0, rtol=0)


class TestTDMI(unittest.TestCase):
    def setUp(self):
        self.X = _var1_data()
        self.lags = [0, 1, 2, 4]

    def test_tdmi_shapes(self):
        tdmi_values, xcorr_values, full_tdmi = tdmi(
            self.X,
            self.lags,
            dtype=torch.float64,
            bias_correction=False,
            return_full=True,
        )
        N = self.X.shape[1]
        self.assertEqual(tdmi_values.shape, (1, len(self.lags), N, N))
        self.assertEqual(xcorr_values.shape, (1, len(self.lags), N, N))
        self.assertEqual(full_tdmi.shape, (1, N, N, 2 * len(self.lags) - 1))
        self.assertTrue(torch.isnan(torch.diagonal(full_tdmi[0], dim1=0, dim2=1)).all())

    def test_tdmi_precomputed_matches_raw(self):
        raw = tdmi(
            self.X,
            self.lags,
            dtype=torch.float64,
            bias_correction=False,
            return_full=False,
        )
        Xg, covmats, T_eff, _, _ = precompute_temporal_embedding(
            self.X,
            self.lags,
            dtype=torch.float64,
        )
        precomputed = tdmi(
            Xg,
            self.lags,
            covmats=covmats,
            T=T_eff,
            dtype=torch.float64,
            bias_correction=False,
            return_full=False,
            precomputed=True,
        )
        for raw_tensor, precomputed_tensor in zip(raw, precomputed):
            torch.testing.assert_close(raw_tensor, precomputed_tensor, atol=0, rtol=0)

    def test_tdmi_bias_correction_changes_values(self):
        no_bias, _ = tdmi(
            self.X,
            self.lags,
            dtype=torch.float64,
            bias_correction=False,
            return_full=False,
        )
        with_bias, _ = tdmi(
            self.X,
            self.lags,
            dtype=torch.float64,
            bias_correction=True,
            return_full=False,
        )
        self.assertGreater(torch.abs(no_bias - with_bias).max().item(), 1e-8)


if __name__ == '__main__':
    unittest.main()
