import torch

from thoi.heuristics import differentiable_hoi


def _block_covariance(rho):
    cov = torch.eye(6, dtype=torch.float64)
    cov[:3, :3] = (
        (1 - rho) * torch.eye(3, dtype=torch.float64)
        + rho * torch.ones(3, 3, dtype=torch.float64)
    )
    return cov


def test_differentiable_hoi_metrics_and_orders():
    expected = {
        'tc': 1.13168219,
        'dtc': 0.73088751,
        'o': 0.40079468,
        's': 1.86256970,
    }

    for metric, expected_score in expected.items():
        torch.manual_seed(0)
        nplets, scores = differentiable_hoi(
            _block_covariance(0.8),
            initial_order=3,
            order=4,
            covmat_precomputed=True,
            metric=metric,
            largest=True,
        )

        assert len(nplets) == 2
        assert nplets[0].shape == (4, 3)
        assert nplets[1].shape == (4, 4)
        assert set(nplets[0][0].tolist()) == {0, 1, 2}
        expected_tensor = torch.tensor(expected_score, dtype=scores.dtype)
        assert torch.isclose(scores[0, 0], expected_tensor, atol=1e-4)


def test_differentiable_hoi_multiple_datasets():
    covmats = torch.stack([
        _block_covariance(-0.45),
        _block_covariance(-0.40),
    ])

    torch.manual_seed(1)
    nplets, scores = differentiable_hoi(
        covmats,
        initial_order=3,
        order=3,
        covmat_precomputed=True,
        metric='o',
        largest=False,
    )

    assert set(nplets[0][0].tolist()) == {0, 1, 2}
    expected = torch.tensor(-0.32351769, dtype=scores.dtype)
    assert torch.isclose(scores[0, 0], expected, atol=1e-4)
