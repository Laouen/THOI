import torch

from thoi.heuristics import differentiable_oinfo
from thoi.heuristics.greedy import _next_order_greedy


def _block_covariance(rho):
    cov = torch.eye(6)
    cov[:3, :3] = (1 - rho) * torch.eye(3) + rho * torch.ones(3, 3)
    return cov


def _greedy_triplet_score(covmats, largest):
    covmats = covmats.unsqueeze(0) if covmats.ndim == 2 else covmats
    candidate, score = _next_order_greedy(
        covmats, None, torch.tensor([[0, 1]]),
        metric='o', largest=largest,
    )
    assert candidate.item() == 2
    return score[0]


def test_differentiable_oinfo_single_dataset():
    cov = _block_covariance(0.8)
    greedy_score = _greedy_triplet_score(cov, largest=True)

    torch.manual_seed(0)
    nplets, scores = differentiable_oinfo(
        cov, initial_order=3, order=3, covmat_precomputed=True,
        repeat=4, largest=True,
    )

    assert set(nplets[0][0].tolist()) == {0, 1, 2}
    assert torch.isclose(scores[0, 0], greedy_score, atol=1e-4)


def test_differentiable_oinfo_multiple_datasets():
    covs = torch.stack([
        _block_covariance(-0.45),
        _block_covariance(-0.40),
    ])
    greedy_score = _greedy_triplet_score(covs, largest=False)

    torch.manual_seed(1)
    nplets, scores = differentiable_oinfo(
        covs, initial_order=3, order=3, covmat_precomputed=True,
        repeat=4, largest=False,
    )

    assert set(nplets[0][0].tolist()) == {0, 1, 2}
    assert torch.isclose(scores[0, 0], greedy_score, atol=1e-4)
