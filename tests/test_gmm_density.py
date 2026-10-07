"""Tests for p3former/utils/gmm_density.py (OCCUQ density score).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_gmm_density.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.functional.ood_eval import binary_ood_metrics  # noqa: E402
from p3former.utils.gmm_density import (GMM_KEYS, density_score,  # noqa: E402
                                        fingerprint, load_gmm, log_density,
                                        no_tf32, point_density_scores)
from p3former.utils.gmm_fit import GaussianStats, finalize  # noqa: E402
# tests/ is on sys.path (the script's directory, or pytest's insertion);
# "from tests..." would hit the site-packages "tests" package instead
from test_ood_eval import _ref_metrics  # noqa: E402


def _fitted(dim=3, seed=0):
    """Two well-separated 3-d Gaussians fitted from samples (float64)."""
    g = torch.Generator().manual_seed(seed)
    stats = GaussianStats(2, dim)
    for c, shift in enumerate((0.0, 6.0)):
        z = torch.randn(500, dim, generator=g, dtype=torch.float64)
        stats.update(z * (1 + c) + shift, torch.full((500, ), c),
                     ignore_index=2)
    return finalize(stats, min_count=10)


def test_log_density_matches_multivariate_normal():
    gmm = _fitted()
    z = torch.randn(20, 3, dtype=torch.float64) * 4
    mvn = torch.distributions.MultivariateNormal(
        gmm['means'], covariance_matrix=gmm['covs'])
    expected = torch.logsumexp(
        mvn.log_prob(z[:, None, :]) + gmm['log_prior'], dim=1)
    assert torch.allclose(log_density(z, gmm, chunk=7), expected, atol=1e-9)
    print('PASS test_log_density_matches_multivariate_normal')


def test_far_points_score_higher():
    gmm = _fitted()
    near = gmm['means'][0][None]
    far = near + 50.0
    scores = density_score(torch.cat([near, far]), gmm)
    assert scores[1] > scores[0]
    print('PASS test_far_points_score_higher')


def test_no_tf32_restores_the_flag():
    for initial in (True, False):
        torch.backends.cuda.matmul.allow_tf32 = initial
        try:
            with no_tf32():
                assert torch.backends.cuda.matmul.allow_tf32 is False
                raise RuntimeError('boom')
        except RuntimeError:
            pass
        assert torch.backends.cuda.matmul.allow_tf32 is initial
    torch.backends.cuda.matmul.allow_tf32 = True  # torch 1.10's default
    print('PASS test_no_tf32_restores_the_flag')


def test_asinh_keeps_binned_metrics_exact_on_heavy_tails():
    rng = np.random.RandomState(0)
    s = np.concatenate([rng.normal(0, 1, 20000), rng.normal(1.5, 1, 1000),
                        np.full(5, 1e12)])  # five far-off ID points
    y = np.concatenate([np.zeros(20000, bool), np.ones(1000, bool),
                        np.zeros(5, bool)])
    exact = _ref_metrics(s, y)
    binned = binary_ood_metrics([np.arcsinh(s)], [y])
    for got, ref in zip((binned['auroc'], binned['ap'], binned['fpr95']),
                        exact):
        assert abs(got - ref) < 1e-3, (got, ref)
    raw = binary_ood_metrics([s], [y])  # why the score is compressed
    assert abs(raw['auroc'] - exact[0]) > 0.05
    print('PASS test_asinh_keeps_binned_metrics_exact_on_heavy_tails')


def test_fingerprint_and_load_gmm_guards():
    a, b = torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
    fp_a = fingerprint([('m', a)])
    assert fp_a == fingerprint([('m', a)])
    assert fp_a != fingerprint([('m', b)])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'gmm.pth')
        torch.save(dict(_fitted(), features='pe', fingerprint=fp_a), path)
        loaded = load_gmm(path, 'pe', fp_a, 'cpu')
        assert set(loaded) == set(GMM_KEYS)
        assert loaded['means'].dtype == torch.float32
        for features, fp in (('pe', fingerprint([('m', b)])),
                             ('head', fp_a)):
            try:
                load_gmm(path, features, fp, 'cpu')
            except ValueError:
                pass
            else:
                raise AssertionError(f'accepted features={features}')
        try:
            load_gmm(os.path.join(tmp, 'missing.pth'), 'pe', fp_a, 'cpu')
        except FileNotFoundError as err:
            assert 'fit_occuq_gmm.py' in str(err)
        else:
            raise AssertionError('a missing file was accepted')
    print('PASS test_fingerprint_and_load_gmm_guards')


def test_point_projection_and_non_finite():
    gmm = {k: v.float() for k, v in _fitted().items() if k in GMM_KEYS}
    voxels = torch.tensor([[0.0, 0.0, 0.0], [40.0, 40.0, 40.0]])
    p2v = torch.tensor([1, 0, 1])
    pts = point_density_scores(voxels, gmm, p2v)
    assert pts.dtype == np.float32 and pts.shape == (3, )
    assert np.allclose(pts, density_score(voxels, gmm)[p2v].numpy())
    bad = dict(gmm, means=gmm['means'] * float('nan'))
    try:
        density_score(voxels, bad)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('a NaN density was accepted')
    print('PASS test_point_projection_and_non_finite')


if __name__ == '__main__':
    test_log_density_matches_multivariate_normal()
    test_far_points_score_higher()
    test_no_tf32_restores_the_flag()
    test_asinh_keeps_binned_metrics_exact_on_heavy_tails()
    test_fingerprint_and_load_gmm_guards()
    test_point_projection_and_non_finite()
    print('ALL TESTS PASSED')
