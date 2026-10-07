"""Tests for p3former/utils/gmm_fit.py (OCCUQ Gaussians from running sums).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_fit_occuq_gmm.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p3former.utils.gmm_fit import (JITTERS, GaussianStats,  # noqa: E402
                                    cholesky_with_jitter, finalize,
                                    voxel_majority_labels)


def _data(n=600, dim=4, seed=0):
    """Features of 3 classes plus ignore (label 3)."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 4, (n, ), generator=g)
    feats = torch.randn(n, dim, generator=g) * 2 + labels[:, None].float()
    return feats, labels


def test_running_sums_match_torch_cov():
    feats, labels = _data()
    stats = GaussianStats(num_classes=3, dim=4)
    for chunk in torch.split(torch.randperm(len(labels)), 97):
        stats.update(feats[chunk], labels[chunk], ignore_index=3)
    gmm = finalize(stats, min_count=10)
    eye = torch.eye(4, dtype=torch.float64)
    for c in range(3):
        zc = feats[labels == c].double()
        assert torch.allclose(gmm['means'][c], zc.mean(0), atol=1e-10)
        assert torch.allclose(gmm['covs'][c], torch.cov(zc.t()), atol=1e-10)
        assert int(gmm['counts'][c]) == len(zc)
        prec = gmm['prec_chol'][c]  # P = L^-T whitens the class
        assert torch.allclose(prec.t() @ gmm['covs'][c] @ prec, eye,
                              atol=1e-8)
        assert torch.isclose(gmm['log_det_prec'][c],
                             -0.5 * torch.logdet(gmm['covs'][c]), atol=1e-10)
    counts = torch.tensor([(labels == c).sum() for c in range(3)],
                          dtype=torch.float64)
    assert torch.allclose(gmm['log_prior'], torch.log(counts / counts.sum()))
    assert (gmm['jitter'] == 0).all()
    print('PASS test_running_sums_match_torch_cov')


def test_ignore_label_never_reaches_the_sums():
    feats, labels = _data()
    clean = GaussianStats(3, 4)
    clean.update(feats[labels != 3], labels[labels != 3], ignore_index=3)
    poisoned = feats.clone()
    poisoned[labels == 3] = 1e6  # ignored voxels, e.g. Stop / Others
    mixed = GaussianStats(3, 4)
    mixed.update(poisoned, labels, ignore_index=3)
    for name in ('count', 'sum', 'outer'):
        assert torch.equal(getattr(clean, name), getattr(mixed, name)), name
    print('PASS test_ignore_label_never_reaches_the_sums')


def test_out_of_range_label_is_rejected():
    stats = GaussianStats(3, 4)
    try:
        stats.update(torch.zeros(2, 4), torch.tensor([0, 7]), ignore_index=3)
    except ValueError as err:
        assert 'labels must be in' in str(err)
    else:
        raise AssertionError('label 7 was accepted')
    print('PASS test_out_of_range_label_is_rejected')


def test_majority_vote():
    p2v = torch.tensor([0, 0, 0, 1, 1, 2])
    labels = torch.tensor([5, 5, 2, 3, 1, 4])
    voxel = voxel_majority_labels(labels, p2v, num_voxels=3, num_labels=25)
    # voxel 0: 5 wins 2-1; voxel 1: 3 and 1 tie -> lowest (1); voxel 2: 4
    assert voxel.tolist() == [5, 1, 4]
    print('PASS test_majority_vote')


def test_majority_vote_rejects_empty_voxels_and_length_mismatch():
    labels = torch.tensor([1, 2])
    for p2v, num_voxels in ((torch.tensor([0, 2]), 3),  # voxel 1 is empty
                            (torch.tensor([0]), 1)):  # 2 labels, 1 entry
        try:
            voxel_majority_labels(labels, p2v, num_voxels, 25)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted p2v={p2v.tolist()}, '
                                 f'V={num_voxels}')
    print('PASS test_majority_vote_rejects_empty_voxels_and_length_mismatch')


def test_jitter_zero_for_positive_definite():
    a = torch.randn(5, 5, dtype=torch.float64)
    cov = a @ a.t() + 0.1 * torch.eye(5, dtype=torch.float64)
    _, jitter = cholesky_with_jitter(cov)
    assert jitter == 0.0
    print('PASS test_jitter_zero_for_positive_definite')


def test_jitter_is_smallest_working_value():
    cov = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0, -1e-3],
                                  dtype=torch.float64))
    chol, jitter = cholesky_with_jitter(cov)
    # 1e-3 only lifts the last eigenvalue to 0, which is not positive
    assert jitter == 1e-2, jitter
    assert JITTERS.index(jitter) == JITTERS.index(1e-3) + 1
    eye = torch.eye(5, dtype=torch.float64)
    assert torch.allclose(chol @ chol.t(), cov + jitter * eye, atol=1e-12)
    print('PASS test_jitter_is_smallest_working_value')


def test_min_count_error_names_the_class():
    feats, labels = _data(n=60)
    stats = GaussianStats(3, 4)
    stats.update(feats, labels, ignore_index=3)
    try:
        finalize(stats, min_count=1000)
    except ValueError as err:
        assert 'fewer than 1000' in str(err) and '(0, ' in str(err)
    else:
        raise AssertionError('no error for too few voxels')
    print('PASS test_min_count_error_names_the_class')


if __name__ == '__main__':
    test_running_sums_match_torch_cov()
    test_ignore_label_never_reaches_the_sums()
    test_out_of_range_label_is_rejected()
    test_majority_vote()
    test_majority_vote_rejects_empty_voxels_and_length_mismatch()
    test_jitter_zero_for_positive_definite()
    test_jitter_is_smallest_working_value()
    test_min_count_error_names_the_class()
    print('ALL TESTS PASSED')
