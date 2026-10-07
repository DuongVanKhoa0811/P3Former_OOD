"""Tests for p3former/utils/gmm_fit.py (OCCUQ Gaussians from running sums).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_fit_occuq_gmm.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p3former.utils.gmm_density import GMM_KEYS, log_density  # noqa: E402
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
    # the pooled ID variance per dimension, over all classes together
    pooled = torch.var(feats[labels != 3].double(), dim=0)
    assert torch.allclose(gmm['pooled_var'], pooled, atol=1e-10)
    assert gmm['ridge'].dim() == 0 and gmm['ridge'].dtype == torch.float64
    assert gmm['ridge'] == 1e-6 * gmm['pooled_var'].mean()
    # one diagonal for every class: the ridge alone, as no jitter is needed
    assert (gmm['jitter'] == gmm['jitter'][0]).all()
    assert (gmm['jitter'] == gmm['ridge']).all()
    for c in range(3):
        zc = feats[labels == c].double()
        assert torch.allclose(gmm['means'][c], zc.mean(0), atol=1e-10)
        # covs stays the raw covariance
        assert torch.allclose(gmm['covs'][c], torch.cov(zc.t()), atol=1e-10)
        assert int(gmm['counts'][c]) == len(zc)
        regularised = gmm['covs'][c] + gmm['jitter'][c] * eye
        prec = gmm['prec_chol'][c]  # P = L^-T whitens the regularised class
        assert torch.allclose(prec.t() @ regularised @ prec, eye, atol=1e-8)
        assert torch.isclose(gmm['log_det_prec'][c],
                             -0.5 * torch.logdet(regularised), atol=1e-10)
    counts = torch.tensor([(labels == c).sum() for c in range(3)],
                          dtype=torch.float64)
    assert torch.allclose(gmm['log_prior'], torch.log(counts / counts.sum()))
    print('PASS test_running_sums_match_torch_cov')


def test_dead_dimension_adds_one_constant_to_every_class():
    """A feature that is 0 in every class, like pe_features dim 190 of the
    base checkpoint, makes every class covariance singular."""
    feats, labels = _data(dim=5)
    feats[:, 2] = 0.0  # the dead dimension
    stats = GaussianStats(3, 5)
    stats.update(feats, labels, ignore_index=3)
    gmm = finalize(stats, min_count=10)
    assert gmm['pooled_var'][2] == 0 and int((gmm['pooled_var'] > 0).sum()) == 4
    assert (gmm['jitter'] == gmm['jitter'][0]).all()
    assert gmm['jitter'][0] == gmm['ridge'] and gmm['ridge'] > 0
    for key in GMM_KEYS:  # what the density score casts to float32
        assert torch.isfinite(gmm[key].float()).all(), key
    # log N_c of a point with 0 in the dead dimension = log N_c of the other
    # four dimensions + one constant, the same for every class
    g = torch.Generator().manual_seed(1)
    z = torch.randn(50, 5, generator=g, dtype=torch.float64) * 3
    z[:, 2] = 0.0
    keep = [0, 1, 3, 4]
    const = 0.5 * 5 * math.log(2 * math.pi)
    offsets = []
    for c in range(3):
        y = (z - gmm['means'][c]) @ gmm['prec_chol'][c]
        log_n = gmm['log_det_prec'][c] - const - 0.5 * y.pow(2).sum(1)
        reduced = torch.distributions.MultivariateNormal(
            gmm['means'][c, keep],
            covariance_matrix=gmm['covs'][c][keep][:, keep] +
            gmm['jitter'][c] * torch.eye(4, dtype=torch.float64))
        offsets.append(log_n - reduced.log_prob(z[:, keep]))
    offsets = torch.stack(offsets)  # [classes, points]
    shared = -0.5 * torch.log(2 * math.pi * gmm['jitter'][0])
    assert torch.allclose(offsets, shared.expand_as(offsets), atol=1e-8)
    # the float32 log-density, NaN under a per-class jitter search, is finite
    gmm32 = {key: gmm[key].float() for key in GMM_KEYS}
    fast = log_density(z.float(), gmm32)
    assert torch.isfinite(fast).all()
    assert torch.allclose(fast.double(), log_density(z, gmm), atol=1e-3)
    print('PASS test_dead_dimension_adds_one_constant_to_every_class')


def test_constant_features_are_rejected():
    stats = GaussianStats(2, 3)
    stats.update(torch.ones(40, 3), torch.arange(40) % 2, ignore_index=2)
    try:
        finalize(stats, min_count=10)
    except ValueError as err:
        assert 'pooled' in str(err), str(err)
    else:
        raise AssertionError('constant features were accepted')
    print('PASS test_constant_features_are_rejected')


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


def test_batched_jitter_is_the_first_that_works_for_all():
    eye = torch.eye(3, dtype=torch.float64)
    covs = torch.stack([
        eye,
        torch.diag(torch.tensor([1.0, 1.0, -5e-5], dtype=torch.float64)),
        torch.diag(torch.tensor([1.0, 1.0, -5e-3], dtype=torch.float64)),
    ])
    for cov, alone in zip(covs, (0.0, 1e-4, 1e-2)):
        assert cholesky_with_jitter(cov)[1] == alone
    chol, jitter = cholesky_with_jitter(covs)
    assert jitter == 1e-2, jitter  # one value, the first that works for all
    assert chol.shape == (3, 3, 3)
    assert torch.allclose(chol @ chol.transpose(1, 2), covs + jitter * eye,
                          atol=1e-12)
    print('PASS test_batched_jitter_is_the_first_that_works_for_all')


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


def test_finalize_accepts_gpu_sums():
    if not torch.cuda.is_available():
        print('SKIP test_finalize_accepts_gpu_sums (no CUDA)')
        return
    feats, labels = _data()
    # Accumulate on GPU
    stats_gpu = GaussianStats(num_classes=3, dim=4, device='cuda')
    for chunk in torch.split(torch.randperm(len(labels)), 97):
        stats_gpu.update(feats[chunk].cuda(), labels[chunk], ignore_index=3)
    gmm_gpu = finalize(stats_gpu, min_count=10)
    # Accumulate on CPU
    stats_cpu = GaussianStats(num_classes=3, dim=4)
    for chunk in torch.split(torch.randperm(len(labels)), 97):
        stats_cpu.update(feats[chunk], labels[chunk], ignore_index=3)
    gmm_cpu = finalize(stats_cpu, min_count=10)
    # GPU-path result must be on CPU and match CPU-path
    for key in gmm_gpu.keys():
        assert gmm_gpu[key].device.type == 'cpu', \
            f'{key} is on {gmm_gpu[key].device}, expected cpu'
        assert torch.allclose(gmm_gpu[key], gmm_cpu[key], atol=1e-9), \
            f'{key} differs: GPU {gmm_gpu[key].shape} vs CPU {gmm_cpu[key].shape}'
    print('PASS test_finalize_accepts_gpu_sums')


if __name__ == '__main__':
    test_running_sums_match_torch_cov()
    test_dead_dimension_adds_one_constant_to_every_class()
    test_constant_features_are_rejected()
    test_ignore_label_never_reaches_the_sums()
    test_out_of_range_label_is_rejected()
    test_majority_vote()
    test_majority_vote_rejects_empty_voxels_and_length_mismatch()
    test_jitter_zero_for_positive_definite()
    test_jitter_is_smallest_working_value()
    test_batched_jitter_is_the_first_that_works_for_all()
    test_min_count_error_names_the_class()
    test_finalize_accepts_gpu_sums()
    print('ALL TESTS PASSED')
