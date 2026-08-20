"""Tests for p3former/utils/ood_scores.py (point-level OOD baseline scores).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_scores.py
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p3former.utils.ood_scores import (OOD_SCORE_KEYS, compute_ood_scores,
                                       point_ood_scores)


def test_score_keys_and_shapes():
    logits = torch.randn(7, 19)
    scores = compute_ood_scores(logits)
    assert tuple(scores.keys()) == OOD_SCORE_KEYS == ('msp', 'maxlogit',
                                                      'odin', 'energy',
                                                      'entropy')
    for key in OOD_SCORE_KEYS:
        assert scores[key].shape == (7, ), key
    print('PASS test_score_keys_and_shapes')


def test_known_values_two_classes():
    # One voxel, two classes, logits [2, 0]. Higher score = more OOD.
    logits = torch.tensor([[2.0, 0.0]])
    scores = compute_ood_scores(logits, odin_temperature=1000.0,
                                energy_temperature=1.0)
    p_max = math.exp(2.0) / (math.exp(2.0) + 1.0)
    assert torch.allclose(scores['msp'], torch.tensor([-p_max]), atol=1e-6)
    assert torch.allclose(scores['maxlogit'], torch.tensor([-2.0]))
    p_max_T = math.exp(2.0 / 1000) / (math.exp(2.0 / 1000) + 1.0)
    assert torch.allclose(scores['odin'], torch.tensor([-p_max_T]), atol=1e-7)
    energy = -math.log(math.exp(2.0) + 1.0)
    assert torch.allclose(scores['energy'], torch.tensor([energy]), atol=1e-6)
    entropy = -(p_max * math.log(p_max) + (1 - p_max) * math.log(1 - p_max))
    assert torch.allclose(scores['entropy'], torch.tensor([entropy]),
                          atol=1e-6)
    print('PASS test_known_values_two_classes')


def test_uniform_logits_are_most_ood():
    # Voxel 0: confident one-hot-ish logits. Voxel 1: flat logits.
    # Every method must rate the flat voxel as MORE OOD (higher score).
    confident = torch.full((1, 19), -5.0)
    confident[0, 3] = 10.0
    flat = torch.zeros(1, 19)
    scores = compute_ood_scores(torch.cat([confident, flat]))
    for key in OOD_SCORE_KEYS:
        assert scores[key][1] > scores[key][0], key
    # Exact values for the flat voxel: softmax = 1/19 each.
    assert torch.allclose(scores['msp'][1], torch.tensor(-1.0 / 19), atol=1e-6)
    assert torch.allclose(scores['maxlogit'][1], torch.tensor(0.0))
    assert torch.allclose(scores['energy'][1],
                          torch.tensor(-math.log(19.0)), atol=1e-6)
    # uniform distribution has the maximum entropy log(C)
    assert torch.allclose(scores['entropy'][1],
                          torch.tensor(math.log(19.0)), atol=1e-6)
    assert scores['entropy'][0] < 1e-3  # near one-hot -> near zero entropy
    print('PASS test_uniform_logits_are_most_ood')


def test_entropy_matches_reference_formula():
    # Reference: trash/Done/eval_ood_from_logits.py (softmax, clip 1e-12).
    rng = np.random.RandomState(0)
    z = rng.randn(50, 19) * 3.0
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    pc = np.clip(p, 1e-12, None)
    expected = -(pc * np.log(pc)).sum(1)
    scores = compute_ood_scores(torch.from_numpy(z).float())
    assert np.allclose(scores['entropy'].numpy(), expected, atol=1e-5)
    print('PASS test_entropy_matches_reference_formula')


def test_energy_temperature():
    logits = torch.randn(5, 19)
    scores = compute_ood_scores(logits, energy_temperature=2.0)
    expected = -2.0 * torch.logsumexp(logits / 2.0, dim=1)
    assert torch.allclose(scores['energy'], expected, atol=1e-6)
    print('PASS test_energy_temperature')


def test_odin_is_temperature_scaled_msp():
    logits = torch.randn(6, 19)
    scores = compute_ood_scores(logits, odin_temperature=1000.0)
    expected = -torch.softmax(logits / 1000.0, dim=1).max(dim=1).values
    assert torch.allclose(scores['odin'], expected, atol=1e-8)
    print('PASS test_odin_is_temperature_scaled_msp')


def test_numerical_stability_huge_logits():
    logits = torch.tensor([[1e4, -1e4, 0.0] + [0.0] * 16])
    scores = compute_ood_scores(logits)
    for key in OOD_SCORE_KEYS:
        assert torch.isfinite(scores[key]).all(), key
    print('PASS test_numerical_stability_huge_logits')


def test_point_projection():
    voxel_logits = torch.tensor([
        [10.0, 0.0],   # voxel 0: confident
        [0.0, 0.0],    # voxel 1: flat
        [0.0, 5.0],    # voxel 2: confident
    ])
    point2voxel_map = torch.tensor([0, 0, 2, 1], dtype=torch.int64)
    pts = point_ood_scores(voxel_logits, point2voxel_map)
    vox = compute_ood_scores(voxel_logits)
    for key in OOD_SCORE_KEYS:
        assert isinstance(pts[key], np.ndarray)
        assert pts[key].dtype == np.float32
        assert pts[key].shape == (4, )
        expected = vox[key][torch.tensor([0, 0, 2, 1])].numpy()
        assert np.allclose(pts[key], expected, atol=1e-6), key
    # the flat voxel's point is the most OOD point
    for key in OOD_SCORE_KEYS:
        assert pts[key].argmax() == 3, key
    # float map (as produced by dynamic_scatter_3d) must also work
    pts2 = point_ood_scores(voxel_logits, point2voxel_map.float())
    assert np.allclose(pts2['msp'], pts['msp'])
    print('PASS test_point_projection')


if __name__ == '__main__':
    test_score_keys_and_shapes()
    test_known_values_two_classes()
    test_uniform_logits_are_most_ood()
    test_entropy_matches_reference_formula()
    test_energy_temperature()
    test_odin_is_temperature_scaled_msp()
    test_numerical_stability_huge_logits()
    test_point_projection()
    print('ALL TESTS PASSED')
