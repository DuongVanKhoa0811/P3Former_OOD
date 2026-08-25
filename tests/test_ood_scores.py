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

from p3former.utils.ood_scores import (ALL_SCORE_KEYS, GN_SCORE_KEYS,
                                       GROUP_SCORE_KEYS, OOD_SCORE_KEYS,
                                       compute_ood_scores, point_ood_scores)

# GroupPaper.pdf Table 2 hierarchy in SemanticKITTI train ids.
SK_GROUPS = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11], [12, 13],
             [14, 15, 16], [17, 18]]


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


def _reference_scores(logits, hier, C, odin_T=1000.0):
    """Verbatim port of trash/Done/eval_ood_from_logits.py::scores_from_logits
    (numpy, float64). Note its MSP/ODIN family uses ``1 - max``."""

    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / np.clip(e.sum(1, keepdims=True), 1e-30, None)

    def _lse(z, axis=1):
        m = z.max(axis, keepdims=True)
        return m.squeeze(axis) + np.log(
            np.clip(np.exp(z - m).sum(axis), 1e-30, None))

    z = logits.astype(np.float64)
    p = _softmax(z)
    pc = np.clip(p, 1e-12, None)
    pT = _softmax(z / odin_T)
    out = {
        'msp': 1 - p.max(1),
        'maxlogit': -z.max(1),
        'energy': -_lse(z, 1),
        'entropy': -(pc * np.log(pc)).sum(1),
        'odin': 1 - pT.max(1),
    }
    gp = np.stack([p[:, g].sum(1) for g in hier], 1)
    gpT = np.stack([pT[:, g].sum(1) for g in hier], 1)
    gmax = np.stack([z[:, g].max(1) for g in hier], 1)
    glse = np.stack([_lse(z[:, g], 1) for g in hier], 1)
    gpc = np.clip(gp, 1e-12, None)
    K = np.array([len(g) for g in hier], dtype=np.float64)
    logK = np.log(K)
    prior = K / C
    Q = np.clip(gp - prior, 0, None)
    QT = np.clip(gpT - prior, 0, None)
    out.update({
        'group_msp': 1 - gp.max(1),
        'group_maxlogit': -gmax.max(1),
        'group_energy': -glse.max(1),
        'group_entropy': -(gpc * np.log(gpc)).sum(1),
        'group_odin': 1 - gpT.max(1),
        'gn_msp': 1 - Q.max(1),
        'gn_maxlogit': -(gmax - logK).max(1),
        'gn_energy': -(glse - logK).max(1),
        'gn_entropy': -((Q + 1e-12) * np.log(Q + 1e-12)).sum(1),
        'gn_odin': 1 - QT.max(1),
    })
    return out


def test_group_keys_and_order():
    logits = torch.randn(7, 19)
    scores = compute_ood_scores(logits, class_groups=SK_GROUPS)
    assert tuple(scores.keys()) == ALL_SCORE_KEYS
    assert ALL_SCORE_KEYS == OOD_SCORE_KEYS + GROUP_SCORE_KEYS + GN_SCORE_KEYS
    assert GROUP_SCORE_KEYS == ('group_msp', 'group_maxlogit', 'group_odin',
                                'group_energy', 'group_entropy')
    assert GN_SCORE_KEYS == ('gn_msp', 'gn_maxlogit', 'gn_odin', 'gn_energy',
                             'gn_entropy')
    for key in ALL_SCORE_KEYS:
        assert scores[key].shape == (7, ), key
    # Without groups only the flat keys are produced.
    assert tuple(compute_ood_scores(logits).keys()) == OOD_SCORE_KEYS
    print('PASS test_group_keys_and_order')


def test_group_scores_match_reference_script():
    # Our convention is -max instead of the script's 1 - max for the
    # MSP/ODIN family (constant shift, identical ranking); every other score
    # must agree exactly.
    rng = np.random.RandomState(0)
    z = rng.randn(300, 19) * 3.0
    ref = _reference_scores(z, SK_GROUPS, 19, odin_T=1000.0)
    ours = compute_ood_scores(torch.from_numpy(z).float(),
                              odin_temperature=1000.0,
                              energy_temperature=1.0,
                              class_groups=SK_GROUPS)
    shifted = {'msp', 'odin', 'group_msp', 'group_odin', 'gn_msp', 'gn_odin'}
    for key in ALL_SCORE_KEYS:
        expected = ref[key] - 1.0 if key in shifted else ref[key]
        assert np.allclose(ours[key].numpy(), expected, atol=1e-5), key
    print('PASS test_group_scores_match_reference_script')


def test_group_structure_properties():
    logits = torch.randn(40, 19) * 2.0
    scores = compute_ood_scores(logits, class_groups=SK_GROUPS)
    # A partition leaves MaxLogit unchanged (GroupPaper, after Eq. 5).
    assert torch.allclose(scores['group_maxlogit'], scores['maxlogit'])
    # Singleton groups covering every class: the probability scores reduce
    # to the flat ones, the logit aggregates reduce to the class max (so the
    # energy variants equal MaxLogit) and the log K_g correction vanishes.
    single = compute_ood_scores(logits, class_groups=[[c] for c in range(19)])
    for key in ('msp', 'odin', 'entropy'):
        assert torch.allclose(single[f'group_{key}'], single[key],
                              atol=1e-6), key
    assert torch.allclose(single['group_energy'], single['maxlogit'])
    assert torch.allclose(single['gn_maxlogit'], single['maxlogit'])
    assert torch.allclose(single['gn_energy'], single['maxlogit'])
    print('PASS test_group_structure_properties')


def test_group_normalization_known_values():
    # Two classes, logits [2, 0]. Groups {0}, {1}: P = softmax, prior 1/2.
    logits = torch.tensor([[2.0, 0.0]])
    s = compute_ood_scores(logits, class_groups=[[0], [1]])
    p0 = math.exp(2.0) / (math.exp(2.0) + 1.0)
    q0 = p0 - 0.5  # class 1 has p1 < 1/2, so Q_1 = 0
    assert torch.allclose(s['gn_msp'], torch.tensor([-q0]), atol=1e-6)
    eps = 1e-12
    gn_ent = -((q0 + eps) * math.log(q0 + eps) + eps * math.log(eps))
    assert torch.allclose(s['gn_entropy'], torch.tensor([gn_ent]), atol=1e-6)
    # One group holding both classes: P_g = 1 -> group_msp = -1,
    # group_entropy = 0, Q_g = 1 - 2/2 = 0; logit aggregates over the
    # whole group with the log K_g = log 2 correction.
    s2 = compute_ood_scores(logits, class_groups=[[0, 1]])
    assert torch.allclose(s2['group_msp'], torch.tensor([-1.0]), atol=1e-6)
    assert torch.allclose(s2['group_entropy'], torch.tensor([0.0]),
                          atol=1e-6)
    assert torch.allclose(s2['gn_msp'], torch.tensor([0.0]), atol=1e-6)
    lse = math.log(math.exp(2.0) + 1.0)
    assert torch.allclose(s2['group_energy'], torch.tensor([-lse]),
                          atol=1e-6)
    assert torch.allclose(s2['gn_energy'],
                          torch.tensor([-(lse - math.log(2.0))]), atol=1e-6)
    assert torch.allclose(s2['gn_maxlogit'],
                          torch.tensor([-(2.0 - math.log(2.0))]), atol=1e-6)
    print('PASS test_group_normalization_known_values')


def test_group_energy_temperature():
    logits = torch.randn(5, 19)
    s = compute_ood_scores(logits, energy_temperature=2.0,
                           class_groups=SK_GROUPS)
    glse = torch.stack([
        2.0 * torch.logsumexp(logits[:, g] / 2.0, dim=1) for g in SK_GROUPS
    ], dim=1)
    log_k = torch.log(torch.tensor([float(len(g)) for g in SK_GROUPS]))
    assert torch.allclose(s['group_energy'], -glse.max(dim=1).values,
                          atol=1e-6)
    assert torch.allclose(s['gn_energy'],
                          -(glse - 2.0 * log_k).max(dim=1).values, atol=1e-6)
    print('PASS test_group_energy_temperature')


def test_invalid_class_groups_rejected():
    logits = torch.randn(3, 19)
    for bad in ([[0, 1], [1, 2]],  # overlapping
                [[0, 19]],  # out of range
                [[0], []]):  # empty group
        try:
            compute_ood_scores(logits, class_groups=bad)
            raise AssertionError(f'expected ValueError for {bad}')
        except ValueError:
            pass
    print('PASS test_invalid_class_groups_rejected')


def test_point_projection_with_groups():
    voxel_logits = torch.randn(3, 19)
    point2voxel_map = torch.tensor([0, 0, 2, 1])
    pts = point_ood_scores(voxel_logits, point2voxel_map,
                           class_groups=SK_GROUPS)
    assert tuple(pts.keys()) == ALL_SCORE_KEYS
    vox = compute_ood_scores(voxel_logits, class_groups=SK_GROUPS)
    for key in ALL_SCORE_KEYS:
        assert pts[key].dtype == np.float32 and pts[key].shape == (4, )
        assert np.allclose(pts[key], vox[key][[0, 0, 2, 1]].numpy(),
                           atol=1e-6), key
    print('PASS test_point_projection_with_groups')


def test_class_groups_variants():
    # Named alternative hierarchies ride along with prefixed keys and must
    # equal a direct class_groups computation with the same hierarchy.
    logits = torch.randn(9, 19)
    variants = {
        'm2': [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11]],
        'c2': [[12, 13], [14, 15, 16]],  # partial coverage is allowed
    }
    scores = compute_ood_scores(logits, class_groups=SK_GROUPS,
                                class_groups_variants=variants)
    for name, groups in variants.items():
        direct = compute_ood_scores(logits, class_groups=groups)
        for key in GROUP_SCORE_KEYS + GN_SCORE_KEYS:
            assert torch.allclose(scores[f'{name}_{key}'], direct[key],
                                  atol=1e-6), (name, key)
    # Base keys unchanged, each variant adds its ten prefixed keys.
    assert len(scores) == len(ALL_SCORE_KEYS) + 10 * len(variants)
    pts = point_ood_scores(logits, torch.tensor([0, 3, 5]),
                           class_groups_variants=variants)
    assert 'm2_gn_energy' in pts and pts['m2_gn_energy'].shape == (3, )
    assert pts['m2_gn_energy'].dtype == np.float32
    print('PASS test_class_groups_variants')


if __name__ == '__main__':
    test_score_keys_and_shapes()
    test_known_values_two_classes()
    test_uniform_logits_are_most_ood()
    test_entropy_matches_reference_formula()
    test_energy_temperature()
    test_odin_is_temperature_scaled_msp()
    test_numerical_stability_huge_logits()
    test_point_projection()
    test_group_keys_and_order()
    test_group_scores_match_reference_script()
    test_group_structure_properties()
    test_group_normalization_known_values()
    test_group_energy_temperature()
    test_invalid_class_groups_rejected()
    test_point_projection_with_groups()
    test_class_groups_variants()
    print('ALL TESTS PASSED')
