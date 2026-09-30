"""Tests for tools/ood_class_resemblance.py (synthetic features, CPU).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_class_resemblance.py
"""
import os
import sys
import tempfile
from collections import OrderedDict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

import divided_mass as dm  # noqa: E402
import ood_class_resemblance as res  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402

C = sb.NUM_CLASSES


def test_knn_shares_find_the_nearest_class():
    rng = np.random.RandomState(0)
    dim = 16
    centres = 3.0 * rng.randn(C, dim)
    bank = np.concatenate([centres[c] + 0.3 * rng.randn(50, dim)
                           for c in range(C)]).astype(np.float32)
    bank_label = np.repeat(np.arange(C), 50)
    ood = (centres[16] + 0.5 + 0.3 * rng.randn(200, dim)).astype(np.float32)
    counts = res.knn_class_counts(ood, bank, bank_label, k=10, device='cpu',
                                  chunk=64)
    assert counts.shape == (200, C) and np.all(counts.sum(axis=1) == 10)
    assert counts[:, 16].sum() > 0.8 * counts.sum()  # next to overhead-bridge
    q = ood[0] / np.linalg.norm(ood[0])
    b = bank / np.linalg.norm(bank, axis=1, keepdims=True)
    top = np.argsort(-(b @ q))[:10]
    assert np.array_equal(np.bincount(bank_label[top], minlength=C), counts[0])
    try:
        res.knn_class_counts(ood, bank[:5], bank_label[:5], k=10,
                             device='cpu')
    except ValueError:
        pass
    else:
        raise AssertionError('k larger than the bank accepted')
    print('test_knn_shares_find_the_nearest_class passed')


def test_profile_weights_and_exclusion():
    k = 10
    ood_counts = np.zeros((2, C), np.int32)
    ood_counts[0, 3] = 10  # weight 3: all neighbours class 3
    ood_counts[1, 5] = 10  # weight 1: all neighbours class 5
    id_label = np.array([0, 0, 1, 1])
    id_counts = np.zeros((4, C), np.int32)
    id_counts[0, 3] = 10  # one class-0 query sits among class-3 points
    id_counts[1, 0] = 10
    id_counts[2:, 1] = 10
    population = np.zeros(C)
    population[0], population[1] = 90.0, 10.0
    id_weight = res.query_weights(id_label, population)
    assert np.allclose(id_weight, [45, 45, 5, 5])
    excluded = np.zeros(C, bool)
    excluded[23] = True
    r_ood, r_id, contrast = res.class_profile(
        ood_counts, np.array([3.0, 1.0]), id_counts, id_label, id_weight, k,
        excluded)
    assert np.isclose(r_ood[3], 75.0) and np.isclose(r_ood[5], 25.0)
    assert np.isclose(r_id[3], 45.0) and np.isclose(contrast[3], 75.0 / 45.0)
    assert r_id[5] == 0 and contrast[5] == float('inf')
    assert np.isnan(r_ood[23]) and np.isnan(r_id[23]) and np.isnan(contrast[23])
    print('test_profile_weights_and_exclusion passed')


def test_split_measures_match_brute_force():
    rng = np.random.RandomState(2)
    k = 10
    counts = rng.multinomial(k, np.ones(C) / C, size=300).astype(np.int32)
    counts[:50] = 0
    counts[:50, 7] = k  # pure neighbourhoods: never divided
    weight = rng.uniform(0.5, 2.0, 300)
    subsets = [(7, ), (0, 1, 2, 3, 4), tuple(range(12))]
    r_ood = 100.0 * rng.dirichlet(np.ones(C))
    R_A, E_A, fd_ood, fd_id = res.split_measures(
        sb.subsets_to_mask(subsets), r_ood, counts, weight, counts, weight, k,
        chunk=64)
    for s, subset in enumerate(subsets):
        n_a = counts[:, list(subset)].sum(axis=1)
        want = 100.0 * weight[(n_a > 0) & (n_a < k)].sum() / weight.sum()
        assert np.isclose(fd_ood[s], want) and np.isclose(fd_id[s], want)
        assert np.isclose(R_A[s], r_ood[list(subset)].sum())
        assert np.isclose(E_A[s], R_A[s] / (100.0 * len(subset) / C))
    print('test_split_measures_match_brute_force passed')


def test_reference_selection():
    label = np.concatenate([np.full(900, 0), np.full(20, 1), np.full(60, 2),
                            np.full(100, 24)])
    ood = label == 24
    bank, query, table = res.select_reference(
        label, ood, np.ones(len(label)), 500, 200, np.random.default_rng(0))
    assert not set(bank.tolist()) & set(query.tolist())
    assert table[0]['bank'] == 500 and table[0]['queries'] == 200
    assert table[1]['excluded'] == 1 and table[1]['bank'] == 0  # 20 < 30
    assert table[2]['queries'] == 20 and table[2]['bank'] == 40
    assert table[5]['samples'] == 0 and table[5]['excluded'] == 1  # absent
    assert np.all(label[bank] != 24) and np.all(label[query] != 24)
    print('test_reference_selection passed')


def _write_features(d, frames, rng, centres, dim=8):
    """Feature samples as tools/extract_point_features.py writes them; the
    OOD points sit next to class 16 (overhead-bridge)."""
    os.makedirs(d)
    for f in range(frames):
        label = np.concatenate([np.repeat(np.arange(C), 20), np.full(30, 24)])
        ood = label == 24
        noise = 0.3 * rng.randn(len(label), dim)
        feat = np.where(ood[:, None], centres[16] + 0.4 + noise,
                        centres[np.minimum(label, C - 1)] + noise)
        np.savez(os.path.join(d, f'f{f:06d}.npz'),
                 feat=feat.astype(np.float16),
                 pos=(0.1 * rng.randn(len(label), dim)).astype(np.float16),
                 logits=np.zeros((len(label), C), np.float16),
                 label=label.astype(np.int16),
                 raw=np.where(ood, 17, 1).astype(np.int16), ood=ood,
                 weight=np.ones(len(label), np.float32),
                 index=np.arange(len(label), dtype=np.int32),
                 lidar_path=f'x{f}', frame=f)


def test_run_end_to_end():
    rng = np.random.RandomState(3)
    centres = 3.0 * rng.randn(C, 8)
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, 'root')
        _write_features(os.path.join(root, 'features_cetran'), 3, rng, centres)
        _write_features(os.path.join(root, 'features_test'), 4, rng, centres)
        divided = os.path.join(tmp, 'divided')
        os.makedirs(divided)
        subsets = sb.singleton_subsets() + [(12, 16), (0, 1, 2, 3, 4)]
        for key in res.SETS:
            dm.write_tsv(os.path.join(divided, f'{key}.tsv'), [
                OrderedDict([
                    ('name', sb.partition_name(s)), ('size_A', len(s)),
                    ('group_A', ', '.join(sb.CLASSES[c] for c in s)),
                    ('ood_div@0.05', float(rng.uniform(0, 50))),
                    ('id_div@0.05', float(rng.uniform(0, 5))),
                    ('improvement', float(rng.uniform(-30, 30))),
                    ('robust', int(s == (16, )))]) for s in subsets])
        out = os.path.join(tmp, 'resemblance')
        profiles, splits, rho = res.run(
            res.resolve_sets(root), divided, out, k=5, bank_per_class=40,
            query_per_class=20, device='cpu')
        for key in res.SETS:
            for space in res.SPACES:
                for kind in ('profile', 'splits'):
                    assert os.path.exists(
                        os.path.join(out, f'{kind}_{key}_{space}.tsv'))
        profile = {r['class']: r for r in profiles[('test_cetran', 'full')]}
        assert max(profile, key=lambda c: profile[c]['r_ood']) == 'overhead-bridge'
        assert abs(sum(r['r_ood'] for r in profile.values()) - 100.0) < 1e-6
        assert len(splits[('cetran', 'full')]) == len(subsets)
        assert len(rho) == 3 * 2 * (len(res.CLASS_PAIRS) + len(res.SPLIT_PAIRS))
        with open(os.path.join(out, 'summary.md')) as fh:
            text = fh.read()
        for section in ('## Reference banks', '## Hypotheses',
                        '## Profile, Test + Cetran (full space)',
                        '## Robust splits, Cetran (full space)'):
            assert section in text, section
    print('test_run_end_to_end passed')


if __name__ == '__main__':
    test_knn_shares_find_the_nearest_class()
    test_profile_weights_and_exclusion()
    test_split_measures_match_brute_force()
    test_reference_selection()
    test_run_end_to_end()
    print('ALL TESTS PASSED')
