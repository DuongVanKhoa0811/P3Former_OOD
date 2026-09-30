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


def test_split_measures_nan_r_ood():
    rng = np.random.RandomState(4)
    k = 10
    counts = rng.multinomial(k, np.ones(C) / C, size=200).astype(np.int32)
    weight = rng.uniform(0.5, 2.0, 200)
    subsets = [(7, ), (0, 1, 2, 3, 4), tuple(range(12))]
    mask = sb.subsets_to_mask(subsets)
    r_ood = 100.0 * rng.dirichlet(np.ones(C))
    baseline = res.split_measures(mask, r_ood, counts, weight, counts, weight,
                                  k)
    nan_r_ood = r_ood.copy()
    nan_r_ood[7] = float('nan')  # class 7 excluded from the bank
    R_A, E_A, fd_ood, fd_id = res.split_measures(
        mask, nan_r_ood, counts, weight, counts, weight, k)
    assert np.all(np.isfinite(R_A)) and np.all(np.isfinite(E_A))
    want = r_ood.copy()
    want[7] = 0.0  # the code counts a NaN r_ood as 0, not as missing
    for s, subset in enumerate(subsets):
        assert np.isclose(R_A[s], want[list(subset)].sum())
        assert np.isclose(E_A[s], R_A[s] / (100.0 * len(subset) / C))
    # the feature-divided shares are computed from counts/weight alone and
    # must not change when r_ood does.
    assert np.allclose(fd_ood, baseline[2]) and np.allclose(fd_id, baseline[3])
    print('test_split_measures_nan_r_ood passed')


def test_placement_rows():
    resembled = {2: 40.0, 5: 30.0, 9: 20.0, 14: 10.0}  # descending r_ood
    profile = [OrderedDict([('class', sb.CLASSES[c]),
                            ('r_ood', resembled.get(c, float('nan')))])
              for c in range(C)]

    def row(subset, improvement):
        return OrderedDict([
            ('name', sb.partition_name(tuple(sorted(subset)))),
            ('improvement', improvement)])

    # S_2 = {2, 5}; S_3 = {2, 5, 9}.
    splits = [
        row((2, 5), 15.0),      # k=2: small side | k=3: apart
        row((2, 3, 5), 25.0),   # k=2: small side | k=3: apart
        row((9, 20), -10.0),    # k=2: large side | k=3: apart
        row((1, 3), 5.0),       # k=2: large side | k=3: large side
        row((2, ), -20.0),      # k=2: apart      | k=3: apart
    ]
    rows = res.placement_rows(profile, splits, 'test', 'full', ks=(2, 3))
    by = {(r['k'], r['position']): r for r in rows}

    s2 = f'{sb.CLASSES[2]}, {sb.CLASSES[5]}'
    s3 = f'{sb.CLASSES[2]}, {sb.CLASSES[5]}, {sb.CLASSES[9]}'
    for position in ('small side', 'large side', 'apart'):
        assert by[(2, position)]['classes'] == s2
        assert by[(3, position)]['classes'] == s3

    small2 = by[(2, 'small side')]
    assert small2['n'] == 2 and np.isclose(small2['median'], 20.0)
    assert np.isclose(small2['positive'], 100.0)
    assert small2['best'] == sb.partition_name((2, 3, 5))
    assert np.isclose(small2['best_improvement'], 25.0)

    large2 = by[(2, 'large side')]
    assert large2['n'] == 2 and np.isclose(large2['median'], -2.5)
    assert np.isclose(large2['positive'], 50.0)
    assert large2['best'] == sb.partition_name((1, 3))
    assert np.isclose(large2['best_improvement'], 5.0)

    apart2 = by[(2, 'apart')]
    assert apart2['n'] == 1 and np.isclose(apart2['median'], -20.0)
    assert np.isclose(apart2['positive'], 0.0)
    assert apart2['best'] == sb.partition_name((2, ))
    assert np.isclose(apart2['best_improvement'], -20.0)

    # the small side is empty at k=3: NaN median/positive, '' best.
    small3 = by[(3, 'small side')]
    assert small3['n'] == 0
    assert np.isnan(small3['median']) and np.isnan(small3['positive'])
    assert small3['best'] == '' and np.isnan(small3['best_improvement'])

    large3 = by[(3, 'large side')]
    assert large3['n'] == 1 and np.isclose(large3['median'], 5.0)
    assert np.isclose(large3['positive'], 100.0)
    assert large3['best'] == sb.partition_name((1, 3))
    assert np.isclose(large3['best_improvement'], 5.0)

    apart3 = by[(3, 'apart')]
    assert apart3['n'] == 4 and np.isclose(apart3['median'], 2.5)
    assert np.isclose(apart3['positive'], 50.0)
    assert apart3['best'] == sb.partition_name((2, 3, 5))
    assert np.isclose(apart3['best_improvement'], 25.0)
    print('test_placement_rows passed')


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


def _write_features(d, frames, rng, centres, dim=8, missing=None, noise=0.3):
    """Feature samples as tools/extract_point_features.py writes them; the
    OOD points sit next to class 16 (overhead-bridge). ``missing`` drops
    that class's rows entirely (the OOD cluster stays at its centre), for a
    set with no ID sample of one class. ``noise`` is the per-point spread
    around each class centre; the default keeps classes cleanly separated,
    as every test but test_cross_set_reference wants."""
    os.makedirs(d)
    for f in range(frames):
        label = np.concatenate([np.repeat(np.arange(C), 20), np.full(30, 24)])
        ood = label == 24
        spread = noise * rng.randn(len(label), dim)
        feat = np.where(ood[:, None], centres[16] + 0.4 + spread,
                        centres[np.minimum(label, C - 1)] + spread)
        pos = (0.1 * rng.randn(len(label), dim)).astype(np.float16)
        raw = np.where(ood, 17, 1).astype(np.int16)
        if missing is not None:
            keep = label != missing
            label, feat, pos, raw, ood = (label[keep], feat[keep], pos[keep],
                                          raw[keep], ood[keep])
        np.savez(os.path.join(d, f'f{f:06d}.npz'),
                 feat=feat.astype(np.float16), pos=pos,
                 logits=np.zeros((len(label), C), np.float16),
                 label=label.astype(np.int16), raw=raw, ood=ood,
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

        placement = dm.read_tsv(os.path.join(out, 'placement.tsv'))
        assert placement
        by_combo = OrderedDict()
        for r in placement:
            by_combo.setdefault((r['set'], r['space'], r['k']),
                                []).append(r)
        for combo, rows_here in by_combo.items():
            assert {r['position'] for r in rows_here} == {
                'small side', 'large side', 'apart'}
            assert sum(r['n'] for r in rows_here) == len(subsets), combo

        with open(os.path.join(out, 'summary.md')) as fh:
            text = fh.read()
        for section in ('## Reference banks', '## Hypotheses',
                        '## Profile, Test + Cetran (full space)',
                        '## Robust splits, Cetran (full space)',
                        '## Hypothesis, literal reading'):
            assert section in text, section
    print('test_run_end_to_end passed')


def test_cross_set_reference():
    rng = np.random.RandomState(5)
    centres = 3.0 * rng.randn(C, 8)
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, 'root')
        # Cetran has no ID sample of class 16 at all -- the real dump's
        # situation for ten classes; the OOD cluster still sits at its
        # centre. Test keeps every class, so Test + Cetran has all 24. A
        # wider noise (not the default's clean separation) gives every
        # class some cross-class neighbours, so the rho row count agrees
        # across x's -- with the default's tight clusters, r_ood and r_id
        # are both exactly 0 for most classes, and contrast (ratio(0, 0))
        # is NaN for more classes than the bank-exclusion alone accounts
        # for, unrelated to what this test is checking.
        _write_features(os.path.join(root, 'features_cetran'), 3, rng,
                        centres, missing=16, noise=2.0)
        _write_features(os.path.join(root, 'features_test'), 4, rng, centres,
                        noise=2.0)
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

        sets = res.resolve_sets(root)
        out_default = os.path.join(tmp, 'resemblance_default')
        _, _, default_rho = res.run(
            sets, divided, out_default, k=5, bank_per_class=40,
            query_per_class=20, device='cpu')
        out_none = os.path.join(tmp, 'resemblance_none')
        res.run(sets, divided, out_none, k=5, bank_per_class=40,
               query_per_class=20, device='cpu', reference=None)
        out_xref = os.path.join(tmp, 'resemblance_xref')
        _, _, xref_rho = res.run(
            sets, divided, out_xref, k=5, bank_per_class=40,
            query_per_class=20, device='cpu', reference='test_cetran')

        def profile_of(out_dir, key='cetran', space='full'):
            return {r['class']: r for r in dm.read_tsv(
                os.path.join(out_dir, f'profile_{key}_{space}.tsv'))}

        default_profile = profile_of(out_default)
        assert np.isnan(default_profile['overhead-bridge']['r_ood'])
        # a class left out of the bank has no measured feature-divided
        # share either (its singleton's 0.0 is structural, not measured).
        assert np.isnan(default_profile['overhead-bridge']['feat_div_ood'])
        assert np.isnan(default_profile['overhead-bridge']['feat_div_id'])

        xref_profile = profile_of(out_xref)
        assert not np.isnan(xref_profile['overhead-bridge']['r_ood'])
        assert not np.isnan(xref_profile['overhead-bridge']['feat_div_ood'])
        assert not np.isnan(xref_profile['overhead-bridge']['feat_div_id'])
        assert max(xref_profile,
                  key=lambda c: xref_profile[c]['r_ood']) == 'overhead-bridge'

        # every class-level rho row agrees on n, once feat_div_ood/id are
        # NaN'd at the same classes as r_ood/r_id/contrast.
        default_cetran_n = {r['n'] for r in default_rho
                            if r['set'] == 'cetran'
                            and r['population'] == 'classes'}
        assert default_cetran_n == {23}, default_cetran_n
        xref_cetran_n = {r['n'] for r in xref_rho
                         if r['set'] == 'cetran'
                         and r['population'] == 'classes'}
        assert xref_cetran_n == {24}, xref_cetran_n

        # reference=None is exactly the no-argument behaviour.
        with open(os.path.join(out_default,
                               'profile_cetran_full.tsv')) as fh:
            a = fh.read()
        with open(os.path.join(out_none, 'profile_cetran_full.tsv')) as fh:
            b = fh.read()
        assert a == b

        with open(os.path.join(out_default, 'summary.md')) as fh:
            default_summary = fh.read()
        with open(os.path.join(out_xref, 'summary.md')) as fh:
            xref_summary = fh.read()
        assert "reference: each set's own ID samples" in default_summary
        assert ('reference: the ID samples of Test + Cetran (all sets)'
               in xref_summary)
        # Cetran lacks class 16 only: 23 measured by default, 24 with the
        # cross-set reference (Test + Cetran has every class).
        assert 'Spearman rho over the 23 measured classes:' in default_summary
        assert 'Spearman rho over the 24 measured classes:' in xref_summary
    print('test_cross_set_reference passed')


if __name__ == '__main__':
    test_knn_shares_find_the_nearest_class()
    test_profile_weights_and_exclusion()
    test_split_measures_match_brute_force()
    test_split_measures_nan_r_ood()
    test_placement_rows()
    test_reference_selection()
    test_run_end_to_end()
    test_cross_set_reference()
    print('ALL TESTS PASSED')
