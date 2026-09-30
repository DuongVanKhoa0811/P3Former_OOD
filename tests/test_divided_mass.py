"""Tests for tools/divided_mass.py.

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_divided_mass.py
"""
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

import divided_mass as dm  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402

RNG = np.random.RandomState(0)


def _logits(n, scale=4.0):
    return (scale * RNG.randn(n, sb.NUM_CLASSES)).astype(np.float32)


def _write_dump(dump_dir, num_frames=3, n=4000, start=0):
    """Frames as _OODLogitsDumpMetric writes them; OOD points flatter."""
    os.makedirs(dump_dir, exist_ok=True)
    for i in range(num_frames):
        z = _logits(n)
        ood = RNG.rand(n) < 0.1
        z[ood] *= 0.3
        np.savez(os.path.join(dump_dir, f'r0_{start + i:06d}.npz'),
                 logits=z.astype(np.float16), ood=ood,
                 valid=RNG.rand(n) < 0.95, mapped=np.zeros(n, np.int16),
                 lidar_path=f'f{start + i}')


def _files(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d))


def _fake_root(tmp):
    """A --root with two dumps and every sweep log divided_mass.py reads."""
    root = os.path.join(tmp, 'root')
    _write_dump(os.path.join(root, 'logits'), num_frames=2)
    _write_dump(os.path.join(root, 'logits_test'), num_frames=3, start=2)
    kwargs = dict(bins=2**16, workers=1, chunk=8, top=1)
    for spec in dm.SETS.values():
        dumps = [os.path.join(root, d) for d in spec['dumps']]
        sb.sweep(dumps, os.path.join(root, spec['sweep']), num=8, seed=3,
                 **kwargs)
        sb.sweep(dumps, os.path.join(root, spec['singletons']),
                 subsets=sb.singleton_subsets(), **kwargs)
    return root


def test_bin_edges():
    edges = dm.bin_edges()
    assert edges[0] == 0.0 and edges[1] == dm.M_MIN and edges[-1] == 0.5
    assert np.all(np.diff(edges) > 0)
    # below TOP_START the grid is log-spaced; at and above it, linear at
    # TOP_STEP (checked away from the deltas, which perturb local spacing)
    i_top = dm.edge_index(edges, dm.TOP_START)
    assert edges[i_top] == dm.TOP_START
    assert abs(edges[-1] - edges[-2] - dm.TOP_STEP) < 1e-12
    for d in dm.DELTAS:
        i = dm.edge_index(edges, d)
        assert edges[i] == d
        # a value exactly at delta opens the bin that starts at delta
        assert dm.bin_index(np.array([d]), edges)[0] == i
    assert list(dm.bin_index(np.array([0.0, 1e-20, 0.5]), edges)) == [
        0, 0, len(edges) - 2]
    # upper is configurable: the u edges reach 1.0, not 0.5
    edges_u = dm.bin_edges(upper=1.0)
    assert edges_u[0] == 0.0 and edges_u[-1] == 1.0
    for bad in (lambda: dm.edge_index(edges, 0.123456),
                lambda: dm.bin_edges(deltas=(0.6, )),
                lambda: dm.bin_edges(upper=1.0, deltas=(1.2, ))):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid edge accepted')
    print('test_bin_edges passed')


def test_u_bins_cover_the_whole_range():
    edges_u = dm.bin_edges(upper=1.0)
    assert edges_u[-1] == 1.0
    # u = 23/24 (uniform over the other 23 classes) is not in the last bin
    i = dm.bin_index(np.array([23.0 / 24.0]), edges_u)[0]
    assert edges_u[i] > 0.9 and i < len(edges_u) - 2

    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 roc_curve)
    rng = np.random.RandomState(2)
    u_id = 10.0**rng.uniform(-9, -1, 200000)
    # top-heavy OOD case: 30% of the OOD points above 0.5, which the old
    # 0.5-clipped edges would all collapse into a single last bin (F1)
    n_ood = 5000
    n_top = int(0.3 * n_ood)
    u_ood = np.concatenate([
        rng.uniform(0.5, 0.95, n_top),
        10.0**rng.uniform(-9, math.log10(0.5), n_ood - n_top)])
    nbins = len(edges_u) - 1
    hist_id = np.bincount(dm.bin_index(u_id, edges_u), minlength=nbins)
    hist_ood = np.bincount(dm.bin_index(u_ood, edges_u), minlength=nbins)
    auroc, ap, fpr95 = dm.histogram_metrics(hist_ood, hist_id)
    scores = np.concatenate([u_id, u_ood])
    labels = np.r_[np.zeros(len(u_id)), np.ones(len(u_ood))]
    fpr, tpr, _ = roc_curve(labels, scores)
    assert abs(auroc - 100.0 * roc_auc_score(labels, scores)) < 0.05
    assert abs(ap - 100.0 * average_precision_score(labels, scores)) < 0.1
    assert abs(fpr95 - 100.0 * fpr[np.searchsorted(tpr, 0.95)]) < 0.5
    print('test_u_bins_cover_the_whole_range passed')


def test_masses_have_no_cancellation():
    z = np.zeros((3, sb.NUM_CLASSES), np.float32)
    z[0, 0] = 30.0  # confident: 1 - max p cancels to 0 in float32
    z[1, 5] = 12.0
    z[2] = _logits(1)[0]
    p = dm.softmax(z)
    p64 = np.exp(z.astype(np.float64) - z.max(axis=1, keepdims=True))
    p64 /= p64.sum(axis=1, keepdims=True)
    u_exact = np.sort(p64, axis=1)[:, :-1].sum(axis=1)
    u = dm.flat_uncertainty(p)
    assert np.allclose(u, u_exact, rtol=1e-5, atol=0), (u, u_exact)
    assert 0 < u[0] < 1e-11  # 23 * exp(-30) = 2.2e-12, not 0
    subsets = [(0, ), (5, 6), tuple(range(12))]
    m = dm.divided_mass(p, sb.subsets_to_mask(subsets))
    for j, subset in enumerate(subsets):
        rest = [c for c in range(sb.NUM_CLASSES) if c not in subset]
        want = np.minimum(p64[:, list(subset)].sum(axis=1),
                          p64[:, rest].sum(axis=1))
        assert np.allclose(m[j], want, rtol=1e-5, atol=0), (subset, m[j], want)
    print('test_masses_have_no_cancellation passed')


def test_float32_masses_reproduce_the_sweep_ties():
    mask = sb.subsets_to_mask([(0, )])

    # a point with m = 1e-9: max(P_A, P_B) saturates to exactly 1.0 in
    # float32, so m32 rounds all the way down to 0
    p = np.zeros((1, sb.NUM_CLASSES), np.float32)
    p[0, 0], p[0, 1] = 1.0 - 1e-9, 1e-9
    m32 = dm.divided_mass32(p, mask)
    assert m32[0, 0] == 0.0

    # a point with m = 0.3: float32 resolves it closely
    p2 = np.zeros((1, sb.NUM_CLASSES), np.float32)
    p2[0, 0], p2[0, 1] = 0.3, 0.7
    m32b = dm.divided_mass32(p2, mask)
    assert abs(m32b[0, 0] - 0.3) < 1e-6

    # a small synthetic frame: three points whose exact m differs by many
    # orders of magnitude (1e-13, 1e-15, 1e-17ish) but whose max(P_A, P_B)
    # all round to exactly 1.0 in float32 -- they must tie in m32, both as
    # computed by divided_mass32 (from the tool's own softmax) and as
    # computed by the sweep's own bipartition_scores (from its own softmax
    # pipeline; the two differ at the last ulp, see sweep_bipartitions.py,
    # but a whole-class saturation is a one-hot matmul, exact regardless of
    # pipeline) -- plus two well-separated points that both must rank the
    # same, clearly above the tied group.
    z = np.zeros((5, sb.NUM_CLASSES), np.float32)
    z[0, 0], z[1, 0], z[2, 0] = 30.0, 35.0, 40.0
    z[3, 0], z[3, 1] = math.log(0.3), math.log(0.7)
    z[3] -= z[3].max()
    z[4, 0], z[4, 1] = math.log(0.1), math.log(0.9)
    z[4] -= z[4].max()

    p_dm = dm.softmax(z)
    m32_dm = dm.divided_mass32(p_dm, mask)[0]
    q_sb = sb.base_quantities(z)
    score_sb = sb.bipartition_scores(q_sb, mask)['group_msp'][0]
    m32_sb = 1.0 + score_sb.astype(np.float64)  # score = -max => m32 = 1+score

    for m32 in (m32_dm, m32_sb):
        u, inv = np.unique(m32, return_inverse=True)
        assert len(u) == 3, (m32, u)  # {saturated group, point 4, point 3}
        assert inv[0] == inv[1] == inv[2]
        assert inv[3] != inv[4] and inv[3] != inv[0] and inv[4] != inv[0]
        assert m32[3] > m32[4] > m32[0]
    print('test_float32_masses_reproduce_the_sweep_ties passed')


def test_divided_counts_match_direct_counts():
    with tempfile.TemporaryDirectory() as tmp:
        _write_dump(tmp)
        files = _files(tmp)
        subsets = [(3, ), (0, 1, 2, 3, 4), tuple(range(12)), (16, )]
        a_mask = sb.subsets_to_mask(subsets)
        edges_m, edges_u = dm.bin_edges(), dm.bin_edges(upper=1.0)
        hist = dm.directory_histograms(files, a_mask, edges_m, edges_u,
                                       chunk=3)
        frames = [sb.load_frame(f) for f in files]
        ood = np.concatenate([f[1] for f in frames])
        assert list(hist['counts']) == [int((~ood).sum()), int(ood.sum())]
        # m exactly as the tool computes it: per frame, in blocks of 3 splits
        # (another matmul shape may round the last ulp differently)
        m = np.concatenate([
            np.concatenate([dm.divided_mass(dm.softmax(z), a_mask[j0:j0 + 3])
                            for j0 in range(0, len(subsets), 3)])
            for z, _ in frames], axis=1).astype(np.float64)
        u = np.concatenate([dm.flat_uncertainty(dm.softmax(z))
                            for z, _ in frames]).astype(np.float64)
        rows = dm.divided_stats(hist, hist['counts'], edges_m, edges_u)
        flat = dm.flat_stats(hist, hist['counts'], edges_u)
        for d in dm.DELTAS:
            key = dm.delta_key(d)
            assert math.isclose(flat[f'ood_unc@{key}'],
                                100.0 * (u[ood] >= d).mean())
            for s in range(len(subsets)):
                assert rows[s][f'ood_div_n@{key}'] == int((m[s][ood] >= d).sum())
                assert rows[s][f'id_div_n@{key}'] == int((m[s][~ood] >= d).sum())
                # the divided points are among the flat-uncertain ones
                assert rows[s][f'ood_div@{key}'] <= flat[f'ood_unc@{key}'] + 1e-9
    print('test_divided_counts_match_direct_counts passed')


def test_histogram_metrics_match_exact():
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 roc_curve)
    rng = np.random.RandomState(1)
    m_id = 10.0**rng.uniform(-12, -1, 200000)
    m_ood = 10.0**rng.uniform(-6, np.log10(0.5), 5000)
    edges = dm.bin_edges()
    nbins = len(edges) - 1
    hist_id = np.bincount(dm.bin_index(m_id, edges), minlength=nbins)
    hist_ood = np.bincount(dm.bin_index(m_ood, edges), minlength=nbins)
    auroc, ap, fpr95 = dm.histogram_metrics(hist_ood, hist_id)
    scores = np.concatenate([m_id, m_ood])
    labels = np.r_[np.zeros(len(m_id)), np.ones(len(m_ood))]
    fpr, tpr, _ = roc_curve(labels, scores)
    assert abs(auroc - 100.0 * roc_auc_score(labels, scores)) < 0.05
    assert abs(ap - 100.0 * average_precision_score(labels, scores)) < 0.1
    assert abs(fpr95 - 100.0 * fpr[np.searchsorted(tpr, 0.95)]) < 0.5
    print('test_histogram_metrics_match_exact passed')


def test_several_directories_sum_and_backends_agree():
    with tempfile.TemporaryDirectory() as tmp:
        a, b, both = (os.path.join(tmp, d) for d in ('a', 'b', 'both'))
        _write_dump(a, num_frames=2)
        _write_dump(b, num_frames=2, start=2)
        os.makedirs(both)
        for d in (a, b):
            for f in _files(d):
                shutil.copy(f, both)
        a_mask = sb.subsets_to_mask([(3, ), (0, 1, 2, 3, 4), (16, 17)])
        edges_m, edges_u = dm.bin_edges(), dm.bin_edges(upper=1.0)
        summed = dm.sum_histograms([
            dm.directory_histograms(_files(a), a_mask, edges_m, edges_u),
            dm.directory_histograms(_files(b), a_mask, edges_m, edges_u)])
        whole = dm.directory_histograms(_files(both), a_mask, edges_m, edges_u)
        for key in ('m', 'm32', 'u', 'u32', 'counts'):
            assert np.array_equal(summed[key], whole[key]), key
        # the torch backend (CPU tensors here) agrees up to last-ulp binning
        torch_hist = dm.directory_histograms(_files(both), a_mask, edges_m,
                                             edges_u, backend='torch',
                                             device='cpu', chunk=2)
        assert np.array_equal(torch_hist['counts'], whole['counts'])
        for key in ('m', 'u'):
            diff = np.abs(torch_hist[key] - whole[key]).sum()
            assert diff <= 1e-3 * whole[key].sum(), (key, diff)
        # m32 / u32 deliberately resolve float32-level ties (F2): numpy's
        # direct e / e.sum() and torch's own softmax kernel legitimately
        # round the last bit differently, and the bins near a tie are only
        # ~2^-24 wide, so a few percent of points land one bin over between
        # backends (confirmed: the mismatches sit exactly at edges that are
        # small integer multiples of 2^-24). A loose bound on the raw counts
        # still catches a gross bug (wrong shape, wrong offset); the
        # invariant that actually matters -- the AUROC/AP/FPR95 the bins
        # feed into -- is checked at the same tolerance as the sweep-log
        # checks, and holds tightly.
        for key in ('m32', 'u32'):
            diff = np.abs(torch_hist[key].astype(np.int64)
                         - whole[key].astype(np.int64)).sum()
            assert diff <= 0.1 * whole[key].sum(), (key, diff)
        for s in range(a_mask.shape[0]):
            np_m = dm.histogram_metrics(whole['m32'][s, 1], whole['m32'][s, 0])
            t_m = dm.histogram_metrics(torch_hist['m32'][s, 1],
                                       torch_hist['m32'][s, 0])
            for x, y, tol in zip(np_m, t_m, dm.CHECK_TOL.values()):
                assert abs(x - y) <= tol, ('m32', s, np_m, t_m)
        np_u = dm.histogram_metrics(whole['u32'][1], whole['u32'][0])
        t_u = dm.histogram_metrics(torch_hist['u32'][1], torch_hist['u32'][0])
        for x, y, tol in zip(np_u, t_u, dm.CHECK_TOL.values()):
            assert abs(x - y) <= tol, ('u32', np_u, t_u)
        try:
            dm.directory_histograms(_files(a), a_mask, edges_m, edges_u,
                                    backend='cuda')
        except ValueError:
            pass
        else:
            raise AssertionError("backend 'cuda' accepted")
    print('test_several_directories_sum_and_backends_agree passed')


def test_zero_divided_counts():
    edges = dm.bin_edges()
    nbins = len(edges) - 1
    at = lambda v: dm.bin_index(np.array([v]), edges)[0]  # noqa: E731
    hist = dict(m=np.zeros((1, 2, nbins), np.int64),
                u=np.zeros((2, nbins), np.int64))
    hist['m'][0, 0, at(1e-7)] = 1000  # no ID point divided at >= 1e-6
    hist['m'][0, 1, at(0.35)] = 5
    hist['m'][0, 1, at(1e-3)] = 5
    hist['u'][0, at(0.4)] = 1000  # u >= m for every point
    hist['u'][1, at(0.35)] = 10
    (row, ) = dm.divided_stats(hist, np.array([1000, 10]), edges, edges)
    assert row['id_div_n@0.3'] == 0 and row['ood_div_n@0.3'] == 5
    assert row['selectivity@0.3'] == float('inf')
    assert row['precision@0.3'] == 100.0
    assert row['ood_retention@0.3'] == 50.0 and row['id_retention@0.3'] == 0.0
    assert math.isfinite(row['log_ratio@0.3'])
    assert row['delta95'] == 1e-3 and row['id_div@delta95'] == 0.0
    rho, n = dm.spearman([1.0, float('nan'), 3.0, 2.0],
                         [1.0, 5.0, float('inf'), 2.0])
    assert n == 3 and abs(rho - 1.0) < 1e-12
    assert math.isnan(dm.spearman([1, 1, 1], [1, 2, 3])[0])
    print('test_zero_divided_counts passed')


def test_tsv_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 't.tsv')
        rows = [dict(name='s16', size_A=1, group_A='car, bicycle', x=0.05,
                     y=float('inf'), z=float('nan'), w=12345678.9)]
        dm.write_tsv(path, rows)
        (back, ) = dm.read_tsv(path)
        assert back['name'] == 's16' and back['size_A'] == 1
        assert back['group_A'] == 'car, bicycle' and back['x'] == 0.05
        assert back['y'] == float('inf') and math.isnan(back['z'])
        assert abs(back['w'] - 12345678.9) / 12345678.9 < 1e-5
        dm.write_tsv(path, [], header=['name', 'worst'])
        assert dm.read_tsv(path) == []
    print('test_tsv_round_trip passed')


def test_run_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_root(tmp)
        sets = dm.resolve_sets(root)
        subsets = dm.default_subsets(sets)
        assert subsets[:24] == sb.singleton_subsets()
        assert len(set(subsets)) == len(subsets)
        out = os.path.join(tmp, 'divided_mass')
        result = dm.run(sets, subsets, out)
        for name in ('histograms.npz', 'cetran.tsv', 'test.tsv',
                     'test_cetran.tsv', 'flat.tsv', 'robust.tsv', 'rho.tsv',
                     'summary.md'):
            assert os.path.exists(os.path.join(out, name)), name
        with np.load(os.path.join(out, 'histograms.npz')) as h:
            assert np.array_equal(h['test_cetran_m'], h['cetran_m'] + h['test_m'])
            assert np.array_equal(h['test_cetran_m32'],
                                  h['cetran_m32'] + h['test_m32'])
            assert np.array_equal(h['test_cetran_u'], h['cetran_u'] + h['test_u'])
            assert np.array_equal(h['test_cetran_u32'],
                                  h['cetran_u32'] + h['test_u32'])
            assert h['edges_m'][-1] == 0.5 and h['edges_u'][-1] == 1.0
            assert list(h['names']) == [sb.partition_name(s) for s in subsets]
        rows = dm.read_tsv(os.path.join(out, 'test_cetran.tsv'))
        assert [r['name'] for r in rows] == [sb.partition_name(s) for s in subsets]
        assert all(c in rows[0] for c in
                  ('f32_auroc', 'f32_ap', 'f32_fpr95', 'exact_auroc',
                   'exact_ap', 'exact_fpr95'))
        assert not any(c in rows[0] for c in
                       ('hist_auroc', 'hist_ap', 'hist_fpr95'))
        improvements = {key: {r['name']: r['improvement'] for r in table}
                        for key, table in result['tables'].items()}
        robust = {r['name'] for r in dm.read_tsv(os.path.join(out, 'robust.tsv'))}
        for r in rows:
            want = all(imp[r['name']] > 0 for imp in improvements.values())
            assert r['robust'] == int(want) and (r['name'] in robust) == want
        rho = dm.read_tsv(os.path.join(out, 'rho.tsv'))
        assert len(rho) == (3 * 2 * len(dm.DELTAS) * len(dm.TARGETS)
                            * len(dm.STATISTICS))
        agreement = [c for c in result['checks'] if 'both sweep logs' in c['check']]
        assert len(agreement) == 3 and all(c['status'] == 'PASS' for c in agreement)
        assert all(c['status'] in ('PASS', 'FAIL') for c in result['checks'])
        # F1/F2 fix: the f32-histogram metrics, compared against the sweep
        # logs, must PASS on this synthetic root. A failure here needs
        # investigation, never a loosened tolerance.
        hist_checks = [c for c in result['checks']
                       if 'histograms vs sweep' in c['check']]
        assert len(hist_checks) == 3 * 2 * len(dm.CHECK_TOL)
        assert all(c['status'] == 'PASS' for c in hist_checks), hist_checks
        with open(os.path.join(out, 'summary.md')) as fh:
            text = fh.read()
        for section in ('## Consistency checks',
                        '## Float32 ties in the implemented Group MSP',
                        '## Flat MSP reference',
                        '## Single-class splits, Cetran', '## Robust splits',
                        '## Spearman rho'):
            assert section in text, section
    print('test_run_end_to_end passed')


def test_missing_inputs_are_explained():
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_root(tmp)
        sets = dm.resolve_sets(root)
        # a sweep of the test set with another list of partitions: refused
        path = os.path.join(root, 'bipartitions_test', 'partitions.json')
        with open(path) as fh:
            parts = json.load(fh)
        with open(path, 'w') as fh:
            json.dump(parts[:-1], fh)
        try:
            dm.default_subsets(sets)
        except ValueError as err:
            assert '--seed' in str(err)
        else:
            raise AssertionError('different partitions accepted')
        # no single-class sweep yet: the error names the tool to run
        shutil.rmtree(os.path.join(root, 'singletons_test'))
        try:
            dm.sweep_rows(sets['test']['sweep_log'],
                          sets['test']['singletons_log'])
        except FileNotFoundError as err:
            assert 'sweep_bipartitions.py' in str(err)
        else:
            raise AssertionError('missing single-class log accepted')
        # a split the logs do not hold: KeyError naming it
        rows = dm.sweep_rows(sets['cetran']['sweep_log'],
                             sets['cetran']['singletons_log'])
        try:
            dm.split_metrics(rows, ['s99'])
        except KeyError as err:
            assert 's99' in str(err)
        else:
            raise AssertionError('missing split accepted')
    print('test_missing_inputs_are_explained passed')


if __name__ == '__main__':
    test_bin_edges()
    test_u_bins_cover_the_whole_range()
    test_masses_have_no_cancellation()
    test_float32_masses_reproduce_the_sweep_ties()
    test_divided_counts_match_direct_counts()
    test_histogram_metrics_match_exact()
    test_several_directories_sum_and_backends_agree()
    test_zero_divided_counts()
    test_tsv_round_trip()
    test_run_end_to_end()
    test_missing_inputs_are_explained()
    print('ALL TESTS PASSED')
