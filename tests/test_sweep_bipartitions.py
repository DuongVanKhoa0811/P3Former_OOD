"""Tests for tools/sweep_bipartitions.py and metrics_from_histograms.

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_sweep_bipartitions.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

from evaluation.functional.ood_eval import (binary_ood_metrics,
                                            metrics_from_histograms)
from p3former.utils.ood_scores import compute_ood_scores
from summarize_hierarchy_ablation import check_family, offline_sweep_logs
import sweep_bipartitions as sb

RNG = np.random.RandomState(0)


def _logits(n, scale=4.0):
    return (scale * RNG.randn(n, sb.NUM_CLASSES)).astype(np.float32)


def test_class_names_match_dataset():
    from datasets.dso_dataset import _DSODataset
    assert sb.CLASSES == tuple(_DSODataset.METAINFO['classes'])
    print('test_class_names_match_dataset passed')


def test_sampling():
    subsets = sb.sample_bipartitions(60, seed=0)
    assert len(subsets) == 60 and len(set(subsets)) == 60
    assert subsets[0] == (0, 1, 2, 3, 4)
    assert sb.partition_name(subsets[0]) == 's0.1.2.3.4'
    for subset in subsets:
        assert 1 <= len(subset) <= 12 and subset == tuple(sorted(subset))
        if len(subset) == 12:
            assert 0 in subset
    assert sb.sample_bipartitions(60, seed=0) == subsets  # deterministic
    assert sb.canonical(range(5, 24)) == (0, 1, 2, 3, 4)  # larger side flips
    assert sb.canonical([c for c in range(12, 24)]) == tuple(range(12))
    for bad in ([], list(range(24)), [1, 1]):
        try:
            sb.canonical(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{bad} accepted')
    print('test_sampling passed')


def test_scores_match_reference():
    z = _logits(400)
    subsets = [(3, ), (0, 1, 2, 3, 4), tuple(range(12)),
               (2, 5, 9, 13, 17, 21, 23)]
    a_mask = sb.subsets_to_mask(subsets)
    q = sb.base_quantities(z)
    scores = sb.bipartition_scores(q, a_mask)
    flat = sb.flat_scores(q)
    ref_flat = compute_ood_scores(torch.from_numpy(z))
    for key in sb.FLAT_KEYS:
        assert np.allclose(flat[key], ref_flat[key].numpy(), atol=1e-4), key
    for i, subset in enumerate(subsets):
        groups = [list(subset),
                  [c for c in range(sb.NUM_CLASSES) if c not in subset]]
        ref = compute_ood_scores(torch.from_numpy(z), class_groups=groups)
        for key in sb.HIER_KEYS:
            got, want = scores[key][i], ref[key].numpy()
            assert np.allclose(got, want, atol=1e-4), (
                key, subset, np.abs(got - want).max())
    # per-row binning of a block equals scalar binning of each row
    s = scores['gn_energy']
    lo, hi = s.min(axis=1).astype(np.float64), s.max(axis=1).astype(np.float64)
    hi[1] = lo[1]  # a constant row must not divide by zero
    for eps in (0, sb.BIN_EPS):
        block = sb.bin_index(s, lo, hi, 1000, eps)
        assert block.min() == 0 and block.max() == 999, eps
        for j in range(s.shape[0]):
            assert np.array_equal(
                block[j], sb.bin_index(s[j], lo[j], hi[j], 1000, eps)), j
    print('test_scores_match_reference passed')


def test_metrics_from_histograms_matches_binary_ood_metrics():
    scores = RNG.randn(5000).astype(np.float32)
    labels = RNG.rand(5000) < 0.2
    scores[labels] += 1.0
    want = binary_ood_metrics([scores], [labels], num_bins=4096)
    lo, hi = float(scores.min()), float(scores.max())
    b = sb.bin_index(scores, lo, hi, 4096, eps=0)
    got = metrics_from_histograms(np.bincount(b[labels], minlength=4096),
                                  np.bincount(b[~labels], minlength=4096))
    assert got == want, (got, want)
    print('test_metrics_from_histograms_matches_binary_ood_metrics passed')


def test_log_bins_resolve_confident_points():
    """Most ID points and >5% of the OOD points within 1e-5 of the minimum:
    equal-width bins put them in one bin (FPR@95 = 100%), log-spaced bins
    recover the exact value."""
    n_id, n_ood = 200000, 20000
    u_id = 10.0**RNG.uniform(-7.0, -4.0, n_id)
    u_ood = np.where(RNG.rand(n_ood) < 0.10,
                     10.0**RNG.uniform(-6.5, -5.5, n_ood),
                     10.0**RNG.uniform(-3.0, -0.4, n_ood))
    scores = (np.concatenate([u_id, u_ood]) - 1.0).astype(np.float32)
    labels = np.arange(n_id + n_ood) >= n_id
    lo, hi = float(scores.min()), float(scores.max())

    # exact values from the sorted scores (float32 ties get half credit)
    threshold = np.sort(scores[labels])[int(np.ceil(0.05 * n_ood)) - 1]
    exact_fpr = float((scores[~labels] >= threshold).mean())
    below = np.searchsorted(np.sort(scores[~labels]), scores[labels], 'left')
    upto = np.searchsorted(np.sort(scores[~labels]), scores[labels], 'right')
    exact_auroc = float(((below + upto) / 2).sum() / (n_id * n_ood))

    def metrics(eps):
        b = sb.bin_index(scores, lo, hi, 2**16, eps)
        return metrics_from_histograms(np.bincount(b[labels], minlength=2**16),
                                       np.bincount(b[~labels],
                                                   minlength=2**16))

    # the bins are symmetric: mirroring the scores mirrors the bins
    b = sb.bin_index(scores, lo, hi, 2**16, sb.BIN_EPS)
    mirrored = sb.bin_index(-scores, -hi, -lo, 2**16, sb.BIN_EPS)
    assert np.abs((2**16 - 1 - mirrored) - b).max() <= 1
    assert b.min() == 0 and b.max() == 2**16 - 1 and np.all(np.diff(
        b[np.argsort(scores, kind='stable')]) >= 0)  # monotone in the score

    linear, logged = metrics(0), metrics(sb.BIN_EPS)
    assert 0.2 < exact_fpr < 0.9, exact_fpr
    assert linear['fpr95'] > 0.99, linear  # saturated
    assert abs(logged['fpr95'] - exact_fpr) < 5e-3, (logged, exact_fpr)
    assert abs(logged['auroc'] - exact_auroc) < 1e-4, (logged, exact_auroc)
    assert abs(linear['auroc'] - exact_auroc) > abs(
        logged['auroc'] - exact_auroc)
    print('test_log_bins_resolve_confident_points passed')


def _write_dump(dump_dir, num_frames=3, n=3000):
    frames = []
    for i in range(num_frames):
        z = _logits(n)
        ood = RNG.rand(n) < 0.15
        z[ood] *= 0.3  # OOD points: flatter logits
        valid = RNG.rand(n) < 0.9
        np.savez(os.path.join(dump_dir, f'r0_{i:06d}.npz'),
                 logits=z.astype(np.float16), ood=ood, valid=valid,
                 mapped=np.zeros(n, np.int16), lidar_path='x')
        frames.append((z.astype(np.float16).astype(np.float32)[valid],
                       ood[valid]))
    return frames


def test_sweep_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        dump_dir = os.path.join(tmp, 'logits')
        os.makedirs(dump_dir)
        frames = _write_dump(dump_dir)
        out_dir = os.path.join(tmp, 'bipartitions')
        rows = sb.sweep(dump_dir, out_dir, num=6, seed=1, bins=2**20,
                        workers=1, chunk=4, top=3, bin_eps=0)
        assert set(rows) == set(sb.FLAT_KEYS) | {
            f'{sb.partition_name(s)}_{k}'
            for s in sb.sample_bipartitions(6, seed=1) for k in sb.HIER_KEYS}
        assert os.path.exists(os.path.join(out_dir, 'partitions.tsv'))
        assert os.path.exists(os.path.join(out_dir, 'partitions.json'))

        # the log is readable by the summariser and reproduces the rows
        log = os.path.join(out_dir, 'bipartitions.log')
        parsed = sb.parse_logs([log])
        assert len(parsed) == len(rows)

        # ... but the summariser refuses the GN family on it (the offline
        # GN MSP / GN Entropy rows are approximate); test.py logs are fine
        assert offline_sweep_logs([log]) == [log]
        check_family('group', [log])
        try:
            check_family('gn', [log])
        except ValueError as err:
            assert 'gn' in str(err) and log in str(err)
        else:
            raise AssertionError('--family gn accepted on a sweep log')
        online = os.path.join(tmp, 'online.log')
        with open(online, 'w') as fh:
            fh.write("work_dir = 'work_dirs/x'\n    gn_msp |  1.00 |  2.00 |  3.00\n")
        assert offline_sweep_logs([online]) == []
        check_family('gn', [online])
        for key, m in rows.items():
            assert parsed[key] == tuple(
                round(100 * m[k], 2) for k in ('auroc', 'ap', 'fpr95')), key

        # ... and matches the chunk-based metric on the same scores: the
        # flat scores bit-for-bit (bin_eps=0: same ranges, bins and
        # equal-width binning as the online metric); the
        # hierarchy scores up to BLAS accumulation order, which only shows
        # for the degenerate ODIN scores whose spread is at float32
        # resolution (their 2^20 bins are narrower than an ulp).
        subsets = sb.sample_bipartitions(6, seed=1)
        a_mask = sb.subsets_to_mask(subsets)
        labels = [ood for _, ood in frames]
        chunks = {k: [] for k in sb.HIER_KEYS + sb.FLAT_KEYS}
        for z, _ in frames:
            q = sb.base_quantities(z)
            for k, s in sb.flat_scores(q).items():
                chunks[k].append(s)
            for k, s in sb.bipartition_scores(q, a_mask).items():
                chunks[k].append(s)
        for key in sb.FLAT_KEYS:
            assert rows[key] == binary_ood_metrics(chunks[key], labels), key
        for i, subset in enumerate(subsets):
            name = sb.partition_name(subset)
            for key in sb.HIER_KEYS:
                want = binary_ood_metrics([s[i] for s in chunks[key]],
                                          labels)
                for k in ('auroc', 'ap', 'fpr95'):
                    assert abs(rows[f'{name}_{key}'][k] - want[k]) < 2e-3, (
                        name, key, k, rows[f'{name}_{key}'][k], want[k])
    print('test_sweep_end_to_end passed')


def test_torch_scores_match_numpy():
    sb.require_torch()
    z = _logits(300)
    a_mask = sb.subsets_to_mask([(3, ), (0, 1, 2, 3, 4), tuple(range(12)),
                                 (2, 5, 9, 13, 17, 21, 23)])
    q = sb.base_quantities(z)
    qt = sb.torch_base_quantities(torch.from_numpy(z))
    for key, s in sb.flat_scores(q).items():
        got = sb.torch_flat_scores(qt)[key].numpy()
        assert np.allclose(got, s, atol=1e-5), key
    want = sb.bipartition_scores(q, a_mask)
    got = sb.torch_bipartition_scores(qt, torch.from_numpy(a_mask))
    for key in sb.HIER_KEYS:
        assert got[key].shape == want[key].shape, key
        assert np.allclose(got[key].numpy(), want[key], atol=1e-5), (
            key, np.abs(got[key].numpy() - want[key]).max())
    # identical scores bin identically, per row and with a constant row
    s = want['gn_energy']
    lo, hi = s.min(axis=1).astype(np.float64), s.max(axis=1).astype(np.float64)
    hi[1] = lo[1]
    for eps in (0, sb.BIN_EPS):
        got_bins = sb.torch_bin_index(
            torch.from_numpy(s), torch.from_numpy(lo), torch.from_numpy(hi),
            1000, eps).numpy()
        want_bins = sb.bin_index(s, lo, hi, 1000, eps)
        if eps == 0:
            assert np.array_equal(got_bins, want_bins)
        else:  # log1p differs by an ulp between the libraries: bin edges
            assert np.abs(got_bins - want_bins).max() <= 1
            assert (got_bins == want_bins).mean() > 0.999
    if torch.cuda.is_available():
        # the sort-based CUDA count equals bincount (peaked, sparse, empty)
        index = torch.randint(0, 5000, (20000, ))
        index[:15000] = 7
        for idx in (index, index[:0]):
            got = sb.torch_counts(idx.cuda(), 5000).cpu()
            assert torch.equal(got, torch.bincount(idx, minlength=5000))
        # CUDA scores match numpy at float32 precision; TF32 matmuls (the
        # torch default, switched off by require_torch) are off by ~4e-4,
        # enough to move FPR@95 by tens of points on confident ID points.
        assert not torch.backends.cuda.matmul.allow_tf32
        zc = _logits(20000, scale=8.0)
        want = sb.bipartition_scores(sb.base_quantities(zc), a_mask)
        got = sb.torch_bipartition_scores(
            sb.torch_base_quantities(torch.from_numpy(zc).cuda()),
            torch.from_numpy(a_mask).cuda())
        for key in sb.HIER_KEYS:
            diff = np.abs(got[key].cpu().numpy() - want[key]).max()
            assert diff < 1e-5, (key, diff)
    print('test_torch_scores_match_numpy passed')


def test_backends_and_several_dump_dirs():
    import shutil
    with tempfile.TemporaryDirectory() as tmp:
        dump_dir = os.path.join(tmp, 'logits')
        os.makedirs(dump_dir)
        _write_dump(dump_dir, num_frames=4)
        kwargs = dict(num=5, seed=2, bins=2**14, workers=1, chunk=3, top=2)
        rows = sb.sweep(dump_dir, os.path.join(tmp, 'out'), **kwargs)

        # the same frames spread over two directories: identical rows
        for name in ('a', 'b'):
            os.makedirs(os.path.join(tmp, name))
        for i, fname in enumerate(sorted(os.listdir(dump_dir))):
            shutil.copy(os.path.join(dump_dir, fname),
                        os.path.join(tmp, 'ab'[i % 2], fname))
        split_rows = sb.sweep([os.path.join(tmp, 'b'), os.path.join(tmp, 'a')],
                              os.path.join(tmp, 'out_ab'), **kwargs)
        assert split_rows == rows
        with open(os.path.join(tmp, 'out_ab', 'bipartitions.log')) as fh:
            assert ' + ' in fh.read().splitlines()[1]  # both dirs recorded

        # the torch backend (here on CPU tensors) agrees with numpy
        torch_rows = sb.sweep(dump_dir, os.path.join(tmp, 'out_torch'),
                              backend='torch', device='cpu', **kwargs)
        assert set(torch_rows) == set(rows)
        for key, m in rows.items():
            for k in ('auroc', 'ap', 'fpr95'):
                assert abs(torch_rows[key][k] - m[k]) < 2e-3, (
                    key, k, torch_rows[key][k], m[k])

        for bad in ('numpi', 'cuda'):
            try:
                sb.sweep(dump_dir, os.path.join(tmp, 'x'), backend=bad,
                         **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(f'backend {bad!r} accepted')
    print('test_backends_and_several_dump_dirs passed')


if __name__ == '__main__':
    test_class_names_match_dataset()
    test_sampling()
    test_scores_match_reference()
    test_metrics_from_histograms_matches_binary_ood_metrics()
    test_log_bins_resolve_confident_points()
    test_sweep_end_to_end()
    test_torch_scores_match_numpy()
    test_backends_and_several_dump_dirs()
    print('ALL TESTS PASSED')
