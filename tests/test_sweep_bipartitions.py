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
    block = sb.bin_index(s, lo, hi, 1000)
    for j in range(s.shape[0]):
        assert np.array_equal(block[j],
                              sb.bin_index(s[j], lo[j], hi[j], 1000)), j
    print('test_scores_match_reference passed')


def test_metrics_from_histograms_matches_binary_ood_metrics():
    scores = RNG.randn(5000).astype(np.float32)
    labels = RNG.rand(5000) < 0.2
    scores[labels] += 1.0
    want = binary_ood_metrics([scores], [labels], num_bins=4096)
    lo, hi = float(scores.min()), float(scores.max())
    b = sb.bin_index(scores, lo, hi, 4096)
    got = metrics_from_histograms(np.bincount(b[labels], minlength=4096),
                                  np.bincount(b[~labels], minlength=4096))
    assert got == want, (got, want)
    print('test_metrics_from_histograms_matches_binary_ood_metrics passed')


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
                        workers=1, chunk=4, top=3)
        assert set(rows) == set(sb.FLAT_KEYS) | {
            f'{sb.partition_name(s)}_{k}'
            for s in sb.sample_bipartitions(6, seed=1) for k in sb.HIER_KEYS}
        assert os.path.exists(os.path.join(out_dir, 'partitions.tsv'))
        assert os.path.exists(os.path.join(out_dir, 'partitions.json'))

        # the log is readable by the summariser and reproduces the rows
        parsed = sb.parse_logs([os.path.join(out_dir, 'bipartitions.log')])
        assert len(parsed) == len(rows)
        for key, m in rows.items():
            assert parsed[key] == tuple(
                round(100 * m[k], 2) for k in ('auroc', 'ap', 'fpr95')), key

        # ... and matches the chunk-based metric on the same scores: the
        # flat scores bit-for-bit (same ranges and bin count); the
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


if __name__ == '__main__':
    test_class_names_match_dataset()
    test_sampling()
    test_scores_match_reference()
    test_metrics_from_histograms_matches_binary_ood_metrics()
    test_sweep_end_to_end()
    print('ALL TESTS PASSED')
