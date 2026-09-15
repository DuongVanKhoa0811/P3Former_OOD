"""Tests for the logits dump path and the distribution plot tool.

Covers evaluation/metrics/ood_logits_dump.py (_OODLogitsDumpMetric),
the 'logits' special case of _P3Former.postprocess_result, and the offline
recomputation in tools/plot_ood_score_distributions.py.

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_logits_dump.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

from evaluation.metrics.ood_logits_dump import _OODLogitsDumpMetric
from p3former.utils.ood_scores import compute_ood_scores
import plot_ood_score_distributions as pod

RNG = np.random.RandomState(0)
NUM_CLASSES = 24
# DSO-style GT: raw 17 (Stop) / 28 (Others) are OOD, mapped 24 is ignore.
RAW_SEM = np.array([3, 1, 17, 28, 0, 29, 8, 17], dtype=np.int64)
MAPPED = np.array([0, 5, 24, 24, 24, 24, 9, 24], dtype=np.int64)
RAW_PANOPTIC = (np.arange(8, dtype=np.int64) << 16) | RAW_SEM
OOD = np.isin(RAW_SEM, [17, 28])
VALID = (MAPPED != 24) | OOD  # excludes points 4 (Noise) and 5 (Sky)


def _sample(logits, lidar_path='seq/frame.bin'):
    return dict(
        pred_pts_seg=dict(
            sem_logits=logits, ood_msp=RNG.rand(len(MAPPED))),
        eval_ann_info=dict(
            pts_instance_mask=RAW_PANOPTIC.copy(),
            pts_semantic_mask=MAPPED.copy()),
        lidar_path=lidar_path)


def test_dump_metric():
    logits = RNG.randn(8, NUM_CLASSES).astype(np.float16)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, 'logits')
        metric = _OODLogitsDumpMetric(
            out_dir=out_dir, ood_raw_ids=[17, 28], seg_offset=2**16,
            ignore_index=24)
        metric.process({}, [_sample(logits), _sample(2.0 * logits, 'b.bin')])
        results = metric.compute_metrics(metric.results)
        assert results == dict(frames=2, points=16), results

        files = sorted(os.listdir(out_dir))
        assert files == ['r0_000000.npz', 'r0_000001.npz'], files
        with np.load(os.path.join(out_dir, files[0])) as data:
            assert data['logits'].dtype == np.float16
            assert np.array_equal(data['logits'], logits)
            assert np.array_equal(data['ood'], OOD)
            assert np.array_equal(data['valid'], VALID)
            assert np.array_equal(data['mapped'], MAPPED)
            assert str(data['lidar_path']) == 'seq/frame.bin'

        # a second dump into the same directory must refuse to mix frames
        try:
            _OODLogitsDumpMetric(out_dir=out_dir, ignore_index=24)
        except FileExistsError:
            pass
        else:
            raise AssertionError('stale dump dir accepted')

        # missing sem_logits points at save_logits
        metric2 = _OODLogitsDumpMetric(
            out_dir=os.path.join(tmp, 'other'), ignore_index=24)
        bad = _sample(logits)
        del bad['pred_pts_seg']['sem_logits']
        try:
            metric2.process({}, [bad])
        except KeyError as err:
            assert 'save_logits' in str(err)
        else:
            raise AssertionError('missing sem_logits accepted')
    print('test_dump_metric passed')


def test_postprocess_keeps_logits_out_of_scores():
    from mmdet3d.structures import Det3DDataSample
    from p3former.segmentors.p3former import _P3Former

    scores = dict(
        msp=RNG.rand(8).astype(np.float32),
        logits=RNG.randn(8, NUM_CLASSES).astype(np.float16))
    samples = _P3Former.postprocess_result(
        None, [MAPPED.copy()], [RAW_PANOPTIC.copy()], [Det3DDataSample()],
        [scores])
    seg = samples[0].pred_pts_seg
    keys = set(seg.keys())
    assert 'sem_logits' in keys and 'ood_msp' in keys, keys
    assert 'ood_logits' not in keys, keys
    assert np.array_equal(seg.sem_logits, scores['logits'])
    # without save_logits nothing extra appears
    samples = _P3Former.postprocess_result(
        None, [MAPPED.copy()], [RAW_PANOPTIC.copy()], [Det3DDataSample()],
        [dict(msp=scores['msp'])])
    assert 'sem_logits' not in set(samples[0].pred_pts_seg.keys())
    print('test_postprocess_keeps_logits_out_of_scores passed')


def test_parse_hierarchy():
    groups = pod.parse_hierarchy('p_v_hgcno')
    assert [label for label, _ in groups] == ['v', 'hgcno']
    assert groups[0][1] == [0, 1, 2, 3, 4]
    covered = sorted(i for _, ids in groups for i in ids)
    assert covered == list(range(NUM_CLASSES)), covered
    assert len(pod.parse_hierarchy(pod.CURRENT)) == 6

    for bad in ('v_hgcno', 'p_', 'p_v__h', 'p_v_x', 'p_v_vh'):
        try:
            pod.parse_hierarchy(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{bad!r} accepted')
    print('test_parse_hierarchy passed')


def test_offline_recomputation_matches_compute_ood_scores():
    logits16 = (3.0 * RNG.randn(500, NUM_CLASSES)).astype(np.float16)
    ood = RNG.rand(500) < 0.3
    groups = pod.parse_hierarchy('p_vh_gn_co')
    probs = pod.softmax(logits16)
    score, claimed = pod.group_msp(probs, groups)

    ref = compute_ood_scores(
        torch.from_numpy(logits16.astype(np.float32)),
        class_groups=[ids for _, ids in groups])['group_msp'].numpy()
    assert np.allclose(score, ref, atol=1e-5), np.abs(score - ref).max()

    hier = pod.HierarchyDistributions('p_vh_gn_co', bins=64)
    hier.update(probs, ood)
    # per-group histogram mass = points claimed by that group, per label
    for lab, mask in ((0, ~ood), (1, ood)):
        counts = np.bincount(claimed[mask], minlength=len(groups))
        assert np.array_equal(hier.hist[lab].sum(axis=1), counts)
    # combined row = sum of the group rows = all points
    assert hier.hist.sum() == 500
    metrics = hier.finalise()
    assert set(metrics) == {'auroc', 'ap', 'fpr95'}
    print('test_offline_recomputation_matches_compute_ood_scores passed')


if __name__ == '__main__':
    test_dump_metric()
    test_postprocess_keeps_logits_out_of_scores()
    test_parse_hierarchy()
    test_offline_recomputation_matches_compute_ood_scores()
    print('ALL TESTS PASSED')
