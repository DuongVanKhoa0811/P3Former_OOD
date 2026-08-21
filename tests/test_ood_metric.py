"""Tests for evaluation/metrics/ood_metric.py (_OODPointMetric).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_metric.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics.ood_metric import _OODPointMetric

# Raw semantic ids: 10 (car -> train 0), 30 (person -> train 5),
# 52 (other-structure -> OOD), 99 (other-object -> OOD),
# 0 (unlabeled -> excluded), 1 (outlier -> excluded).
RAW_SEM = np.array([10, 30, 52, 99, 0, 1, 10, 52], dtype=np.int64)
MAPPED = np.array([0, 5, 19, 19, 19, 19, 0, 19], dtype=np.int64)
INSTANCE_BITS = np.array([7, 3, 0, 0, 0, 0, 8, 0], dtype=np.int64)
RAW_PANOPTIC = (INSTANCE_BITS << 16) | RAW_SEM
# expected: OOD mask [F,F,T,T,-,-,F,T], excluded points 4 and 5


def _sample(scores):
    pred = {f'ood_{key}': np.asarray(val, dtype=np.float32)
            for key, val in scores.items()}
    return {
        'pred_pts_seg': pred,
        'eval_ann_info': {
            'pts_instance_mask': RAW_PANOPTIC.copy(),
            'pts_semantic_mask': MAPPED.copy(),
        },
    }


def test_gt_derivation_and_perfect_scores():
    # OOD points (2, 3, 7) get high scores; excluded points (4, 5) get the
    # HIGHEST scores of all -- they must be dropped, else AUROC < 1.
    good = [0.0, 0.1, 5.0, 6.0, 99.0, 98.0, 0.2, 7.0]
    bad = [-s for s in good]
    metric = _OODPointMetric(
        score_keys=('msp', 'maxlogit', 'odin', 'energy'))
    metric.process({}, [_sample({'msp': good, 'maxlogit': bad,
                                 'odin': good, 'energy': good})])
    assert len(metric.results) == 1
    labels, scores = metric.results[0]
    assert labels.tolist() == [False, False, True, True, False, True]
    assert scores['msp'].shape == (6, )
    results = metric.compute_metrics(metric.results)
    assert results['msp_AUROC'] == 100.0, results
    assert results['msp_AP'] == 100.0
    assert results['msp_FPR95'] == 0.0
    assert results['maxlogit_AUROC'] == 0.0
    assert results['odin_AUROC'] == 100.0
    assert results['energy_AUROC'] == 100.0
    print('PASS test_gt_derivation_and_perfect_scores')


def test_accumulates_across_scans():
    metric = _OODPointMetric(score_keys=('msp', ))
    s = [0.0, 0.1, 5.0, 6.0, 9.0, 9.0, 0.2, 7.0]
    metric.process({}, [_sample({'msp': s})])
    metric.process({}, [_sample({'msp': s})])
    assert len(metric.results) == 2
    results = metric.compute_metrics(metric.results)
    assert results['msp_AUROC'] == 100.0
    print('PASS test_accumulates_across_scans')


def test_missing_score_key_message():
    metric = _OODPointMetric(score_keys=('msp', ))
    try:
        metric.process({}, [{
            'pred_pts_seg': {},
            'eval_ann_info': {
                'pts_instance_mask': RAW_PANOPTIC.copy(),
                'pts_semantic_mask': MAPPED.copy(),
            },
        }])
        raise AssertionError('expected KeyError')
    except KeyError as e:
        assert 'ood_cfg' in str(e), e
    print('PASS test_missing_score_key_message')


def test_default_score_keys_are_discovered():
    # The configs rely on the default: every ood_* key the model emits is
    # evaluated, in the canonical ALL_SCORE_KEYS order (unknown keys last),
    # fixed from the first processed sample.
    from p3former.utils.ood_scores import OOD_SCORE_KEYS
    metric = _OODPointMetric()
    assert metric.score_keys is None
    s = [0.0, 0.1, 5.0, 6.0, 9.0, 9.0, 0.2, 7.0]
    metric.process({}, [_sample({key: s for key in OOD_SCORE_KEYS})])
    assert metric.score_keys == OOD_SCORE_KEYS == (
        'msp', 'maxlogit', 'odin', 'energy', 'entropy')
    # Group keys are picked up in canonical order; unknown ones are appended.
    metric = _OODPointMetric()
    metric.process({}, [_sample({'gn_energy': s, 'zz_custom': s, 'msp': s,
                                 'group_msp': s, 'entropy': s})])
    assert metric.score_keys == ('msp', 'entropy', 'group_msp', 'gn_energy',
                                 'zz_custom'), metric.score_keys
    results = metric.compute_metrics(metric.results)
    assert results['zz_custom_AUROC'] == 100.0
    assert set(results) == {f'{k}_{m}' for k in metric.score_keys
                            for m in ('AUROC', 'AP', 'FPR95')}
    # With no ood_* key at all the error points at ood_cfg.
    metric = _OODPointMetric()
    try:
        metric.process({}, [{
            'pred_pts_seg': {'pts_semantic_mask': np.zeros(8)},
            'eval_ann_info': {
                'pts_instance_mask': RAW_PANOPTIC.copy(),
                'pts_semantic_mask': MAPPED.copy(),
            },
        }])
        raise AssertionError('expected KeyError')
    except KeyError as e:
        assert 'ood_cfg' in str(e), e
    print('PASS test_default_score_keys_are_discovered')


def test_length_mismatch_rejected():
    metric = _OODPointMetric(score_keys=('msp', ))
    try:
        metric.process({}, [_sample({'msp': [0.0, 1.0]})])
        raise AssertionError('expected ValueError')
    except ValueError:
        pass
    print('PASS test_length_mismatch_rejected')


if __name__ == '__main__':
    test_gt_derivation_and_perfect_scores()
    test_accumulates_across_scans()
    test_missing_score_key_message()
    test_default_score_keys_are_discovered()
    test_length_mismatch_rejected()
    print('ALL TESTS PASSED')
