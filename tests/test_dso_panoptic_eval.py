"""Tests for the 'dso' branch of evaluation/functional/panoptic_seg_eval.py.

Run from the repo root:
    conda run -n p3former python tests/test_dso_panoptic_eval.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.functional.panoptic_seg_eval import EvalPanoptic


def _evaluator():
    return EvalPanoptic(
        classes=['car', 'road'],
        thing_classes=['car'],
        stuff_classes=['road'],
        include=[0, 1],
        dataset_type='dso',
        min_num_points=5,
        id_offset=2**16,
        label2cat={0: 'car', 1: 'road'},
        ignore_index=2,
        logger=None)


def test_perfect_prediction():
    n = 40
    gt_sem = np.zeros(n, dtype=int)          # train id 0 = car
    gt_sem[20:] = 1                          # train id 1 = road
    gt_inst = np.zeros(n, dtype=np.int64)
    gt_inst[:10] = (5 << 16) | 3             # car instance 5 (raw sem 3)
    gt_inst[10:20] = (6 << 16) | 3           # car instance 6
    gt_inst[20:] = 8                         # road, instance bits zeroed
    pred_sem = gt_sem.copy()
    pred_inst = np.zeros(n, dtype=np.int64)
    pred_inst[:10] = 101
    pred_inst[10:20] = 102                   # matching split of car points
    ret = _evaluator().evaluate(
        [{'pts_semantic_mask': gt_sem, 'pts_instance_mask': gt_inst}],
        [{'pts_semantic_mask': pred_sem, 'pts_instance_mask': pred_inst}])
    assert abs(ret['pq'] - 1.0) < 1e-6, ret['pq']
    assert abs(ret['miou'] - 1.0) < 1e-6, ret['miou']
    print('PASS test_perfect_prediction')


def test_merged_raw_classes_form_one_segment():
    """One GT object whose points carry two different raw semantics that map
    to the same train class (a class-merge setup). The 'dso' re-pack replaces
    the raw low bits with the train id, fusing them into ONE GT segment; the
    single matching prediction then has IoU 1. Without the branch the GT
    splits into two segments and the prediction (IoU 0.5 each) matches
    neither."""
    n = 20
    gt_sem = np.zeros(n, dtype=int)                   # all train id 0
    gt_inst = np.empty(n, dtype=np.int64)
    gt_inst[:10] = (5 << 16) | 3                      # instance 5, raw sem 3
    gt_inst[10:] = (5 << 16) | 7                      # instance 5, raw sem 7
    pred_sem = np.zeros(n, dtype=int)
    pred_inst = np.full(n, 42, dtype=np.int64)        # one predicted segment
    ev = _evaluator()
    ev.evaluate(
        [{'pts_semantic_mask': gt_sem, 'pts_instance_mask': gt_inst}],
        [{'pts_semantic_mask': pred_sem, 'pts_instance_mask': pred_inst}])
    assert ev.pan_tp[0] == 1, ev.pan_tp
    assert ev.pan_fn[0] == 0 and ev.pan_fp[0] == 0
    print('PASS test_merged_raw_classes_form_one_segment')


if __name__ == '__main__':
    test_perfect_prediction()
    test_merged_raw_classes_form_one_segment()
    print('ALL TESTS PASSED')
