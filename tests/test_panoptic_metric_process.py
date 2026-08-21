"""_PanopticSegMetric.process keeps only the two predicted masks.

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_panoptic_metric_process.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics.panoptic_seg_metric import _PanopticSegMetric


def test_process_keeps_only_masks():
    metric = _PanopticSegMetric(
        thing_class_inds=[0],
        stuff_class_inds=[1],
        min_num_points=5,
        id_offset=2**16,
        dataset_type='semantickitti')
    sample = {
        'pred_pts_seg': {
            'pts_semantic_mask': np.array([0, 1, 1]),
            'pts_instance_mask': torch.tensor([1, 0, 0]),
            # OOD scores ride along in pred_pts_seg but must not be stored.
            'ood_msp': np.array([-0.9, -0.5, -0.1], dtype=np.float32),
            'ood_group_msp': np.array([-0.9, -0.5, -0.1], dtype=np.float32),
        },
        'eval_ann_info': {
            'pts_semantic_mask': np.array([0, 1, 1]),
            'pts_instance_mask': np.array([(3 << 16) | 10, 40, 40]),
        },
    }
    metric.process({}, [sample])
    assert len(metric.results) == 1
    eval_ann, pred = metric.results[0]
    assert set(pred) == {'pts_semantic_mask', 'pts_instance_mask'}, pred.keys()
    assert isinstance(pred['pts_instance_mask'], np.ndarray)  # tensor -> numpy
    assert pred['pts_instance_mask'].tolist() == [1, 0, 0]
    assert pred['pts_semantic_mask'].tolist() == [0, 1, 1]
    assert eval_ann is sample['eval_ann_info']
    print('PASS test_process_keeps_only_masks')


if __name__ == '__main__':
    test_process_keeps_only_masks()
    print('ALL TESTS PASSED')
