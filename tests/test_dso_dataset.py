"""Tests for datasets/dso_dataset.py.

Run from the repo root:  conda run -n p3former python tests/test_dso_dataset.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmengine
from mmengine.registry import init_default_scope

init_default_scope('mmdet3d')

import datasets.transforms.dso_loading  # noqa: F401  register transform
from datasets.dso_dataset import _DSODataset

# sibling import via the script dir (site-packages ships a shadowing top-level
# 'tests' package, so 'from tests.test_dso_loading import ...' won't resolve)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dso_loading import make_frame, write_ply  # noqa: E402

CLASS_NAMES = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'rider',
    'traffic-sign', 'traffic-cone',
    'paved-road', 'unpaved-road', 'sidewalk', 'building', 'window',
    'perimeter-barrier', 'other-barrier', 'overhead-bridge', 'gate',
    'pole-like-object', 'drain', 'terrain', 'trunks', 'vegetation',
    'obscurant'
]
LABELS_MAP = {
    0: 24, 1: 5, 2: 6, 3: 0, 4: 3, 5: 4, 6: 2, 7: 1, 8: 9, 9: 10, 10: 11,
    11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 24, 18: 18, 19: 7,
    20: 8, 21: 19, 22: 20, 23: 21, 24: 22, 27: 23, 28: 24, 29: 24, 30: 24,
    255: 24,
}
PIPELINE = [
    dict(type='_LoadDSOPointsAndAnnotations'),
    dict(type='PointSegClassMapping'),
]


def _make_data_root(tmp):
    seq_dir = os.path.join(tmp, 'annotations', 'seqA')
    os.makedirs(seq_dir)
    write_ply(os.path.join(seq_dir, '00000000.ply'), make_frame())
    rel = os.path.join('annotations', 'seqA', '00000000.ply')
    infos = dict(
        metainfo=dict(DATASET='DSO'),
        data_list=[{
            'lidar_points': {'lidar_path': rel, 'num_pts_feats': 4},
            'pts_panoptic_mask_path': rel,
            'sample_id': 'seqA/00000000',
        }])
    mmengine.dump(infos, os.path.join(tmp, 'dso_infos_tiny.pkl'))


def _build(tmp, test_mode):
    return _DSODataset(
        data_root=tmp,
        ann_file='dso_infos_tiny.pkl',
        pipeline=PIPELINE,
        metainfo=dict(classes=CLASS_NAMES, seg_label_mapping=LABELS_MAP,
                      max_label=255),
        ignore_index=24,
        test_mode=test_mode)


def test_seg_label_mapping():
    with tempfile.TemporaryDirectory() as tmp:
        _make_data_root(tmp)
        ds = _build(tmp, test_mode=False)
    m = ds.seg_label_mapping
    assert m.shape == (256,)
    expected = {3: 0, 7: 1, 6: 2, 4: 3, 5: 4, 1: 5, 2: 6, 19: 7, 20: 8,
                8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16,
                16: 17, 18: 18, 21: 19, 22: 20, 23: 21, 24: 22, 27: 23}
    for raw, train in expected.items():
        assert m[raw] == train, (raw, m[raw], train)
    # explicit ignores AND never-listed ids all fall to 24
    for raw in (0, 17, 28, 29, 30, 255, 25, 26, 200):
        assert m[raw] == 24, (raw, m[raw])
    print('PASS test_seg_label_mapping')


def test_train_item_mapped():
    with tempfile.TemporaryDirectory() as tmp:
        _make_data_root(tmp)
        ds = _build(tmp, test_mode=False)
        assert len(ds) == 1
        results = ds.prepare_data(0)
    # raw sems [3, 3, 1, 8, 19, 200] -> train [0, 0, 5, 9, 7, 24]
    np.testing.assert_array_equal(results['pts_semantic_mask'],
                                  [0, 0, 5, 9, 7, 24])
    np.testing.assert_array_equal(
        results['pts_instance_mask'],
        [(7 << 16) | 3, (7 << 16) | 3, (12 << 16) | 1, 8, (9 << 16) | 19,
         200])
    assert results['dataset'] is ds  # LaserMix/PolarMix need this handle
    print('PASS test_train_item_mapped')


def test_eval_ann_info_mapped():
    with tempfile.TemporaryDirectory() as tmp:
        _make_data_root(tmp)
        ds = _build(tmp, test_mode=True)
        results = ds.prepare_data(0)
    eval_ann = results['eval_ann_info']
    # PointSegClassMapping must have overwritten the eval semantic mask
    np.testing.assert_array_equal(eval_ann['pts_semantic_mask'],
                                  [0, 0, 5, 9, 7, 24])
    np.testing.assert_array_equal(eval_ann['pts_instance_mask'],
                                  results['pts_instance_mask'])
    print('PASS test_eval_ann_info_mapped')


if __name__ == '__main__':
    test_seg_label_mapping()
    test_train_item_mapped()
    test_eval_ann_info_mapped()
    print('ALL TESTS PASSED')
