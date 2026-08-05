"""Tests for tools/dataset_converters/dso_converter.py.

Run from the repo root:  conda run -n p3former python tests/test_dso_converter.py
"""
import os
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mmengine

from dataset_converters import dso_converter
from test_dso_loading import make_frame, write_ply

ALL_SEQUENCES = (dso_converter.TRAIN_SEQUENCES +
                 dso_converter.VAL_SEQUENCES + dso_converter.TEST_SEQUENCES)


def _fake_annotations(tmp, frames_per_seq):
    ann = os.path.join(tmp, 'annotations')
    for seq in ALL_SEQUENCES:
        os.makedirs(os.path.join(ann, seq))
        for i in range(frames_per_seq):
            write_ply(os.path.join(ann, seq, f'{i:08d}.ply'), make_frame())


def test_split_membership_and_structure():
    with tempfile.TemporaryDirectory() as tmp:
        _fake_annotations(tmp, frames_per_seq=3)
        real_totals = dso_converter.EXPECTED_TOTALS
        real_mini = dso_converter.MINI_NUM_FRAMES
        dso_converter.EXPECTED_TOTALS = {'train': 24, 'val': 3, 'test': 9}
        dso_converter.MINI_NUM_FRAMES = 2
        try:
            dso_converter.create_dso_info_file('dso', tmp)
        finally:
            dso_converter.EXPECTED_TOTALS = real_totals
            dso_converter.MINI_NUM_FRAMES = real_mini
        train = mmengine.load(os.path.join(tmp, 'dso_infos_train.pkl'))
        val = mmengine.load(os.path.join(tmp, 'dso_infos_val.pkl'))
        test = mmengine.load(os.path.join(tmp, 'dso_infos_test.pkl'))
        mini = mmengine.load(os.path.join(tmp, 'dso_infos_mini.pkl'))
    assert train['metainfo'] == {'DATASET': 'DSO'}
    assert len(train['data_list']) == 24
    assert len(val['data_list']) == 3 and len(test['data_list']) == 9
    assert len(mini['data_list']) == 2
    val_seqs = {e['sample_id'].split('/')[0] for e in val['data_list']}
    assert val_seqs == {'(2024-06-10) Chinatown Route 1 Day'}
    test_seqs = {e['sample_id'].split('/')[0] for e in test['data_list']}
    assert test_seqs == {'(2024-04-29) One-North Route 2 Day',
                         '(2025-03-02) Ubin Route 1',
                         '(2025-03-02) Ubin Route 3'}
    entry = train['data_list'][0]
    assert entry['lidar_points']['num_pts_feats'] == 4
    assert entry['lidar_points']['lidar_path'].startswith('annotations/')
    assert entry['lidar_points']['lidar_path'].endswith('.ply')
    assert entry['pts_panoptic_mask_path'] == \
        entry['lidar_points']['lidar_path']
    print('PASS test_split_membership_and_structure')


def test_missing_sequence_raises():
    with tempfile.TemporaryDirectory() as tmp:
        _fake_annotations(tmp, frames_per_seq=1)
        removed = dso_converter.VAL_SEQUENCES[0]
        os.remove(os.path.join(tmp, 'annotations', removed, '00000000.ply'))
        os.rmdir(os.path.join(tmp, 'annotations', removed))
        try:
            dso_converter.create_dso_info_file('dso', tmp)
        except RuntimeError as exc:
            assert removed in str(exc)
            print('PASS test_missing_sequence_raises')
            return
    raise AssertionError('missing split sequence did not raise')


def test_extra_sequence_ignored():
    """Extra dirs (e.g. newly added recordings) are ignored with a notice;
    the fixed 12-sequence split is authoritative."""
    with tempfile.TemporaryDirectory() as tmp:
        _fake_annotations(tmp, frames_per_seq=3)
        extra = os.path.join(tmp, 'annotations', '(2026-01-28) Cetran Run AM')
        os.makedirs(extra)
        write_ply(os.path.join(extra, '00000000.ply'), make_frame())
        real_totals = dso_converter.EXPECTED_TOTALS
        real_mini = dso_converter.MINI_NUM_FRAMES
        dso_converter.EXPECTED_TOTALS = {'train': 24, 'val': 3, 'test': 9}
        dso_converter.MINI_NUM_FRAMES = 2
        try:
            dso_converter.create_dso_info_file('dso', tmp)
        finally:
            dso_converter.EXPECTED_TOTALS = real_totals
            dso_converter.MINI_NUM_FRAMES = real_mini
        train = mmengine.load(os.path.join(tmp, 'dso_infos_train.pkl'))
    seqs = {e['sample_id'].split('/')[0] for e in train['data_list']}
    assert '(2026-01-28) Cetran Run AM' not in seqs
    assert len(train['data_list']) == 24
    print('PASS test_extra_sequence_ignored')


def test_wrong_frame_count_raises():
    with tempfile.TemporaryDirectory() as tmp:
        _fake_annotations(tmp, frames_per_seq=1)  # real totals expect 8474
        try:
            dso_converter.create_dso_info_file('dso', tmp)
        except RuntimeError as exc:
            assert 'expected' in str(exc)
            print('PASS test_wrong_frame_count_raises')
            return
    raise AssertionError('wrong frame totals did not raise')


if __name__ == '__main__':
    test_split_membership_and_structure()
    test_missing_sequence_raises()
    test_extra_sequence_ignored()
    test_wrong_frame_count_raises()
    print('ALL TESTS PASSED')
