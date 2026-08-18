# Copyright (c) OpenMMLab. All rights reserved.
"""Generate DSO info pkls (native annotation-PLY layout).

Expects ``<save_path>/annotations`` to (sym)link the DSO ``Annotation_Final``
directory: 12 sequence dirs of ``*.ply`` frames that hold points and panoptic
labels together. Splits are fixed by sequence name. The mini split is a small
subset of one train sequence, used only for smoke tests.
"""
from os import path as osp
from pathlib import Path

import mmengine

# Annotation-directory sequence names. The user's split was specified in raw
# LiDAR_INS naming (hyphenated, "Pulau Ubin RouteN"); it maps 1:1 onto these.
VAL_SEQUENCES = ['(2024-06-10) Chinatown Route 1 Day']
TEST_SEQUENCES = [
    '(2024-04-29) One-North Route 2 Day',
    '(2025-03-02) Ubin Route 1',
    '(2025-03-02) Ubin Route 3',
]
TRAIN_SEQUENCES = [
    '(2024-04-29) One-North Route 1 Day',
    '(2024-04-29) One-North Route 1 Night',
    '(2024-05-13) One-North Route 2 Rain',
    '(2024-05-13) Science Park Rain',
    '(2024-05-17) Science Park Day',
    '(2024-06-10) Chinatown Route 2 Day',
    '(2024-06-10) Chinatown Route 2 Night',
    '(2025-02-27) Tiong Bahru Rerun AM',
]
# Optional Cetran AV-test-centre sequences (added to Annotation_Final after
# the main 12; verified non-overlapping by manual review of the per-frame
# PNGs — the 01-28 "AM-clean" re-export shares some frame stems with "AM"
# but covers different content). They feed the optional 'cetran' and
# 'test_cetran' sets, never train/val, and are only generated when all
# three dirs are present.
CETRAN_SEQUENCES = [
    '(2026-01-28) Cetran Run AM',
    '(2026-01-28) Cetran Run AM-clean',
    '(2026-02-04) Cetran Run AM',
]
EXPECTED_TOTALS = {
    'train': 8474,
    'val': 401,
    'test': 2625,
    'cetran': 980,
    'test_cetran': 3605,
}
MINI_SEQUENCE = '(2024-04-29) One-North Route 1 Day'
MINI_NUM_FRAMES = 32


def _frame_infos(ann_dir, sequence, limit=None):
    frames = sorted((ann_dir / sequence).glob('*.ply'))
    if not frames:
        raise FileNotFoundError(f'no .ply frames under {ann_dir / sequence}')
    if limit is not None:
        frames = frames[:limit]
    infos = []
    for frame in frames:
        rel = osp.join('annotations', sequence, frame.name)
        infos.append({
            'lidar_points': {
                'lidar_path': rel,
                'num_pts_feats': 4
            },
            'pts_panoptic_mask_path': rel,
            'sample_id': f'{sequence}/{frame.stem}',
        })
    return infos


def create_dso_info_file(pkl_prefix, save_path):
    """Create train/val/test/mini info files for the DSO dataset.

    Args:
        pkl_prefix (str): Prefix of the info files to be generated.
        save_path (str): Directory holding the ``annotations`` (sym)link;
            the pkls are written here.
    """
    save_path = Path(save_path)
    ann_dir = save_path / 'annotations'
    if not ann_dir.is_dir():
        raise FileNotFoundError(
            f'{ann_dir} missing; symlink it to the DSO Annotation_Final dir')
    found = {p.name for p in ann_dir.iterdir() if p.is_dir()}
    expected = set(TRAIN_SEQUENCES + VAL_SEQUENCES + TEST_SEQUENCES)
    missing = sorted(expected - found)
    if missing:
        raise RuntimeError(
            f'missing split sequences under {ann_dir}: {missing}')
    extra = sorted(found - expected - set(CETRAN_SEQUENCES))
    if extra:
        # The 12-sequence split is fixed; new recordings dropped into the
        # annotation dir are not silently absorbed into any split.
        print(f'ignoring {len(extra)} sequence dir(s) outside the fixed '
              f'split: {extra}')

    splits = {
        'train': TRAIN_SEQUENCES,
        'val': VAL_SEQUENCES,
        'test': TEST_SEQUENCES,
    }
    for split, sequences in splits.items():
        _write_split(ann_dir, save_path, pkl_prefix, split, sequences)

    mini_list = _frame_infos(ann_dir, MINI_SEQUENCE, limit=MINI_NUM_FRAMES)
    infos = dict(metainfo=dict(DATASET='DSO'), data_list=mini_list)
    filename = save_path / f'{pkl_prefix}_infos_mini.pkl'
    mmengine.dump(infos, filename)
    print(f'DSO info mini ({len(mini_list)} frames) is saved to {filename}')

    missing_cetran = sorted(set(CETRAN_SEQUENCES) - found)
    if missing_cetran:
        print('skipping cetran/test_cetran pkls: missing sequence dirs '
              f'{missing_cetran}')
    else:
        _write_split(ann_dir, save_path, pkl_prefix, 'cetran',
                     CETRAN_SEQUENCES)
        _write_split(ann_dir, save_path, pkl_prefix, 'test_cetran',
                     TEST_SEQUENCES + CETRAN_SEQUENCES)


def _write_split(ann_dir, save_path, pkl_prefix, split, sequences):
    """Collect the frames of one split, validate the total, dump the pkl."""
    data_list = []
    for sequence in sequences:
        data_list.extend(_frame_infos(ann_dir, sequence))
    if len(data_list) != EXPECTED_TOTALS[split]:
        raise RuntimeError(f'{split}: found {len(data_list)} frames, '
                           f'expected {EXPECTED_TOTALS[split]}')
    infos = dict(metainfo=dict(DATASET='DSO'), data_list=data_list)
    filename = save_path / f'{pkl_prefix}_infos_{split}.pkl'
    mmengine.dump(infos, filename)
    print(f'DSO info {split} ({len(data_list)} frames) '
          f'is saved to {filename}')
