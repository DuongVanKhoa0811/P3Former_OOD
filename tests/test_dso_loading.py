"""Tests for datasets/transforms/dso_loading.py.

Run from the repo root:  conda run -n p3former python tests/test_dso_loading.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.transforms.dso_loading import (DSO_THING_RAW_IDS,
                                             dso_points_and_masks,
                                             read_dso_ply)

# Mirrors the real annotation PLY record (27 B/point).
DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                  ('intensity', '<f4'), ('device_id', '<f4'),
                  ('semantic', 'u1'), ('instance', '<u2'), ('isVisible', 'u1'),
                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

HEADER = (b'ply\n'
          b'format binary_little_endian 1.0\n'
          b'element vertex %d\n'
          b'property float x\nproperty float y\nproperty float z\n'
          b'property float intensity\nproperty float device_id\n'
          b'property uchar semantic\nproperty ushort instance\n'
          b'property uchar isVisible\n'
          b'property uchar red\nproperty uchar green\nproperty uchar blue\n'
          b'end_header\n')


def write_ply(path, arr):
    with open(path, 'wb') as fh:
        fh.write(HEADER % len(arr))
        arr.tofile(fh)


def make_frame():
    """6 points: 2 car points (one instance), 1 person, 1 road, 1 traffic-sign,
    1 unknown raw id — road/sign/unknown carry bogus instance ids that must be
    dropped by the stuff-zeroing rule."""
    arr = np.zeros(6, dtype=DTYPE)
    arr['x'] = [1, 2, 3, 4, 5, 6]
    arr['y'] = [0, 1, -1, 2, -2, 3]
    arr['z'] = [0.5, -1.0, 2.0, 3.0, 0.0, 10.0]
    arr['intensity'] = [255.0, 51.0, 0.0, 102.0, 255.0, 25.5]
    arr['device_id'] = 71.0
    arr['semantic'] = [3, 3, 1, 8, 19, 200]
    arr['instance'] = [7, 7, 12, 4, 9, 33]
    return arr


def test_read_roundtrip():
    arr = make_frame()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'frame.ply')
        write_ply(path, arr)
        data = read_dso_ply(path)
    assert data.shape == (6,)
    for field in ('x', 'y', 'z', 'intensity', 'semantic', 'instance'):
        np.testing.assert_array_equal(data[field], arr[field])
    print('PASS test_read_roundtrip')


def test_size_mismatch_raises():
    arr = make_frame()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'frame.ply')
        write_ply(path, arr)
        with open(path, 'rb') as fh:
            blob = fh.read()
        with open(path, 'wb') as fh:
            fh.write(blob[:-5])  # truncate
        try:
            read_dso_ply(path)
        except ValueError as exc:
            assert 'size mismatch' in str(exc)
            print('PASS test_size_mismatch_raises')
            return
    raise AssertionError('truncated PLY did not raise ValueError')


def test_points_and_masks():
    arr = make_frame()
    points, sem, packed = dso_points_and_masks(arr)
    assert points.shape == (6, 4) and points.dtype == np.float32
    np.testing.assert_allclose(points[:, 3],
                               arr['intensity'] / 255.0, rtol=1e-6)
    assert sem.dtype == np.int64 and packed.dtype == np.int64
    np.testing.assert_array_equal(sem, [3, 3, 1, 8, 19, 200])
    # things keep their instance ids
    assert packed[0] == (7 << 16) | 3
    assert packed[1] == (7 << 16) | 3
    assert packed[2] == (12 << 16) | 1
    # stuff / unknown classes have instance bits zeroed
    assert packed[3] == 8
    assert packed[4] == 19
    assert packed[5] == 200
    assert set(np.asarray(DSO_THING_RAW_IDS).tolist()) == {1, 2, 3, 4, 5, 6, 7}
    print('PASS test_points_and_masks')


def test_no_norm_intensity():
    arr = make_frame()
    points, _, _ = dso_points_and_masks(arr, norm_intensity=False)
    np.testing.assert_allclose(points[:, 3], arr['intensity'], rtol=1e-6)
    print('PASS test_no_norm_intensity')


def test_transform_synthetic():
    from datasets.transforms.dso_loading import _LoadDSOPointsAndAnnotations
    from mmdet3d.structures.points import LiDARPoints
    arr = make_frame()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'frame.ply')
        write_ply(path, arr)
        loader = _LoadDSOPointsAndAnnotations()
        results = loader.transform({'lidar_path': path})
    assert isinstance(results['points'], LiDARPoints)
    assert tuple(results['points'].tensor.shape) == (6, 4)
    np.testing.assert_array_equal(results['pts_semantic_mask'],
                                  [3, 3, 1, 8, 19, 200])
    np.testing.assert_array_equal(
        results['pts_instance_mask'],
        [(7 << 16) | 3, (7 << 16) | 3, (12 << 16) | 1, 8, 19, 200])
    print('PASS test_transform_synthetic')


def test_transform_eval_ann_and_no_ann():
    from datasets.transforms.dso_loading import _LoadDSOPointsAndAnnotations
    arr = make_frame()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'frame.ply')
        write_ply(path, arr)
        # test mode: dataset created an empty eval_ann_info dict
        results = _LoadDSOPointsAndAnnotations().transform(
            {'lidar_path': path, 'eval_ann_info': {}})
        assert 'pts_semantic_mask' in results['eval_ann_info']
        assert 'pts_instance_mask' in results['eval_ann_info']
        np.testing.assert_array_equal(
            results['eval_ann_info']['pts_semantic_mask'],
            results['pts_semantic_mask'])
        # with_ann=False: points only
        results = _LoadDSOPointsAndAnnotations(with_ann=False).transform(
            {'lidar_path': path})
        assert 'points' in results
        assert 'pts_semantic_mask' not in results
        assert 'pts_instance_mask' not in results
    print('PASS test_transform_eval_ann_and_no_ann')


REAL_SEQ = ('/mnt/ssd/khoadv/projects/OOD_PanSeg_3D/data/DSO_Dataset/'
            'Annotation_Final/(2024-04-29) One-North Route 1 Day')


def test_transform_real_frame():
    if not os.path.isdir(REAL_SEQ):
        print('SKIP test_transform_real_frame (data not mounted)')
        return
    from datasets.transforms.dso_loading import _LoadDSOPointsAndAnnotations
    path = sorted(os.path.join(REAL_SEQ, f) for f in os.listdir(REAL_SEQ)
                  if f.endswith('.ply'))[0]
    results = _LoadDSOPointsAndAnnotations().transform({'lidar_path': path})
    pts = results['points'].tensor.numpy()
    sem = results['pts_semantic_mask']
    packed = results['pts_instance_mask']
    assert pts.shape[0] > 100_000 and pts.shape[1] == 4
    assert 0.0 < pts[:, 3].max() <= 1.0
    # semantic histogram must match an independent re-read of the raw file
    raw = read_dso_ply(path)
    np.testing.assert_array_equal(np.bincount(sem, minlength=256),
                                  np.bincount(raw['semantic'], minlength=256))
    # packing invariants
    np.testing.assert_array_equal(packed & 0xFFFF, sem)
    thing = np.isin(sem, DSO_THING_RAW_IDS)
    assert (packed[~thing] >> 16 == 0).all()
    # thing instances survive: count unique (instance, class) pairs two ways
    n_inst_loader = len(np.unique(packed[thing & (packed >> 16 != 0)]))
    raw_inst = raw['instance'].astype(np.int64)
    raw_thing = np.isin(raw['semantic'], DSO_THING_RAW_IDS) & (raw_inst != 0)
    n_inst_raw = len(np.unique((raw_inst[raw_thing] << 16)
                               | raw['semantic'][raw_thing].astype(np.int64)))
    assert n_inst_loader == n_inst_raw
    print(f'PASS test_transform_real_frame ({pts.shape[0]:,} pts, '
          f'{n_inst_raw} thing instances)')


if __name__ == '__main__':
    test_read_roundtrip()
    test_size_mismatch_raises()
    test_points_and_masks()
    test_no_norm_intensity()
    test_transform_synthetic()
    test_transform_eval_ann_and_no_ann()
    test_transform_real_frame()
    print('ALL TESTS PASSED')
