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


if __name__ == '__main__':
    test_read_roundtrip()
    test_size_mismatch_raises()
    test_points_and_masks()
    test_no_norm_intensity()
    print('ALL TESTS PASSED')
