# P3Former on DSO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate P3Former panoptic segmentation on the DSO dataset (16 classes, native annotation-PLY loading), per spec `docs/superpowers/specs/2026-08-05-dso-dataset-design.md`.

**Architecture:** New namespace-package modules (`datasets/transforms/dso_loading.py`, `datasets/dso_dataset.py`, `tools/dataset_converters/dso_converter.py`), a one-word extension to the panoptic evaluator, and two configs mirroring the SemanticKITTI ones with DSO-specific values (16 classes, cylindrical z-range [−3, 13], LaserMix pitch [−12, 45]). No model-code changes.

**Tech Stack:** Python 3.8, torch 1.10.1+cu111, mmengine 0.7.4, mmdet3d 1.1.0 (OpenMMLab registry/config system), numpy structured arrays for PLY IO.

## Global Constraints

- Conda env `p3former` for every python command: `conda run -n p3former python …` (or an activated shell). Plain `python3` is only for numpy-only scripts.
- **All commands run from the repo root** (`/home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD`); custom modules import by path.
- GPU work uses `CUDA_VISIBLE_DEVICES=1` — GPU 0 belongs to another job. CPU-only checks need no prefix.
- Never write into `/mnt/ssd/.../DSO_Dataset/Annotation_Final` (read-only source data). Generated pkls live in `data/dso/` which is gitignored (`data/` rule) — never commit data files.
- A new module only takes effect if registered (`@TRANSFORMS/…register_module()`) **and** listed in the top-level config's `custom_imports` (CLAUDE.md registry gotcha).
- Class order is fixed: things 0–6 = car, bicycle, motorcycle, truck, bus, person, rider; stuff 7–15 = road, sidewalk, building, fence, vegetation, trunk, terrain, pole, traffic-sign; `ignore_index=16`; raw thing ids = {1..7}.
- Instance packing everywhere: `(instance << 16) | raw_semantic`, int64, instance bits zeroed for non-thing raw classes.
- Tests are plain-python scripts under `tests/` (repo has no pytest setup): `python tests/<file>.py` must end with `ALL TESTS PASSED`.
- Work on branch `dso-dataset`. Commit after each task with the message given in the task.

---

### Task 1: DSO PLY reading + mask building (pure functions)

**Files:**
- Create: `datasets/transforms/dso_loading.py`
- Test: `tests/test_dso_loading.py`

**Interfaces:**
- Consumes: nothing (leaf module; numpy only for these functions).
- Produces:
  - `DSO_THING_RAW_IDS: tuple = (1, 2, 3, 4, 5, 6, 7)`
  - `read_dso_ply(path: str) -> np.ndarray` — structured array with fields x, y, z, intensity, device_id, semantic, instance, isVisible, red, green, blue; raises `ValueError` on bad magic/format/size.
  - `dso_points_and_masks(data: np.ndarray, norm_intensity: bool = True) -> (points N×4 float32, sem N int64, packed N int64)`
  - Test module exports reused by later tests: `DTYPE`, `HEADER`, `write_ply(path, arr)`, `make_frame() -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dso_loading.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n p3former python tests/test_dso_loading.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'datasets.transforms.dso_loading'`

- [ ] **Step 3: Write the pure functions**

Create `datasets/transforms/dso_loading.py` (no `__init__.py` anywhere — `datasets` is a namespace package):

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Native loading of DSO annotation PLY frames.

Each annotated DSO frame is one binary-little-endian PLY holding the point
cloud and its panoptic labels together, with a fixed 27-byte per-point record:

    float32 x, y, z, intensity, device_id
    uint8   semantic            raw class id (0-255)
    uint16  instance            instance id, 0 = no instance
    uint8   isVisible
    uint8   red, green, blue

The transform at the bottom replaces the usual LoadPointsFromFile +
_LoadAnnotations3D pair, reading the file once.
"""
import os
from typing import Tuple

import numpy as np
from mmcv.transforms.base import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import get_points_type

# Raw semantic ids of the 7 thing classes (person, rider, car, truck, bus,
# motorcycle, bicycle). Instance ids of every other class are dropped so that
# a stuff class forms exactly one GT segment per frame (traffic-sign carries
# ~17 real instance ids per frame in the raw labels).
DSO_THING_RAW_IDS = (1, 2, 3, 4, 5, 6, 7)

_PLY_TO_NUMPY = {
    b'int8': 'i1', b'char': 'i1',
    b'uint8': 'u1', b'uchar': 'u1',
    b'int16': 'i2', b'short': 'i2',
    b'uint16': 'u2', b'ushort': 'u2',
    b'int32': 'i4', b'int': 'i4',
    b'uint32': 'u4', b'uint': 'u4',
    b'float32': 'f4', b'float': 'f4',
    b'float64': 'f8', b'double': 'f8',
}


def read_dso_ply(path: str) -> np.ndarray:
    """Read one DSO annotation PLY into a structured numpy array.

    Validates the magic, the binary-little-endian format and that the file
    size matches the header's vertex count before reading.
    """
    with open(path, 'rb') as fh:
        if b'ply' not in fh.readline():
            raise ValueError(f'{path}: not a PLY file')
        fmt = fh.readline().split()[1]
        if fmt != b'binary_little_endian':
            raise ValueError(
                f'{path}: expected binary_little_endian, got {fmt.decode()}')
        num = None
        props = []
        element = None
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f'{path}: header ended without end_header')
            tok = line.split()
            if not tok or tok[0] == b'comment':
                continue
            if tok[0] == b'end_header':
                break
            if tok[0] == b'element':
                element = tok[1]
                if element == b'vertex':
                    num = int(tok[2])
            elif tok[0] == b'property' and element == b'vertex':
                props.append((tok[2].decode(), '<' + _PLY_TO_NUMPY[tok[1]]))
        if num is None:
            raise ValueError(f'{path}: header has no vertex element')
        dtype = np.dtype(props)
        expected = fh.tell() + num * dtype.itemsize
        actual = os.path.getsize(path)
        if actual != expected:
            raise ValueError(
                f'{path}: size mismatch, header implies {expected} B '
                f'({num} points x {dtype.itemsize} B), file is {actual} B')
        data = np.fromfile(fh, dtype=dtype, count=num)
    for field in ('x', 'y', 'z', 'intensity', 'semantic', 'instance'):
        if field not in data.dtype.names:
            raise ValueError(f'{path}: missing {field!r} property')
    return data


def dso_points_and_masks(
        data: np.ndarray,
        norm_intensity: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a structured DSO frame into model inputs and panoptic masks.

    Returns:
        points (float32, N x 4): x, y, z, intensity (/255 when norm_intensity).
        pts_semantic_mask (int64, N): raw uint8 semantic ids.
        pts_instance_mask (int64, N): ``(instance << 16) | raw_semantic``
            (SemanticKITTI packing, id_offset 2**16), with instance bits
            zeroed for every non-thing class.
    """
    intensity = data['intensity'].astype(np.float32)
    if norm_intensity:
        intensity = intensity / 255.0
    points = np.stack(
        [data['x'], data['y'], data['z'], intensity], axis=-1).astype(
            np.float32)
    sem = data['semantic'].astype(np.int64)
    inst = data['instance'].astype(np.int64)
    inst[~np.isin(sem, DSO_THING_RAW_IDS)] = 0
    packed = (inst << 16) | sem
    return points, sem, packed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n p3former python tests/test_dso_loading.py`
Expected: 4 PASS lines then `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add datasets/transforms/dso_loading.py tests/test_dso_loading.py
git commit -m "Add DSO annotation-PLY reader and panoptic mask packing"
```

---

### Task 2: `_LoadDSOPointsAndAnnotations` transform

**Files:**
- Modify: `datasets/transforms/dso_loading.py` (append the transform class)
- Test: `tests/test_dso_loading.py` (append tests)

**Interfaces:**
- Consumes: `read_dso_ply`, `dso_points_and_masks` (Task 1).
- Produces: registered TRANSFORMS class `_LoadDSOPointsAndAnnotations(with_ann: bool = True, norm_intensity: bool = True, coord_type: str = 'LIDAR')`. `transform(results)` reads `results['lidar_path']`, sets `results['points']` (LiDARPoints N×4), and when `with_ann`: `results['pts_semantic_mask']` (int64 raw ids), `results['pts_instance_mask']` (int64 packed), mirroring both into `results['eval_ann_info']` when that key exists.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_dso_loading.py` (above the `__main__` block) and add the three calls to the `__main__` block before the final print:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n p3former python tests/test_dso_loading.py`
Expected: FAIL with `ImportError: cannot import name '_LoadDSOPointsAndAnnotations'`

- [ ] **Step 3: Append the transform class**

Append to `datasets/transforms/dso_loading.py`:

```python
@TRANSFORMS.register_module()
class _LoadDSOPointsAndAnnotations(BaseTransform):
    """Load points and panoptic labels from one DSO annotation PLY.

    Replaces the ``LoadPointsFromFile`` + ``_LoadAnnotations3D`` pair: the PLY
    at ``results['lidar_path']`` already holds both. Emits:

    - ``points``: 4-dim LiDARPoints (x, y, z, intensity).
    - ``pts_semantic_mask`` (when ``with_ann``): raw uint8 semantic ids as
      int64, mapped to train ids later by ``PointSegClassMapping``.
    - ``pts_instance_mask`` (when ``with_ann``): ``(instance << 16) | raw
      semantic`` as int64, instance bits zeroed for non-thing classes.

    Both masks are mirrored into ``eval_ann_info`` when present (test mode),
    matching the behaviour of ``_LoadAnnotations3D``.

    Args:
        with_ann (bool): Whether to produce the two masks. Defaults to True.
        norm_intensity (bool): Divide intensity by 255 into (0, 1].
            Defaults to True.
        coord_type (str): Point coordinate frame. Defaults to 'LIDAR'.
    """

    def __init__(self,
                 with_ann: bool = True,
                 norm_intensity: bool = True,
                 coord_type: str = 'LIDAR') -> None:
        self.with_ann = with_ann
        self.norm_intensity = norm_intensity
        self.coord_type = coord_type

    def transform(self, results: dict) -> dict:
        data = read_dso_ply(results['lidar_path'])
        points, sem, packed = dso_points_and_masks(
            data, norm_intensity=self.norm_intensity)
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None)
        if self.with_ann:
            results['pts_semantic_mask'] = sem
            results['pts_instance_mask'] = packed
            if 'eval_ann_info' in results:
                results['eval_ann_info']['pts_semantic_mask'] = sem
                results['eval_ann_info']['pts_instance_mask'] = packed
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(with_ann={self.with_ann}, '
                f'norm_intensity={self.norm_intensity}, '
                f'coord_type={self.coord_type})')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n p3former python tests/test_dso_loading.py`
Expected: 7 PASS lines (incl. `test_transform_real_frame` with a 6-figure point count) then `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add datasets/transforms/dso_loading.py tests/test_dso_loading.py
git commit -m "Add _LoadDSOPointsAndAnnotations transform"
```

---

### Task 3: `_DSODataset` dataset class

**Files:**
- Create: `datasets/dso_dataset.py`
- Test: `tests/test_dso_dataset.py`

**Interfaces:**
- Consumes: `_LoadDSOPointsAndAnnotations` (Task 2), test helpers `DTYPE/write_ply/make_frame` from `tests/test_dso_loading.py`.
- Produces: registered DATASETS class `_DSODataset(Seg3DDataset)` with the 16-class METAINFO; `get_seg_label_mapping(metainfo) -> np.ndarray` (256 entries, default `ignore_index`); `parse_data_info` joining `pts_panoptic_mask_path`. Config contract: `metainfo=dict(classes=…16 names…, seg_label_mapping=<dict>, max_label=255)`, `ignore_index=16`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dso_dataset.py`:

```python
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
from tests.test_dso_loading import make_frame, write_ply

CLASS_NAMES = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'rider',
    'road', 'sidewalk', 'building', 'fence', 'vegetation', 'trunk',
    'terrain', 'pole', 'traffic-sign'
]
LABELS_MAP = {
    0: 16, 1: 5, 2: 6, 3: 0, 4: 3, 5: 4, 6: 2, 7: 1, 8: 7, 9: 16, 10: 8,
    11: 9, 12: 16, 13: 16, 14: 10, 15: 16, 16: 16, 17: 16, 18: 14, 19: 15,
    20: 16, 21: 16, 22: 13, 23: 12, 24: 11, 27: 16, 28: 16, 255: 16,
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
        ignore_index=16,
        test_mode=test_mode)


def test_seg_label_mapping():
    with tempfile.TemporaryDirectory() as tmp:
        _make_data_root(tmp)
        ds = _build(tmp, test_mode=False)
    m = ds.seg_label_mapping
    assert m.shape == (256,)
    expected = {3: 0, 7: 1, 6: 2, 4: 3, 5: 4, 1: 5, 2: 6, 8: 7, 10: 8,
                11: 9, 14: 10, 24: 11, 23: 12, 22: 13, 18: 14, 19: 15}
    for raw, train in expected.items():
        assert m[raw] == train, (raw, m[raw], train)
    # explicit ignores AND never-listed ids all fall to 16
    for raw in (0, 9, 12, 13, 15, 16, 17, 20, 21, 27, 28, 255, 25, 26, 200):
        assert m[raw] == 16, (raw, m[raw])
    print('PASS test_seg_label_mapping')


def test_train_item_mapped():
    with tempfile.TemporaryDirectory() as tmp:
        _make_data_root(tmp)
        ds = _build(tmp, test_mode=False)
        assert len(ds) == 1
        results = ds.prepare_data(0)
    # raw sems [3, 3, 1, 8, 19, 200] -> train [0, 0, 5, 7, 15, 16]
    np.testing.assert_array_equal(results['pts_semantic_mask'],
                                  [0, 0, 5, 7, 15, 16])
    np.testing.assert_array_equal(
        results['pts_instance_mask'],
        [(7 << 16) | 3, (7 << 16) | 3, (12 << 16) | 1, 8, 19, 200])
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
                                  [0, 0, 5, 7, 15, 16])
    np.testing.assert_array_equal(eval_ann['pts_instance_mask'],
                                  results['pts_instance_mask'])
    print('PASS test_eval_ann_info_mapped')


if __name__ == '__main__':
    test_seg_label_mapping()
    test_train_item_mapped()
    test_eval_ann_info_mapped()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n p3former python tests/test_dso_dataset.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'datasets.dso_dataset'`

- [ ] **Step 3: Write the dataset class**

Create `datasets/dso_dataset.py`:

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""DSO dataset for panoptic segmentation (native annotation-PLY loading)."""
from os import path as osp
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.datasets.seg3d_dataset import Seg3DDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class _DSODataset(Seg3DDataset):
    r"""DSO Dataset.

    16 train classes in a fixed order: 7 things (car, bicycle, motorcycle,
    truck, bus, person, rider) then 9 stuff (road, sidewalk, building, fence,
    vegetation, trunk, terrain, pole, traffic-sign); ``ignore_index`` is 16.
    The raw->train mapping comes from ``metainfo['seg_label_mapping']`` in the
    config; every raw id not listed there falls to ignore (the SemanticKITTI
    variant zero-fills instead, which would silently map stray ids to car).
    """
    METAINFO = {
        'classes': ('car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
                    'rider', 'road', 'sidewalk', 'building', 'fence',
                    'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'),
        'palette': [[125, 46, 141], [255, 127, 0], [255, 0, 0],
                    [118, 171, 47], [161, 19, 46], [216, 82, 24],
                    [236, 176, 31], [190, 190, 0], [0, 0, 255],
                    [170, 0, 255], [84, 255, 0], [84, 0, 127], [0, 255, 127],
                    [0, 170, 127], [255, 84, 0], [255, 170, 0]],
        'seg_valid_class_ids':
        (3, 7, 6, 4, 5, 1, 2, 8, 10, 11, 14, 24, 23, 22, 18, 19),
        'seg_all_class_ids':
        tuple(range(256)),
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='',
                     img='',
                     pts_instance_mask='',
                     pts_semantic_mask='',
                     pts_panoptic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = None,
                 scene_idxs: Optional[Union[str, np.ndarray]] = None,
                 test_mode: bool = False,
                 **kwargs) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            test_mode=test_mode,
            **kwargs)

    def get_seg_label_mapping(self, metainfo):
        """Raw uint8 id -> train id lookup, defaulting to ignore_index."""
        seg_label_mapping = np.full(
            metainfo['max_label'] + 1, self.ignore_index, dtype=np.int64)
        for raw_id, train_id in metainfo['seg_label_mapping'].items():
            seg_label_mapping[raw_id] = train_id
        return seg_label_mapping

    def parse_data_info(self, info: dict) -> dict:
        """Join the panoptic mask path with its data prefix."""
        info = super().parse_data_info(info)
        if 'pts_panoptic_mask_path' in info:
            info['pts_panoptic_mask_path'] = \
                osp.join(self.data_prefix.get('pts_panoptic_mask', ''),
                         info['pts_panoptic_mask_path'])
        return info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n p3former python tests/test_dso_dataset.py`
Expected: 3 PASS lines then `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add datasets/dso_dataset.py tests/test_dso_dataset.py
git commit -m "Add _DSODataset with ignore-defaulted label mapping"
```

---

### Task 4: Info generator + real pkl generation

**Files:**
- Create: `tools/dataset_converters/dso_converter.py`
- Modify: `tools/create_data.py` (add `dso` branch)
- Test: `tests/test_dso_converter.py`

**Interfaces:**
- Consumes: filesystem layout `data/dso/annotations/<seq>/*.ply` (symlink created in Step 5).
- Produces: `create_dso_info_file(pkl_prefix: str, save_path: str) -> None` writing `<save_path>/<prefix>_infos_{train,val,test,mini}.pkl`; module constants `TRAIN_SEQUENCES`, `VAL_SEQUENCES`, `TEST_SEQUENCES`, `EXPECTED_TOTALS`, `MINI_SEQUENCE`, `MINI_NUM_FRAMES`. Info entry shape consumed by `_DSODataset` (Task 3): `{'lidar_points': {'lidar_path': 'annotations/<seq>/<frame>.ply', 'num_pts_feats': 4}, 'pts_panoptic_mask_path': <same>, 'sample_id': '<seq>/<stem>'}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dso_converter.py`:

```python
"""Tests for tools/dataset_converters/dso_converter.py.

Run from the repo root:  conda run -n p3former python tests/test_dso_converter.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

import mmengine

from dataset_converters import dso_converter
from tests.test_dso_loading import make_frame, write_ply

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n p3former python tests/test_dso_converter.py`
Expected: FAIL with `ImportError: cannot import name 'dso_converter'` (or ModuleNotFoundError)

- [ ] **Step 3: Write the converter and wire it into create_data.py**

Create `tools/dataset_converters/dso_converter.py`:

```python
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
EXPECTED_TOTALS = {'train': 8474, 'val': 401, 'test': 2625}
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
    extra = sorted(found - expected)
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

    mini_list = _frame_infos(ann_dir, MINI_SEQUENCE, limit=MINI_NUM_FRAMES)
    infos = dict(metainfo=dict(DATASET='DSO'), data_list=mini_list)
    filename = save_path / f'{pkl_prefix}_infos_mini.pkl'
    mmengine.dump(infos, filename)
    print(f'DSO info mini ({len(mini_list)} frames) is saved to {filename}')
```

Modify `tools/create_data.py` — add the import after the existing converter imports (line 6, `from dataset_converters import semantickitti_converter`):

```python
from dataset_converters import dso_converter
```

Add after `semantickitti_data_prep` (line 42-50):

```python
def dso_data_prep(info_prefix, out_dir):
    """Prepare the info files for the DSO dataset.

    Args:
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info files.
    """
    dso_converter.create_dso_info_file(info_prefix, out_dir)
```

Add a branch to the `__main__` dispatch, before the final `else` (line 117-121):

```python
    elif args.dataset == 'dso':
        dso_data_prep(info_prefix=args.extra_tag, out_dir=args.out_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n p3former python tests/test_dso_converter.py`
Expected: 4 PASS lines then `ALL TESTS PASSED`

- [ ] **Step 5: Create the data symlink and generate the real pkls**

```bash
mkdir -p data/dso
ln -sfn "/mnt/ssd/khoadv/projects/OOD_PanSeg_3D/data/DSO_Dataset/Annotation_Final" data/dso/annotations
conda run -n p3former python tools/create_data.py dso --root-path data/dso --out-dir data/dso --extra-tag dso
```

Expected output: four `DSO info <split> (<N> frames) is saved to …` lines with N = 8474, 401, 2625, 32 (plus an `ignoring …` notice for any sequence dirs outside the fixed split, e.g. the 2026 Cetran runs). (`--root-path` is accepted but unused; `--out-dir` drives everything, matching the semantickitti branch's style.)

- [ ] **Step 6: Verify the real pkls against the filesystem**

```bash
conda run -n p3former python - <<'EOF'
import mmengine, os
for split, expected in [('train', 8474), ('val', 401), ('test', 2625), ('mini', 32)]:
    infos = mmengine.load(f'data/dso/dso_infos_{split}.pkl')
    n = len(infos['data_list'])
    assert n == expected, (split, n, expected)
    sample = infos['data_list'][n // 2]
    path = os.path.join('data/dso', sample['lidar_points']['lidar_path'])
    assert os.path.isfile(path), path
    print(split, n, 'ok  e.g.', sample['sample_id'])
EOF
```

Expected: four `<split> <N> ok e.g. …` lines, no assertion errors.

- [ ] **Step 7: Commit**

```bash
git add tools/dataset_converters/dso_converter.py tools/create_data.py tests/test_dso_converter.py
git commit -m "Add DSO info generator with fixed sequence splits"
```

---

### Task 5: `'dso'` branch in the panoptic evaluator

**Files:**
- Modify: `evaluation/functional/panoptic_seg_eval.py:298`
- Test: `tests/test_dso_panoptic_eval.py`

**Interfaces:**
- Consumes: `EvalPanoptic` (existing).
- Produces: `add_panoptic_sample` re-packs GT ids for `dataset_type='dso'` exactly as for `'semantickitti'`: `gt_instances // id_offset * id_offset + gt_semantics` (train-id low bits). Config contract used later: `_PanopticSegMetric(dataset_type='dso', …)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dso_panoptic_eval.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n p3former python tests/test_dso_panoptic_eval.py`
Expected: `test_merged_raw_classes_form_one_segment` FAILS (`ev.pan_tp[0] == 0`) because `dataset_type='dso'` currently matches no re-pack branch. (`test_perfect_prediction` may pass — the re-pack is an identity there.)

- [ ] **Step 3: Extend the branch**

In `evaluation/functional/panoptic_seg_eval.py` change line 298:

```python
        elif self.dataset_type == 'semantickitti':
```

to:

```python
        elif self.dataset_type in ('semantickitti', 'dso'):
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n p3former python tests/test_dso_panoptic_eval.py`
Expected: 2 PASS lines then `ALL TESTS PASSED` (the evaluator also prints its metric table — that's normal).

- [ ] **Step 5: Commit**

```bash
git add evaluation/functional/panoptic_seg_eval.py tests/test_dso_panoptic_eval.py
git commit -m "Handle dataset_type 'dso' in panoptic GT re-packing"
```

---

### Task 6: DSO base dataset config

**Files:**
- Create: `configs/_base_/datasets/dso_panoptic_lpmix.py`
- Test: inline integration check (Step 2) — builds real train/val datasets, so Task 4's pkls must exist.

**Interfaces:**
- Consumes: `_DSODataset`, `_LoadDSOPointsAndAnnotations`, `_LaserMix`/`_PolarMix` (existing), pkls from Task 4.
- Produces: config variables `train_dataloader`, `val_dataloader`, `test_dataloader`, `val_evaluator`, `test_evaluator`, `labels_map`, `learning_map_inv`, `metainfo` for the top-level config (Task 7).

- [ ] **Step 1: Write the config**

Create `configs/_base_/datasets/dso_panoptic_lpmix.py`:

```python
# 16-class panoptic segmentation on the DSO dataset, loading points and
# panoptic labels natively from the annotation PLYs. Things are train ids
# 0-6, stuff 7-15; following the MMDet3D convention the ignore class is the
# last one (16). Raw ids are DSO uint8 semantic codes.
dataset_type = '_DSODataset'
data_root = 'data/dso/'
class_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'rider',
    'road', 'sidewalk', 'building', 'fence', 'vegetation', 'trunk',
    'terrain', 'pole', 'traffic-sign'
]
labels_map = {
    0: 16,  # miss label -> ignore
    1: 5,  # person
    2: 6,  # rider
    3: 0,  # car
    4: 3,  # truck
    5: 4,  # bus
    6: 2,  # motorcycle
    7: 1,  # bicycle
    8: 7,  # road
    9: 16,  # unpaved-road -> ignore
    10: 8,  # sidewalk
    11: 9,  # building
    12: 16,  # window -> ignore
    13: 16,  # net-fence -> ignore
    14: 10,  # fence
    15: 16,  # overhead-bridge -> ignore
    16: 16,  # gate -> ignore
    17: 16,  # bus-stop -> ignore
    18: 14,  # pole
    19: 15,  # traffic-sign
    20: 16,  # traffic-cone -> ignore
    21: 16,  # drain -> ignore
    22: 13,  # terrain
    23: 12,  # trunk
    24: 11,  # vegetation
    27: 16,  # airborne-raindrops -> ignore
    28: 16,  # barrier -> ignore
    255: 16,  # noise -> ignore
}

learning_map_inv = {  # train id -> raw id
    0: 3,  # car
    1: 7,  # bicycle
    2: 6,  # motorcycle
    3: 4,  # truck
    4: 5,  # bus
    5: 1,  # person
    6: 2,  # rider
    7: 8,  # road
    8: 10,  # sidewalk
    9: 11,  # building
    10: 14,  # fence
    11: 24,  # vegetation
    12: 23,  # trunk
    13: 22,  # terrain
    14: 18,  # pole
    15: 19,  # traffic-sign
    16: 0,  # ignore -> miss label
}

metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=255)

input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

pre_transform = [
    dict(type='_LoadDSOPointsAndAnnotations'),
    dict(type='PointSegClassMapping'),
]

train_pipeline = [
    dict(type='_LoadDSOPointsAndAnnotations'),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='_LaserMix',
                    num_areas=[3, 4, 5, 6],
                    # DSO pitch span (two merged LiDARs, tall structure):
                    # measured p1 ~ -8 deg, p99 ~ +36 deg.
                    pitch_angles=[-12, 45],
                    pre_transform=pre_transform,
                    prob=0.5)
            ],
            [
                dict(
                    type='_PolarMix',
                    instance_classes=[0, 1, 2, 3, 4, 5, 6],
                    swap_ratio=0.5,
                    rotate_paste_ratio=1.0,
                    pre_transform=pre_transform,
                    prob=0.5)
            ],
        ],
        prob=[0.2, 0.8]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
]

test_pipeline = [
    dict(type='_LoadDSOPointsAndAnnotations'),
    dict(type='PointSegClassMapping'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                pts='',
                img='',
                pts_instance_mask='',
                pts_semantic_mask='',
                pts_panoptic_mask=''),
            ann_file='dso_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            ignore_index=16,
            backend_args=backend_args)),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                pts='',
                img='',
                pts_instance_mask='',
                pts_semantic_mask='',
                pts_panoptic_mask=''),
            ann_file='dso_infos_val.pkl',
            pipeline=test_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            ignore_index=16,
            test_mode=True,
            backend_args=backend_args)),
)

val_dataloader = test_dataloader

val_evaluator = dict(
    type='_PanopticSegMetric',
    thing_class_inds=[0, 1, 2, 3, 4, 5, 6],
    stuff_class_inds=[7, 8, 9, 10, 11, 12, 13, 14, 15],
    min_num_points=50,
    id_offset=2**16,
    dataset_type='dso',
    learning_map_inv=learning_map_inv)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

- [ ] **Step 2: Integration check — build datasets and pull one sample through each pipeline**

```bash
conda run -n p3former python - <<'EOF'
import sys
sys.path.insert(0, '.')
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
init_default_scope('mmdet3d')
import datasets.dso_dataset            # noqa: F401
import datasets.transforms.dso_loading  # noqa: F401
import datasets.transforms.transforms_3d  # noqa: F401
from mmdet3d.registry import DATASETS

cfg = Config.fromfile('configs/_base_/datasets/dso_panoptic_lpmix.py')

train_ds = DATASETS.build(cfg.train_dataloader['dataset']['dataset'])
assert len(train_ds) == 8474, len(train_ds)
item = train_ds[0]  # full train pipeline incl. RandomChoice lp-mix
sample = item['data_samples']
pts = item['inputs']['points']
sem = sample.gt_pts_seg.pts_semantic_mask.numpy()
inst = sample.gt_pts_seg.pts_instance_mask.numpy()
assert pts.shape[1] == 4, pts.shape
assert pts.shape[0] == sem.shape[0] == inst.shape[0]
assert sem.min() >= 0 and sem.max() <= 16, (sem.min(), sem.max())
# stuff/ignore (train id >= 7): instance bits are 0 from the loader, or
# exactly 1000 for points contributed by LaserMix/PolarMix's mixed-in scan
# (the +1000<<16 collision-avoidance offset in transforms_3d).
hi = np.unique(inst[sem >= 7] >> 16)
assert set(hi.tolist()) <= {0, 1000}, hi
print('train sample ok:', pts.shape[0], 'points,',
      len(np.unique(inst[(sem <= 6) & (inst >> 16 != 0)])), 'thing instances')

val_ds = DATASETS.build(cfg.val_dataloader['dataset']['dataset'])
assert len(val_ds) == 401, len(val_ds)
vitem = val_ds[0]
vsample = vitem['data_samples']
eval_ann = vsample.eval_ann_info
assert eval_ann['pts_semantic_mask'].max() <= 16
assert eval_ann['pts_instance_mask'].dtype == np.int64
# no mixes at test time: the loader invariant must hold exactly
vsem = eval_ann['pts_semantic_mask']
vinst = eval_ann['pts_instance_mask']
assert (vinst[vsem >= 7] >> 16 == 0).all()
print('val sample ok:', vitem['inputs']['points'].shape[0], 'points')
print('CONFIG CHECK PASSED')
EOF
```

Expected: `train sample ok: …`, `val sample ok: …`, `CONFIG CHECK PASSED`. (First run takes ~1 min: full-size frames, and RandomChoice may run LaserMix/PolarMix which loads a second frame.)

- [ ] **Step 3: Commit**

```bash
git add configs/_base_/datasets/dso_panoptic_lpmix.py
git commit -m "Add DSO panoptic lp-mix base dataset config"
```

---

### Task 7: P3Former DSO top-level config

**Files:**
- Create: `configs/p3former/p3former_1xb2_3x_dso.py`
- Test: inline config + model-build check (Step 2).

**Interfaces:**
- Consumes: base configs (`dso_panoptic_lpmix.py` from Task 6, existing `_base_/models/p3former.py`, `_base_/default_runtime.py`), all registered modules.
- Produces: the config used by `train.py`/`test.py` in Tasks 8–10.

- [ ] **Step 1: Write the config**

Create `configs/p3former/p3former_1xb2_3x_dso.py`:

```python
_base_ = [
    '../_base_/datasets/dso_panoptic_lpmix.py',
    '../_base_/models/p3former.py',
    '../_base_/default_runtime.py'
]

# DSO cylindrical range: ground sits at z ~ -1.5..-0.8 and ~40% of labeled
# points (buildings, vegetation) lie above z = 2, so the SemanticKITTI range
# [-4, 2] covers only 54% of them. [-3, 13] covers ~95% with 0.5 m z-bins on
# the unchanged [480, 360, 32] grid; the preprocessor clamps the tails into
# the boundary bins.
point_cloud_range = [0, -3.14159265359, -3, 50, 3.14159265359, 13]

model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(point_cloud_range=point_cloud_range)),
    voxel_encoder=dict(
        feat_channels=[64, 128, 256, 256],
        in_channels=6,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    backbone=dict(
        input_channels=16,
        base_channels=32,
        more_conv=True,
        out_channels=256),
    decode_head=dict(
        num_classes=17,
        num_decoder_layers=6,
        num_queries=128,
        embed_dims=256,
        point_cloud_range=point_cloud_range,
        cls_channels=(256, 256, 17),
        mask_channels=(256, 256, 256, 256, 256),
        thing_class=[0, 1, 2, 3, 4, 5, 6],
        stuff_class=[7, 8, 9, 10, 11, 12, 13, 14, 15],
        ignore_index=16))

lr = 0.0008
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 32],
        gamma=0.2)
]

train_dataloader = dict(batch_size=2, )

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))

custom_imports = dict(
    imports=[
        'p3former.backbones.cylinder3d',
        'p3former.data_preprocessors.data_preprocessor',
        'p3former.decode_heads.p3former_head',
        'p3former.segmentors.p3former',
        'p3former.task_modules.samplers.mask_pseduo_sampler',
        'evaluation.metrics.panoptic_seg_metric',
        'datasets.dso_dataset',
        'datasets.transforms.dso_loading',
        'datasets.transforms.transforms_3d',
    ],
    allow_failed_imports=False)
```

- [ ] **Step 2: Config + model-build check**

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n p3former python - <<'EOF'
import sys
sys.path.insert(0, '.')
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings

cfg = Config.fromfile('configs/p3former/p3former_1xb2_3x_dso.py')
import_modules_from_strings(**cfg.custom_imports)
init_default_scope('mmdet3d')

pcr = [0, -3.14159265359, -3, 50, 3.14159265359, 13]
assert cfg.model.data_preprocessor.voxel_layer.point_cloud_range == pcr
assert cfg.model.data_preprocessor.voxel_layer.grid_shape == [480, 360, 32]
assert cfg.model.decode_head.point_cloud_range == pcr
assert cfg.model.decode_head.num_classes == 17
assert cfg.model.decode_head.cls_channels == (256, 256, 17)
assert cfg.model.decode_head.thing_class == [0, 1, 2, 3, 4, 5, 6]
assert cfg.model.decode_head.stuff_class == [7, 8, 9, 10, 11, 12, 13, 14, 15]
assert cfg.model.decode_head.ignore_index == 16
assert cfg.train_dataloader.batch_size == 2
assert cfg.train_cfg.max_epochs == 36
assert cfg.val_evaluator.dataset_type == 'dso'

from mmdet3d.registry import MODELS
model = MODELS.build(cfg.model)
n_params = sum(p.numel() for p in model.parameters())
print(f'model built: {type(model).__name__}, {n_params/1e6:.1f}M params')
print('MODEL CONFIG CHECK PASSED')
EOF
```

Expected: `model built: _P3Former, …M params` and `MODEL CONFIG CHECK PASSED`.

- [ ] **Step 3: Commit**

```bash
git add configs/p3former/p3former_1xb2_3x_dso.py
git commit -m "Add P3Former DSO training config (1 GPU, 36 epochs)"
```

---

### Task 8: Smoke train on the mini split (GPU 1)

**Files:**
- No source changes — validates Tasks 1–7 end to end on real data.

**Interfaces:**
- Consumes: `configs/p3former/p3former_1xb2_3x_dso.py`, `data/dso/dso_infos_mini.pkl`.
- Produces: go/no-go for the full run, plus the working smoke command documented in the final report.

- [ ] **Step 1: Run one epoch on the 32-frame mini split**

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n p3former python train.py \
  configs/p3former/p3former_1xb2_3x_dso.py \
  --work-dir work_dirs/smoke_dso \
  --cfg-options \
    train_dataloader.dataset.dataset.ann_file=dso_infos_mini.pkl \
    val_dataloader.dataset.dataset.ann_file=dso_infos_mini.pkl \
    train_cfg.max_epochs=1
```

Note the nested `dataset.dataset` override (RepeatDataset wrapping — CLAUDE.md gotcha). Runtime ~5–10 min (16 train iters + 32-frame val). Run with a generous Bash timeout (600000 ms); if it exceeds that, re-run with `run_in_background` and poll the newest log under `work_dirs/smoke_dso/`.

Expected in output:
- `Epoch(train) [1][16/16]` with finite loss values (no `nan`/`inf`).
- A metric table `|        |   IoU   |   PQ   |   RQ   |  SQ   |` with an `all` row plus 16 class rows — values near 0 are fine after 16 iters.
- Process exits 0.

- [ ] **Step 2: If (and only if) it OOMs**

Apply fallbacks in this order, re-running Step 1 each time, and carry the working combination into Task 9:

1. `train_dataloader.batch_size=1 optim_wrapper.accumulative_counts=2` (keeps the effective batch at 2)
2. additionally pass `--amp`

- [ ] **Step 3: Confirm GPU memory headroom and clean up**

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv
rm -rf work_dirs/smoke_dso
```

Record the peak `memory.used` for GPU 1 from during the run if observed; the smoke work dir is disposable.

- [ ] **Step 4: Commit (checkpoint the plan progress only)**

No file changes to commit. Mark the task done in this plan and report the observed iteration time (s/iter from the log) — it calibrates the Task 9 duration estimate.

---

### Task 9: Full training run (GPU 1)

**Files:**
- No source changes. Output: `work_dirs/p3former_1xb2_3x_dso/`.

**Interfaces:**
- Consumes: everything above; the batch/AMP combination proven in Task 8.
- Produces: checkpoints `epoch_5.pth` … `epoch_36.pth`, per-epoch val PQ in the log, used by Task 10.

- [ ] **Step 1: Launch training in the background**

```bash
cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD
CUDA_VISIBLE_DEVICES=1 nohup conda run -n p3former python train.py \
  configs/p3former/p3former_1xb2_3x_dso.py \
  > work_dirs/dso_train_launch.log 2>&1 &
echo "launched, pid $!"
```

(If Task 8 needed the batch-1 fallback, append `--cfg-options train_dataloader.batch_size=1 optim_wrapper.accumulative_counts=2` — and `--amp` if that was needed too.)

- [ ] **Step 2: Verify healthy start (~10 min after launch)**

```bash
tail -n 30 work_dirs/dso_train_launch.log
grep -m 5 "Epoch(train)" work_dirs/p3former_1xb2_3x_dso/*/*.log 2>/dev/null | tail -n 5
```

Expected: iterations progressing, losses finite and broadly decreasing over the first ~200 iters (loss_cls + loss_mask + loss_dice + sem losses; initial total typically double-digit, trending down). No `nan`, no OOM traceback. At batch 2, one epoch is 4,237 iters; expect roughly 1.2–2 h/epoch → 2–3 days for 36 epochs.

- [ ] **Step 3: Monitor per-epoch val**

After each epoch the log contains the PQ table. Health criteria: `miou` and `pq` rise over the first epochs (mIoU after epoch 1 usually well above random; PQ_stuff leads PQ_things early). If loss goes NaN or PQ stays ~0 after 3 epochs, stop and fall back to the Cylinder3D pretraining recipe (spec §10) — that is a new plan, not an ad-hoc change.

- [ ] **Step 4: On completion**

Confirm `work_dirs/p3former_1xb2_3x_dso/epoch_36.pth` exists and note the best-val-PQ epoch from the log for Task 10.

---

### Task 10: Test-set evaluation (3 held-out sequences)

**Files:**
- No source changes. Output: final metric table.

**Interfaces:**
- Consumes: best checkpoint from Task 9 (default `epoch_36.pth`; substitute the best-val-PQ epoch if different).
- Produces: the reported DSO test PQ/RQ/SQ/mIoU baseline numbers.

- [ ] **Step 1: Evaluate on the held-out test sequences**

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n p3former python test.py \
  configs/p3former/p3former_1xb2_3x_dso.py \
  work_dirs/p3former_1xb2_3x_dso/epoch_36.pth \
  --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test.pkl
```

Expected: 2,625 frames evaluated (One-North Route 2 Day + the two rural Ubin routes — note the test split carries a deliberate urban→rural domain shift); final table with per-class IoU/PQ/RQ/SQ and the summary metrics (`pq`, `pq_dagger`, `miou`, `pq_things`, `pq_stuff`, …). DSO test infos contain labels, so this is a real metric run (no submission writing).

- [ ] **Step 2: Record results**

Copy the val (epoch of best PQ) and test tables into the final report to the user, alongside the config path and checkpoint path. Do not edit `DOCs.md` (user's personal log) unless asked.

---

## Self-review notes (completed)

- Spec coverage: §5 layout → Task 4 Step 5; §6.1 → Task 4; §6.2 → Tasks 1–2; §6.3 → Task 3; §6.4 → Task 5 + Task 6 evaluator block; §6.5 → Tasks 6–7; §9.1 → Task 4 Step 6; §9.2 → Task 2 real-frame test + Task 6 Step 2; §9.3 → Task 8; §9.4 → Task 9; §9.5 → Task 10; §10 fallbacks → Task 8 Step 2 / Task 9 Step 3.
- Names/types cross-checked: `read_dso_ply`, `dso_points_and_masks`, `DSO_THING_RAW_IDS`, `_LoadDSOPointsAndAnnotations`, `_DSODataset`, `create_dso_info_file`, config keys — consistent across tasks.
- No placeholders: every code step contains the full content.
