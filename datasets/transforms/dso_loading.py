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
