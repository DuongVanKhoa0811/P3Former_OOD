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

    24 train classes in a fixed order (2026-08-14 class-set revision):
    9 things (car, bicycle, motorcycle, truck, bus, person, rider,
    traffic-sign, traffic-cone) then 15 stuff in ascending raw-id order
    (paved-road, unpaved-road, sidewalk, building, window,
    perimeter-barrier, other-barrier, overhead-bridge, gate,
    pole-like-object, drain, terrain, trunks, vegetation, obscurant);
    ``ignore_index`` is 24. Raw 17 (Stop) and 28 (Others) are reserved as
    future OOD classes; 29/30 are 2D-only.
    The raw->train mapping comes from ``metainfo['seg_label_mapping']`` in the
    config; every raw id not listed there falls to ignore (the SemanticKITTI
    variant zero-fills instead, which would silently map stray ids to car).
    """
    METAINFO = {
        'classes': ('car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
                    'rider', 'traffic-sign', 'traffic-cone', 'paved-road',
                    'unpaved-road', 'sidewalk', 'building', 'window',
                    'perimeter-barrier', 'other-barrier', 'overhead-bridge',
                    'gate', 'pole-like-object', 'drain', 'terrain', 'trunks',
                    'vegetation', 'obscurant'),
        'palette': [[125, 46, 141], [255, 127, 0], [255, 0, 0],
                    [118, 171, 47], [161, 19, 46], [216, 82, 24],
                    [236, 176, 31], [255, 170, 0], [255, 255, 0],
                    [190, 190, 0], [0, 255, 0], [0, 0, 255], [170, 0, 255],
                    [84, 84, 0], [84, 170, 0], [84, 255, 0], [170, 84, 0],
                    [170, 170, 0], [255, 84, 0], [0, 84, 127], [0, 170, 127],
                    [0, 255, 127], [84, 0, 127], [84, 255, 127]],
        'seg_valid_class_ids':
        (3, 7, 6, 4, 5, 1, 2, 19, 20, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18,
         21, 22, 23, 24, 27),
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
