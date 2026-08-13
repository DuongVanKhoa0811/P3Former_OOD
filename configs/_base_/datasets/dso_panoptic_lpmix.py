# 24-class panoptic segmentation on the DSO dataset (2026-08-14 class-set
# revision), loading points and panoptic labels natively from the annotation
# PLYs. Things are train ids 0-8 (incl. traffic-sign and traffic-cone, which
# carry real instance ids), stuff 9-23; following the MMDet3D convention the
# ignore class is the last one (24). Raw ids are DSO uint8 semantic codes.
# Raw 17 (Stop) and 28 (Others) are reserved as future OOD classes; raw 29
# (Sky) and 30 (Water Body) exist only in the 2D annotation.
dataset_type = '_DSODataset'
data_root = 'data/dso/'
class_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'rider',
    'traffic-sign', 'traffic-cone',
    'paved-road', 'unpaved-road', 'sidewalk', 'building', 'window',
    'perimeter-barrier', 'other-barrier', 'overhead-bridge', 'gate',
    'pole-like-object', 'drain', 'terrain', 'trunks', 'vegetation',
    'obscurant'
]
labels_map = {
    0: 24,  # Noise -> ignore
    1: 5,  # Person
    2: 6,  # Rider
    3: 0,  # Car
    4: 3,  # Truck
    5: 4,  # Bus
    6: 2,  # Motorcycle
    7: 1,  # Bicycle
    8: 9,  # Paved Road
    9: 10,  # Unpaved Road
    10: 11,  # Sidewalk
    11: 12,  # Building
    12: 13,  # Window
    13: 14,  # Perimeter Barrier
    14: 15,  # Other Barrier
    15: 16,  # Overhead Bridge
    16: 17,  # Gate
    17: 24,  # Stop -> ignore (future OOD class)
    18: 18,  # Pole-like Object
    19: 7,  # Traffic Sign (thing)
    20: 8,  # Traffic Cone (thing)
    21: 19,  # Drain
    22: 20,  # Terrain
    23: 21,  # Trunks
    24: 22,  # Vegetation
    27: 23,  # Obscurant
    28: 24,  # Others -> ignore (future OOD class)
    29: 24,  # Sky (2D-only) -> ignore
    30: 24,  # Water Body (2D-only) -> ignore
    255: 24,  # Unlabelled -> ignore
}

learning_map_inv = {  # train id -> raw id
    0: 3,  # car
    1: 7,  # bicycle
    2: 6,  # motorcycle
    3: 4,  # truck
    4: 5,  # bus
    5: 1,  # person
    6: 2,  # rider
    7: 19,  # traffic-sign
    8: 20,  # traffic-cone
    9: 8,  # paved-road
    10: 9,  # unpaved-road
    11: 10,  # sidewalk
    12: 11,  # building
    13: 12,  # window
    14: 13,  # perimeter-barrier
    15: 14,  # other-barrier
    16: 15,  # overhead-bridge
    17: 16,  # gate
    18: 18,  # pole-like-object
    19: 21,  # drain
    20: 22,  # terrain
    21: 23,  # trunks
    22: 24,  # vegetation
    23: 27,  # obscurant
    24: 0,  # ignore -> Noise
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
                    instance_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8],
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
            ignore_index=24,
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
            ignore_index=24,
            test_mode=True,
            backend_args=backend_args)),
)

val_dataloader = test_dataloader

val_evaluator = dict(
    type='_PanopticSegMetric',
    thing_class_inds=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    stuff_class_inds=[
        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    ],
    min_num_points=50,
    id_offset=2**16,
    dataset_type='dso',
    learning_map_inv=learning_map_inv)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
