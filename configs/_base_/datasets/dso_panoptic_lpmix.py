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
