dataset_type = '_DSODataset'
data_root = 'data/dso/'
class_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'rider',
    'traffic-sign', 'traffic-cone', 'paved-road', 'unpaved-road', 'sidewalk',
    'building', 'window', 'perimeter-barrier', 'other-barrier',
    'overhead-bridge', 'gate', 'pole-like-object', 'drain', 'terrain',
    'trunks', 'vegetation', 'obscurant'
]
labels_map = dict({
    0: 24,
    1: 5,
    2: 6,
    3: 0,
    4: 3,
    5: 4,
    6: 2,
    7: 1,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
    17: 24,
    18: 18,
    19: 7,
    20: 8,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    27: 23,
    28: 24,
    29: 24,
    30: 24,
    255: 24
})
learning_map_inv = dict({
    0: 3,
    1: 7,
    2: 6,
    3: 4,
    4: 5,
    5: 1,
    6: 2,
    7: 19,
    8: 20,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 18,
    19: 21,
    20: 22,
    21: 23,
    22: 24,
    23: 27,
    24: 0
})
metainfo = dict(
    classes=[
        'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'rider',
        'traffic-sign', 'traffic-cone', 'paved-road', 'unpaved-road',
        'sidewalk', 'building', 'window', 'perimeter-barrier', 'other-barrier',
        'overhead-bridge', 'gate', 'pole-like-object', 'drain', 'terrain',
        'trunks', 'vegetation', 'obscurant'
    ],
    seg_label_mapping=dict({
        0: 24,
        1: 5,
        2: 6,
        3: 0,
        4: 3,
        5: 4,
        6: 2,
        7: 1,
        8: 9,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 14,
        14: 15,
        15: 16,
        16: 17,
        17: 24,
        18: 18,
        19: 7,
        20: 8,
        21: 19,
        22: 20,
        23: 21,
        24: 22,
        27: 23,
        28: 24,
        29: 24,
        30: 24,
        255: 24
    }),
    max_label=255)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None
pre_transform = [
    dict(type='_LoadDSOPointsAndAnnotations'),
    dict(type='PointSegClassMapping')
]
train_pipeline = [
    dict(type='_LoadDSOPointsAndAnnotations'),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomChoice',
        transforms=[[{
            'type':
            '_LaserMix',
            'num_areas': [3, 4, 5, 6],
            'pitch_angles': [-12, 45],
            'pre_transform': [{
                'type': '_LoadDSOPointsAndAnnotations'
            }, {
                'type': 'PointSegClassMapping'
            }],
            'prob':
            0.5
        }],
                    [{
                        'type':
                        '_PolarMix',
                        'instance_classes': [0, 1, 2, 3, 4, 5, 6, 7, 8],
                        'swap_ratio':
                        0.5,
                        'rotate_paste_ratio':
                        1.0,
                        'pre_transform': [{
                            'type':
                            '_LoadDSOPointsAndAnnotations'
                        }, {
                            'type': 'PointSegClassMapping'
                        }],
                        'prob':
                        0.5
                    }]],
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
        translation_std=[0.1, 0.1, 0.1]),
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
    batch_size=1,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='_DSODataset',
            data_root='data/dso/',
            data_prefix=dict(
                pts='',
                img='',
                pts_instance_mask='',
                pts_semantic_mask='',
                pts_panoptic_mask=''),
            ann_file='dso_infos_train.pkl',
            pipeline=[
                dict(type='_LoadDSOPointsAndAnnotations'),
                dict(type='PointSegClassMapping'),
                dict(
                    type='RandomChoice',
                    transforms=[[{
                        'type':
                        '_LaserMix',
                        'num_areas': [3, 4, 5, 6],
                        'pitch_angles': [-12, 45],
                        'pre_transform': [{
                            'type':
                            '_LoadDSOPointsAndAnnotations'
                        }, {
                            'type': 'PointSegClassMapping'
                        }],
                        'prob':
                        0.5
                    }],
                                [{
                                    'type':
                                    '_PolarMix',
                                    'instance_classes':
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                    'swap_ratio':
                                    0.5,
                                    'rotate_paste_ratio':
                                    1.0,
                                    'pre_transform': [{
                                        'type':
                                        '_LoadDSOPointsAndAnnotations'
                                    }, {
                                        'type':
                                        'PointSegClassMapping'
                                    }],
                                    'prob':
                                    0.5
                                }]],
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
                    translation_std=[0.1, 0.1, 0.1]),
                dict(
                    type='Pack3DDetInputs',
                    keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
            ],
            metainfo=dict(
                classes=[
                    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
                    'rider', 'traffic-sign', 'traffic-cone', 'paved-road',
                    'unpaved-road', 'sidewalk', 'building', 'window',
                    'perimeter-barrier', 'other-barrier', 'overhead-bridge',
                    'gate', 'pole-like-object', 'drain', 'terrain', 'trunks',
                    'vegetation', 'obscurant'
                ],
                seg_label_mapping=dict({
                    0: 24,
                    1: 5,
                    2: 6,
                    3: 0,
                    4: 3,
                    5: 4,
                    6: 2,
                    7: 1,
                    8: 9,
                    9: 10,
                    10: 11,
                    11: 12,
                    12: 13,
                    13: 14,
                    14: 15,
                    15: 16,
                    16: 17,
                    17: 24,
                    18: 18,
                    19: 7,
                    20: 8,
                    21: 19,
                    22: 20,
                    23: 21,
                    24: 22,
                    27: 23,
                    28: 24,
                    29: 24,
                    30: 24,
                    255: 24
                }),
                max_label=255),
            modality=dict(use_lidar=True, use_camera=False),
            ignore_index=24,
            backend_args=None)))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='_DSODataset',
            data_root='data/dso/',
            data_prefix=dict(
                pts='',
                img='',
                pts_instance_mask='',
                pts_semantic_mask='',
                pts_panoptic_mask=''),
            ann_file='dso_infos_cetran.pkl',
            pipeline=[
                dict(type='_LoadDSOPointsAndAnnotations'),
                dict(type='PointSegClassMapping'),
                dict(
                    type='Pack3DDetInputs',
                    keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
            ],
            metainfo=dict(
                classes=[
                    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
                    'rider', 'traffic-sign', 'traffic-cone', 'paved-road',
                    'unpaved-road', 'sidewalk', 'building', 'window',
                    'perimeter-barrier', 'other-barrier', 'overhead-bridge',
                    'gate', 'pole-like-object', 'drain', 'terrain', 'trunks',
                    'vegetation', 'obscurant'
                ],
                seg_label_mapping=dict({
                    0: 24,
                    1: 5,
                    2: 6,
                    3: 0,
                    4: 3,
                    5: 4,
                    6: 2,
                    7: 1,
                    8: 9,
                    9: 10,
                    10: 11,
                    11: 12,
                    12: 13,
                    13: 14,
                    14: 15,
                    15: 16,
                    16: 17,
                    17: 24,
                    18: 18,
                    19: 7,
                    20: 8,
                    21: 19,
                    22: 20,
                    23: 21,
                    24: 22,
                    27: 23,
                    28: 24,
                    29: 24,
                    30: 24,
                    255: 24
                }),
                max_label=255),
            modality=dict(use_lidar=True, use_camera=False),
            ignore_index=24,
            test_mode=True,
            backend_args=None)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='_DSODataset',
            data_root='data/dso/',
            data_prefix=dict(
                pts='',
                img='',
                pts_instance_mask='',
                pts_semantic_mask='',
                pts_panoptic_mask=''),
            ann_file='dso_infos_cetran.pkl',
            pipeline=[
                dict(type='_LoadDSOPointsAndAnnotations'),
                dict(type='PointSegClassMapping'),
                dict(
                    type='Pack3DDetInputs',
                    keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
            ],
            metainfo=dict(
                classes=[
                    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
                    'rider', 'traffic-sign', 'traffic-cone', 'paved-road',
                    'unpaved-road', 'sidewalk', 'building', 'window',
                    'perimeter-barrier', 'other-barrier', 'overhead-bridge',
                    'gate', 'pole-like-object', 'drain', 'terrain', 'trunks',
                    'vegetation', 'obscurant'
                ],
                seg_label_mapping=dict({
                    0: 24,
                    1: 5,
                    2: 6,
                    3: 0,
                    4: 3,
                    5: 4,
                    6: 2,
                    7: 1,
                    8: 9,
                    9: 10,
                    10: 11,
                    11: 12,
                    12: 13,
                    13: 14,
                    14: 15,
                    15: 16,
                    16: 17,
                    17: 24,
                    18: 18,
                    19: 7,
                    20: 8,
                    21: 19,
                    22: 20,
                    23: 21,
                    24: 22,
                    27: 23,
                    28: 24,
                    29: 24,
                    30: 24,
                    255: 24
                }),
                max_label=255),
            modality=dict(use_lidar=True, use_camera=False),
            ignore_index=24,
            test_mode=True,
            backend_args=None)))
val_evaluator = [
    dict(
        type='_OODPointMetric',
        ood_raw_ids=[17, 28],
        seg_offset=65536,
        ignore_index=24)
]
test_evaluator = [
    dict(
        type='_OODPointMetric',
        ood_raw_ids=[17, 28],
        seg_offset=65536,
        ignore_index=24)
]
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
grid_shape = [480, 360, 32]
model = dict(
    type='_P3Former',
    data_preprocessor=dict(
        type='_Det3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=[480, 360, 32],
            point_cloud_range=[0, -3.14159265359, -3, 50, 3.14159265359, 13],
            max_num_points=-1,
            max_voxels=-1)),
    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=6,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    backbone=dict(
        type='_Asymm3DSpconv',
        grid_size=[480, 360, 32],
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-05, momentum=0.1),
        more_conv=True,
        out_channels=256),
    decode_head=dict(
        type='_P3FormerHead',
        num_classes=25,
        num_queries=128,
        embed_dims=256,
        point_cloud_range=[0, -3.14159265359, -3, 50, 3.14159265359, 13],
        assigner_zero_layer_cfg=dict(
            type='mmdet.HungarianAssigner',
            match_costs=[
                dict(
                    type='mmdet.FocalLossCost',
                    weight=1.0,
                    binary_input=True,
                    gamma=2.0,
                    alpha=0.25),
                dict(type='mmdet.DiceCost', weight=2.0, pred_act=True)
            ]),
        assigner_cfg=dict(
            type='mmdet.HungarianAssigner',
            match_costs=[
                dict(
                    type='mmdet.FocalLossCost',
                    gamma=4.0,
                    alpha=0.25,
                    weight=1.0),
                dict(
                    type='mmdet.FocalLossCost',
                    weight=1.0,
                    binary_input=True,
                    gamma=2.0,
                    alpha=0.25),
                dict(type='mmdet.DiceCost', weight=2.0, pred_act=True)
            ]),
        sampler_cfg=dict(type='_MaskPseudoSampler'),
        loss_mask=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_dice=dict(type='mmdet.DiceLoss', loss_weight=2.0),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=4.0,
            alpha=0.25,
            loss_weight=1.0),
        num_decoder_layers=6,
        cls_channels=(256, 256, 25),
        mask_channels=(256, 256, 256, 256, 256),
        thing_class=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        stuff_class=[
            9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
        ],
        ignore_index=24,
        ood_cfg=dict(
            num_ood_logits=24,
            odin_temperature=1000.0,
            energy_temperature=1.0,
            class_groups=[[0, 1, 2, 3, 4], [5, 6], [9, 10, 11, 19],
                          [12, 13, 14, 15, 16, 17], [20, 21, 22],
                          [7, 8, 18, 23]],
            class_groups_variants=dict(
                m4_vhgn=[[0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 19, 20, 21, 22],
                         [12, 13, 14, 15, 16, 17], [7, 8, 18, 23]],
                m4_vhgo=[[0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 19, 7, 8, 18, 23],
                         [12, 13, 14, 15, 16, 17], [20, 21, 22]],
                m4_vhcn=[[
                    0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 20, 21, 22
                ], [9, 10, 11, 19], [7, 8, 18, 23]],
                m4_vhco=[[
                    0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 7, 8, 18, 23
                ], [9, 10, 11, 19], [20, 21, 22]],
                m4_vhno=[[0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 7, 8, 18, 23],
                         [9, 10, 11, 19], [12, 13, 14, 15, 16, 17]],
                m4_vgcn=[[
                    0, 1, 2, 3, 4, 9, 10, 11, 19, 12, 13, 14, 15, 16, 17, 20,
                    21, 22
                ], [5, 6], [7, 8, 18, 23]],
                m4_vgco=[[
                    0, 1, 2, 3, 4, 9, 10, 11, 19, 12, 13, 14, 15, 16, 17, 7, 8,
                    18, 23
                ], [5, 6], [20, 21, 22]],
                m4_vgno=[[
                    0, 1, 2, 3, 4, 9, 10, 11, 19, 20, 21, 22, 7, 8, 18, 23
                ], [5, 6], [12, 13, 14, 15, 16, 17]],
                m4_vcno=[[
                    0, 1, 2, 3, 4, 12, 13, 14, 15, 16, 17, 20, 21, 22, 7, 8,
                    18, 23
                ], [5, 6], [9, 10, 11, 19]],
                m4_hgcn=[[
                    5, 6, 9, 10, 11, 19, 12, 13, 14, 15, 16, 17, 20, 21, 22
                ], [0, 1, 2, 3, 4], [7, 8, 18, 23]],
                m4_hgco=[[
                    5, 6, 9, 10, 11, 19, 12, 13, 14, 15, 16, 17, 7, 8, 18, 23
                ], [0, 1, 2, 3, 4], [20, 21, 22]],
                m4_hgno=[[5, 6, 9, 10, 11, 19, 20, 21, 22, 7, 8, 18, 23],
                         [0, 1, 2, 3, 4], [12, 13, 14, 15, 16, 17]]))),
    train_cfg=None,
    test_cfg=dict(mode='whole'))
default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'work_dirs/p3former_2xb1_3x_dso/epoch_36.pth'
resume = False
point_cloud_range = [0, -3.14159265359, -3, 50, 3.14159265359, 13]
lr = 0.0008
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0008, weight_decay=0.01))
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
custom_imports = dict(
    imports=[
        'p3former.backbones.cylinder3d',
        'p3former.data_preprocessors.data_preprocessor',
        'p3former.decode_heads.p3former_head', 'p3former.segmentors.p3former',
        'p3former.task_modules.samplers.mask_pseduo_sampler',
        'evaluation.metrics.panoptic_seg_metric',
        'evaluation.metrics.ood_metric', 'datasets.dso_dataset',
        'datasets.transforms.dso_loading', 'datasets.transforms.transforms_3d'
    ],
    allow_failed_imports=False)
launcher = 'none'
work_dir = 'work_dirs/p3former_2xb1_3x_dso_ood_hier/b4'
