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
