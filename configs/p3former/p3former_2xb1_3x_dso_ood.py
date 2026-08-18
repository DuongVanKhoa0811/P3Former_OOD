_base_ = ['./p3former_2xb1_3x_dso.py']

# Post-hoc point-level OOD scoring (MSP / MaxLogit / ODIN / Energy) from the
# auxiliary semantic branch, DSO 24-class edition. OOD classes are the two
# ignore-mapped raw ids reserved for OOD: 17 (Stop) and 28 (Others); the
# other ignore-mapped raw ids (0 Noise, 29 Sky, 30 Water Body, 255
# Unlabelled) are excluded from the evaluation. Protocol otherwise identical
# to docs/superpowers/specs/2026-08-12-ood-baselines-design.md.
model = dict(
    decode_head=dict(
        ood_cfg=dict(
            num_ood_logits=24,  # drop the ignore channel (24) of the 25-way head
            odin_temperature=1000.0,
            energy_temperature=1.0)))

learning_map_inv = {  # copied from the DSO base dataset config
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

val_evaluator = [
    dict(
        type='_PanopticSegMetric',
        thing_class_inds=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        stuff_class_inds=[
            9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
        ],
        min_num_points=50,
        id_offset=2**16,
        dataset_type='dso',
        learning_map_inv=learning_map_inv),
    dict(
        type='_OODPointMetric',
        ood_raw_ids=[17, 28],
        seg_offset=2**16,
        ignore_index=24),
]
test_evaluator = val_evaluator

custom_imports = dict(
    imports=[
        'p3former.backbones.cylinder3d',
        'p3former.data_preprocessors.data_preprocessor',
        'p3former.decode_heads.p3former_head',
        'p3former.segmentors.p3former',
        'p3former.task_modules.samplers.mask_pseduo_sampler',
        'evaluation.metrics.panoptic_seg_metric',
        'evaluation.metrics.ood_metric',
        'datasets.dso_dataset',
        'datasets.transforms.dso_loading',
        'datasets.transforms.transforms_3d',
    ],
    allow_failed_imports=False)
