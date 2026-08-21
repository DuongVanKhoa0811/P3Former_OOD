_base_ = ['./p3former_2xb1_3x_semantickitti.py']

# Post-hoc point-level OOD scoring (MSP / MaxLogit / ODIN / Energy / Entropy
# + group and group-normalised variants) from the auxiliary semantic branch,
# for the 2-GPU batch-1 SemanticKITTI model. Identical evaluation setup to
# p3former_8xb2_3x_semantickitti_ood.py; protocol and hyperparameters:
# docs/superpowers/specs/2026-08-12-ood-baselines-design.md
model = dict(
    decode_head=dict(
        ood_cfg=dict(
            num_ood_logits=19,
            odin_temperature=1000.0,
            energy_temperature=1.0,
            # Semantic hierarchy for the group / group-normalised scores
            # (papers/RelatedPapers/GroupPaper.pdf, Table 2), train ids.
            class_groups=[
                [0, 1, 2, 3, 4],  # vehicle: car, bicycle, motorcycle, truck, other-vehicle
                [5, 6, 7],  # human: person, bicyclist, motorcyclist
                [8, 9, 10, 11],  # ground: road, parking, sidewalk, other-ground
                [12, 13],  # construction: building, fence
                [14, 15, 16],  # nature: vegetation, trunk, terrain
                [17, 18],  # object: pole, traffic-sign
            ])))

learning_map_inv = {  # copied from the base dataset config
    0: 10,
    1: 11,
    2: 15,
    3: 18,
    4: 20,
    5: 30,
    6: 31,
    7: 32,
    8: 40,
    9: 44,
    10: 48,
    11: 49,
    12: 50,
    13: 51,
    14: 70,
    15: 71,
    16: 72,
    17: 80,
    18: 81,
    19: 0
}

val_evaluator = [
    dict(
        type='_PanopticSegMetric',
        thing_class_inds=[0, 1, 2, 3, 4, 5, 6, 7],
        stuff_class_inds=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        min_num_points=50,
        id_offset=2**16,
        dataset_type='semantickitti',
        learning_map_inv=learning_map_inv),
    dict(
        type='_OODPointMetric',
        ood_raw_ids=[52, 99],
        seg_offset=2**16,
        ignore_index=19),
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
        'datasets.semantickitti_dataset',
        'datasets.transforms.loading',
        'datasets.transforms.transforms_3d',
    ],
    allow_failed_imports=False)
