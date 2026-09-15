_base_ = ['./p3former_2xb1_3x_dso_ood.py']

# Record the per-point 24-class semantic logits on the Cetran-only split so
# that ID/OOD score distributions — for any class hierarchy — can be
# recomputed offline (tools/plot_ood_score_distributions.py) without another
# GPU pass. The float16 dump is ~15 MB per frame, ~14 GB for the 980 Cetran
# frames. _OODPointMetric still runs (flat + current hierarchy +
# ``p_v_hgcno``, the best hierarchy of the 2026-08-25 ablation) so the
# logged table cross-checks the offline recomputation; PQ is dropped.
model = dict(
    decode_head=dict(
        ood_cfg=dict(
            save_logits=True,
            class_groups_variants=dict(
                # {vehicle} | {human+ground+construction+nature+object}
                p_v_hgcno=[
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                     20, 21, 22, 23],
                ]))))

test_dataloader = dict(
    dataset=dict(dataset=dict(ann_file='dso_infos_cetran.pkl')))
val_dataloader = test_dataloader

val_evaluator = [
    dict(
        type='_OODPointMetric',
        ood_raw_ids=[17, 28],
        seg_offset=2**16,
        ignore_index=24),
    dict(
        type='_OODLogitsDumpMetric',
        out_dir='work_dirs/p3former_2xb1_3x_dso_ood_dump/logits',
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
        'evaluation.metrics.ood_logits_dump',
        'datasets.dso_dataset',
        'datasets.transforms.dso_loading',
        'datasets.transforms.transforms_3d',
    ],
    allow_failed_imports=False)
