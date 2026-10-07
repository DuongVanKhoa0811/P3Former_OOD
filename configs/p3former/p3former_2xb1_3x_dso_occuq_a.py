_base_ = ['./p3former_2xb1_3x_dso_ood.py']

# OCCUQ, variant A (docs/superpowers/specs/2026-10-07-occuq-density-design.md).
# The OCCUQ head (_OCCUQHead, OCCUQ's spectrally normalised residual MLP) is
# trained alone on top of the frozen base model, with OCCUQ's fine-tuning
# recipe scaled to P3Former: a quarter of the base schedule's epochs and
# learning rate, cosine with a 500-iteration warmup.
# Train (for seeds 1 and 2, add --cfg-options randomness.seed=N):
#   bash dist_train.sh configs/p3former/p3former_2xb1_3x_dso_occuq_a.py 2 \
#       --work-dir work_dirs/p3former_2xb1_3x_dso_occuq_a/seed0
# Then fit the Gaussians (tools/fit_occuq_gmm.py --features head) and pass
# the file to test.py:
#   --cfg-options model.decode_head.occuq_cfg.gmm_file=<seed dir>/gmm_head.pth
model = dict(decode_head=dict(occuq_cfg=dict(freeze_base=True)))

load_from = 'work_dirs/p3former_2xb1_3x_dso/epoch_36.pth'
randomness = dict(seed=0)

optim_wrapper = dict(optimizer=dict(lr=2e-4))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=9, val_interval=9)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=9,
        eta_min=2e-7,
        by_epoch=True,
        begin=0,
        end=9,
        convert_to_iter_based=True),
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=9, max_keep_ckpts=1))

pq_metric = dict(
    type='_PanopticSegMetric',
    thing_class_inds=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    stuff_class_inds=[
        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    ],
    min_num_points=50,
    id_offset=2**16,
    dataset_type='dso',
    learning_map_inv={{_base_.learning_map_inv}})
# validation at the end of training: no Gaussians exist yet, so PQ only
val_evaluator = [pq_metric]
test_evaluator = [
    pq_metric,
    dict(
        type='_OODPointMetric',
        ood_raw_ids=[17, 28],
        seg_offset=2**16,
        ignore_index=24,
        score_keys=('msp', 'maxlogit', 'odin', 'energy', 'entropy',
                    'density')),
]
