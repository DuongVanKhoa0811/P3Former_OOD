_base_ = ['./p3former_2xb1_3x_dso_ood.py']

# OCCUQ, variant pe (docs/superpowers/specs/2026-10-07-occuq-density-design.md).
# Class-conditional Gaussians on the base checkpoint's pe_features, scored as
# density_pe. It is the signal check before any training, and the ablation
# without the spectrally normalised head. Evaluation only.
# Fit the Gaussians:
#   python tools/fit_occuq_gmm.py configs/p3former/p3former_2xb1_3x_dso_occuq_pe.py \
#       work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --features pe \
#       --out work_dirs/p3former_2xb1_3x_dso_occuq_pe/gmm_pe.pth --check 5
# then pass the file to test.py:
#   --cfg-options model.decode_head.occuq_cfg.pe_gmm_file=work_dirs/p3former_2xb1_3x_dso_occuq_pe/gmm_pe.pth
model = dict(decode_head=dict(occuq_cfg=dict(head=False)))

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
val_evaluator = [pq_metric]
test_evaluator = [
    pq_metric,
    dict(
        type='_OODPointMetric',
        ood_raw_ids=[17, 28],
        seg_offset=2**16,
        ignore_index=24,
        score_keys=('msp', 'maxlogit', 'odin', 'energy', 'entropy',
                    'density_pe')),
]
