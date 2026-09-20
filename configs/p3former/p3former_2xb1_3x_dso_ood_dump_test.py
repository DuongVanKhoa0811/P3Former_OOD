_base_ = ['./p3former_2xb1_3x_dso_ood_dump.py']

# The same per-point logits dump for the held-out test split (2,625 frames,
# ~54 GB of float16 logits). dso_infos_test_cetran.pkl is exactly test +
# Cetran (same frames, same order), so this dump plus the Cetran dump of the
# base config covers it: the offline tools take one or several dump
# directories and so evaluate test, Cetran or test + Cetran without another
# GPU pass. _OODPointMetric still runs (flat + current hierarchy +
# p_v_hgcno, ~103 GB RAM for the 1.02B test points) and cross-checks the
# offline numbers.
test_dataloader = dict(
    dataset=dict(dataset=dict(ann_file='dso_infos_test.pkl')))
val_dataloader = test_dataloader

val_evaluator = [
    dict(
        type='_OODPointMetric',
        ood_raw_ids=[17, 28],
        seg_offset=2**16,
        ignore_index=24),
    dict(
        type='_OODLogitsDumpMetric',
        out_dir='work_dirs/p3former_2xb1_3x_dso_ood_dump/logits_test',
        ood_raw_ids=[17, 28],
        seg_offset=2**16,
        ignore_index=24),
]
test_evaluator = val_evaluator
