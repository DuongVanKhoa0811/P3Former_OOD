_base_ = ['./p3former_2xb1_3x_dso_occuq_a.py']

# OCCUQ, variant C (docs/superpowers/specs/2026-10-07-occuq-density-design.md).
# As variant A, but the whole model is fine-tuned end to end with the OCCUQ
# head attached, as OCCUQ does: P3Former's losses plus the head's CE and
# Lovasz. The commands are variant A's, with this config and
# work_dirs/p3former_2xb1_3x_dso_occuq_c/seed<N>.
model = dict(decode_head=dict(occuq_cfg=dict(freeze_base=False)))
