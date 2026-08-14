_base_ = ['./p3former_1xb2_3x_dso.py']

# 2 GPUs, 1 sample per GPU -> effective batch 2: identical recipe to the
# 1xb2 config (lr/schedule unchanged) with roughly half the per-GPU
# activation memory. The 24-class revision made traffic-sign/cone things,
# so instance-dense frames spike the thing-mask loss well past the ~19.6 GB
# steady state (observed ~27 GB allocation at OOM) — batch 1 per GPU keeps
# a safe margin on the 48 GB cards.
train_dataloader = dict(batch_size=1, )
