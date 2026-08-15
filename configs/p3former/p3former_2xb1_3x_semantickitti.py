_base_ = ['./p3former_8xb2_3x_semantickitti.py']

# 2 GPUs, 1 sample per GPU -> effective batch 2 (the paper recipe is
# 8 x 2 = 16): same recipe as the 1xb2 config with half the per-GPU
# memory and roughly half the wall-clock. lr stays at the recipe's 8e-4
# (not scaled with batch).
train_dataloader = dict(batch_size=1, )
