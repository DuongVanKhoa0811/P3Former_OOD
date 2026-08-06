_base_ = ['./p3former_8xb2_3x_semantickitti.py']

# 4 x 24 GB GPUs (e.g. A5000), 1 sample per GPU -> effective batch 4
# (the paper recipe is 8 x 2 = 16). The BN1d layers normalize over
# voxels/points (tens of thousands per sample), so per-GPU batch 1 still
# has healthy statistics without SyncBN. lr stays at the recipe's 8e-4
# (not scaled with batch).
train_dataloader = dict(batch_size=1, )
