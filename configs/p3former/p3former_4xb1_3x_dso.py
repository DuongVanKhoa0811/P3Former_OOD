_base_ = ['./p3former_1xb2_3x_dso.py']

# 4 x 24 GB GPUs (e.g. A5000), 1 sample per GPU -> effective batch 4.
# Batch 2 needs ~18.8 GB on DSO frames (~416k pts/frame), so 24 GB cards
# take batch 1. The BN1d layers normalize over voxels/points (tens of
# thousands per sample), so per-GPU batch 1 still has healthy statistics
# without SyncBN. lr stays at the recipe's 8e-4 (not scaled with batch).
train_dataloader = dict(batch_size=1, )
