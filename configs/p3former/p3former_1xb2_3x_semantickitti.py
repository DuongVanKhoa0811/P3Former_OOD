_base_ = ['./p3former_8xb2_3x_semantickitti.py']

# Single-GPU run, 2 samples per GPU -> effective batch 2 (the paper recipe
# is 8 x 2 = 16). Identical to the parent config except for the launcher:
# use `python train.py` instead of dist_train.sh. lr stays at the recipe's
# 8e-4 (not scaled with batch).
train_dataloader = dict(batch_size=2, )
