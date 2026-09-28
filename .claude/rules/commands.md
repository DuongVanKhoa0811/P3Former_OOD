# Training, evaluation and data preparation commands

Every `python` below means the `p3former` interpreter (see `environment.md`).

```bash
# single GPU (both accept --cfg-options key=value; train.py --resume continues from the latest checkpoint)
CUDA_VISIBLE_DEVICES=0 python train.py configs/p3former/p3former_2xb1_3x_dso.py
CUDA_VISIBLE_DEVICES=0 python test.py <config> <checkpoint> [--work-dir DIR]
# test.py evaluates the val split; switch split via the RepeatDataset-nested ann_file
python test.py configs/p3former/p3former_2xb1_3x_dso_ood.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth \
    --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test.pkl   # or dso_infos_test_cetran.pkl

# multi-GPU: bash, not sh (the scripts use bash-only ${@:3} / ${@:4}); PORT=... overrides 29503 (train) / 29500 (test)
CUDA_VISIBLE_DEVICES=0,1 bash dist_train.sh configs/p3former/p3former_2xb1_3x_dso.py 2
bash dist_test.sh <config> <checkpoint> <NUM_GPUS>

# dataset info pkls
python tools/create_data.py semantickitti --root-path data/semantickitti --out-dir data/semantickitti --extra-tag semantickitti
python tools/create_data.py dso --out-dir data/dso --extra-tag dso   # expects data/dso/annotations -> DSO Annotation_Final
python tools/create_data.py nuscenes --root-path data/nuscenes --out-dir data/nuscenes --extra-tag nuscenes --version v1.0
```

Outputs go to `work_dirs/<config-name>/`, with the log in a timestamped subdirectory; the OOD result tables are in that log. The standard checkpoints are:

- `work_dirs/p3former_2xb1_3x_{dso,semantickitti}/epoch_36.pth`: our own runs.
- `checkpoint/semantickitti_val_62.6.pth`: the official checkpoint, used with the `8xb2` configs.
