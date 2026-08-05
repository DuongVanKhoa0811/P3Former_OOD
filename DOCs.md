# DOCs

Working log for P3Former_OOD (OOD panoptic segmentation baseline).

## 2026-08-05 — Environment setup

Conda env `p3former` (base conda's libmamba solver is broken, so use `--solver=classic`):

```bash
conda create -n p3former python=3.8 -y --solver=classic
conda activate p3former
```

Install the pinned stack (the README wraps the PyTorch URL across two lines — it must be one line):

```bash
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install openmim
mim install mmengine==0.7.4
mim install mmcv==2.0.0rc4
mim install mmdet==3.0.0
mim install mmdet3d==1.1.0
wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install yapf==0.40.1   # newer yapf breaks mmengine 0.7.4 (FormatCode 'verify' kwarg)
```



## 2026-08-05 — Data setup

SemanticKITTI lives on the SSD; the repo expects `data/semantickitti`, so symlink it:

```bash
mkdir -p data
ln -s /mnt/ssd/khoadv/projects/OOD_PanSeg_3D/data/SemanticKITTI data/semantickitti
```

Info pkls (`semantickitti_infos_{train,val,trainval,test}.pkl`) are generated with:

```bash
python tools/create_data.py semantickitti --root-path data/semantickitti --out-dir data/semantickitti --extra-tag semantickitti
```



## 2026-08-05 — Validate pretrained checkpoint on SemanticKITTI val

Checkpoint: `checkpoint/semantickitti_val_62.6.pth` (official release, reference PQ 62.6 on val).

```bash
CUDA_VISIBLE_DEVICES=1 bash dist_test.sh configs/p3former/p3former_8xb2_3x_semantickitti.py checkpoint/semantickitti_val_62.6.pth 1
```

Notes:

- Use `bash`, not `sh` (script uses bash-only `${@:4}`; Ubuntu's `sh` is dash → "Bad substitution").
- `CUDA_VISIBLE_DEVICES` picks the GPU; `PORT=...` overrides the default 29500 if the port is busy.
- Logs and results go to `work_dirs/p3former_8xb2_3x_semantickitti/`.

Results (4071 val scans, ~6.5 min on one RTX 6000 Ada, ~1.2 GB GPU memory):


| PQ    | PQ†   | RQ    | SQ    | mIoU  | PQ_things | PQ_stuff |
| ----- | ----- | ----- | ----- | ----- | --------- | -------- |
| 62.63 | 66.25 | 72.42 | 76.17 | 66.77 | 69.36     | 57.74    |


Matches the reference PQ 62.6 for this checkpoint.