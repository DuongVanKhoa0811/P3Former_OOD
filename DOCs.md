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

## 2026-08-06 — Training commands

DSO (this machine, single GPU — ~0.9 s/iter, ~18.8 GB at batch 2, ≈40 h for 36 epochs):

```bash
CUDA_VISIBLE_DEVICES=1 python train.py configs/p3former/p3former_1xb2_3x_dso.py
```

DSO (4x A5000 24 GB server, batch 1 per GPU, effective batch 4):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train.sh configs/p3former/p3former_4xb1_3x_dso.py 4
```

SemanticKITTI (single GPU, batch 2):

```bash
CUDA_VISIBLE_DEVICES=1 python train.py configs/p3former/p3former_1xb2_3x_semantickitti.py
```

SemanticKITTI (4x A5000, batch 1 per GPU):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train.sh configs/p3former/p3former_4xb1_3x_semantickitti.py 4
```

Notes:

- Append `--resume` to continue an interrupted run from the latest checkpoint.
- `PORT=29511 ...` in front of dist_train.sh if the default port is busy.
- Outputs land in `work_dirs/<config-name>/`; checkpoints every 5 epochs, val PQ table every epoch.
- Run status: DSO 1xb2 ✅ done · SemanticKITTI 4xb1 ✅ done · SemanticKITTI 1xb2 🔄 running ·
DSO 4xb1 ❌ out-of-memory on 24 GB A5000s (DSO frames are ~416k pts; no checkpoint produced).



## 2026-08-11 — Testing commands

`epoch_36.pth` is the final checkpoint; check the training log's per-epoch val PQ and substitute
the best epoch's checkpoint (saved at 5, 10, ..., 35, 36) if it isn't the last one.

DSO 1xb2 — val split (Chinatown Route 1 Day, 401 frames; this is what test.py evaluates by default):

```bash
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_1xb2_3x_dso.py work_dirs/p3former_1xb2_3x_dso/epoch_36.pth
```

DSO 1xb2 — held-out test split (One-North Route 2 Day + Ubin Route 1 + Ubin Route 3, 2625 frames;
the two rural Ubin routes are a deliberate urban→rural domain shift):

```bash
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_1xb2_3x_dso.py work_dirs/p3former_1xb2_3x_dso/epoch_36.pth --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test.pkl
```

SemanticKITTI 1xb2 — val split (4071 scans; run after training finishes):

```bash
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_1xb2_3x_semantickitti.py work_dirs/p3former_1xb2_3x_semantickitti/epoch_36.pth
```

SemanticKITTI 4xb1 — val split:

```bash
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_4xb1_3x_semantickitti.py work_dirs/p3former_4xb1_3x_semantickitti/epoch_36.pth
```

Notes:

- SemanticKITTI "testing" means the val split — the official test split is unlabeled
(submission-only, via the `_submit` config).
- The two DSO evals contend with whatever is training on GPU 1; run them after the
SemanticKITTI 1xb2 run finishes (or on GPU 0 when it is free — eval is light).

## 2026-08-12 — Point-level OOD baselines (MSP / MaxLogit / ODIN / Energy)

Post-hoc OOD scoring from the aux semantic branch; OOD classes = raw 52
(other-structure) + 99 (other-object); raw 0/1 (unlabeled/outlier) excluded. Spec:
`docs/superpowers/specs/2026-08-12-ood-baselines-design.md`, branch `ood-baselines`.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_8xb2_3x_semantickitti_ood.py checkpoint/semantickitti_val_62.6.pth
```

Results (val, 4071 scans, official checkpoint; PQ table unchanged at 62.63;
476.8M ID / 9.4M OOD points = 1.94% OOD):

| method   | AUROC | AP    | FPR@95 |
| -------- | ----- | ----- | ------ |
| MSP      | 87.48 | 16.29 | 43.45  |
| MaxLogit | 91.28 | 40.28 | 39.98  |
| ODIN     | 90.87 | 32.42 | 40.70  |
| Energy   | 91.59 | 43.71 | 39.98  |

ODIN = temperature-scaled MSP (T=1000, ε=0 per docs/others/baselines/OOD_Baseline.pdf);
Energy uses T=1. Higher score = more OOD everywhere. Smoke test: add
`--cfg-options test_dataloader.dataset.dataset.ann_file=semantickitti_infos_mini.pkl`
(20-scan mini pkl generated from the val infos). Unit tests: `tests/test_ood_scores.py`,
`tests/test_ood_eval.py`, `tests/test_ood_metric.py`.



## 2026-08-14 — DSO class-set revision (24 classes)

The DSO class definitions changed (new names; e.g. road→Paved Road, fence→Other Barrier,
net-fence→Perimeter Barrier). Training now covers **24 classes** instead of 16:

- **Things (train 0–8):** car, bicycle, motorcycle, truck, bus, person, rider,
  **traffic-sign** (raw 19), **traffic-cone** (raw 20) — signs/cones carry real instance ids
  (~17.6 and ~4.2 per frame) and are now things; the loader keeps their instance bits
  (`DSO_THING_RAW_IDS` gained 19, 20).
- **Stuff (train 9–23):** paved-road, unpaved-road, sidewalk, building, window,
  perimeter-barrier, other-barrier, overhead-bridge, gate, pole-like-object, drain, terrain,
  trunks, vegetation, obscurant (ascending raw id).
- **Ignored (→ 24):** Noise (0), **Stop (17)** and **Others (28)** — reserved as future OOD
  classes — Sky (29) and Water Body (30) — 2D-only — Unlabelled (255).

Head is now 25-way (`num_classes=25`, `cls_channels=(256,256,25)`); pkls/splits unchanged.
**Old 16-class checkpoints are incompatible — DSO must be retrained** (same commands as
2026-08-06; smoke-verified: 19.6 GB at batch 2, so the 4xb1 A5000 OOM situation is unchanged).
Committed on `dso-dataset`, merged into `ood-baselines`.

Caveat for eval: many traffic-cone instances are below the `min_num_points=50` PQ cutoff
(~40 pts/instance on average), so cone PQ reflects only the larger instances — standard
SemanticKITTI convention, left as is.

Update: the 24-class 1xb2 run OOM'd mid-epoch-3 in the thing-mask loss (~27 GB allocation
spike on instance-dense frames — signs/cones roughly double the thing masks; the run also
shared GPU 0 with a ~15 GB job). Use the 2-GPU batch-1 config instead (same effective
batch 2, ~half the per-GPU activations):

```bash
CUDA_VISIBLE_DEVICES=0,1 bash dist_train.sh configs/p3former/p3former_2xb1_3x_dso.py 2
```


## 2026-08-15 — 2xb1 SemanticKITTI config; DSO 24-class test commands

SemanticKITTI on both of this machine's GPUs (batch 1 per GPU, effective batch 2 — same
recipe as 1xb2, ~half the per-GPU memory and wall-clock):

```bash
CUDA_VISIBLE_DEVICES=0,1 bash dist_train.sh configs/p3former/p3former_2xb1_3x_semantickitti.py 2
```

DSO 24-class run — held-out test split (2,625 frames; One-North Route 2 Day + the two rural
Ubin routes). Without the `--cfg-options` override, test.py evaluates the val split:

```bash
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_2xb1_3x_dso.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test.pkl
```



## 2026-08-18 — Cetran test sets

The three Cetran AV-test-centre sequences (401 + 202 + 377 = 980 frames, ~300k pts/frame,
fully labeled; both future-OOD classes Stop/Others present in all three) are now optional
test sets. Manual review of the per-frame PNGs confirmed the three runs cover
non-overlapping content (the 01-28 "AM-clean" re-export shares frame stems with "AM" but
different scenes). `create_data.py dso` writes them only when all three dirs exist and they
never enter train/val:

- `dso_infos_cetran.pkl` — Cetran only (980 frames)
- `dso_infos_test_cetran.pkl` — held-out test + Cetran (3,605 frames)

Evaluate the 24-class model on them with the usual override, e.g.:

```bash
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_2xb1_3x_dso.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test_cetran.pkl
```
