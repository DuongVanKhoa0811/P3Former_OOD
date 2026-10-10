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
| Entropy  | 88.57 | 24.70 | 42.91  |

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

## 2026-08-19 — Point-level OOD baselines on DSO (MSP / MaxLogit / ODIN / Energy)

Same post-hoc protocol as the 2026-08-12 SemanticKITTI entry, ported to the 24-class DSO
model via `configs/p3former/p3former_2xb1_3x_dso_ood.py`: scores from the first 24 channels
of the aux semantic branch; OOD = raw 17 (Stop) + 28 (Others); raw 0/29/30/255
(Noise/Sky/Water Body/Unlabelled) excluded. Checkpoint:
`work_dirs/p3former_2xb1_3x_dso/epoch_36.pth`.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_2xb1_3x_dso_ood.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --work-dir work_dirs/p3former_2xb1_3x_dso_ood/test --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test.pkl
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_2xb1_3x_dso_ood.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --work-dir work_dirs/p3former_2xb1_3x_dso_ood/test_cetran --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test_cetran.pkl
```

Held-out test split (2,625 frames; PQ 46.50, mIoU 48.66; 1.008B ID / 15.9M OOD points
= 1.55% OOD):

| method   | AUROC | AP    | FPR@95 |
| -------- | ----- | ----- | ------ |
| MSP      | 86.26 | 13.76 | 51.15  |
| MaxLogit | 92.08 | 34.24 | 42.70  |
| ODIN     | 86.53 | 19.51 | 69.63  |
| Energy   | 92.36 | 35.47 | 42.55  |
| Entropy  | 87.84 | 20.23 | 50.40  |

Test + Cetran (3,605 frames; PQ 46.19, mIoU 47.94; 1.261B ID / 23.6M OOD points
= 1.84% OOD):

| method   | AUROC | AP    | FPR@95 |
| -------- | ----- | ----- | ------ |
| MSP      | 87.76 | 18.06 | 45.76  |
| MaxLogit | 92.67 | 35.90 | 38.12  |
| ODIN     | 89.36 | 27.19 | 55.64  |
| Energy   | 92.88 | 35.55 | 37.94  |
| Entropy  | 89.38 | 26.12 | 44.90  |

Energy and MaxLogit are the strongest on both splits (as on SemanticKITTI); ODIN's
T=1000 flattening hurts noticeably more here than on SemanticKITTI. Smoke test: 5-frame
`dso_infos_mini.pkl` (frames of `dso_infos_test.pkl` verified to contain Stop/Others).

## 2026-08-21 — Entropy added as a fifth OOD score

`entropy` = softmax Shannon entropy `−Σ_c p_c log p_c` (natural log, higher = more OOD;
as in `trash/Done/eval_ood_from_logits.py`). Emitted as `ood_entropy` and included in
`_OODPointMetric`'s default keys — no config changes. The Entropy rows in the tables above
come from re-running the three evals on 2026-08-21 (the other rows reproduced exactly).
Note: one eval needs ~17 GB (SemanticKITTI) to ~25 GB (DSO) of GPU memory; it OOMs in
spconv (`cuda execution failed with error 2`) when less is free.

## 2026-08-21 — OOD baselines for the 2xb1 SemanticKITTI model

`configs/p3former/p3former_2xb1_3x_semantickitti_ood.py` mirrors the 8xb2 OOD config on top
of `p3former_2xb1_3x_semantickitti.py` (our own 2-GPU batch-1 run, last epoch):

```bash
CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_2xb1_3x_semantickitti_ood.py work_dirs/p3former_2xb1_3x_semantickitti/epoch_36.pth
```

Val (4071 scans; PQ 60.32, PQ† 62.99, mIoU 62.49 vs 62.63 / 66.25 / 66.77 for the official
checkpoint; same 476.8M ID / 9.4M OOD points):

| method   | AUROC | AP    | FPR@95 |
| -------- | ----- | ----- | ------ |
| MSP      | 87.25 | 12.79 | 41.39  |
| MaxLogit | 90.03 | 32.50 | 44.38  |
| ODIN     | 91.46 | 26.94 | 38.33  |
| Energy   | 90.20 | 33.26 | 44.47  |
| Entropy  | 88.33 | 17.40 | 40.99  |

Same picture as the official checkpoint, ~1 point lower across the board; here ODIN has the
best AUROC and the lowest FPR@95.

## 2026-10-10 — OCCUQ density scores vs the robust grouping splits (DSO)

OCCUQ (Heidrich, Beemelmanns et al., ICRA 2025) scores a point by the density of its
voxel feature under one full-covariance Gaussian per class, fitted on the training split.
This entry ports it to DSO and compares it with the grouping method (spec
`docs/superpowers/specs/2026-10-07-occuq-density-design.md`, plan
`docs/superpowers/plans/2026-10-07-occuq-density.md`, branch `ood-baselines/occuq`).
Three variants: **pe** fits the Gaussians on the base checkpoint's `pe_features`, with no
training (score `density_pe`); **A** trains a spectrally normalized residual-MLP head
(`_OCCUQHead`) on the frozen base model and fits the Gaussians on the head's last block
(score `density`); **C** trains the same head while fine-tuning the whole model end to end
(score `density`). A and C ran with seeds 0, 1 and 2, for 9 epochs from
`work_dirs/p3former_2xb1_3x_dso/epoch_36.pth`. Every score is reduced to a like-for-like
`improvement` I = (AUROC − R_AUROC) + (AP − R_AP) − (FPR@95 − R_FPR@95), where R is the
mean of the base model's flat MSP, Energy and Entropy rows; a robust split counts with
the I of its best single Group score. The verdict of a variant on a set is against the
best of the 33 robust splits: `better` if every run is above it, `worse` if every run is
below it, `too close to call` otherwise.

### Commands

The runs went through `../store/occuq/run_variant.sh` on server C. It lives in the store
next to the repo, outside git; `work_dirs/p3former_2xb1_3x_dso_occuq_{pe,a,c,compare}`
are symlinks into `../store/occuq/`. It trains (A and C), fits the Gaussians and
evaluates on Cetran, test and test + Cetran, and skips every step whose output exists.
Its content, with the login user written `<user>`:

```bash
#!/usr/bin/env bash
# OCCUQ runs on server C (docs/superpowers/plans/2026-10-07-occuq-density.md).
# Run only inside a PBS GPU job, never on the login node:
#   run_variant.sh pe            1 GPU
#   run_variant.sh a|c SEED...   2 GPUs to train; the fit and evaluations use the first
#   TRAIN_ONLY=1 run_variant.sh a|c SEED...   2 GPUs: train only, no fit and no evaluation (a seed with an epoch_9.pth does nothing)
# Training saves a checkpoint every epoch (default_hooks.checkpoint.interval=1, max_keep_ckpts stays 1) and, after a failed attempt, resumes from the last one (5 attempts in all); don't pass --resume by hand.
# Variant c also empties PyTorch's CUDA cache after every training iteration (mmengine's EmptyCacheHook): the 32 GiB V100s ran out of memory.
set -euo pipefail
cd /home/users/<user>/projects/OOD_PanSeg_3D/P3Former_OOD
PY=/home/users/<user>/miniconda3/envs/p3former/bin/python
export PATH=/home/users/<user>/miniconda3/envs/p3former/bin:$PATH
GPUS=${CUDA_VISIBLE_DEVICES:?run inside a PBS GPU job, which sets CUDA_VISIBLE_DEVICES}
GPU0=${GPUS%%,*}  # fits and evaluations run on the job's first GPU
V=$1; shift
CFG=configs/p3former/p3former_2xb1_3x_dso_occuq_$V.py
OUT=work_dirs/p3former_2xb1_3x_dso_occuq_$V
EXTRA=()  # variant c: EmptyCacheHook after every training iteration, against the out-of-memory crashes
if [ "$V" = c ]; then EXTRA=(default_hooks.empty_cache.type=EmptyCacheHook default_hooks.empty_cache.after_iter=True); fi
fit() {  # fit CKPT FEATURES GMM_FILE
  if [ -f $3 ]; then return 0; fi
  local SUMS=${3%.pth}.sums.pth  # the pass saves its sums there before finalizing
  if [ -f $SUMS ]; then  # an earlier fit failed after its pass: finalize the saved sums
    CUDA_VISIBLE_DEVICES=$GPU0 $PY tools/fit_occuq_gmm.py $CFG $1 --features $2 --out $3 --check 5 --from-sums $SUMS
  else
    CUDA_VISIBLE_DEVICES=$GPU0 $PY tools/fit_occuq_gmm.py $CFG $1 --features $2 --out $3 --check 5
  fi
}
evaluate() {  # evaluate CKPT RUN_DIR KEY OPTION=GMM_FILE
  local GMM=${4#*=} LOG
  for S in cetran test test_cetran; do
    # a set is done when its newest test.py log with the KEY row is newer than the Gaussian file
    LOG=$(grep -ls " $3 | " $2/$S/*/*.log 2>/dev/null | xargs -r ls -t | head -1 || true)
    if [ -n "$LOG" ] && [ "$LOG" -nt "$GMM" ]; then continue; fi
    CUDA_VISIBLE_DEVICES=$GPU0 $PY test.py $CFG $1 --work-dir $2/$S \
      --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_$S.pkl model.decode_head.occuq_cfg.$4
  done
}
# Steps whose output exists are skipped, so a rerun after a failure resumes.
if [ "$V" = pe ]; then
  CKPT=work_dirs/p3former_2xb1_3x_dso/epoch_36.pth
  fit $CKPT pe $OUT/gmm_pe.pth
  evaluate $CKPT $OUT density_pe pe_gmm_file=$OUT/gmm_pe.pth
  exit 0
fi
for N in "$@"; do
  if [ ! -f $OUT/seed$N/epoch_9.pth ]; then
    if [ "${GPUS//[^,]/}" != , ]; then
      echo "training needs a job with 2 GPUs, got CUDA_VISIBLE_DEVICES=$GPUS" >&2; exit 1
    fi
    for K in 1 2 3 4 5; do  # up to 5 attempts; each one after a failure resumes from the last epoch checkpoint, if there is one
      LAST=
      if [ -f $OUT/seed$N/last_checkpoint ]; then LAST=$(cat $OUT/seed$N/last_checkpoint); fi
      RESUME=()  # --resume goes before --cfg-options, which swallows every argument after it
      FROM="nothing, so the base checkpoint of the config"
      if [ -n "$LAST" ] && [ -f "$LAST" ]; then RESUME=(--resume "$LAST"); FROM=$LAST; fi
      echo "-- seed $N: training attempt $K of 5, resuming from $FROM"
      if PORT=$((29500 + RANDOM % 1000)) bash dist_train.sh $CFG 2 --work-dir $OUT/seed$N ${RESUME[@]+"${RESUME[@]}"} \
          --cfg-options randomness.seed=$N default_hooks.checkpoint.interval=1 ${EXTRA[@]+"${EXTRA[@]}"}; then break; fi
      echo "== training attempt $K of seed $N failed"
      if [ $K -eq 5 ]; then exit 1; fi
      sleep 30
    done
  fi
  if [ "${TRAIN_ONLY:-0}" = 1 ]; then continue; fi
  fit $OUT/seed$N/epoch_9.pth head $OUT/seed$N/gmm_head.pth
  evaluate $OUT/seed$N/epoch_9.pth $OUT/seed$N density gmm_file=$OUT/seed$N/gmm_head.pth
done
```

The script changed twice while the seeds ran, and the block above is its last version.
Variant A's three seeds ran version 1, whose training step was one call,
`PORT=$((29500 + RANDOM % 1000)) bash dist_train.sh $CFG 2 --work-dir $OUT/seed$N --cfg-options randomness.seed=$N`,
so a checkpoint at epoch 9 only and no retry. Seed c0 and epochs 1–3 of seeds c1 and c2
ran version 2 (a checkpoint every epoch, up to 3 attempts, each resuming from the last
checkpoint); epochs 4–9 of c1 and c2 ran version 3, the block above (up to 5 attempts
and, for variant c, mmengine's `EmptyCacheHook` after every training iteration). The
three versions differ only in the training step.

The runs did not use three detached invocations (`run_variant.sh pe`, `a 0 1 2`,
`c 0 1 2`). They ran as PBS jobs on the cluster's V100s, each with the job header of
`.claude/memory.md` and its own queue, `select` line and walltime. `run_variant.sh` is
the script above, and the one fit that does not go through it is the pe fit:

```bash
python tools/fit_occuq_gmm.py configs/p3former/p3former_2xb1_3x_dso_occuq_pe.py \
    work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --features pe \
    --out work_dirs/p3former_2xb1_3x_dso_occuq_pe/gmm_pe.pth --check 5
```

| job (script in `../store/occuq/`) | seeds | queue | `select` | walltime | command |
| --- | --- | --- | --- | --- | --- |
| pe fit (`pe/occuq_pe_fit.pbs`) | pe | v100q | `1:ncpus=6:ngpus=1:mem=60gb` | 03:00:00 | the fit above, after a 20-frame smoke fit (`--limit 20`) that gates it, then a 5-frame smoke evaluation |
| pe evaluation (`pe/occuq_pe_eval.pbs`) | pe | v100q | `1:ncpus=4:ngpus=1:mem=160gb` | 05:00:00 | `run_variant.sh pe` |
| a0 training (`a/occuq_a0_train.pbs`) | A seed 0 | v100q | `1:ncpus=10:ngpus=2:mem=120gb` | 16:00:00 | `TRAIN_ONLY=1 run_variant.sh a 0` |
| a0 fit and evaluation (`a/occuq_a0_eval.pbs`) | A seed 0 | aimcq | `1:ncpus=4:ngpus=1:mem=160gb` | 06:00:00 | `run_variant.sh a 0`, submitted with `qsub -W depend=afterok:<training job>` |
| a12 (`a/occuq_a12.pbs`) | A seeds 1, 2 | v100q | `1:ncpus=20:ngpus=4:mem=240gb` | 16:00:00 | side by side, `CUDA_VISIBLE_DEVICES=<first two GPUs> run_variant.sh a 1` and `CUDA_VISIBLE_DEVICES=<last two GPUs> run_variant.sh a 2` |
| c0 (`c/occuq_c0.pbs`) | C seed 0 | aimcq | `1:ncpus=10:ngpus=2:mem=120gb` | 36:00:00 | `run_variant.sh c 0`, from training to the last evaluation |
| c12 (`c/occuq_c12.pbs`) | C seeds 1, 2 | v100q | `1:ncpus=20:ngpus=4:mem=240gb` | 36:00:00 | as a12, with `c`; restarted once from the epoch-3 checkpoints to switch to version 3 |

Both C jobs export `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`. The `aimcq` jobs stop
unless they got a V100 (sm_70): PBS cannot pin the GPU type in that queue, and its nodes
include an A100, which would train with TF32. The job logs of a12
(`../store/occuq/a/pair12.out`) and c0 (`../store/occuq/c/seed0.out`) end with
`== JOB FAILED` although every step finished: the script file was replaced while they ran,
and on this NFS share the running bash then failed its one read after the last command
(`error reading input file: Stale file handle`, exit status 2). The error is the line
right after the run's last `Epoch(test)` line: line 4443 of `a/seed1.out` and
`a/seed2.out` (a12) and line 4487 of `c/seed0.out`. The numbers come from server C's
V100s: the flat rows of the pe runs match server A's logged values within 0.011 points
(the plan's Task 9 check), while the grouping rows of the comparison come from server A's
sweep logs. Each `improvement` is measured against its own runs' R, so a small server
offset does not bias the verdict.

Why C's training restarted, so that a rerun on a V100 does not repeat it. The first runs
of C died of CUDA out-of-memory errors on the 32 GiB V100s: seed c1 in epoch 1 (18.03 GiB
allocated and 20.30 GiB reserved by PyTorch, of 31.73 GiB) and seed c0 in epoch 6 (17.69
and 20.33 GiB); c2, which shared c1's job, was restarted with it. The ~11.4 GiB that
PyTorch's allocator did not hold matches mmcv 2.0.0rc4's sparse-convolution index kernels
(`mmcv/ops/csrc/pytorch/cuda/sparse_indice.cu`, `KernelMaxVolume` 4096): their 64 KiB
local array per thread keeps about 10 GiB of a V100 for the whole run, which leaves C's
training (logged peak allocation up to 18.8 GiB) too little room for fragmentation. A
peaks at about 3.7 GiB and is not at risk. The reruns set the allocator variable and save
a checkpoint every epoch, from which `run_variant.sh` resumes after a failed attempt. The
allocator setting alone did not lower the crash rate: seeds c1 and c2 each crashed once
more, in epoch 3, and each resumed from its `epoch_2.pth` (`seed1/epoch_2.pth`,
`seed2/epoch_2.pth`). So from epoch 4 their job ran version 3, which empties PyTorch's
cache after every iteration and costs about 8 % in speed (median logged time per
iteration 1.25 s against 1.16 s); that job started from each seed's `epoch_3.pth`. Seeds
c1 and c2 therefore trained epochs 1–2 from the base checkpoint, epoch 3 after a resume
from epoch 2 and epochs 4–9 after a resume from epoch 3, with no further failure; seed
c0 is the only C seed that trained without a resume (one attempt, no failure). The
archived first attempts (`../store/occuq/c/failed_oom_20261009/`) are not results.

The comparison reads logs only, so it ran on the login node:

```bash
python tools/compare_occuq_grouping.py \
  --pe work_dirs/p3former_2xb1_3x_dso_occuq_pe \
  --a work_dirs/p3former_2xb1_3x_dso_occuq_a/seed0 work_dirs/p3former_2xb1_3x_dso_occuq_a/seed1 work_dirs/p3former_2xb1_3x_dso_occuq_a/seed2 \
  --c work_dirs/p3former_2xb1_3x_dso_occuq_c/seed0 work_dirs/p3former_2xb1_3x_dso_occuq_c/seed1 work_dirs/p3former_2xb1_3x_dso_occuq_c/seed2 \
  --out work_dirs/p3former_2xb1_3x_dso_occuq_compare/occuq_vs_grouping.md
```

### Results

Numbers are AUROC / AP / FPR@95 in %. The verdict table and the three set tables are
copied from `occuq_vs_grouping.md`. Verdict per variant and set, against the best robust
split:

| variant | Cetran | test | test + Cetran |
| --- | --- | --- | --- |
| pe | worse | worse | worse |
| A | worse | worse | worse |
| C | worse | worse | worse |

#### Cetran

R (mean of the base model's flat MSP, Energy and Entropy): 91.89 / 35.77 / 29.99

| method | AUROC | AP | FPR@95 | I (like-for-like) | averaged improvement | splits beaten | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat Energy (base model) | 93.14 | 40.33 | 26.35 | 9.45 | | 0/33 | |
| best robust split `s3.4.8.17` (group_msp) | 96.41 | 52.08 | 18.83 | 31.99 | 24.73 | | |
| OCCUQ pe (1 run) | 89.35 | 15.09 | 28.89 | -22.12 | | 0/33 | worse |
| OCCUQ A (3 runs) | 92.53 | 35.67 | 28.89 | 1.64 [0.16, 2.93] | | 0/33 | worse |
| OCCUQ C (3 runs) | 93.50 | 37.03 | 25.56 | 7.30 [5.97, 9.19] | | 0/33 | worse |

C after fine-tuning (mean over runs): PQ 35.14, mIoU 33.71, own flat Energy 93.87 | 46.58 | 25.64; base model PQ 35.35, mIoU 33.44.

#### test

R (mean of the base model's flat MSP, Energy and Entropy): 88.82 / 23.15 / 48.03

| method | AUROC | AP | FPR@95 | I (like-for-like) | averaged improvement | splits beaten | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat Energy (base model) | 92.36 | 35.47 | 42.55 | 21.34 | | 6/33 | |
| best robust split `s16` (group_entropy) | 94.22 | 39.75 | 29.04 | 41.00 | 34.78 | | |
| OCCUQ pe (1 run) | 82.26 | 6.83 | 62.32 | -37.17 | | 0/33 | worse |
| OCCUQ A (3 runs) | 86.90 | 12.51 | 51.18 | -15.70 [-17.04, -13.19] | | 0/33 | worse |
| OCCUQ C (3 runs) | 87.33 | 12.75 | 50.85 | -14.70 [-18.31, -10.15] | | 0/33 | worse |

C after fine-tuning (mean over runs): PQ 47.05, mIoU 48.80, own flat Energy 92.25 | 36.76 | 42.71; base model PQ 46.50, mIoU 48.68.

#### test + Cetran

R (mean of the base model's flat MSP, Energy and Entropy): 90.01 / 26.57 / 42.87

| method | AUROC | AP | FPR@95 | I (like-for-like) | averaged improvement | splits beaten | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat Energy (base model) | 92.88 | 35.54 | 37.95 | 16.77 | | 3/33 | |
| best robust split `s16` (group_entropy) | 94.06 | 40.45 | 28.43 | 32.38 | 27.42 | | |
| OCCUQ pe (1 run) | 84.91 | 9.06 | 53.40 | -33.13 | | 0/33 | worse |
| OCCUQ A (3 runs) | 88.75 | 17.93 | 45.27 | -12.29 [-13.66, -9.74] | | 0/33 | worse |
| OCCUQ C (3 runs) | 89.39 | 18.44 | 43.94 | -9.81 [-12.51, -6.13] | | 0/33 | worse |

C after fine-tuning (mean over runs): PQ 46.61, mIoU 48.28, own flat Energy 92.94 | 39.63 | 37.87; base model PQ 46.20, mIoU 47.95.

C per seed, PQ / mIoU in % (the final `Epoch(test)` line of each run's log; the base
model is the pe run):

| model | Cetran | test | test + Cetran |
| --- | --- | --- | --- |
| base model (pe run) | 35.35 / 33.44 | 46.50 / 48.68 | 46.20 / 47.95 |
| C seed 0 | 36.45 / 34.02 | 47.17 / 48.78 | 46.79 / 48.27 |
| C seed 1 | 34.26 / 33.50 | 46.80 / 48.99 | 46.34 / 48.42 |
| C seed 2 | 34.72 / 33.62 | 47.19 / 48.63 | 46.70 / 48.15 |

**Reading.** No variant beats grouping on any set: all nine verdicts are `worse`, no
variant beats any of the 33 splits (0/33), and every mean improvement also stays below the
base model's flat Energy (9.45 / 21.34 / 16.77 on Cetran / test / test + Cetran); the
highest is C on Cetran, 7.30. The spectrally normalized head matters: A's mean
improvement is 21 to 24 points above pe's on every set and each of its three seeds is
above pe, but A still does not reach flat Energy, and C adds only 1.0 to 5.7 points over
A. C's PQ cost is nil on average on test and test + Cetran (mean PQ 0.55 and 0.41 points
above the base model) and 0.21 points on Cetran (35.14 against 35.35).

**Per seed.** PQ holds for every C seed on test (+0.30 to +0.69 over the base model) and
on test + Cetran (+0.14 to +0.59), but on Cetran alone it moved +1.10, −1.09 and −0.63
points for seeds 0, 1 and 2. Seed 1's validation mIoU after epoch 9 was 37.22, against
43.31 for seed 0, 40.61 for seed 2 and 41.24 for the base model, while its validation PQ,
39.18, stayed above the base model's 38.83; seed 0 is the only C seed that trained
without a resume, so seed and resume cannot be told apart.

The report states two caveats that both favour grouping: the robust splits were selected
on these same sets and their numbers are offline, while OCCUQ's settings were fixed in
advance and its numbers come from the online metric.

### Dead-dimension check

The fit found three channels of `pe_features` that are 0 on every training voxel (116,
190 and 252), so `density_pe`'s Gaussians carry the shared ridge there (6.76e-8), and a
voxel that fired one of them would be charged x² / (2 × 6.76e-8) nats for an activation
x. Whether these dimensions, rather than the Gaussians over the others, explain
`density_pe`'s weak AP was checked on 2026-10-09 at the user's request, with
`../store/occuq/deaddims/deaddims.py` (untracked; one 1-GPU job of about 5 minutes on a
V100; results in `../store/occuq/deaddims/results.json`). It runs the base model on every
10th frame of Cetran (98 frames: 25.3 M ID and 0.75 M OOD points, 5.9 M voxels) and of
test (263 frames: 101.0 M ID and 1.65 M OOD points, 22.3 M voxels); its features
reproduce the shipped `density_pe` exactly on every frame. No voxel fires dim 116, 190
or 252 (0 of 28.1 M), so the three dimensions shift every voxel's −log q by the same
−22.007 nats and leave the ranking unchanged. `density_pe` as shipped, with the three
dimensions marginalized out, and with the firing points dropped are therefore identical
to two decimals, against flat Energy on the same points (AUROC / AP / FPR@95):

| sample | density_pe (all three ways) | flat Energy |
| --- | --- | --- |
| Cetran, 98 frames | 89.26 / 14.76 / 30.05 | 92.75 / 41.48 / 29.45 |
| test, 263 frames | 82.36 / 7.23 / 60.97 | 92.44 / 37.56 / 42.16 |

The sample rows are within 0.4 AUROC, 2.1 AP and 3.1 FPR@95 of the full-set rows. So the
weak AP of `density_pe` comes from the Gaussians over the other 253 dimensions. The
limits: only the three exactly dead dimensions were tested, not near-ridge
eigen-directions, and zero firing in 361 frames bounds the share of firing frames at
about 0.8 % (rule of three).
