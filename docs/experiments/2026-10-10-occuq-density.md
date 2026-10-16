# 2026-10-10 — OCCUQ density scores vs the robust grouping splits (DSO)

Full record of the `DOCs.md` entry of the same date ([DOCs.md](../../DOCs.md)). Paths are relative to the repo root.

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

## Commands

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
three versions differ only in the training step, its header comments and, in version 3,
the `EXTRA` array that feeds it.

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

C's training had to be restarted; the causes are below, so that a rerun on a V100 does
not repeat them.

- **The crashes.** The first runs of C died of CUDA out-of-memory errors on the 32 GiB
  V100s: seed c1 in epoch 1 (18.03 GiB allocated and 20.30 GiB reserved by PyTorch, of
  31.73 GiB) and seed c0 in epoch 6 (17.69 and 20.33 GiB); c2, which shared c1's job, was
  restarted with it. The archived first attempts (`../store/occuq/c/failed_oom_20261009/`)
  are not results.
- **The cause.** The ~11.4 GiB that PyTorch's allocator did not hold matches mmcv
  2.0.0rc4's sparse-convolution index kernels
  (`mmcv/ops/csrc/pytorch/cuda/sparse_indice.cu`, `KernelMaxVolume` 4096): their 64 KiB
  local array per thread keeps about 10 GiB of a V100 for the whole run, which leaves C's
  training (logged peak allocation up to 18.8 GiB) too little room for fragmentation. A
  peaks at about 3.7 GiB and is not at risk.
- **The remedies.** The reruns set the allocator variable and save a checkpoint every
  epoch, from which `run_variant.sh` resumes after a failed attempt. The allocator
  setting alone did not lower the crash rate: seeds c1 and c2 each crashed once more, in
  epoch 3. So from epoch 4 their job ran version 3, which empties PyTorch's cache after
  every iteration and costs about 8 % in speed (median logged time per iteration 1.25 s
  against 1.16 s).
- **The resumes.** After the epoch-3 crash, seeds c1 and c2 each resumed from their
  `epoch_2.pth` (`seed1/epoch_2.pth`, `seed2/epoch_2.pth`), and the version-3 job started
  from each seed's `epoch_3.pth`. So c1 and c2 trained epochs 1–2 from the base
  checkpoint, epoch 3 after a resume from epoch 2 and epochs 4–9 after a resume from
  epoch 3, with no further failure; seed c0 is the only C seed that trained without a
  resume (one attempt, no failure).

The comparison reads logs only, so it ran on the login node:

```bash
python tools/compare_occuq_grouping.py \
  --pe work_dirs/p3former_2xb1_3x_dso_occuq_pe \
  --a work_dirs/p3former_2xb1_3x_dso_occuq_a/seed0 work_dirs/p3former_2xb1_3x_dso_occuq_a/seed1 work_dirs/p3former_2xb1_3x_dso_occuq_a/seed2 \
  --c work_dirs/p3former_2xb1_3x_dso_occuq_c/seed0 work_dirs/p3former_2xb1_3x_dso_occuq_c/seed1 work_dirs/p3former_2xb1_3x_dso_occuq_c/seed2 \
  --out work_dirs/p3former_2xb1_3x_dso_occuq_compare/occuq_vs_grouping.md
```

## Results

Numbers are AUROC / AP / FPR@95 in %. The verdict table and the three set tables are
copied from `occuq_vs_grouping.md`. Each set table lists the base model's five flat
scores, the best robust split's three Group scores and the OCCUQ variants. `(best)` marks
the Group score the verdict compares against; the averaged improvement is the split's mean
I over its three Group scores (grouping's `robust.tsv`), so it repeats on all three rows.
Verdict per variant and set, against the best robust split:

| variant | Cetran | test | test + Cetran |
| --- | --- | --- | --- |
| pe | worse | worse | worse |
| A | worse | worse | worse |
| C | worse | worse | worse |

### Cetran

R (mean of the base model's flat MSP, Energy and Entropy): 91.89 / 35.77 / 29.99

| method | AUROC | AP | FPR@95 | I (like-for-like) | averaged improvement | splits beaten | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat MSP (base model) | 90.42 | 28.27 | 32.32 | -11.30 | | 0/33 | |
| flat MaxLogit (base model) | 93.19 | 41.72 | 26.63 | 10.61 | | 6/33 | |
| flat ODIN (base model) | 94.86 | 44.58 | 23.39 | 18.38 | | 27/33 | |
| flat Energy (base model) | 93.14 | 40.33 | 26.35 | 9.45 | | 0/33 | |
| flat Entropy (base model) | 92.10 | 38.70 | 31.29 | 1.84 | | 0/33 | |
| best robust split `s3.4.8.17`: group_msp (best) | 96.41 | 52.08 | 18.83 | 31.99 | 24.73 | | |
| best robust split `s3.4.8.17`: group_energy | 93.29 | 40.81 | 26.21 | 10.22 | 24.73 | | |
| best robust split `s3.4.8.17`: group_entropy | 96.41 | 52.08 | 18.83 | 31.99 | 24.73 | | |
| OCCUQ pe (1 run) | 89.35 | 15.09 | 28.89 | -22.12 | | 0/33 | worse |
| OCCUQ A (3 runs) | 92.53 | 35.67 | 28.89 | 1.64 [0.16, 2.93] | | 0/33 | worse |
| OCCUQ C (3 runs) | 93.50 | 37.03 | 25.56 | 7.30 [5.97, 9.19] | | 0/33 | worse |

C after fine-tuning (mean over runs): PQ 35.14, mIoU 33.71, own flat Energy 93.87 | 46.58 | 25.64; base model PQ 35.35, mIoU 33.44.

### test

R (mean of the base model's flat MSP, Energy and Entropy): 88.82 / 23.15 / 48.03

| method | AUROC | AP | FPR@95 | I (like-for-like) | averaged improvement | splits beaten | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat MSP (base model) | 86.26 | 13.75 | 51.15 | -15.08 | | 0/33 | |
| flat MaxLogit (base model) | 92.08 | 34.23 | 42.71 | 19.66 | | 0/33 | |
| flat ODIN (base model) | 86.53 | 19.51 | 69.62 | -27.52 | | 0/33 | |
| flat Energy (base model) | 92.36 | 35.47 | 42.55 | 21.34 | | 6/33 | |
| flat Entropy (base model) | 87.84 | 20.23 | 50.40 | -6.27 | | 0/33 | |
| best robust split `s16`: group_msp | 94.21 | 39.75 | 29.06 | 40.97 | 34.78 | | |
| best robust split `s16`: group_energy | 92.39 | 36.47 | 42.56 | 22.37 | 34.78 | | |
| best robust split `s16`: group_entropy (best) | 94.22 | 39.75 | 29.04 | 41.00 | 34.78 | | |
| OCCUQ pe (1 run) | 82.26 | 6.83 | 62.32 | -37.17 | | 0/33 | worse |
| OCCUQ A (3 runs) | 86.90 | 12.51 | 51.18 | -15.70 [-17.04, -13.19] | | 0/33 | worse |
| OCCUQ C (3 runs) | 87.33 | 12.75 | 50.85 | -14.70 [-18.31, -10.15] | | 0/33 | worse |

C after fine-tuning (mean over runs): PQ 47.05, mIoU 48.80, own flat Energy 92.25 | 36.76 | 42.71; base model PQ 46.50, mIoU 48.68.

### test + Cetran

R (mean of the base model's flat MSP, Energy and Entropy): 90.01 / 26.57 / 42.87

| method | AUROC | AP | FPR@95 | I (like-for-like) | averaged improvement | splits beaten | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat MSP (base model) | 87.76 | 18.06 | 45.76 | -13.64 | | 0/33 | |
| flat MaxLogit (base model) | 92.67 | 35.89 | 38.12 | 16.74 | | 3/33 | |
| flat ODIN (base model) | 89.36 | 27.19 | 55.63 | -12.78 | | 0/33 | |
| flat Energy (base model) | 92.88 | 35.54 | 37.95 | 16.77 | | 3/33 | |
| flat Entropy (base model) | 89.38 | 26.11 | 44.91 | -3.12 | | 0/33 | |
| best robust split `s16`: group_msp | 94.04 | 40.45 | 28.52 | 32.27 | 27.42 | | |
| best robust split `s16`: group_energy | 92.90 | 36.40 | 37.97 | 17.63 | 27.42 | | |
| best robust split `s16`: group_entropy (best) | 94.06 | 40.45 | 28.43 | 32.38 | 27.42 | | |
| OCCUQ pe (1 run) | 84.91 | 9.06 | 53.40 | -33.13 | | 0/33 | worse |
| OCCUQ A (3 runs) | 88.75 | 17.93 | 45.27 | -12.29 [-13.66, -9.74] | | 0/33 | worse |
| OCCUQ C (3 runs) | 89.39 | 18.44 | 43.94 | -9.81 [-12.51, -6.13] | | 0/33 | worse |

C after fine-tuning (mean over runs): PQ 46.61, mIoU 48.28, own flat Energy 92.94 | 39.63 | 37.87; base model PQ 46.20, mIoU 47.95.

### Per seed, and reading

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
highest is C on Cetran, 7.30. The density on A's head features scores far better than on
pe's `pe_features`: A's mean improvement is 21 to 24 points above pe's on every set, and
each of its three seeds is above pe. That comparison changes the training and the feature
layer together (a trained head's features against the frozen base's `pe_features`), so
it does not isolate spectral normalization. A still does not reach flat Energy. C's mean
improvement is 1.0 to 5.7 points above A's, but the seed ranges are disjoint only on
Cetran ([5.97, 9.19] against [0.16, 2.93]) and overlap on test and test + Cetran. C's PQ
cost is nil on average on test and test + Cetran (mean PQ 0.55 and 0.41 points above the
base model) and 0.21 points on Cetran (35.14 against 35.35).

**Per seed.** PQ holds for every C seed on test (+0.30 to +0.69 over the base model) and
on test + Cetran (+0.14 to +0.59), but on Cetran alone it moved +1.10, −1.09 and −0.63
points for seeds 0, 1 and 2. Seed 1's validation mIoU after epoch 9 was 37.22, against
43.31 for seed 0, 40.61 for seed 2 and 41.24 for the base model, while its validation PQ,
39.18, stayed above the base model's 38.83; seed 0 is the only C seed that trained
without a resume, so seed and resume cannot be told apart.

The report states two caveats that both favour grouping: the robust splits were selected
on these same sets and their numbers are offline, while OCCUQ's settings were fixed in
advance and its numbers come from the online metric.

## Dead-dimension check

The fit found three channels of `pe_features` that are 0 on every training voxel (116,
190 and 252), so `density_pe`'s Gaussians carry the shared ridge there (6.761e-08), and a
voxel that fired one of them would be charged x² / (2 × 6.761e-08) nats for an
activation x. Whether these dimensions, rather than the Gaussians over the others, explain
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
about 0.8 % (rule of three). The check also did not cover two dimensions that are
exactly 0 within a single class, and both are among those 253 dimensions: in `gmm_pe.pth`,
unpaved-road (class 10, 39,373 voxels) in dimension 191 (class mean 0, pooled variance
0.712) and drain (class 19, 155,261 voxels) in dimension 233 (class mean 0, pooled
variance 0.000615), where the other classes vary; those two Gaussians have only the
ridge, 6.761e-08, as variance there (`../store/occuq/pe/fit.out`: every class gets
6.761e-08 on its diagonal), so a point that is exactly 0 in that dimension gets
−0.5 ln(2π · 6.761e-08) = 7.34 nats of log-density from it and a point anywhere else
gets a steep penalty. The head fits of A and C (all six seeds) have no dimension with
zero variance in any class.
