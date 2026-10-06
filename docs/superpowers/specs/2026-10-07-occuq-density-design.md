# OCCUQ feature-density OOD scores for P3Former, compared with grouping

Date: 2026-10-07
Status: draft, awaiting review
Branch: `ood-baselines/occuq`
Sources:
- Paper: `papers/RelatedPapers/OCCUQ_Exploring_Efficient_Uncertainty_Quantification_for_3D_Occupancy_Prediction.pdf` (Heidrich, Beemelmanns et al., ICRA 2025).
- Code: https://github.com/ika-rwth-aachen/OCCUQ at commit `2aa5429`, read but not run.

## Goal

Port OCCUQ to P3Former as a point-level OOD score on DSO, and decide whether it beats the grouping method of `ood-baselines/grouping`. Grouping here means the 33 robust two-group splits of the bipartition sweep: the splits whose `improvement` is positive on Cetran, on test and on test + Cetran. Their numbers are collected from the existing logs, and no grouping experiment is re-run.

## What OCCUQ does

OCCUQ is DDU (Mukhoti et al.) applied to 3D occupancy, built on SurroundOcc. From the paper and the code:

- **Head** (`MLPHeadv5`, `occ_head.py:199-233`):
  - First, a bias-free linear map C→C with no spectral normalization: SurroundOcc's 1×1×1 conv, widened.
  - Then 4 residual blocks, `x ← x + ReLU(SN-Linear(x))`, and an SN-Linear classifier.
  - Spectral normalization is `torch.nn.utils.spectral_norm` with its defaults (σ normalized to 1, one power iteration). It is on all five layers and stays active at inference, using the stored u and v.
  - There are no normalization layers and no dropout.
  - The paper's "5 layers with skip connections" (Table II) are these 4 blocks plus the classifier.
- **Training.** A fine-tune of the converged SurroundOcc, end to end, for ¼ of its schedule at ¼ of its learning rate: 6 epochs at 5e-5 instead of 24 at 2e-4, cosine, with a 500-iteration warmup. The losses are SurroundOcc's, applied to the new head's logits. There is no uncertainty-specific loss.
- **Gaussians.** After training, one full-covariance Gaussian per class is fitted on the output of block 4 over the training set, using ground-truth labels. The score is log q(z) = logsumexp_c [log N(z; μ_c, Σ_c) + log π_c], with empirical class priors π_c. Low density means high epistemic uncertainty.
- **Evaluation.** Scene-level detection of corrupted camera images. It is not ported (see Non-goals).

## Protocol

Unchanged from the flat baselines and the grouping work:

- Dataset: DSO.
  - OOD: raw 17 (Stop) and 28 (Others).
  - ID: points whose mapped label is not 24.
  - Every other point is excluded.
- `_OODPointMetric`: point-level AUROC, AP and FPR@95 from 2^20-bin histograms, where a higher score means more OOD.
- Sets:
  - Cetran: `dso_infos_cetran.pkl`, 980 frames;
  - test: `dso_infos_test.pkl`, 2,625 frames;
  - test + Cetran: `dso_infos_test_cetran.pkl`, 3,605 frames.
- Base checkpoint: `work_dirs/p3former_2xb1_3x_dso/epoch_36.pth`.
- This spec fixes every OCCUQ setting. None is chosen on an evaluation set.

## Variants

| Variant | What is trained | Feature | Score key |
| --- | --- | --- | --- |
| `pe` | nothing | `pe_features` of the base checkpoint | `density_pe` |
| A, frozen side branch | only the OCCUQ head; the rest of the base model is frozen | output of the OCCUQ head's last block | `density` |
| C, end-to-end fine-tune | the whole model, with the OCCUQ head attached | output of the OCCUQ head's last block | `density` |

`pe` is DDU without spectral normalization.
- It runs first, to check that a density signal exists.
- It stays in the results as the ablation that shows whether the spectrally normalized head matters.
- It depends only on the base model, so it is evaluated once rather than inside every A run.

A and C each run with seeds 0, 1 and 2.

## Model-side changes

### OCCUQ head (`p3former/decode_heads/occuq_head.py`, new)

`_OCCUQHead(in_channels=256, num_classes=25, num_blocks=4)` is a plain `nn.Module` built by `_P3FormerHead`:

- `input_proj = nn.Linear(256, 256, bias=False)`, without spectral normalization.
- `blocks`: 4 × `spectral_norm(nn.Linear(256, 256))`, each applied as `x = x + relu(block(x))`.
- `classifier = spectral_norm(nn.Linear(256, 25))`.
- `forward(x [V, 256]) -> (logits [V, 25], feature [V, 256])`, where `feature` is the output of the last block.

### `_P3FormerHead` (`p3former/decode_heads/p3former_head.py`)

- **New argument** `occuq_cfg: dict | None = None`. With `None`, every existing config behaves exactly as today. Its keys:
  - `head` (bool, default `True`): build `_OCCUQHead` on `pe_features`, parallel to `sem_queries`. `sem_queries` stays because its weights also seed the stuff queries (`p3former_head.py:334`). So the OCCUQ head is added next to it, not swapped in.
  - `freeze_base` (bool, default `False`): variant A. It requires `head=True`; otherwise nothing would train.
  - `loss_weight` (float, default 1.0): the weight of the head's cross-entropy and Lovász losses.
  - `gmm_file` and `pe_gmm_file` (str or `None`): the Gaussians for `density` and for `density_pe`.
  - `score_chunk` (int, default 65,536): voxels per scoring chunk.
- **`forward`** also returns the OCCUQ head's per-sample `(logits, feature)`, or `None` without the head. Both are computed from the same `pe_features`.
- **`loss`:**
  - With `freeze_base`, it returns only the OCCUQ head's losses, `loss_occuq_ce` and `loss_occuq_lovasz`: cross-entropy + Lovász on `voxel_semantic_mask`, ignoring 24, as for the auxiliary branch. `pe_features` come from `mpe`. The transformer decoder, the Hungarian matching and the panoptic losses are not run.
  - Otherwise, it returns the existing losses, plus the OCCUQ head's two losses when the head exists.
- **`predict`:**
  - When `gmm_file` or `pe_gmm_file` is set, it adds `density` or `density_pe` to each sample's OOD score dict, projected to points through `point2voxel_map` like the flat scores.
  - `postprocess_result` already stores every key as `ood_<key>`.
  - Setting either file requires `ood_cfg`.
- **Gaussians at test time:** loaded on the first `predict` call and kept as non-persistent buffers, so they never enter a checkpoint.

### `_P3Former` (`p3former/segmentors/p3former.py`)

When `occuq_cfg.freeze_base` is set:

- every parameter outside `decode_head.occuq_head` gets `requires_grad=False`;
- `train()` keeps every module except `decode_head.occuq_head` in eval mode, so the BatchNorm statistics of the voxel encoder and backbone can't drift;
- `loss` runs `extract_feat` under `torch.no_grad()`.

## Fitting the Gaussians (`tools/fit_occuq_gmm.py`, new)

```bash
python tools/fit_occuq_gmm.py CONFIG CKPT --features head|pe --out FILE [--ann dso_infos_train.pkl] [--check N]
```

- **Data.** The training split (8,474 frames) through the config's **test** pipeline, at batch 1, in eval mode. OCCUQ collects its features with training augmentation; here that would mean LaserMix/PolarMix-mixed scenes.
- **Samples.** One sample per voxel:
  - Feature: `pe_features`, or the output of the OCCUQ head's last block.
  - Label: the majority vote of its points' labels (`pts_semantic_mask` through `point2voxel_map`), with ties going to the lowest class id.
  - Voxels labelled 24 are skipped. Stop and Others map to 24, so the Gaussians never see OOD points.
  - There is no "unoccupied" component, so there are 24 components in total.
- **Statistics.** Per class c in 0–23, running sums on the GPU: the count n_c, Σz and Σzzᵀ. The features are cast to float64 before the products. Every voxel is used: OCCUQ caps the samples at 200M per class and 100k per class per frame only to bound memory, and running sums make that unnecessary.
- **Gaussians**, in float64:
  - Mean and covariance: μ_c = Σz / n_c and Σ_c = (Σzzᵀ − n_c μ_c μ_cᵀ) / (n_c − 1), which is `torch.cov`'s N−1 convention.
  - Jitter, per class: the smallest value from OCCUQ's list (0, 2.2e-308, 1e-308, 1e-307, …, 1e-1) for which `torch.linalg.cholesky(Σ_c + jitter·I)` succeeds. OCCUQ shares one jitter across all classes.
  - Precision Cholesky factor P_c = L_c^{-T}, computed with `torch.triangular_solve`, because torch 1.10 has no `linalg.solve_triangular`.
  - Priors: log π_c = log(n_c / Σ_k n_k), over all voxels. OCCUQ takes them from its capped counts.
- **Output.**
  - Contents: a `torch.save` of a dict of plain tensors, not a pickled distribution object.
    - Tensors: `means` [24, 256], `prec_chol` [24, 256, 256], `log_det_prec` [24] (Σ log diag P_c), `log_prior` [24], `counts` and `jitter`.
    - Metadata: `features`, `checkpoint`, `ann_file`, `class_names` and `fingerprint` (see Error handling).
  - The tool refuses to overwrite an existing file.
  - It logs the per-class counts and jitters. With `--features head` it also logs the head's per-class voxel accuracy on the training split (argmax of the first 24 logits), to show that the head learned.
- **`--check N`.** Re-scores N training frames twice, with the production float32 GPU path and with float64 on the CPU. It fails if log q differs by more than 0.05 nats anywhere.

## Density score (`p3former/utils/gmm_density.py`, new)

Pure functions, unit-testable without a model, like `ood_scores.py`:

- **`log_density(z, gmm)`.**
  - For each class: y = (z − μ_c) · P_c, centred before the product; m_c = Σ y²; and log N_c = log_det_prec_c − ½ (D log 2π + m_c).
  - It returns logsumexp_c (log N_c + log π_c).
  - It runs in float32 on the GPU, with μ_c and P_c cast once, in chunks of voxels.
- **`no_tf32()`.** A context manager that sets `torch.backends.cuda.matmul.allow_tf32 = False` and restores the previous value on exit, even after an exception. Only the density computation runs inside it, so every other score stays bit-identical to the logged runs.
- **Emitted score:** `asinh(−log q)`.
  - It is strictly increasing in −log q, so the exact AUROC, AP and FPR@95 are those of −log q.
  - The compression keeps the metric's equal-width bins fine when a few far-off points have a very large −log q.
- **Non-finite values** raise an error.

## Configs and training recipes

There are three new configs, each built on `p3former_2xb1_3x_dso_ood.py`. Each sets `randomness=dict(seed=0)`. Each has an `_OODPointMetric` whose `score_keys` add the new key to the five flat scores.

- `p3former_2xb1_3x_dso_occuq_pe.py`: evaluation only; `occuq_cfg=dict(head=False)`; key `density_pe`.
- `p3former_2xb1_3x_dso_occuq_a.py`: `occuq_cfg=dict(freeze_base=True)`; key `density`.
- `p3former_2xb1_3x_dso_occuq_c.py`: `occuq_cfg=dict(freeze_base=False)`; key `density`.

The Gaussian file is not in the configs. Each evaluation sets it with `--cfg-options model.decode_head.occuq_cfg.gmm_file=...` (or `pe_gmm_file` for `pe`), because each seed has its own file.

A and C share OCCUQ's fine-tuning recipe, scaled to P3Former (¼ of the base schedule's epochs and learning rate), with P3Former's optimizer settings:

- **Initialization:** `load_from = 'work_dirs/p3former_2xb1_3x_dso/epoch_36.pth'`. The OCCUQ head starts from random weights, through the non-strict load.
- **Optimizer:** AdamW, lr 2e-4, weight decay 0.01. There is no backbone learning-rate multiplier and no gradient clipping: OCCUQ's 0.1× backbone LR and clip at 35 are SurroundOcc's base settings, not part of the method.
- **Schedule:** 9 epochs.
  - Warmup: `LinearLR` over the first 500 iterations, starting at ⅓ of the LR.
  - Then `CosineAnnealingLR` over the 9 epochs, updated every iteration (`convert_to_iter_based=True`), down to `eta_min=2e-7` (10⁻³ of the LR, as in OCCUQ).
- **Data:** the train pipeline unchanged, on 2 GPUs × batch 1.
- **Validation and checkpoints:** validation only after the last epoch (`val_interval=9`), and `CheckpointHook(interval=9, max_keep_ckpts=1)`.
- **Seeds:** 1 and 2 are set with `--cfg-options randomness.seed=N`.

## Runs

- **Storage.** `/` has 32 GB free (99% used), and a checkpoint is ~0.95 GB. So `work_dirs/p3former_2xb1_3x_dso_occuq_{pe,a,c,compare}` are symlinks to directories under `/mnt/sandisk/khoadv/occuq/`.
- **Jobs.** Long jobs start detached (`nohup setsid`), with one evaluation per GPU.
- **Evaluation runs.** `<set>` stands for `cetran`, `test` and `test_cetran`. Every evaluation passes `test_dataloader.dataset.dataset.ann_file=dso_infos_<set>.pkl` and its Gaussian file through `--cfg-options`.

1. **Signal check (`pe`).** Fit with `--features pe` on `epoch_36.pth`, then evaluate `p3former_2xb1_3x_dso_occuq_pe.py` on the three sets.
2. **Variant A, seeds 0–2.**
   1. Train with `CUDA_VISIBLE_DEVICES=0,1 bash dist_train.sh configs/p3former/p3former_2xb1_3x_dso_occuq_a.py 2 --work-dir work_dirs/p3former_2xb1_3x_dso_occuq_a/seed<N> --cfg-options randomness.seed=<N>`.
   2. Fit with `--features head` on `seed<N>/epoch_9.pth`.
   3. Evaluate on the three sets.
3. **Variant C, seeds 0–2.** The same steps, with `p3former_2xb1_3x_dso_occuq_c.py`.

## Comparison and verdict (`tools/compare_occuq_grouping.py`, new)

The tool only reads logs.

- **Grouping inputs.** The 33 splits listed in `work_dirs/p3former_2xb1_3x_dso_ood_dump/divided_mass/robust.tsv`.
  - For each split, its `<split>_group_msp`, `_group_energy` and `_group_entropy` rows, plus the flat `msp`, `energy` and `entropy` rows.
  - These come from `bipartitions{,_test,_test_cetran}/bipartitions.log` and `singletons{,_test,_test_cetran}/bipartitions.log`, for Cetran, test and test + Cetran.
- **OCCUQ inputs.** The `test.py` logs of the runs above. All logs use the same `method | AUROC | AP | FPR@95` row format.
- **Single-score improvement.** For a score x on a set, I(x) = (AUROC_x − R_AUROC) + (AP_x − R_AP) − (FPR@95_x − R_FPR@95).
  - R is the mean of the flat MSP, Energy and Entropy rows from the same estimator: the sweep's flat rows for grouping, and the base model's online flat rows (from the `pe` runs) for every OCCUQ variant, C included.
  - So every method is measured from the same starting point.
  - The two references differ by at most 0.01 per metric.
- **Consistency check.** For every robust split and set, the mean of the split's three I values must reproduce `robust.tsv`'s `improvement` within 0.02. Otherwise the tool stops.
- **Like-for-like rule.** A split's grouping value on a set is the I of its best single Group score: the maximum over Group MSP, Group Energy and Group Entropy.
  - `robust.tsv`'s averaged `improvement` is shown, but it doesn't decide the verdict.
  - It averages in Group Energy, which grouping barely changes, so it favours any single strong score. Under it, flat Energy alone (I = 9.45 / 21.34 / 16.77 on Cetran / test / test + Cetran) would beat 18 / 25 / 21 of the 33 splits.
  - Example, `s16` on test + Cetran:

    | Score | I |
    | --- | --- |
    | Group MSP | 32.27 |
    | Group Energy | 17.63 |
    | Group Entropy | 32.38 |
    | averaged `improvement` | 27.42 |
    | like-for-like value | 32.38 |
- **Verdict.** Per variant and set, against the best robust split on that set:
  - **better** if every seed's I(density) is above the best split's value;
  - **worse** if every seed's I(density) is below it;
  - **too close to call** otherwise.

  `pe` is a single deterministic run, so its verdict is either better or worse.
- **Table, per set.**
  - Rows: flat Energy; the best robust split (its name and its best Group score); `pe`; A and C, each as the mean with the min–max over seeds.
  - Columns: AUROC, AP, FPR@95, the like-for-like I, the averaged `improvement` (splits only), and the rank among the 33, meaning how many splits OCCUQ's mean beats.
  - C also gets its own flat Energy row, PQ and mIoU.
  - The table is printed and written to `work_dirs/p3former_2xb1_3x_dso_occuq_compare/occuq_vs_grouping.md`.
- **Stated in the report.** Both of these points favour grouping:
  - The robust splits were selected on these same sets, and their numbers are offline. As `divided_mass/summary.md` notes, some of their FPR@95 values are limited by float32 ties.
  - OCCUQ's settings were fixed in advance, and its numbers are online.

## Deviations from OCCUQ

1. **Head placement.** The head is added next to the auxiliary branch instead of replacing it, because `sem_queries` also seeds the stuff queries.
2. **Feature level.** One feature level at width 256. P3Former has a single per-voxel feature level. OCCUQ's best of four scales was the 32-d one, but scale and dimension are confounded there (paper Table IV).
3. **Training.** Variant A, with the base model frozen, is ours. C follows OCCUQ's fine-tune, with P3Former's optimizer settings.
4. **Fitting.**
   - The test pipeline instead of training augmentation.
   - Every voxel, in float64 running sums, instead of capped float32 samples.
   - A per-class jitter.
   - Priors from all voxels.
   - Voxel labels by majority vote of their points' class labels. With instance masks, P3Former's training targets vote by panoptic segment first and take that segment's class; the two differ only in voxels shared by several segments.
   - No "unoccupied" class.
5. **Scoring.** Centred before the product, in float32 with TF32 off (OCCUQ's code leaves TF32 on), and emitted as asinh(−log q).
6. **Evaluation.** The project's point-level protocol, not OCCUQ's scene-level one.

## Error handling

Every failure raises an error with a message that says what to do:

- `gmm_file` or `pe_gmm_file` is set but the file is missing: the message names the `fit_occuq_gmm.py` command to run.
- Fingerprint mismatch:
  - The Gaussian file stores a SHA-256 over the `state_dict` entries (sorted by name, as float32 bytes) of the modules that produce its feature:
    - the head's MPE layers `polar_proj`, `polar_norm`, `cart_proj`, `cart_norm` and `pe_conv`;
    - plus `occuq_head` for `--features head`. Its entries include the spectral-norm `weight_orig`, `weight_u` and `weight_v`.
  - `predict` recomputes it and refuses a file fitted for other weights, for example seed 0's Gaussians with seed 1's head.
- Non-finite density.
- Fitting: a class with fewer than 2,560 voxels (10 × 256), a covariance that is still not positive definite at the largest jitter, or an existing output file.
- Comparison: a missing log, set, split row or seed, or a failed `robust.tsv` consistency check.

## Verification

1. **Unit tests**, on the CPU with synthetic tensors; like the existing tests, each file also runs as a script.
   - `tests/test_occuq_head.py`:
     - shapes and layout;
     - every spectrally normalized layer's top singular value is about 1 after a few training-mode passes;
     - with `freeze_base`, `train()` leaves the frozen modules in eval mode, and only the `occuq_head` parameters are trainable.
   - `tests/test_gmm_density.py`:
     - the log-density matches `torch.distributions.MultivariateNormal.log_prob` in float64;
     - the prior-weighted logsumexp matches a direct computation;
     - a point at a class mean scores lower than a far point;
     - `no_tf32()` restores the flag after an exception;
     - on heavy-tailed synthetic scores, the binned metrics of `asinh(s)` match exact brute-force metrics within 1e-3;
     - a fingerprint mismatch raises an error.
   - `tests/test_fit_occuq_gmm.py`:
     - running sums over random chunks reproduce `torch.mean` and `torch.cov` of the concatenated data;
     - the majority vote is correct on a hand-built case;
     - label-24 voxels never reach the sums;
     - the jitter search gives 0 for a positive-definite covariance and the smallest working value for a singular one;
     - the minimum-count error fires.
   - `tests/test_compare_occuq_grouping.py`: row parsing; I and the best single Group score; the verdict rule on toy seed values; the consistency check.
2. **Smoke run** of each evaluation config on the 5-frame `dso_infos_mini.pkl`: the new keys are present and finite.
3. **Frozen-path check.** A's first evaluation reproduces the logged PQ (46.50 on test, 46.19 on test + Cetran) and the flat rows exactly.
4. **Precision.** `--check 5` passes for every Gaussian file.
5. **Documentation.**
   - `DOCs.md` gets an entry with the commands and tables.
   - On `ood-baselines/occuq`, these are updated: `.claude/CLAUDE.md` (it still describes `flat`), `rules/architecture/ood-pipeline.md`, `rules/configs.md`, and the test count in `rules/testing.md`.

## Budget

- **Training:** about 2 h per seed for A and 5 h for C, on both GPUs. A full step of the base model takes 0.45 s; A runs no decoder and no backbone backward.
- **Fitting:** one forward pass over 8,474 frames, about 15–30 min on one GPU, for each of the 7 Gaussian files.
- **Evaluation:** about 5 / 12 / 15 min on Cetran / test / test + Cetran, for each of the 7 models.
- **Total:** about 28 h of wall-clock time if the runs go back to back.
- **Memory:** an evaluation needs about 25 GB of GPU memory. The evaluator holds (4 × 6 + 1) bytes per point, about 32 GB on test + Cetran.
- **Disk:** about 1 GB per run (one checkpoint) plus 12.6 MB per Gaussian file, all on `/mnt/sandisk`.

## Files

New:

- `p3former/decode_heads/occuq_head.py`
- `p3former/utils/gmm_density.py`
- `tools/fit_occuq_gmm.py`
- `tools/compare_occuq_grouping.py`
- `configs/p3former/p3former_2xb1_3x_dso_occuq_{pe,a,c}.py`
- `tests/test_occuq_head.py`, `tests/test_gmm_density.py`, `tests/test_fit_occuq_gmm.py` and `tests/test_compare_occuq_grouping.py`

Modified: `p3former/decode_heads/p3former_head.py`, `p3former/segmentors/p3former.py`, `DOCs.md`, and the `.claude/` files listed under Verification.

Everything stays on `ood-baselines/occuq`; none of it is shared with `flat` or `grouping`.

## Non-goals

- OCCUQ's scene-level and region-level corruption evaluation, its uncertainty-guided temperature scaling (UGTS), and its MC Dropout and Deep Ensembles baselines.
- SemanticKITTI: the grouping comparison exists only for DSO.
- Re-running or confirming any grouping split online, and combining OCCUQ with grouping.
- A uniform-prior variant. The paper is silent, the code uses priors, and choosing between the two on the evaluation sets would be selection.
- Changes to the shared metric, `evaluation/functional/ood_eval.py`.
