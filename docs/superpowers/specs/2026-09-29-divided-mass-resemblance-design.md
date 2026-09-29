# Why two-group splits beat flat: divided mass and OOD–class resemblance

Date: 2026-09-29
Status: approved in chat (2026-09-29); written spec awaiting review
Branch: `ood-baselines/grouping`

## Goal

Two explorations, both reported on Cetran, Test and Test + Cetran:

1. **Divided mass.** Explain from the logit dumps why some two-group splits A | B beat the
   flat scores, focusing on the single-class splits {c} | rest. Measure the divided mass of
   all 24 single-class splits and of the 500 sweep splits, relate it to ΔAUROC, ΔAP and
   ΔFPR@95, vary the threshold, and draw bubble charts in the style of
   `trash/bubble_chart.py`, one for the single-class splits and one for the robust splits.
2. **Resemblance.** Comment on the hypothesis "the best split puts the classes that the OOD
   objects resemble on one side and everything else on the other". Measure, in the
   penultimate feature space of P3Former, how much the OOD points overlap each ID class,
   and check whether this agrees with (1).

There is no retraining and no change to the model code or configs. Part 2 is correlational:
apart from completing the 24 single-class splits, no new split is built or scored.

## Background

- **Group MSP.** For a split A | B the score is −max(P_A, P_B), where P_g is the softmax
  mass of group g. A point is flagged only when its mass is divided across the boundary,
  and confusion within one side is absorbed.
- **Sweep.** The 2026-09-15 sweep scored 500 random splits (seed 0) on each set, into
  `bipartitions{,_test,_test_cetran}/bipartitions.log`. It sampled 21 of the 24
  single-class splits; {unpaved-road}, {sidewalk} and {building} are missing.
- **Robust splits.** 32 splits have `improvement` > 0 on all three sets. Among the
  single-class splits only {overhead-bridge} (+7.0 / +34.8 / +27.4 on Cetran / Test /
  Test + Cetran) and {gate} (+17.5 / +6.6 / +10.1) are robust.
- **Mechanical link.** For a two-group split, m = min(P_A, P_B) equals Group MSP + 1.
  "Divided at δ" (m > δ, i.e. δ < P_A < 1 − δ) is therefore the Group MSP detector at
  threshold δ − 1: the OOD divided share is its TPR and the ID divided share its FPR. One δ
  gives one ROC operating point, and sweeping δ from 0.5 to 0 traces the whole curve, so a
  correlation between divided mass and ΔAUROC is partly true by construction. The
  explanation therefore rests on two comparisons:
  - with flat MSP at the same δ. Its analogue u = 1 − max_c p_c satisfies u ≥ m for every
    point, so the divided points are a subset of the flat-uncertain ones;
  - on δ95, the depth to which the threshold must go before 95% of the OOD points are
    divided. The ID divided share at δ95 is the split's FPR@95.

## Sets, splits and metrics

All paths are under `work_dirs/p3former_2xb1_3x_dso_ood_dump/`.

| set | key | logit dumps | feature samples | info pkl |
| --- | --- | --- | --- | --- |
| Cetran | `cetran` | `logits/` (980 frames) | `features_cetran/` | `dso_infos_cetran.pkl` |
| Test | `test` | `logits_test/` (2,625 frames) | `features_test/` | `dso_infos_test.pkl` |
| Test + Cetran | `test_cetran` | both | both | = `dso_infos_test_cetran.pkl` |

- **Splits.** The 500 sweep partitions plus the 24 single-class splits make 503 distinct
  splits. They are named as in the sweep: `s<ids of the smaller side>`, with A the smaller
  side.
- **Deltas.** Δ = Group score − flat score for AUROC, AP and FPR@95.
- **Improvement.** mean ΔAUROC + mean ΔAP − mean ΔFPR@95 over Group MSP, Group Energy
  and Group Entropy, with ODIN and MaxLogit excluded. This is exactly
  `summarize_hierarchy_ablation.py --family group --exclude odin`.
- **Robust split.** improvement > 0 on all three sets.

## Part 1 — divided mass

### 1a. Complete the single-class splits (`tools/sweep_bipartitions.py --subsets`)

The new option `--subsets SPEC` replaces the random sample, and the always-included
vehicle-vs-rest reference, with an explicit list. `SPEC` is either `singletons` (the 24
single-class splits) or the path to a JSON list of class-id lists. Scores, bins and the
log format are unchanged. It is run with the torch backend for the 24 single-class splits
on each set, writing `singletons{,_test,_test_cetran}/`. The 21 splits present in both the
sweep and the singletons logs must agree to within 0.02: blocks of different shape may
round the group masses differently in the last ulp.

### 1b. Divided mass (`tools/divided_mass.py`)

Per valid point, from the float16 logits upcast to float32:

- p = softmax(z). P_A and P_B are summed directly, never as 1 − P_A, which cancels for
  confident points. m = min(P_A, P_B).
- The flat reference is u = Σ_{c ≠ argmax} p_c, which is 1 − max_c p_c without the
  cancellation.
- m (for every split) and u are histogrammed separately for ID and OOD points. The bins
  are log-spaced over [1e-15, 0.5] at 200 per decade, with an underflow bin [0, 1e-15).
  Every reported δ is an exact bin edge. The reported grid is
  δ ∈ {1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3}.
- There is one pass per dump directory, with torch on a GPU or numpy on CPU, as in the
  sweep. A set's histograms are the sum of its directories' histograms, so Test + Cetran
  costs no extra pass.

Per split, set and δ:

- **OOD / ID divided %**: the share of OOD / ID points with m > δ, plus the counts.
- **Divided precision**: the OOD divided count / (OOD + ID divided count). It uses
  absolute counts, since ID points outnumber OOD points 33× on Cetran and 63× on Test.
- **Retention vs flat**: OOD divided / OOD uncertain and ID divided / ID uncertain, where
  uncertain means u > δ. Their ratio is the **selectivity**; above 1, the split removes
  relatively more ID uncertainty than OOD uncertainty.
- **δ95**: the largest bin edge at which ≥ 95% of the OOD points are divided, and the ID
  divided % there (≈ the FPR@95 of Group MSP). Flat MSP gets the same (u95).

From the sweep and singletons logs, each split also gets its Group MSP AUROC / AP /
FPR@95, the Δ against flat MSP, the improvement and the robust flag. For the 21 splits in
both logs the 500-split sweep log wins, so the existing numbers stay as they are; the
singletons log supplies the three missing splits. The robust count (32 today) is
recomputed with those three included.

**Correlations.** Spearman ρ of each statistic at each δ against ΔAUROC, ΔAP, ΔFPR@95
(Group MSP − MSP) and improvement. The statistics are OOD divided %, ID divided %, the log
OOD/ID divided ratio (with +0.5 added to zero counts) and divided precision. ρ is computed
per set, over the 24 single-class splits and over all 503 splits.

**Checks.** These go to `summary.md`, where a check beyond tolerance shows as FAIL and is
never hidden:

- Group MSP AUROC / AP / FPR@95 recomputed from the m histograms against the sweep logs,
  within 0.1 for AUROC and AP and 0.5 for FPR@95. The m bins are coarser than the sweep's
  2^16.
- Flat MSP recomputed from the u histograms against the sweep's `msp` row.
- The 21 single-class splits that appear in both logs.

**Outputs** (`divided_mass/`):

- `histograms.npz`;
- `cetran.tsv`, `test.tsv` and `test_cetran.tsv`, one row per split;
- `flat.tsv`, the flat reference per set and δ;
- `robust.tsv`;
- `summary.md`, with the checks, the single-class table at δ = 0.05, the robust table and
  the ρ tables.

A module-level `SETS` table maps each set to its dump directories and sweep outputs under
`--root` (default `work_dirs/p3former_2xb1_3x_dso_ood_dump`), in the spirit of `GROUPS` in
`make_dso_hierarchy_variants.py`. The computing functions take explicit arguments, and the
tests call them directly.

### 1c. Figures (`tools/plot_divided_mass.py`)

The tool reads `divided_mass/` and writes each figure as a PDF (vector) and a PNG (300 dpi)
in the style of `trash/bubble_chart.py`. That style is:

- serif (Times) text, with TrueType fonts embedded;
- Okabe–Ito blue for a gain and orange for a drop;
- bubble area ∝ |improvement|;
- log–log axes.

The panels are (a) Cetran, (b) Test and (c) Test + Cetran.

- **`bubble_singletons`**: x = ID divided %, y = OOD divided % at `--threshold` (default
  0.05).
  - Each of the 24 bubbles is labelled "class / ±improvement". The labels are placed
    automatically, at the candidate position around each bubble with the least overlap
    with other labels and bubbles.
  - Each panel also marks flat MSP at the same δ with a star (ID uncertain %, OOD
    uncertain %). An iso-ratio diagonal runs through the star: a point above it has a
    higher OOD:ID ratio than flat.
  - A value of 0 sits on the axis floor with an open marker.
- **`bubble_robust`**: the same axes. The robust splits are bubbles and every other split
  is a small grey dot. The top-N robust splits (default 8, ranked by their worst-set
  improvement) are labelled by name; their compositions are in `robust.tsv`.
- **`rho_vs_threshold`**: ρ against δ (log x) for the four statistics, with one row per
  target (ΔAUROC, ΔAP, ΔFPR@95, improvement) and one column per set. Single-class splits
  are solid lines and all splits dashed.

## Part 2 — resemblance in the feature space

### 2a. Feature samples (`tools/extract_point_features.py`)

- **Layer.** `pe_features` is the per-voxel 256-d input of the auxiliary semantic
  classifier: `sem_preds` = `pe_features` · `sem_queries`ᵀ, a bias-free 1×1 convolution.
  It is the backbone feature after `pe_conv` plus the positional embedding `mpe`
  (cartesian + polar). `_P3FormerHead.init_inputs` returns
  `(queries, pe_features, mpe, sem_preds)`, so both are in its return value.
  The tool wraps that method on the model instance at runtime, with no change to the model
  code or the configs.
- **Run.** The model is built from the config as in `test.py` (`custom_imports`, default
  scope, checkpoint). The test dataloader reads `--ann` with batch size 1, on one GPU,
  under `torch.no_grad` and with TF32 off.
- **Per frame.** Voxel tensors are mapped to points through `gt_pts_seg.point2voxel_map`.
  Ground truth follows `_OODPointMetric`: a point is OOD when its raw id
  `pts_instance_mask % 2**16` is 17 (Stop) or 28 (Others), and ID when its mapped label is
  not 24.
- **Sampling.** Each frame uses an RNG seeded by (seed, frame index) and draws, uniformly
  without replacement within each stratum:
  - up to `--per-class` (64) ID points per mapped class;
  - up to `--ood-per-frame` (512) OOD points.

  Each sampled point gets the weight "stratum size / points drawn", so weighted means
  estimate population means.
- **Stored per sampled point**, one `f<frame:06d>.npz` per frame:
  - `feat`: f16 [n, 256];
  - `pos`: f16 [n, 256], the `mpe` part;
  - `logits`: f16 [n, 24], the first 24 channels;
  - `label`: int16, the mapped id, or 24 for OOD;
  - `raw`: int16, the raw semantic id;
  - `ood`: bool;
  - `weight`: float32;
  - `index`: int32, the point's index in the frame;
  - plus `lidar_path` and `frame`.

  A `meta.json` records the config, checkpoint, ann file, sampling parameters and counts.
  The tool refuses an out dir that is not empty.
- **Checks.**
  - On every frame, feat @ W[:24]ᵀ (W = `sem_queries.weight`) must reproduce the captured
    logits, within max |diff| ≤ 1e-3 · max(1, max |z|) in float32, before the float16
    cast. This proves the layer.
  - `--check-dump DIR`: frames are matched to a logits dump by `lidar_path`, and at
    `index` the sampled logits must equal the dump's to within 0.05, and `mapped` and
    `ood` exactly.
- **Cost.** About 10 min (Cetran) and 25 min (Test) at ~25 GB of GPU memory each, on both
  GPUs in parallel. The output, ~2 GB and ~6 GB, goes to
  `/mnt/sandisk/khoadv/P3Former_OOD/p3former_2xb1_3x_dso_ood_dump/`, symlinked as
  `features_cetran` and `features_test`.

### 2b. Resemblance (`tools/ood_class_resemblance.py`)

Each set is analysed in two spaces: `full` (`feat`, what the classifier sees) and
`appearance` (`feat − pos`). Test + Cetran is the union of both sample sets.

- **Reference bank.** Per ID class, up to `--bank-per-class` (4,000) samples, drawn in
  proportion to their weight and L2-normalised. The remaining ID samples provide up to
  `--query-per-class` (2,000) ID queries per class, disjoint from the bank. A class with
  fewer than 30 samples is left out of the bank and flagged.
- **kNN.** Cosine similarity with `--k` 10 on a GPU, in chunks; a k = 50 run serves as the
  sensitivity check. Each query gets the class counts n_c(x) of its k neighbours.
- **Per class c:**
  - r_OOD(c): the weighted mean of n_c(x)/k over OOD points, i.e. the share of the OOD
    neighbourhood that class c occupies. 1/24 ≈ 4.2% means no preference.
  - r_ID(c): the same over ID points of the other classes, weighted by population (each
    class's queries carry its population share). It measures how much class c's region
    overlaps the other ID classes and is the feature-space analogue of ID divided.
  - the contrast r_OOD(c) / r_ID(c);
  - as a robustness check, the nearest-centroid share for OOD points (cosine to the
    weighted class means);
  - the classifier's view: the weighted mean softmax P_c and the argmax share over OOD
    points, and the mean P_c over ID points of other classes;
  - the OOD columns again per raw OOD class (Stop, Others).
- **Per split A | B** (all 503):
  - R_A = Σ_{c∈A} r_OOD(c), the OOD resemblance mass on the smaller side, and the
    enrichment E_A = R_A / (|A|/24);
  - **feature-divided**: a point is feature-divided when its k neighbours include classes
    from both sides. This gives an OOD and an ID (population-weighted) feature-divided %.
- **Alignment with Part 1.** Spearman ρ per set and space:
  - over the 24 classes: r_OOD(c), r_ID(c), r_OOD / r_ID and the OOD / ID feature-divided
    % of {c} | rest, against the OOD / ID divided % (δ = 0.05) and the improvement of
    {c} | rest;
  - over the 503 splits: OOD / ID feature-divided % against OOD / ID divided %
    (δ = 0.05);
  - **the hypothesis as stated** ("together"): improvement against R_A and E_A;
  - **the alternative that Part 1 suggests** ("cut through"): improvement against the OOD
    feature-divided % and the OOD/ID feature-divided ratio. For example, {overhead-bridge}
    scores +34.8 on Test while {building, overhead-bridge} scores only +14.3.

  The robust splits get their own table.
- **Outputs** (`resemblance/`):
  - `profile_<set>_<space>.tsv`, one row per class;
  - `splits_<set>_<space>.tsv`, one row per split;
  - `summary.md`, with the bank sizes, profiles, ρ tables, robust table and the evidence
    for or against each hypothesis.

### 2c. Figures (`tools/plot_ood_class_resemblance.py`)

- **`profile`**: per set, the 24 classes sorted by r_OOD. Bars show r_OOD and r_ID in the
  full space, hollow markers the appearance space. The bar colour and label give the
  improvement of {c} | rest.
- **`bubble_feature_singletons`**: the feature-space twin of `bubble_singletons`, with
  x = ID feature-divided %, y = OOD feature-divided % of {c} | rest and area =
  |improvement|. Style and panels match.
- **`hypothesis`**: per set, improvement against R_A ("together") and against the OOD
  feature-divided % ("cut through") over the 503 splits, with the robust splits
  highlighted and ρ in the panel titles.

## Testing and verification

- **Unit tests** follow the repo convention (pytest functions plus a
  `__main__` runner) and use synthetic data:
  - `tests/test_sweep_bipartitions.py`, extended: `--subsets` (singletons, JSON), naming,
    end-to-end rows.
  - `tests/test_divided_mass.py`:
    - m and u against brute force, including confident points (no cancellation);
    - divided counts from the histograms against direct counts at every δ;
    - histogram metrics against `binary_ood_metrics`;
    - several directories give the sum of the histograms;
    - the torch and numpy backends agree;
    - robust selection and the table join.
  - `tests/test_extract_point_features.py`: stratified sampling (counts, weights,
    determinism), the capture wrapper on a stand-in head, the voxel → point gather, and
    the logit-reconstruction check (it must raise on a mismatch).
  - `tests/test_ood_class_resemblance.py`:
    - kNN shares on a Gaussian mixture where the OOD cluster sits next to one class;
    - weighting;
    - feature-divided against brute force;
    - centroid shares.
  - Plot smoke tests run both plotting tools on synthetic outputs with the Agg backend.
- **Smoke run** of the extraction on `dso_infos_mini.pkl` (5 test frames with Stop /
  Others), with `--check-dump logits_test`.
- **Review.** An agent reviews the new scripts for formality and maintainability. Its
  findings are addressed before the full runs are reported.

## Documentation

- `DOCs.md`: two dated entries, divided mass and resemblance, each with its commands,
  tables and reading.
- `.claude/CLAUDE.md`: the progress log and the next step.
- This spec, and the plan in `docs/superpowers/plans/`.
- One commit per task on `ood-baselines/grouping`. Nothing is pushed.

## Out of scope

- Scoring new splits built from the resemblance (the test is correlational, by choice).
- The GN family: the offline GN MSP and GN Entropy rows are invalid (2026-09-24).
- Per-sequence transfer, online confirmation, and t-SNE / UMAP views.

## Risks and caveats

- The divided-mass statistics are ROC operating points of Group MSP, so their correlation
  with ΔAUROC is partly mechanical. The explanation rests on the flat comparison and on
  δ95.
- The kNN profile depends on k and on the class-balanced bank; k = 50 checks it.
- `appearance` = `feat − pos` removes the added positional embedding, but not position
  information already present in the backbone features.
- The reference is the evaluation split's own ID points, by choice: the measure is
  resemblance to the classes as they appear in that split, not as learned in training.
- Automatic label placement may need manual touches for a camera-ready figure.
- The three sets are not independent (Test + Cetran contains the other two), which the
  robust definition inherits.
