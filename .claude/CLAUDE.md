# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Fork of [P3Former](https://github.com/InternRobotics/P3Former) (DETR-style transformer for LiDAR point cloud panoptic segmentation, on the OpenMMLab mmdetection3d stack), used as the baseline for out-of-distribution (OOD) panoptic segmentation research (`origin` = DuongVanKhoa0811/P3Former_OOD, `upstream` = InternRobotics/P3Former). On top of upstream it adds:

- **DSO dataset** support: a 24-class LiDAR dataset with a native PLY loader, an info-pkl converter, and extra Cetran test sets.
- **Post-hoc point-level OOD scoring** from the auxiliary semantic branch: MSP, MaxLogit, ODIN, Energy and Entropy, each also in hierarchy-aware Group and Group-Normalised (GN) variants, plus an OOD metric.
- **Offline analysis tools** that recompute any score or class hierarchy from dumped per-point logits, without running the model.

This is branch `ood-baselines/grouping`: the flat baselines of `ood-baselines/flat` plus the Group / GN work; `rules/branches.md` lists the OOD branches and how changes move between them. `trash/`, `work_dirs/`, `data/`, `checkpoint/`, `papers/`, a root-level `CLAUDE.md` and `AGENTS.md` are gitignored.

## Rules

The detailed instructions are split by topic into `.claude/rules/`. Rules without `paths` frontmatter load at the start of every session; the scoped ones load when Claude reads a matching file.

| Rule | Topic |
| --- | --- |
| `branches.md` | the OOD branches and how changes move between them |
| `environment.md` | Python interpreter, pinned stack, run from the repo root |
| `commands.md` | train, test, multi-GPU and data-preparation commands; outputs and checkpoints |
| `testing.md` | unit-test suite |
| `configs.md` | config naming, variants, generated configs, overrides |
| `experiment-log.md` | `DOCs.md` working log and reference documents |
| `machine-resources.md` | GPU, RAM and disk budgets; detached long jobs |
| `dso-dataset.md` | DSO class set, OOD classes, splits |
| `offline-tools.md` | analysis scripts in `tools/` |
| `torch-cuda-pitfalls.md` | TF32 and CUDA `bincount` in torch 1.10 (scoped: Python code) |
| `architecture/registry.md` | registry pattern and `custom_imports` |
| `architecture/data-pipeline.md` | datasets, loading, class mapping, voxelization (scoped: data code) |
| `architecture/model.md` | segmentor and P3Former head internals (scoped: `p3former/`) |
| `architecture/ood-pipeline.md` | how OOD scores flow from the head to the metric |

## Progress log (updated 2026-10-06)

Numbers are AUROC / AP / FPR@95 in %. The dates in brackets are the `DOCs.md` entries with the commands and full tables.

### Completed from 2026-08-12 to 2026-10-06

- **Flat OOD baselines**: MSP, MaxLogit, ODIN, Energy and Entropy, computed from the auxiliary semantic branch. They were evaluated on SemanticKITTI val (official and our 2xb1 checkpoint) and on DSO test and test + Cetran [08-12, 08-19, 08-21].
- **Group and GN scores** (GroupPaper) with the six-group hierarchy [08-21].
- **Hierarchy ablation on Cetran**: all 202 set partitions of the six groups plus one split hierarchy, ranked by `improvement`. `p_v_hgcno` (vehicle vs rest) won [08-25, 09-10].
- **Per-point logit dumps**:
  - Cetran: 980 frames, 15 GB, in `logits/`.
  - Held-out test: 2,625 frames, 55 GB on `/mnt/sandisk`, symlinked as `logits_test/`.

  The two together are test + Cetran [09-10, 09-15].
- **ID/OOD score-distribution figure** comparing the current hierarchy with `p_v_hgcno`, plus a report caption [09-10].
- **Bipartition sweep**: 500 random two-group class partitions scored offline on Cetran, test and test + Cetran, with a GPU backend. Three precision fixes went in (TF32 off, sort-based counts, log-spaced bins), and the results were validated against the online metric [09-15].
- **Codex review**: the offline GN MSP and GN Entropy rows are invalid. The GN ranking was withdrawn, and the summarizer now refuses `--family gn` on sweep logs [09-24].
- **Housekeeping**:
  - The project instructions are split into `.claude/rules/`.
  - About 0.9 GB of obsolete runs were removed from `work_dirs/`.
  - Branch `duy/ood-baselines` was created for the collaborator.
- **Branch split** (2026-09-29): the flat baselines now live on `ood-baselines/flat`, and this branch replays the grouping commits on top of it. The old `ood-baselines` is kept as `archive/ood-baselines-pre-split`.
- **Single-class sweeps and divided mass** [09-30]:
  - all 24 single-class splits scored on the three sets;
  - the divided mass m = min(P_A, P_B) of all 503 splits, validated against the sweep, with all 21 checks passing;
  - 33 splits beat flat on all three sets.
- **Feature samples and OOD resemblance** [09-30]:
  - `pe_features` samples of Cetran (1.25 M) and test (3.0 M), on `/mnt/sandisk` and symlinked as `features_{cetran,test}`;
  - the kNN resemblance per class and per split, the literal placement test, a cross-set bank and a t-SNE view;
  - re-run on 2026-10-04 after a sampling fix (the bank and the ID queries come from one shuffled weighted draw), with seeds 0–2.
- **Reviews of the split-analysis tools** [09-30]:
  - a maintainability review of the six new tools found 13 issues, all fixed, with a test for each behavioural fix;
  - a final whole-branch review (2026-10-04) corrected two documented conclusions, the Cetran → test transfer mechanism and the online metric's resolution, and removed a sampling bias in the kNN bank;
  - a re-review then recomputed about 840 documented numbers from the outputs;
  - the branch was pushed to origin on 2026-10-06.

### Current status

| Component | Status |
| --- | --- |
| Trained models | `work_dirs/p3former_2xb1_3x_{dso,semantickitti}/epoch_36.pth`. DSO test PQ 46.50, mIoU 48.66. SemanticKITTI val PQ 60.32, against 62.63 for the official checkpoint. |
| Online OOD scoring and metric | Complete; the suite passes (count in `rules/testing.md`). Energy is the best flat score, 92.88 / 35.55 / 37.94 on DSO test + Cetran. GN Energy (online, valid) reaches 93.94 / 36.31 / 29.96. Known limit: the metric's 2^20 equal-width bins cannot separate Group MSP points with divided mass m below ≈ 4.8e-7, which affects splits whose OOD points carry almost no mass on the far side (see "Divided mass"). |
| Six-group hierarchy ablation | Run on Cetran only. The winner `p_v_hgcno` (96.00 / 47.23 / 18.42, against 90.42 / 28.27 / 32.32 for flat MSP) **does not transfer**. On test its Group MSP is 87.21 / 17.23 / 68.02, against 86.26 / 13.76 / 51.15 for flat MSP. |
| Bipartition sweep, Group family | Done on all three splits; the outputs are in `work_dirs/p3former_2xb1_3x_dso_ood_dump/bipartitions{,_test,_test_cetran}/`. Best on Cetran: `s1.3.17` {bicycle, truck, gate}, 96.84 / 54.34 / 17.13. Best on test and on test + Cetran: `s16` {overhead-bridge}, 94.04 / 40.45 / 28.52 on test + Cetran, against 87.76 / 18.06 / 45.76 for flat MSP. 32 splits beat flat on both Cetran and test. With the 24 single-class splits added, 33 beat flat on all three sets. The Cetran~test rank correlation of the improvement is only +0.61. **None is confirmed online yet.** |
| Bipartition sweep, GN family | Offline GN MSP and GN Entropy are invalid, because those scores pile up at an interior value that the bins don't resolve. They are not ranked. Fixing this needs interior-adaptive bins or online runs. |
| Divided mass | Done on all three sets, in `work_dirs/p3former_2xb1_3x_dso_ood_dump/divided_mass/`. Divided precision, the OOD share of the points with mass on both sides, predicts the improvement: ρ 0.72–0.74 over all splits at δ = 0.05, and 0.79–0.84 at δ = 0.3. The exact FPR@95 equals the ID divided share at δ95. The implemented float32 Group MSP ties below m ≈ 6e-8, and `test.py`'s 2^20 equal-width bins tie below m ≈ 4.8e-7. Its FPR@95 is therefore unreliable on 21 / 50 / 43 splits; the tool reports exact values next to it. |
| OOD resemblance | Done: `resemblance/` (k = 10), `resemblance_k50/` and `resemblance_xref/` (the Test + Cetran bank). Re-run on 2026-10-04 after a sampling fix, with seeds 0–2. The OOD points resemble building, perimeter-barrier and gate. The class-level ρ(r_OOD, improvement) is 0.76 on test, and 0.92 without the positional embedding (seed spread ≈ 0.03). Grouping the resembled classes gets mixed support; the cut-through log ratio predicts on every set (ρ 0.48–0.70). |
| Collaborator branch | `origin/duy/ood-baselines` forks from the pre-split history at `90af8ea`, whose counterpart on this branch is `749328a`. It adds `tools/ood_distance.py` (Mahalanobis feature-distance OOD plus a 203-partition sweep, 2026-09-07). It is not merged, and it lacks everything after that commit. |

### Findings: why some two-group splits beat the flat scores (2026-09-30)

The full tables and commands are in the two `DOCs.md` entries of 2026-09-30. The spec is `docs/superpowers/specs/2026-09-29-divided-mass-resemblance-design.md`.

- **Mechanism.** Group MSP flags only the points whose mass is divided across the split. A split beats flat when it keeps far more of flat MSP's uncertain OOD points than of its uncertain ID points. Divided precision is the best single predictor.
- **FPR@95 is set by depth.** The exact FPR@95 equals the ID divided share at δ95, the depth at which 95 % of the OOD points are divided. {bicycle} on test looks clean at δ = 0.05 but has δ95 ≈ 1e-9, so it scores −40.1.
- **Why Cetran's winners do not transfer: the OOD population shifts.** Test's OOD points lean far less toward truck, gate and bicycle (truck 15.6 → 6.4 % of the OOD points divided), while the ID divided share of these classes stays tiny on both sets. The ID:OOD ratio also doubles. Truck and bicycle fail on test; gate weakens but stays robust, and overhead-bridge becomes the best split.
- **Resemblance.** The OOD points look like structures: building, perimeter-barrier and gate.
  - Resemblance predicts the single-class winners.
  - Overhead-bridge is the exception: the kNN sees it only without the positional embedding.
  - The literal hypothesis, grouping the resembled classes on one side, gets mixed support. On test it is about as good as separating them, on Cetran it is worse, and lumping them with everything else is the worst placement. The cut-through log feature-divided ratio is the most consistent predictor (ρ 0.48–0.70 on every set).
- **Open.**
  - Confirm the robust splits online (`class_groups_variants`) on data they were not selected on.
  - Before that, fix the online resolution. Either score Group MSP as log min(P_A, P_B), which is monotone in m and survives float32 storage and equal-width bins, or give `evaluation/functional/ood_eval.py` log-spaced bins. Until then, {gate} on test (δ95 = 4.0e-7) cannot be confirmed online, and its test + Cetran FPR@95 reads about 5 points high.
  - Divided precision is a cheaper proxy, not a label-free selector: it needs the OOD labels of the set it is computed on. Test whether a split picked by it on one set transfers to another.

### Key decisions and how they were handled

- **Score source.** Scores are post-hoc, from the auxiliary semantic branch's first `num_ood_logits` channels with the ignore channel dropped. There is no retraining; this was the approved design on 08-12.
- **Protocol.** REL-style point-level AUROC / AP / FPR@95, where a higher score means more OOD.
  - OOD classes: SemanticKITTI raw 52/99 and DSO raw 17/28. Other ignored points are excluded.
  - Hyperparameters come from `OOD_Baseline.pdf`: ODIN T=1000 with ε=0, and Energy T=1.
- **Sign convention.** When porting Group/GN, the scores use `-max` rather than the reference script's `1 - max`. This was the user's choice, and the rankings are unchanged.
- **MaxLogit excluded** from the Group/GN summaries, because `group_maxlogit` ≡ `maxlogit` for a partition. This was the user's call.
- **Ranking metric.** improvement = mean ΔAUROC + mean ΔAP − mean ΔFPR@95, as defined by the user.
- **Memory.**
  - The metrics are histogram-based (2^20 bins).
  - The PQ metric stores only its two masks.
  - The hierarchy runs are batched 12 per config on the Cetran-only split.
- **Offline analysis.** Float16 logits are dumped once and analysed without the model. The test dump went to `/mnt/sandisk` because `/` is nearly full.
- **Precision of the offline sweep.** Each defect was caught by checking against the online numbers and fixed with a regression test. The defects were the slow CUDA `bincount`, TF32 errors that moved FPR@95 by up to 57 points, and saturated equal-width bins.
- **GN offline.** After the Codex review it was withdrawn rather than patched, and the summarizer now guards against it.
- **Long jobs.** They are started detached (`nohup setsid`), after a session interruption killed two background sweeps.
- **Selection bias.** A split picked on one evaluation split must be confirmed on data it was not picked on before it is reported.
- **Feature layer and resemblance.**
  - The features are `pe_features`, the semantic classifier's input, captured by wrapping `_P3FormerHead.init_inputs`.
  - Resemblance is a cosine kNN (k = 10, with k = 50 as a check) against a class-balanced bank of the evaluation set's own ID points. That bank was the user's choice; `--reference` adds a cross-set bank.
  - A class missing from a bank is reported as NaN, never 0.
  - The hypothesis test is correlational only, also the user's choice.
- **Float32 Group MSP.** The divided-mass tool checks against the sweep only on well-conditioned splits (δ95 ≥ 1e-5), and reports float32 and exact metrics side by side.
- **Git.** Commits and pushes happen only when asked. Pushes to protected branches use the owner bypass of the pull-request rule (see `rules/branches.md`). `CLAUDE.md` lives in `.claude/` and is tracked.

### Open housekeeping

- The docstrings in `p3former/utils/ood_scores.py` still cite the deleted `trash/Done/eval_ood_from_logits.py`. A verbatim port lives in `tests/test_ood_scores.py`.
- Cleanup candidates are still on disk:
  - `work_dirs/less_use/p3former_{1xb2,4xb1}_3x_semantickitti/`, 8.3 GB.
  - The intermediate checkpoints (epochs 5–35) of the two 2xb1 runs, 12.4 GB.
