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

## Progress log (updated 2026-09-29)

Numbers are AUROC / AP / FPR@95 in %. The dates in brackets are the `DOCs.md` entries with the commands and full tables.

### Completed from 2026-08-12 to 2026-09-29

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

### Current status

| Component | Status |
| --- | --- |
| Trained models | `work_dirs/p3former_2xb1_3x_{dso,semantickitti}/epoch_36.pth`. DSO test PQ 46.50, mIoU 48.66. SemanticKITTI val PQ 60.32, against 62.63 for the official checkpoint. |
| Online OOD scoring and metric | Complete; 59 tests pass. Energy is the best flat score, 92.88 / 35.55 / 37.94 on DSO test + Cetran. GN Energy (online, valid) reaches 93.94 / 36.31 / 29.96. |
| Six-group hierarchy ablation | Run on Cetran only. The winner `p_v_hgcno` (96.00 / 47.23 / 18.42, against 90.42 / 28.27 / 32.32 for flat MSP) **does not transfer**. On test its Group MSP is 87.21 / 17.23 / 68.02, against 86.26 / 13.76 / 51.15 for flat MSP. |
| Bipartition sweep, Group family | Done on all three splits; the outputs are in `work_dirs/p3former_2xb1_3x_dso_ood_dump/bipartitions{,_test,_test_cetran}/`. Best on Cetran: `s1.3.17` {bicycle, truck, gate}, 96.84 / 54.34 / 17.13. Best on test and on test + Cetran: `s16` {overhead-bridge}, 94.04 / 40.45 / 28.52 on test + Cetran, against 87.76 / 18.06 / 45.76 for flat MSP. 32 splits beat flat on both Cetran and test. The Cetran~test rank correlation of the improvement is only +0.61. **None is confirmed online yet.** |
| Bipartition sweep, GN family | Offline GN MSP and GN Entropy are invalid, because those scores pile up at an interior value that the bins don't resolve. They are not ranked. Fixing this needs interior-adaptive bins or online runs. |
| Collaborator branch | `origin/duy/ood-baselines` forks from the pre-split history at `90af8ea`, whose counterpart on this branch is `749328a`. It adds `tools/ood_distance.py` (Mahalanobis feature-distance OOD plus a 203-partition sweep, 2026-09-07). It is not merged, and it lacks everything after that commit. |

### Next step: why do some two-group splits beat the flat scores?

The goal is to explain, from the dumps and without the GPU model, why some bipartitions beat the flat scores. Ideally the explanation should predict good splits without a sweep.

**Working hypothesis.** With two groups, Group MSP = −max(P_A, 1 − P_A). A point scores as OOD only when its probability mass is divided across the boundary, and confusion between classes on the same side is absorbed. A split should therefore help when two things hold:

- it absorbs the within-ID confusions behind flat MSP's false positives;
- OOD points still divide their mass across it.

**First test case: the divided-mass ("straddling") table.** It comes from session `65236754` (2026-09-23) and is not in `DOCs.md`. For a single-class split {c} | rest, a valid point has divided mass when 0.05 < P_c < 0.95, using the softmax of the dumped logits. The percentages are over OOD points and over ID points.

| split | Cetran OOD / ID divided | Cetran improvement | test OOD / ID divided | test improvement |
| --- | --- | --- | --- | --- |
| {truck} | 15.6 % / 0.29 % | +25.3 | 6.4 % / 0.25 % | −2.9 |
| {gate} | 11.4 % / 0.10 % | +17.5 | 0.5 % / 0.07 % | +6.6 |
| {overhead-bridge} | 11.1 % / 0.10 % | +7.0 | 18.2 % / 0.14 % | +34.8 |
| {bicycle} | 1.6 % / 0.04 % | +5.8 | 0.5 % / 0.07 % | −40.1 |
| {vegetation} | 33.0 % / 7.41 % | −40.4 | 38.5 % / 11.96 % | −28.6 |
| {building} | 52.3 % / 2.80 % | not sampled | 51.4 % / 2.56 % | not sampled |

What the table does not explain yet:

- On test, {bicycle} has a clean OOD/ID ratio yet scores −40.1, with FPR@95 at 89.20.
- ID points outnumber OOD points about 33× on Cetran and 63× on test, so absolute counts may matter more than percentages. For building on test, about 26 M ID points have divided mass, against 8 M OOD points.

**Analysis plan** (spec: `docs/superpowers/specs/2026-09-29-divided-mass-resemblance-design.md`). Both items are reported on Cetran, test and test + Cetran, for the single-class splits and for the robust splits (improvement > 0 on all three sets).

1. **Extend the measure.** Compute divided mass for all 24 single-class splits and all 500 bipartitions, with P_A = the summed mass of the smaller group. Relate the OOD and ID divided counts to ΔAUROC, ΔAP and ΔFPR@95 on each split, and vary the 0.05/0.95 threshold. Compare every split with flat MSP at the same threshold, and draw bubble charts in the style of `trash/bubble_chart.py`.
2. **Resemblance in feature space.** Measure how much the OOD points overlap each ID class in the penultimate features (`pe_features`, the input of the semantic classifier). The measure is a kNN share against the evaluation split's own ID points. Then test the hypothesis that the best split puts the classes the OOD objects resemble on one side, against the alternative from item 1 that it cuts through them. The test is correlational only.

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
- **Git.** Commits and pushes happen only when asked. Pushes to protected branches use the owner bypass of the pull-request rule (see `rules/branches.md`). `CLAUDE.md` lives in `.claude/` and is tracked.

### Open housekeeping

- The docstrings in `p3former/utils/ood_scores.py` still cite the deleted `trash/Done/eval_ood_from_logits.py`. A verbatim port lives in `tests/test_ood_scores.py`.
- Cleanup candidates are still on disk:
  - `work_dirs/less_use/p3former_{1xb2,4xb1}_3x_semantickitti/`, 8.3 GB.
  - The intermediate checkpoints (epochs 5–35) of the two 2xb1 runs, 12.4 GB.
